//! `dream-tts-serve` — the HTTP surface, wire-compatible with the Python service it
//! replaces.
//!
//! The Python service (`CosyVoice/serve.py`, uvicorn on `PORT`, default 3003) is the
//! contract clients already speak, so this reproduces it rather than inventing a new one:
//! same paths, same request bodies, same response headers, same auth. Point a client at
//! this instead and it should not notice, except in latency.
//!
//! # What this does not carry over, and why
//!
//! Most of that service's complexity exists to work around PyTorch. `run.sh` sets
//! `TTS_WORKER_MAX_GROWTH_MB`, `MAX_FOOTPRINT_MB`, `MAX_REQUESTS` and `IDLE_SECONDS`, and
//! its own comment says why: *"PyTorch's MPS backend never frees its compiled-graph cache,
//! so ending the process is the only way to reclaim it."* Hence a subprocess worker, a
//! recycle budget, a supervisor, and a ~15 s reload whenever the budget trips.
//!
//! None of that applies here. There is no MPSGraph cache, memory is flat by construction,
//! and both engines are held resident in this process for its lifetime. So the supervisor,
//! the worker, `modelWorker` bookkeeping and the reload cost are all simply gone — the
//! endpoints that reported on them still exist and answer honestly, they just have much
//! less to say.
//!
//! Two things are **not implemented** and say so with `501` rather than pretending:
//! the Python service's job routes (`/v1/tts-jobs`, superseded here by `/v1/jobs`) and
//! forced alignment (`/v1/alignment-jobs`).
//! Alignment in particular runs a separate whisper environment in the Python service; it
//! is a subprocess call from here, not a port. `GET /` lists what is live.
//!
//! # Two audiences on one route
//!
//! `GET /` answers JSON to a client and an HTML page to a browser, chosen by `Accept`. A
//! local service that cannot explain itself in the browser someone inevitably points at
//! it is a service whose routes get guessed at; and a JSON blob is the wrong thing to hand
//! a person who typed the address by hand. Neither audience is made to read the other's
//! format. `?format=json` and `?format=html` override the negotiation.
//!
//! # Concurrency
//!
//! One GPU, so synthesis is serialised behind a permit rather than run concurrently —
//! two requests interleaving on one Metal queue would make both slower and neither
//! faster. Requests queue; the semaphore is the queue. Synthesis itself is blocking and
//! runs on `spawn_blocking` so it never occupies an async worker.

mod jobs;
mod stream;

use anyhow::{Context, Result};
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;
use tts_core::{Engine, EngineConfig, Sampling, SynthesisRequest, Voice};

/// The Python service's default. Overridable the same way: `PORT=…`.
const DEFAULT_PORT: u16 = 3003;

#[derive(Parser)]
#[command(
    name = "dream-tts-serve",
    about = "HTTP TTS, wire-compatible with the Python service"
)]
struct Args {
    /// Settings file; same resolution as the CLI's `--config`.
    #[arg(long, value_name = "PATH")]
    config: Option<std::path::PathBuf>,

    /// Defaults to `$PORT`, then 3003.
    #[arg(long)]
    port: Option<u16>,
    /// Loopback by default. A model this size should not be exposed casually.
    /// Config: `serve.host`.
    #[arg(long)]
    host: Option<String>,
    /// Engine id; `tts engines` lists them. Defaults to the registry's own choice
    /// rather than a string here, so the two cannot drift apart.
    #[arg(long)]
    engine: Option<String>,
    /// Voice asset directory used when a request does not name one. Defaults to the
    /// asset shipped for the selected engine, and is unused by an engine that cannot clone.
    #[arg(long)]
    voice: Option<String>,
    #[arg(long)]
    model_root: Option<String>,
    /// Engine setting, e.g. `--set voice=am_michael` for kokoro's built-in voices.
    #[arg(long = "set", value_parser = parse_override)]
    overrides: Vec<(String, std::path::PathBuf)>,
    #[arg(long)]
    quant: Option<String>,
    #[arg(long)]
    cpu: bool,
    /// Per-request character ceiling, mirroring the Python service's `TTS_MAX_CHARS`.
    /// Config: `serve.max_chars`.
    #[arg(long)]
    max_chars: Option<usize>,
    /// Segment budget handed to the engine's own segmenter. Config: `serve.segment_chars`.
    #[arg(long)]
    segment_chars: Option<usize>,
    /// Do not take the advisory GPU lock. See `tts_core::lock`.
    #[arg(long)]
    no_gpu_lock: bool,
    /// Fallback when `DREAM_TTS_API_KEY` and `TTS_API_KEY` are both unset, mirroring the
    /// Python service's `.api_key`.
    #[arg(long, default_value = ".api_key")]
    api_key_file: String,
}

pub struct App {
    engine: Box<dyn Engine>,
    /// `None` for an engine that cannot clone: its voices are inside the checkpoint, so
    /// there is no asset to hold and a request must not be given one.
    voice: Option<Voice>,
    engine_id: String,
    sample_rate: u32,
    max_chars: usize,
    segment_chars: usize,
    api_key: Option<String>,
    /// One GPU: synthesis takes this before running.
    gpu: tokio::sync::Semaphore,
    started: Instant,
    /// The default voice asset's path, reported so a browser can see what it will get.
    voice_name: String,
    /// Which settings file this process read, if any. Shown on the HTML page because
    /// "why is it on the wrong engine" is almost always this.
    config_source: Option<std::path::PathBuf>,
    /// The bound port, so the page's curl example is copy-pasteable as printed.
    port: u16,
    /// The narration queue. Its records live in the data directory rather than here, so a
    /// process that is not this one can answer "is something already narrating?".
    jobs: jobs::Jobs,
}

// ---------------------------------------------------------------- errors

/// An error that serialises the way FastAPI's `HTTPException` does, so a client's
/// error handling does not have to change either.
pub struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "detail": self.1 }))).into_response()
    }
}

pub fn bad(code: StatusCode, msg: impl Into<String>) -> ApiError {
    ApiError(code, msg.into())
}

// ---------------------------------------------------------------- auth

/// `Authorization: Bearer <key>` or `X-API-Key: <key>`, matching the Python service.
///
/// Compared with a length-checked constant-time equality rather than `==`, for the same
/// reason `serve.py` reaches for `secrets.compare_digest`.
pub fn require_key(app: &App, headers: &HeaderMap) -> Result<(), ApiError> {
    let Some(expected) = app.api_key.as_deref() else {
        // The Python service refuses rather than running open. Same here: a missing key
        // is a misconfiguration, and defaulting to "no auth" is the wrong guess.
        return Err(bad(
            StatusCode::SERVICE_UNAVAILABLE,
            "server API key not configured — set DREAM_TTS_API_KEY",
        ));
    };
    let provided = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| {
            let (scheme, rest) = v.split_at(v.len().min(7));
            scheme.eq_ignore_ascii_case("bearer ").then(|| rest.trim())
        })
        .or_else(|| {
            headers
                .get("x-api-key")
                .and_then(|v| v.to_str().ok())
                .map(str::trim)
        });

    match provided {
        Some(got) if constant_time_eq(got.as_bytes(), expected.as_bytes()) => Ok(()),
        _ => Err(ApiError(
            StatusCode::UNAUTHORIZED,
            "invalid or missing API key".into(),
        )),
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

// ---------------------------------------------------------------- request body

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TtsRequest {
    pub text: String,
    #[serde(default = "default_mode")]
    pub mode: String,
    #[serde(default)]
    pub instruct_text: Option<String>,
    #[serde(default = "default_speed")]
    pub speed: f32,
    /// Not in the Python schema. Additive, so an existing client is unaffected: it names
    /// a voice asset directory per request instead of using the one this was started with.
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
}

/// A request body, as JSON or as the text itself.
///
/// JSON unless the caller says `text/plain`, which keeps the Python service's contract
/// exactly: a client that sends no content-type still gets JSON parsing.
///
/// `text/plain` exists so a shell script can post a narration file with
/// `curl --data-binary @chapter.txt` instead of building JSON around arbitrary prose.
/// `narrate-book.sh` did that with a `python3 -c "import json…"` per chapter, which is a
/// Python dependency on the critical path of the one feature this project promises can run
/// with nothing but curl — and quoting arbitrary text into JSON from bash is a bug waiting
/// for an apostrophe. `X-Seed` and `X-Voice` carry what the JSON fields would have.
fn parse_request(headers: &HeaderMap, body: String) -> Result<TtsRequest, ApiError> {
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !content_type.starts_with("text/plain") {
        return serde_json::from_str(&body).map_err(|e| {
            bad(
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("invalid request body: {e}"),
            )
        });
    }
    let header_str = |name: &str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
    };
    let seed = match header_str("x-seed") {
        Some(raw) => Some(raw.trim().parse::<u64>().map_err(|_| {
            bad(
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("X-Seed is not a number: {raw}"),
            )
        })?),
        None => None,
    };
    Ok(TtsRequest {
        text: body,
        mode: default_mode(),
        instruct_text: None,
        speed: default_speed(),
        voice: header_str("x-voice"),
        seed,
    })
}

fn default_mode() -> String {
    "zero_shot".into()
}
fn default_speed() -> f32 {
    1.0
}

/// The validation `serve.py::_validate` does, plus the knobs this port cannot honour.
///
/// Rejecting rather than ignoring is deliberate and is the rule the engine trait already
/// states: an engine documents the controls it ignores instead of silently accepting them.
/// Quietly returning `speed: 1.0` audio to a client that asked for 1.5 is worse than a 501,
/// because nothing in the response says the request was not honoured.
fn validate(app: &App, req: &TtsRequest) -> Result<(), ApiError> {
    if req.text.trim().is_empty() {
        return Err(bad(StatusCode::BAD_REQUEST, "text is empty"));
    }
    if req.text.chars().count() > app.max_chars {
        return Err(bad(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "text too long: {} chars > max_chars={}. Split into multiple requests.",
                req.text.chars().count(),
                app.max_chars
            ),
        ));
    }
    match req.mode.as_str() {
        "zero_shot" => {}
        m @ ("instruct" | "cross_lingual") => {
            return Err(bad(
                StatusCode::NOT_IMPLEMENTED,
                format!(
                    "mode='{m}' is not implemented in the Rust port — only 'zero_shot'. \
                     The port has no instruction-prompt path; see docs/reference.md#porting-traps."
                ),
            ))
        }
        other => {
            return Err(bad(
                StatusCode::BAD_REQUEST,
                format!("unknown mode '{other}'"),
            ))
        }
    }
    if (req.speed - 1.0).abs() > f32::EPSILON {
        return Err(bad(
            StatusCode::NOT_IMPLEMENTED,
            format!(
                "speed={} is not implemented — the port synthesizes at 1.0 only, and \
                 returning unmodified audio would misreport the request as honoured.",
                req.speed
            ),
        ));
    }
    if req.instruct_text.is_some() {
        return Err(bad(
            StatusCode::NOT_IMPLEMENTED,
            "instruct_text is not implemented (mode='instruct' is unsupported)",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------- synthesis

struct Rendered {
    wav: Vec<u8>,
    seconds: f64,
    wall: f64,
    stages: Vec<(&'static str, f64)>,
}

/// The voice a request runs with: the one it named, else the process default. An engine
/// that cannot clone rejects a named voice here rather than after the text is segmented.
pub fn request_voice(app: &Arc<App>, named: &Option<String>) -> Result<Option<Voice>, ApiError> {
    match named {
        None => Ok(app.voice.clone()),
        Some(dir) if app.voice.is_none() => Err(bad(
            StatusCode::BAD_REQUEST,
            format!(
                "engine `{}` cannot clone, so it takes no voice asset",
                app.engine_id
            ),
        )),
        Some(dir) => Voice::load(dir)
            .map(Some)
            .map_err(|e| bad(StatusCode::BAD_REQUEST, format!("loading voice {dir}: {e}"))),
    }
}

pub fn with_optional_voice(request: SynthesisRequest, voice: Option<Voice>) -> SynthesisRequest {
    match voice {
        Some(v) => request.with_voice(v),
        None => request,
    }
}

fn parse_override(s: &str) -> Result<(String, std::path::PathBuf), String> {
    let (k, v) = s
        .split_once('=')
        .ok_or_else(|| format!("expected key=value, got {s:?}"))?;
    Ok((k.to_string(), std::path::PathBuf::from(v)))
}

async fn render(app: &Arc<App>, req: TtsRequest) -> Result<Rendered, ApiError> {
    let voice = request_voice(app, &req.voice)?;

    let text = req.text.clone();
    let seed = req.seed;

    // One GPU: queue rather than contend. Held across the blocking call below, so the
    // permit — not the thread pool — is what bounds concurrent synthesis.
    let _permit = app
        .gpu
        .acquire()
        .await
        .map_err(|_| bad(StatusCode::SERVICE_UNAVAILABLE, "server shutting down"))?;

    let app2 = Arc::clone(app);
    let started = Instant::now();
    let out = tokio::task::spawn_blocking(move || {
        let mut request = with_optional_voice(SynthesisRequest::new(text), voice);
        request.max_chars = app2.segment_chars;
        if let Some(s) = seed {
            request.sampling = Sampling {
                seed: s,
                ..request.sampling
            };
        }
        app2.engine.validate(&request)?;
        app2.engine.synthesize(&request)
    })
    .await
    .map_err(|e| {
        bad(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("worker panic: {e}"),
        )
    })?
    .map_err(|e| bad(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;

    let wall = started.elapsed().as_secs_f64();
    let seconds = out.audio.seconds();
    // One encoder, in tts-core: a second copy here silently disagreed with the CLI
    // by 1 LSB because it truncated where that one rounds.
    let wav = tts_core::wav::to_bytes(&out.audio);
    Ok(Rendered {
        wav,
        seconds,
        wall,
        stages: out.stats.stages,
    })
}

// ---------------------------------------------------------------- handlers

async fn post_tts(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    body: String,
) -> Result<Response, ApiError> {
    require_key(&app, &headers)?;
    let req = parse_request(&headers, body)?;
    validate(&app, &req)?;
    let r = render(&app, req).await?;

    let hdr = |n: &'static str, v: String| (HeaderName::from_static(n), HeaderValue::from_str(&v));
    let mut out = Response::builder()
        .header(header::CONTENT_TYPE, "audio/wav")
        .header(header::CONTENT_DISPOSITION, "inline; filename=\"tts.wav\"");
    for (name, value) in [
        hdr("x-audio-seconds", format!("{:.2}", r.seconds)),
        hdr("x-wall-seconds", format!("{:.2}", r.wall)),
        hdr(
            "x-rtf",
            format!(
                "{:.2}",
                if r.seconds > 0.0 {
                    r.wall / r.seconds
                } else {
                    0.0
                }
            ),
        ),
        hdr("x-audio-format", "pcm_s16le_mono".into()),
        // Additive: the per-stage split the CLI prints, so a client can see where the
        // time went without a second request.
        hdr(
            "x-stages",
            r.stages
                .iter()
                .map(|(n, s)| format!("{n}={s:.3}"))
                .collect::<Vec<_>>()
                .join(","),
        ),
    ] {
        if let Ok(v) = value {
            out = out.header(name, v);
        }
    }
    out.body(r.wav.into())
        .map_err(|e| bad(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

/// Raw little-endian 16-bit mono PCM, as the Python service's `/tts/stream` emits.
///
/// **Honest limitation:** the Python service streams from the model as it decodes, so its
/// point is time-to-first-audio. Neither engine here exposes an incremental decode yet
/// (`Capabilities::streaming` is false for both), so this synthesizes fully and then
/// writes the body. The bytes on the wire are identical and a client needs no change —
/// but it does not get the latency benefit, and saying otherwise would be a lie the
/// header cannot carry.
async fn post_tts_stream(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    body: String,
) -> Result<Response, ApiError> {
    let req = parse_request(&headers, body)?;
    validate(&app, &req)?;
    stream::post_tts_stream(app, headers, req).await
}

/// The live-runs view, kept out of the page's `format!` so its braces need no escaping.
///
/// Polling rather than one SSE stream per job: a tab left open for an eight-hour book should
/// cost a request every few seconds, not a connection per run, and a list is what this view
/// is for. The per-segment stream is for a watcher following one run.
const JOBS_SCRIPT: &str = r##"<script>
// The same three endpoints the CLI uses. Polling rather than one SSE stream per job: a
// browser tab left open for an eight-hour book should cost a request every few seconds, not
// an open connection per run, and the list is what this view is for.
const box = document.getElementById('jobs');
const fmt = (s) => {
  if (!isFinite(s) || s < 0) return '—';
  const t = Math.round(s);
  if (t < 60) return t + 's';
  if (t < 3600) return Math.floor(t / 60) + 'm ' + String(t % 60).padStart(2, '0') + 's';
  return Math.floor(t / 3600) + 'h ' + String(Math.floor((t % 3600) / 60)).padStart(2, '0') + 'm';
};
async function refresh() {
  try {
    const r = await fetch('/v1/jobs', { headers: { 'accept': 'application/json' } });
    const { jobs } = await r.json();
    if (!jobs.length) { box.innerHTML = '<p class="lede">No narration runs yet.</p>'; return; }
    box.innerHTML = '<table><tr><th>id</th><th>state</th><th>chapters</th><th>audio</th>' +
      '<th>eta</th><th>source</th></tr>' + jobs.map((j) => {
        const done = j.chapters.filter((c) => c.state === 'done').length;
        const audio = j.chapters.reduce((a, c) => a + (c.audio_seconds || 0), 0);
        const spent = j.chapters.reduce((a, c) => a + (c.wall_seconds || 0), 0);
        const words = j.chapters.reduce((a, c) => a + c.words, 0);
        const wordsDone = j.chapters.filter((c) => c.state === 'done')
          .reduce((a, c) => a + c.words, 0);
        const eta = wordsDone > 0 && spent > 0 ? fmt(spent / wordsDone * (words - wordsDone)) : '—';
        // A queued pause is shown as `pausing`: a control that appears to have done nothing
        // is worse than one that refused.
        let state = j.state;
        if (j.state === 'running' && j.request === 'pause') state = 'pausing';
        if (j.state === 'running' && j.request === 'cancel') state = 'cancelling';
        const cls = state === 'running' ? 'ok' : state === 'failed' ? 'warn'
          : (state === 'paused' || state === 'pausing' || state === 'cancelling') ? 'warn' : 'no';
        const pct = words ? Math.round(100 * wordsDone / words) : 0;
        return `<tr><td class="m">${j.id}</td><td><span class="${cls}">${state}</span></td>` +
          `<td class="m">${done}/${j.chapters.length} <span class="no">(${pct}%)</span></td>` +
          `<td class="m">${fmt(audio)}</td><td class="m">${eta}</td>` +
          `<td>${j.source.replace(/[&<>]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}</td></tr>`;
      }).join('') + '</table>';
  } catch (e) {
    box.innerHTML = '<p class="lede">could not reach /v1/jobs</p>';
  }
}
refresh();
setInterval(refresh, 3000);
</script>
"##;

/// The page a browser gets at `/`.
///
/// Deliberately one self-contained string: no CDN, no build step, no asset routes. A local
/// service that needs the network to explain itself is broken in exactly the situation
/// someone is most likely to be reading it — offline, on a laptop, wondering why a request
/// 401s. It is also why the auth rule and the error shapes are stated here rather than
/// linked to.
///
/// Escaping: everything interpolated is either a number, an engine-supplied `&'static str`,
/// or a filesystem path. Paths are the only caller-influenced values, and they go through
/// `escape_html`.
fn render_index(app: &Arc<App>) -> String {
    let caps = app.engine.capabilities();
    let uptime = app.started.elapsed().as_secs();
    let languages = match caps.languages {
        Some(l) => format!("{} <em>(closed list)</em>", l.join(", ")),
        None => "unrestricted".to_string(),
    };
    let auth = match &app.api_key {
        Some(_) => r#"<span class="ok">required</span> — send <code>X-API-Key</code>"#,
        None => r#"<span class="warn">no key configured</span> — authenticated routes answer 503"#,
    };
    // Bound before the template: a `format!` inside a `format!` argument list is both a
    // clippy error and harder to read than a named binding.
    let addr = format!("127.0.0.1:{}", app.port);
    let config = match &app.config_source {
        Some(p) => format!("<code>{}</code>", escape_html(&p.display().to_string())),
        None => "none — built-in defaults".to_string(),
    };

    format!(
        r#"<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>dream-tts-serve</title>
<style>
  :root {{
    --bg: #fbfbfa; --fg: #1a1a19; --dim: #6b6b68; --line: #e2e2df;
    --card: #ffffff; --accent: #1f6f4a; --warn: #8a5a00; --code: #f4f4f2;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #16161a; --fg: #e8e8e4; --dim: #9a9a94; --line: #2c2c31;
      --card: #1d1d22; --accent: #6fd39b; --warn: #e0b060; --code: #24242a;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 2.5rem 1.25rem 4rem; background: var(--bg); color: var(--fg);
    font: 15px/1.6 ui-sans-serif, -apple-system, "Helvetica Neue", sans-serif;
  }}
  main {{ max-width: 62rem; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 .25rem; letter-spacing: -.01em; }}
  h2 {{ font-size: .8rem; text-transform: uppercase; letter-spacing: .08em;
        color: var(--dim); margin: 2.5rem 0 .75rem; font-weight: 600; }}
  p.lede {{ color: var(--dim); margin: 0 0 2rem; }}
  code {{ background: var(--code); padding: .1em .35em; border-radius: 3px;
          font: 13px/1.5 ui-monospace, "SF Mono", Menlo, monospace; }}
  pre {{ background: var(--code); padding: .9rem 1rem; border-radius: 6px;
         overflow-x: auto; margin: 0; }}
  pre code {{ background: none; padding: 0; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ text-align: left; padding: .55rem .6rem; border-bottom: 1px solid var(--line);
            vertical-align: top; }}
  th {{ font-size: .75rem; text-transform: uppercase; letter-spacing: .06em;
        color: var(--dim); font-weight: 600; }}
  td.m {{ font: 13px ui-monospace, "SF Mono", Menlo, monospace; white-space: nowrap; }}
  .grid {{ display: grid; gap: .75rem; grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr)); }}
  .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 8px;
           padding: .8rem .9rem; }}
  .card dt {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .06em;
              color: var(--dim); margin: 0 0 .3rem; }}
  .card dd {{ margin: 0; font-size: 1rem; font-weight: 600;
              font-family: ui-monospace, "SF Mono", Menlo, monospace; word-break: break-word; }}
  .ok {{ color: var(--accent); font-weight: 600; }}
  .warn {{ color: var(--warn); font-weight: 600; }}
  .no {{ color: var(--dim); }}
  footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--line);
            color: var(--dim); font-size: .85rem; }}
  a {{ color: var(--accent); }}
</style>
</head><body><main>

<h1>dream-tts-serve</h1>
<p class="lede">Local text-to-speech over HTTP. One engine, resident, on this machine.
This page is what a browser gets; every other client gets JSON from the same URL.</p>

<div class="grid">
  <div class="card"><dt>engine</dt><dd>{engine}</dd></div>
  <div class="card"><dt>sample rate</dt><dd>{sample_rate} Hz</dd></div>
  <div class="card"><dt>uptime</dt><dd>{uptime}s</dd></div>
  <div class="card"><dt>max chars / request</dt><dd>{max_chars}</dd></div>
</div>

<h2>this engine</h2>
<table>
  <tr><th>model</th><td>{description}</td></tr>
  <tr><th>languages</th><td>{languages}</td></tr>
  <tr><th>weight formats</th><td class="m">{quantization}</td></tr>
  <tr><th>streaming</th><td>{streaming}</td></tr>
  <tr><th>default voice</th><td class="m">{voice}</td></tr>
  <tr><th>segment budget</th><td>{segment_chars} characters</td></tr>
  <tr><th>settings from</th><td>{config}</td></tr>
  <tr><th>auth</th><td>{auth}</td></tr>
</table>

<h2 id="runs">narration runs</h2>
<div id="jobs"><p class="lede">loading…</p></div>

<h2>routes</h2>
<table>
  <tr><th>method</th><th>path</th><th></th></tr>
  <tr><td class="m">GET</td><td class="m"><a href="/">/</a></td>
      <td>This page, or JSON. <code>?format=json</code> forces JSON.</td></tr>
  <tr><td class="m">GET</td><td class="m"><a href="/health">/health</a></td>
      <td>Liveness. Answers only once the model is resident, so a reply means ready.</td></tr>
  <tr><td class="m">GET</td><td class="m"><a href="/v1/capabilities">/v1/capabilities</a></td>
      <td>Engine, rates, languages, weight formats.</td></tr>
  <tr><td class="m">POST</td><td class="m">/tts</td>
      <td>WAV body, PCM s16le mono. Needs <code>X-API-Key</code>.</td></tr>
  <tr><td class="m">POST</td><td class="m">/tts/stream</td>
      <td><strong>Incremental.</strong> Raw PCM as each segment lands, chunked, so first
          audio arrives after one segment rather than after all of them.</td></tr>
  <tr><td class="m">GET</td><td class="m"><a href="/v1/jobs">/v1/jobs</a></td>
      <td>Narration runs. <code>POST</code> submits one; the same document resubmitted
          resumes it.</td></tr>
  <tr><td class="m">GET</td><td class="m">/v1/jobs/&lt;id&gt;/events</td>
      <td>Server-sent events: per-segment progress and every state change.</td></tr>
  <tr><td class="m">POST</td><td class="m">/v1/jobs/&lt;id&gt;/&lt;action&gt;</td>
      <td><code>pause</code>, <code>pause-now</code>, <code>resume</code>,
          <code>cancel</code>, <code>cancel-now</code>.</td></tr>
  <tr><td class="m">*</td><td class="m">/v1/tts-jobs, /v1/alignment-jobs</td>
      <td class="no">501. Superseded by <code>/v1/jobs</code>; forced alignment lives
          only in the Python service.</td></tr>
</table>

<h2>synthesize</h2>
<pre><code>curl -X POST http://{addr}/tts \
     -H 'X-API-Key: &lt;your key&gt;' \
     -H 'content-type: application/json' \
     -d '{{"text":"Hello from Rust.","seed":7}}' \
     -o out.wav -D headers.txt</code></pre>

<h2>request body</h2>
<table>
  <tr><th>field</th><th>type</th><th></th></tr>
  <tr><td class="m">text</td><td class="m">string</td>
      <td>Required. Up to {max_chars} characters.</td></tr>
  <tr><td class="m">voice</td><td class="m">string</td>
      <td>A voice asset directory. Omit for the default above; no restart needed.</td></tr>
  <tr><td class="m">seed</td><td class="m">integer</td>
      <td>Makes a render reproducible within this implementation.</td></tr>
</table>

<h2>response headers</h2>
<p class="lede" style="margin:0 0 .75rem">Every response carries its own cost, so a client
sees where the time went without a second request.</p>
<table>
  <tr><td class="m">x-audio-seconds</td><td>audio produced</td></tr>
  <tr><td class="m">x-wall-seconds</td><td>time taken</td></tr>
  <tr><td class="m">x-rtf</td><td>the ratio of the two</td></tr>
  <tr><td class="m">x-stages</td><td>per-stage split, e.g. <code>talker=10.3,codec=2.9</code></td></tr>
</table>

{script}

<footer>
One GPU, so requests queue rather than interleave — two renders on one Metal queue make
both slower and neither faster. Synthesis is blocking and runs off the async workers.
<br>Docs: <a href="https://github.com/drmhse/dream-tts">github.com/drmhse/dream-tts</a>
</footer>

</main></body></html>
"#,
        engine = caps.id,
        sample_rate = caps.sample_rate,
        uptime = uptime,
        max_chars = app.max_chars,
        segment_chars = app.segment_chars,
        description = caps.description,
        languages = languages,
        quantization = caps.quantization.join(", "),
        streaming = if caps.streaming {
            "yes"
        } else {
            "not implemented in this port"
        },
        voice = escape_html(&app.voice_name),
        config = config,
        auth = auth,
        addr = addr,
        script = JOBS_SCRIPT,
    )
}

/// The five characters that matter in element text and double-quoted attributes.
fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// Which representation `GET /` should return.
///
/// `?format=` wins, because a browser cannot easily send a different `Accept` and a script
/// should not have to. Otherwise: HTML only if the client asked for it *by name*. A
/// wildcard `*/*` — what curl sends — means "anything", and for a machine that is JSON.
fn wants_html(headers: &HeaderMap, query: &str) -> bool {
    for pair in query.split('&') {
        match pair.split_once('=') {
            Some(("format", "html")) => return true,
            Some(("format", _)) => return false,
            _ => {}
        }
    }
    headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|a| a.contains("text/html"))
}

async fn get_root(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    axum::extract::RawQuery(query): axum::extract::RawQuery,
) -> Response {
    if wants_html(&headers, query.as_deref().unwrap_or("")) {
        return (
            [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
            render_index(&app),
        )
            .into_response();
    }
    root_json(&app).into_response()
}

fn root_json(app: &Arc<App>) -> Json<serde_json::Value> {
    Json(json!({
        "service": "dream-tts",
        "engine": app.engine_id,
        "sample_rate": app.sample_rate,
        "max_chars": app.max_chars,
        "uptime_seconds": app.started.elapsed().as_secs(),
        "endpoints": [
            "/health", "/v1/capabilities", "POST /tts", "POST /tts/stream",
            "GET /v1/jobs", "POST /v1/jobs", "GET /v1/jobs/{id}",
            "GET /v1/jobs/{id}/events", "POST /v1/jobs/{id}/{action}",
        ],
        "not_implemented": {
            "POST /v1/tts-jobs": "the Python service's job shape; this one serves /v1/jobs",
            "POST /v1/alignment-jobs": "forced alignment (needs a whisper environment)",
            "GET /v1/artifacts/{job_id}/{filename}": "job artifacts",
        },
        "guide": "https://github.com/drmhse/dream-tts",
        "html": "GET / with Accept: text/html, or /?format=html",
    }))
}

async fn get_health(State(app): State<Arc<App>>) -> Response {
    // Models are loaded before the listener binds, so if this answers at all it is ready.
    // The Python service can be up-but-loading because its worker starts lazily.
    (
        StatusCode::OK,
        Json(json!({
            "status": "ok",
            "model_loaded": true,
            "engine": app.engine_id,
            "device": if cfg!(feature = "metal") { "metal" } else { "cpu" },
            "uptime_seconds": app.started.elapsed().as_secs(),
        })),
    )
        .into_response()
}

async fn get_capabilities(State(app): State<Arc<App>>) -> Json<serde_json::Value> {
    let caps = app.engine.capabilities();
    Json(json!({
        "apiVersion": "v1",
        "service": "dream-tts",
        "engine": caps.id,
        "description": caps.description,
        "sampleRate": caps.sample_rate,
        "frameRate": caps.frame_rate,
        "quantization": caps.quantization,
        // `null` is unrestricted, not unknown. A client that cares which languages it
        // may send should not have to parse the description to find out.
        "languages": caps.languages,
        "maxCharsPerRequest": app.max_chars,
        "maxCharsPerSegment": app.segment_chars,
        "modes": ["zero_shot"],
        "audio": {"container": "wav", "encoding": "pcm_s16le", "channels": 1},
        "streaming": caps.streaming,
        "cloning": format!("{:?}", caps.cloning),
        "jobs": {"durable": true, "route": "/v1/jobs"},
        "alignment": {"available": false, "reason": "needs a separate whisper environment"},
        // No supervisor, no recycle budget, no reload: there is no MPSGraph cache to
        // reclaim, so the model stays resident for the process lifetime.
        "modelWorker": {"inProcess": true, "recycles": 0, "reason": "memory is flat by construction"},
    }))
}

async fn not_implemented(Path(rest): Path<String>) -> ApiError {
    bad(
        StatusCode::NOT_IMPLEMENTED,
        format!(
            "/v1/{rest} is not implemented by tts-serve. Narration runs are served at \
             /v1/jobs, not the Python service's route; forced alignment and artifact \
             retrieval live only there. GET / lists what this one serves."
        ),
    )
}

// ---------------------------------------------------------------- main

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let cfg = tts_core::Config::load(args.config.as_deref())?;
    let serve = cfg.settings.serve.clone().unwrap_or_default();

    let port = args
        .port
        .or_else(|| std::env::var("PORT").ok()?.parse().ok())
        .or(serve.port)
        .unwrap_or(DEFAULT_PORT);
    let host = args
        .host
        .clone()
        .or(serve.host.clone())
        .unwrap_or_else(|| "127.0.0.1".to_string());
    let max_chars = args.max_chars.or(serve.max_chars).unwrap_or(1200);
    let segment_chars = args.segment_chars.or(serve.segment_chars).unwrap_or(220);
    // Same resolution order as `serve.py::_load_api_key`: the environment, then a
    // `.api_key` file beside the service. Matching it means an existing deployment can
    // point at this binary without moving its secret.
    // `DREAM_TTS_API_KEY` first; `TTS_API_KEY` still works, because an existing
    // deployment's environment is part of the wire compatibility this service claims.
    let api_key = std::env::var("DREAM_TTS_API_KEY")
        .or_else(|_| std::env::var("TTS_API_KEY"))
        .ok()
        .or_else(|| std::fs::read_to_string(&args.api_key_file).ok())
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty());
    if api_key.is_none() {
        eprintln!(
            "warning: no API key found (checked $DREAM_TTS_API_KEY, $TTS_API_KEY and {}) — every\n\
             \x20        route will answer 503, exactly as the Python service does. Point\n\
             \x20        --api-key-file at the existing deployment's .api_key so clients\n\
             \x20        need no change.",
            args.api_key_file
        );
    }

    let engine_id = match args.engine.clone().or_else(|| cfg.settings.engine.clone()) {
        Some(id) => id,
        None => {
            if let Some(caveat) = tts_engines::default_caveat() {
                eprintln!("note: {caveat}");
            }
            tts_engines::default_id().to_string()
        }
    };
    let clones = tts_engines::catalogue()
        .into_iter()
        .find(|c| c.id == engine_id)
        .is_some_and(|c| c.cloning == tts_core::Cloning::PrecomputedAsset);
    // Same lenient resolution as the CLI: as typed, else against the install root, so a
    // service started from anywhere finds the voices that shipped with it.
    let voice_path = match args
        .voice
        .clone()
        .or_else(|| cfg.settings.voice.as_ref().map(|p| p.display().to_string()))
    {
        Some(v) if !clones => anyhow::bail!(
            "engine `{engine_id}` cannot clone, so --voice {v} has nothing to load. \
             Its own voices are selected with --set voice=<name>"
        ),
        Some(v) => Some(cfg.locate_or_err(std::path::Path::new(&v), "voice asset")?),
        None if clones => Some(cfg.locate_or_err(
            std::path::Path::new(tts_engines::default_voice(&engine_id)),
            &format!("shipped voice for `{engine_id}`"),
        )?),
        None => None,
    }
    .map(|p| p.display().to_string());
    let root = match &args.model_root {
        Some(p) => std::path::PathBuf::from(p),
        None => cfg.data_path(tts_engines::default_root(&engine_id)),
    };
    let mut overrides = BTreeMap::new();
    for (k, v) in &args.overrides {
        overrides.insert(k.clone(), v.clone());
    }
    let config = EngineConfig {
        model_root: root,
        quant: args.quant.clone().or_else(|| cfg.settings.quant.clone()),
        cpu: args.cpu,
        overrides,
    };

    // Before the weights: the load itself is most of the memory pressure this guards, and
    // the lock is held for the process lifetime because the engine stays resident.
    let want_lock = !args.no_gpu_lock && cfg.gpu_lock();
    let _gpu_lock = tts_core::GpuLock::maybe(
        want_lock,
        &cfg.lock_path(),
        &format!("dream-tts-serve --engine {engine_id} on port {port}"),
    )?;

    eprintln!("loading engine `{engine_id}` from {:?}…", config.model_root);
    let load = Instant::now();
    let engine = tts_engines::load(&engine_id, &config)
        .with_context(|| format!("loading engine {engine_id}"))?;
    let voice = match &voice_path {
        Some(p) => Some(Voice::load(p).with_context(|| format!("loading voice asset {p}"))?),
        None => None,
    };
    let caps = engine.capabilities();
    eprintln!(
        "loaded in {:.2}s — {} at {} Hz",
        load.elapsed().as_secs_f64(),
        caps.id,
        caps.sample_rate
    );

    let app = Arc::new(App {
        engine_id: caps.id.to_string(),
        sample_rate: caps.sample_rate,
        engine,
        voice,
        max_chars,
        segment_chars,
        api_key,
        gpu: tokio::sync::Semaphore::new(1),
        started: Instant::now(),
        // What the page and a job record report. An engine without assets names the
        // built-in voice it was started with instead of a path that does not exist.
        voice_name: voice_path.clone().unwrap_or_else(|| {
            let name = args
                .overrides
                .iter()
                .find(|(k, _)| k == "voice")
                .map(|(_, v)| v.display().to_string())
                .unwrap_or_else(|| "engine default".to_string());
            format!("{name} (built-in)")
        }),
        config_source: cfg.source.clone(),
        port,
        jobs: jobs::Jobs::open(&cfg.data_dir)?,
    });

    // One chapter at a time, behind the same GPU permit one-off requests take.
    tokio::spawn(jobs::run(Arc::clone(&app)));

    let router = Router::new()
        .route("/", get(get_root))
        .route("/health", get(get_health))
        .route("/v1/capabilities", get(get_capabilities))
        .route("/tts", post(post_tts))
        .route("/tts/stream", post(post_tts_stream))
        .route("/v1/jobs", get(jobs::get_jobs).post(jobs::post_jobs))
        .route("/v1/jobs/:id", get(jobs::get_job))
        .route("/v1/jobs/:id/events", get(jobs::get_job_events))
        .route("/v1/jobs/:id/:action", post(jobs::post_job_control))
        // Answer the unimplemented v1 surface explicitly. A 404 would look like a wrong
        // URL; a 501 says the route is real and this server does not serve it.
        .route("/v1/*rest", get(not_implemented).post(not_implemented))
        .with_state(Arc::clone(&app));

    let addr = format!("{host}:{port}");
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            // The common case on a developer machine, and the one where a bare
            // "Address already in use" sends someone hunting. Name the fix.
            anyhow::bail!(
                "port {port} is already in use.\n                   Something else holds {addr} — the Python service, another dream-tts-serve, \
                 or an unrelated app.\n                   Find it:      lsof -nP -iTCP:{port} -sTCP:LISTEN\n                   Use another:  --port <n>, PORT=<n>, or \"serve\": {{\"port\": <n>}} in \
                 dream-tts.json"
            );
        }
        Err(e) => return Err(e).with_context(|| format!("binding {addr}")),
    };
    eprintln!("listening on http://{addr}  (open it in a browser for the route reference)");

    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            eprintln!("\nshutting down");
        })
        .await?;
    Ok(())
}
