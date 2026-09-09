//! `dream-tts book` and `dream-tts jobs`: starting a long narration and watching it.
//!
//! # Why a client and a server at all
//!
//! A thousand-page book is hours of synthesis, and three things follow from that length.
//! It has to survive being interrupted, so the state cannot live only in the process doing
//! the work. It has to be watchable at a granularity worth watching, so something has to be
//! emitting progress while it runs. And a second run must not be started on top of the
//! first by accident, so a process that did not start the work has to be able to find it.
//!
//! All three want the work to outlive the terminal that asked for it. So the engine and the
//! queue live in `dream-tts-serve` and this is a client — which is also what makes the
//! observatory story real: this is one client, and a status bar or a script is another,
//! reading the same job files and the same event stream.
//!
//! # Discovery before work
//!
//! Before submitting, this reports any run already in flight. The store is files under the
//! data directory precisely so that this question is answerable with no server up, which is
//! exactly the moment someone is about to start a second one.

use crate::ui;
use anyhow::{bail, Context, Result};
use clap::Args;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tts_core::Config;
use tts_jobs::{Job, State, Store};

#[derive(Args)]
pub struct Book {
    /// The document, or a directory of markdown chapters.
    pub source: PathBuf,
    /// Where the narration text and audio go.
    #[arg(long, value_name = "DIR")]
    pub out: PathBuf,
    /// Voice asset directory. Defaults to the engine's shipped voice.
    #[arg(long, value_name = "PATH")]
    pub voice: Option<PathBuf>,
    /// Weight format. Rarely worth setting: qwen3tts already defaults to `f16`, the only
    /// format that batches across segments, which is worth 4.5x on book-length text.
    #[arg(long)]
    pub quant: Option<String>,
    #[arg(long)]
    pub seed: Option<u64>,
    /// Submit and exit instead of watching. The run continues in the service.
    #[arg(long)]
    pub detach: bool,
    /// Do not start a service if none is running.
    #[arg(long)]
    pub no_serve: bool,
}

#[derive(Args)]
pub struct Jobs {
    /// Watch this job's progress. Omit to list every job.
    pub id: Option<String>,
    /// `pause`, `resume` or `cancel`.
    #[arg(long, value_name = "ACTION")]
    pub control: Option<String>,
    /// Act on the chapter in flight instead of waiting for it. Its work is discarded and it
    /// is narrated again from the start on resume — the alternative is waiting out a
    /// chapter, which on a reference book can be twenty minutes.
    #[arg(long, requires = "control")]
    pub now: bool,
}

/// Where the service is, from the same settings the service itself reads.
fn base_url(cfg: &Config) -> String {
    let serve = cfg.settings.serve.clone().unwrap_or_default();
    let host = serve.host.unwrap_or_else(|| "127.0.0.1".into());
    // The service binds `0.0.0.0` to accept from anywhere; a client cannot connect *to*
    // that, so it becomes loopback here.
    let host = if host == "0.0.0.0" {
        "127.0.0.1".to_string()
    } else {
        host
    };
    format!("http://{host}:{}", serve.port.unwrap_or(3003))
}

fn api_key() -> Option<String> {
    std::env::var("DREAM_TTS_API_KEY")
        .or_else(|_| std::env::var("TTS_API_KEY"))
        .ok()
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty())
}

fn healthy(base: &str) -> bool {
    ureq::get(format!("{base}/health"))
        .config()
        .timeout_global(Some(Duration::from_millis(700)))
        .build()
        .call()
        .is_ok()
}

/// Start a service and wait for it to be ready.
///
/// Spawned detached rather than as a child: the point of the split is that the work outlives
/// the terminal that asked for it, and a child would die with this process.
fn start_service(cfg: &Config, base: &str) -> Result<()> {
    let binary = crate::documents::sibling_binary("dream-tts-serve")?;
    ui::field("service", format!("starting {}", binary.display()));
    let log = cfg.data_path("jobs/service.log");
    if let Some(parent) = log.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let out = std::fs::File::create(&log).with_context(|| format!("creating {}", log.display()))?;
    std::process::Command::new(&binary)
        .stdout(out.try_clone()?)
        .stderr(out)
        .stdin(std::process::Stdio::null())
        .spawn()
        .with_context(|| format!("starting {}", binary.display()))?;

    // The engine loads before the listener binds, so a healthy answer means ready to work.
    for _ in 0..120 {
        if healthy(base) {
            ui::field_note("service", base, &format!("log: {}", log.display()));
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    bail!(
        "the service did not become healthy within 60s. Its log is at {}",
        log.display()
    )
}

/// Report anything already narrating, so a second run is a decision rather than an accident.
fn report_active(store: &Store) -> Result<usize> {
    let active = store.active()?;
    if active.is_empty() {
        return Ok(0);
    }
    ui::heading("already narrating");
    for job in &active {
        println!(
            "  {} {}  {}",
            ui::cell(&job.id, 14, ui::bold),
            ui::cell(
                &format!("{}/{}", job.chapters_done(), job.chapters.len()),
                8,
                ui::dim
            ),
            ui::dim(&job.source)
        );
    }
    // What the machine can take, so this is a decision rather than a warning to ignore.
    // One engine is gigabytes and two on a 16 GB machine swap, which presents as the models
    // getting slower rather than as a mistake — but refusing a second on a 64 GB machine
    // would be wrong, so the numbers are reported and the choice is the user's.
    let capacity = tts_core::system::engines_that_fit();
    let total = tts_core::system::total_memory().map(tts_core::system::human_bytes);
    match (capacity, total) {
        (Some(fits), Some(total)) if active.len() >= fits => println!(
            "{}",
            ui::yellow(&format!(
                "note: {} run(s) in flight and this machine ({total}) has room for about \
                 {fits} resident engine(s). Another will contend for memory, and two \
                 engines that do not fit swap — which looks like the models getting slower \
                 rather than like a mistake.",
                active.len()
            ))
        ),
        (Some(fits), Some(total)) => println!(
            "{}",
            ui::dim(&format!(
                "{} run(s) in flight; this machine ({total}) has room for about {fits}. \
                 They still queue on one GPU, so a second book interleaves rather than \
                 going faster.",
                active.len()
            ))
        ),
        _ => println!(
            "{}",
            ui::yellow(&format!(
                "note: {} run(s) in flight. Chapters queue behind each other on one GPU, \
                 so a second book will not go faster — it will interleave.",
                active.len()
            ))
        ),
    }
    Ok(active.len())
}

pub fn cmd_book(args: &Book, cfg: &Config) -> Result<()> {
    let store = Store::open(&cfg.data_dir)?;
    // Jobs left by a killed service say "running" for ever; turn them into resumable paused
    // ones before reporting what is in flight, or the report is a lie.
    store.reap_stale()?;
    report_active(&store)?;

    // Text first, locally: the service takes narration text, and doing the conversion here
    // means a failure in it costs seconds rather than being discovered an hour in.
    let texts = prepare(&args.source, &args.out)?;
    let base = base_url(cfg);
    if !healthy(&base) {
        if args.no_serve {
            bail!("no service at {base}, and --no-serve was given");
        }
        start_service(cfg, &base)?;
    }

    let voice = match &args.voice {
        Some(v) => Some(cfg.locate_or_err(v, "voice asset")?.display().to_string()),
        None => None,
    };
    let body = serde_json::json!({
        "source": args.source.display().to_string(),
        "out_dir": args.out.display().to_string(),
        "texts": texts.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "voice": voice,
        "quant": args.quant,
        "seed": args.seed,
    });

    let mut request = ureq::post(format!("{base}/v1/jobs"));
    if let Some(key) = api_key() {
        request = request.header("X-API-Key", &key);
    }
    let job: Job = request
        .send_json(&body)
        .map_err(|e| anyhow::anyhow!("submitting the job: {e}"))?
        .body_mut()
        .read_json()
        .context("reading the submitted job")?;

    ui::heading("job");
    ui::field("id", ui::bold(&job.id));
    ui::field_note(
        "chapters",
        format!("{}", job.chapters.len()),
        &format!(
            "{} words, {} already done",
            job.words_total(),
            job.chapters_done()
        ),
    );
    if job.chapters_done() > 0 {
        println!(
            "{}",
            ui::green(&format!(
                "resuming: {} of {} chapters were already narrated",
                job.chapters_done(),
                job.chapters.len()
            ))
        );
    }

    if args.detach || job.state == State::Done {
        if job.state == State::Done {
            println!("\n{} nothing left to narrate", ui::green("done"));
        } else {
            println!(
                "\n{}",
                ui::dim(&format!(
                    "running in the background. Watch: dream-tts jobs {}",
                    job.id
                ))
            );
        }
        return Ok(());
    }
    watch(&base, &job.id, &store)
}

/// The narration text for each chapter, produced here rather than in the service.
fn prepare(source: &Path, out: &Path) -> Result<Vec<PathBuf>> {
    let text_dir = out.join("text");
    std::fs::create_dir_all(&text_dir)
        .with_context(|| format!("creating {}", text_dir.display()))?;

    // A directory of markdown is taken as-is; anything else is imported first.
    let markdown: Vec<PathBuf> = if source.is_dir() {
        let mut found: Vec<PathBuf> = std::fs::read_dir(source)
            .with_context(|| format!("reading {}", source.display()))?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "md"))
            .collect();
        found.sort();
        found
    } else {
        let chapters_dir = out.join("source");
        std::fs::create_dir_all(&chapters_dir)?;
        let document = tts_import::import(source)?;
        ui::field_note(
            "imported",
            source.display().to_string(),
            &format!(
                "{}, {} chapters",
                document.format.name(),
                document.chapters.len()
            ),
        );
        document
            .chapters
            .iter()
            .enumerate()
            .map(|(i, chapter)| -> Result<PathBuf> {
                let path = chapters_dir.join(format!("chapter-{:03}.md", i + 1));
                std::fs::write(&path, chapter.markdown())
                    .with_context(|| format!("writing {}", path.display()))?;
                Ok(path)
            })
            .collect::<Result<Vec<_>>>()?
    };
    anyhow::ensure!(
        !markdown.is_empty(),
        "no chapters found in {}",
        source.display()
    );

    let mut texts = Vec::with_capacity(markdown.len());
    let mut warned = 0usize;
    for path in &markdown {
        let source_text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let narration = tts_narrate::convert(&source_text, &tts_narrate::Options::default());
        let stem = path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let out_path = text_dir.join(format!("{stem}.txt"));
        std::fs::write(&out_path, &narration)
            .with_context(|| format!("writing {}", out_path.display()))?;
        for finding in tts_narrate::lint::findings(&narration) {
            eprintln!("{} {stem}: {finding}", ui::yellow("warning:"));
            warned += 1;
        }
        texts.push(out_path);
    }
    if warned > 0 {
        eprintln!(
            "{}",
            ui::dim(&format!(
                "{warned} warning(s) above. Worth reading before an hour of synthesis: they \
                 name passages the voice will read badly."
            ))
        );
    }
    Ok(texts)
}

/// Follow the event stream, drawing one line that updates in place.
fn watch(base: &str, id: &str, store: &Store) -> Result<()> {
    ui::heading("progress");
    let response = ureq::get(format!("{base}/v1/jobs/{id}/events"))
        .config()
        // No global timeout: this stream is open for the length of a book.
        .timeout_global(None)
        .build()
        .call()
        .map_err(|e| anyhow::anyhow!("opening the event stream: {e}"))?;

    let mut bar = ui::Bar::new();
    // The stream opens with a snapshot. Rendering it matters most when attaching *late*:
    // segment events only arrive when a stage advances, so a watcher joining a long
    // chapter would otherwise sit blank for minutes with no way to tell that from a stall.
    let mut opened = false;
    let reader = BufReader::new(response.into_body().into_reader());
    for line in reader.lines() {
        let line = line.context("reading the event stream")?;
        // Server-sent events: `data: {...}`, blank lines and `:` keep-alives between.
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        let Ok(update) = serde_json::from_str::<serde_json::Value>(payload) else {
            continue;
        };
        match update.get("type").and_then(|t| t.as_str()) {
            Some("progress") => {
                let stage = update.get("stage");
                let (name, done, total) = (
                    stage
                        .and_then(|s| s.get("stage"))
                        .and_then(|s| s.as_str())
                        .unwrap_or(""),
                    stage
                        .and_then(|s| s.get("done"))
                        .and_then(|d| d.as_u64())
                        .unwrap_or(0),
                    stage
                        .and_then(|s| s.get("total"))
                        .and_then(|t| t.as_u64())
                        .unwrap_or(0),
                );
                let chapter = update.get("chapter").and_then(|c| c.as_str()).unwrap_or("");
                let index = update
                    .get("chapter_index")
                    .and_then(|c| c.as_u64())
                    .unwrap_or(0);
                let chapters = update
                    .get("chapters_total")
                    .and_then(|c| c.as_u64())
                    .unwrap_or(0);
                let eta = update.get("eta_seconds").and_then(|e| e.as_f64());
                bar.advance_labelled(
                    &format!("{}/{} {chapter} {name}", index + 1, chapters),
                    done as usize,
                    total as usize,
                    eta,
                );
            }
            Some("job") => {
                let job: Job =
                    serde_json::from_value(update.clone()).context("reading a job update")?;
                if job.state.is_terminal() || job.state == State::Paused {
                    bar.finish();
                    return report_end(&job);
                }
                if !opened {
                    opened = true;
                    let eta = job
                        .eta_seconds()
                        .map(|s| format!(", about {} left", ui::duration(s)))
                        .unwrap_or_default();
                    println!(
                        "{}",
                        ui::dim(&format!(
                            "{} — {}/{} chapters, {} of audio so far{eta}",
                            job.state.name(),
                            job.chapters_done(),
                            job.chapters.len(),
                            ui::duration(job.audio_seconds()),
                        ))
                    );
                }
            }
            _ => {}
        }
    }
    // The stream ended without a terminal state: the service stopped. Say what is on disk.
    bar.finish();
    match store.get(id)? {
        Some(job) => report_end(&job),
        None => bail!("the event stream closed and job {id} is gone"),
    }
}

fn report_end(job: &Job) -> Result<()> {
    println!();
    let done = job.chapters_done();
    let audio = ui::duration(job.audio_seconds());
    match job.state {
        State::Done => println!(
            "{} {done} chapters, {audio} of audio in {}",
            ui::green("done"),
            job.out_dir
        ),
        State::Paused => println!(
            "{} at {done}/{} chapters. Run the same command again to continue.",
            ui::yellow("paused"),
            job.chapters.len()
        ),
        State::Cancelled => println!(
            "{} at {done}/{} chapters. Finished chapters are kept in {}.",
            ui::yellow("cancelled"),
            job.chapters.len(),
            job.out_dir
        ),
        State::Failed => {
            println!(
                "{} {}",
                ui::red("failed:"),
                job.error.as_deref().unwrap_or("unknown")
            );
            for chapter in job.chapters.iter().filter(|c| c.error.is_some()) {
                println!(
                    "  {} {}",
                    chapter.name,
                    chapter.error.as_deref().unwrap_or("")
                );
            }
        }
        State::Queued | State::Running => println!("{} still running", ui::dim("note:")),
    }
    Ok(())
}

pub fn cmd_jobs(args: &Jobs, cfg: &Config) -> Result<()> {
    let store = Store::open(&cfg.data_dir)?;
    store.reap_stale()?;

    let Some(id) = &args.id else {
        let jobs = store.list()?;
        if jobs.is_empty() {
            println!("{}", ui::dim("no narration jobs"));
            return Ok(());
        }
        println!(
            "{} {} {} {}",
            ui::cell("id", 14, ui::dim),
            ui::cell("state", 10, ui::dim),
            ui::cell("chapters", 10, ui::dim),
            ui::dim("source")
        );
        for job in &jobs {
            let state = match (job.state, job.request) {
                (State::Running, tts_jobs::Request::Pause) => ui::yellow("pausing"),
                (State::Running, tts_jobs::Request::Cancel) => ui::yellow("cancelling"),
                (State::Running, _) => ui::green("running"),
                (State::Failed, _) => ui::red("failed"),
                (State::Paused | State::Cancelled, _) => ui::yellow(job.state.name()),
                (s, _) => s.name().to_string(),
            };
            println!(
                "{} {} {} {}",
                ui::cell(&job.id, 14, ui::bold),
                ui::cell(&state, 10, |s| s.to_string()),
                ui::cell(
                    &format!("{}/{}", job.chapters_done(), job.chapters.len()),
                    10,
                    ui::dim
                ),
                ui::dim(&job.source)
            );
        }
        return Ok(());
    };

    if let Some(action) = &args.control {
        let base = base_url(cfg);
        anyhow::ensure!(
            healthy(&base),
            "no service at {base}; a control needs the process that owns the run"
        );
        let action = if args.now {
            format!("{action}-now")
        } else {
            action.clone()
        };
        let mut request = ureq::post(format!("{base}/v1/jobs/{id}/{action}"));
        if let Some(key) = api_key() {
            request = request.header("X-API-Key", &key);
        }
        let response: serde_json::Value = request
            .send_empty()
            .map_err(|e| anyhow::anyhow!("{action}: {e}"))?
            .body_mut()
            .read_json()
            .context("reading the response")?;
        println!(
            "{} {}",
            ui::green(&action),
            response.get("note").and_then(|n| n.as_str()).unwrap_or("")
        );
        return Ok(());
    }

    let base = base_url(cfg);
    if healthy(&base) {
        return watch(&base, id, &store);
    }
    // No service: report what is on disk rather than refusing. This is the observatory
    // case — a job can be inspected long after the run.
    match store.get(id)? {
        Some(job) => report_end(&job),
        None => bail!("no job {id}"),
    }
}
