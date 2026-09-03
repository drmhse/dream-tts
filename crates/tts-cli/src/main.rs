//! `dream-tts` — synthesize with a chosen engine.
//!
//! ```text
//! dream-tts engines                             # what exists, and what works
//! dream-tts speak --text "hello" --out out.wav  # the default engine, qwen3tts
//! dream-tts speak --engine audio8 --text-file book.txt --voice voices/cosy-default --out out.wav
//! dream-tts import book.epub --out prep/        # any document into chapter-NNN.md
//! dream-tts narrate prep/chapter-001.md         # markdown into speakable text
//! dream-tts config                              # the resolved settings, and where each came from
//! dream-tts storage                             # what is on disk, and what removes it
//! ```
//!
//! Defaults come from [`tts_core::config`]: flag, then environment, then `tts.json`, then
//! the built-in. Every flag below still wins over all of it.

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
mod book;
mod documents;
mod ui;

use tts_core::config::Origin;
use tts_core::{Cloning, Config, EngineConfig, Gaps, GpuLock, Sampling, SynthesisRequest, Voice};

#[derive(Parser)]
#[command(
    name = "dream-tts",
    version,
    about = "Local text-to-speech in a cloned voice, on your own machine, offline",
    after_help = "\
Examples:
  dream-tts speak --text \"Hello.\" --out hello.wav        the default engine and voice
  dream-tts speak --text-file book.md --quant f16 --out book.wav
  dream-tts speak --engine audio8 --out x.wav --text \"…\"  44.1 kHz
  dream-tts voice voices/my-voice                         what an asset holds

Settings: dream-tts.json beside the install, or ~/.config/dream-tts/config.json.
`dream-tts config` shows what resolved and what decided it."
)]
struct Cli {
    /// Settings file. Default: `dream-tts.json` in the install root, else
    /// `~/.config/dream-tts/config.json`, else built-in defaults.
    #[arg(long, global = true, value_name = "PATH")]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

// `Speak` carries every synthesis flag and dwarfs the other two variants. Boxing it
// would buy a few bytes on a value constructed once per process.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum Command {
    /// List engines and their capabilities.
    Engines,
    /// Describe a voice asset without synthesizing.
    Voice {
        /// The voice directory, e.g. `voices/cosy-default-qwen3tts`.
        path: PathBuf,
    },
    /// Synthesize speech.
    Speak(Speak),
    /// Narrate a whole document: import, convert, and synthesize every chapter.
    Book(book::Book),
    /// List narration jobs, watch one, or pause, resume and cancel.
    Jobs(book::Jobs),
    /// Split any document into the `chapter-NNN.md` files the pipeline narrates.
    Import(documents::Import),
    /// Convert markdown to the text an engine should speak.
    Narrate(documents::Narrate),
    /// Print the resolved settings and where each value came from.
    Config,
    /// What this install has on disk, and what removes each part.
    Storage,
}

#[derive(Args)]
struct Speak {
    /// Engine id; omit for the first available one (`dream-tts engines` names it).
    #[arg(long)]
    engine: Option<String>,

    /// The text to speak.
    #[arg(long, value_name = "TEXT")]
    text: Option<String>,
    /// Read the text from a file instead. Any document `dream-tts import` reads is
    /// imported and narrated; a `.txt` file is spoken literally.
    #[arg(long, value_name = "PATH")]
    text_file: Option<PathBuf>,
    /// Speak `--text-file` exactly as written, without narrating it. For text that is
    /// already narration output.
    #[arg(long)]
    raw: bool,
    /// Where to write the WAV.
    #[arg(long, value_name = "PATH")]
    out: PathBuf,

    /// Voice asset directory (`voice.json` + `voice.safetensors`). Omit for the one
    /// shipped with the chosen engine.
    #[arg(long, value_name = "PATH")]
    voice: Option<PathBuf>,

    /// Engine model root; defaults per engine.
    #[arg(long)]
    model_root: Option<PathBuf>,
    /// Override a specific file, e.g. `--set codec=references/audio8/weights/codec.safetensors`.
    #[arg(long = "set", value_parser = parse_override)]
    overrides: Vec<(String, PathBuf)>,

    /// Weight format; engine-specific, see `dream-tts engines`.
    #[arg(long)]
    quant: Option<String>,
    /// Run on the CPU. Correct, and roughly 4x slower — the Metal kernels are the point.
    #[arg(long)]
    cpu: bool,

    /// Segment length budget in characters. Config: `max_chars`. Default 220.
    #[arg(long)]
    max_chars: Option<usize>,
    /// Per-segment generation ceiling. A segment that hits it is spoken incompletely, so
    /// the fix is usually a lower `--max-chars`.
    #[arg(long, default_value_t = 512)]
    max_new_tokens: usize,
    /// Sampling temperature. Engines that document their own default use it unless this
    /// is given; `dream-tts engines` names them.
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,
    #[arg(long, default_value_t = 50)]
    top_k: usize,
    /// Makes a render reproducible within this implementation. Not across
    /// implementations: the PyTorch references draw from torch's RNG.
    #[arg(long, default_value_t = 1234)]
    seed: u64,
    /// Take the most likely token every step. The path to use when comparing against a
    /// reference implementation.
    #[arg(long)]
    greedy: bool,
    /// Config: `gaps.segment_ms`. Default 90.
    #[arg(long)]
    gap_ms: Option<usize>,
    /// Config: `gaps.paragraph_ms`. Default 320.
    #[arg(long)]
    para_gap_ms: Option<usize>,

    /// Do not take the advisory GPU lock. Two resident engines will not fit in 16 GB;
    /// this exists for the case where you know the other process is not on the GPU.
    #[arg(long)]
    no_gpu_lock: bool,
}

fn parse_override(s: &str) -> Result<(String, PathBuf), String> {
    let (k, v) = s
        .split_once('=')
        .ok_or_else(|| format!("expected key=path, got {s:?}"))?;
    Ok((k.to_string(), PathBuf::from(v)))
}

fn cmd_engines() -> Result<()> {
    let default = tts_engines::default_id();
    // The catalogue is preference order, so it prints in that order and the default is
    // marked rather than re-sorted to the top — the order is information.
    for c in tts_engines::catalogue() {
        let marker = if c.id == default {
            ui::green("*")
        } else {
            " ".into()
        };
        println!(
            "\n{} {} {}  {}",
            marker,
            ui::cell(c.id, 11, ui::bold),
            ui::cell(if c.available { "ready" } else { "staged" }, 7, |s| {
                if s.starts_with("ready") {
                    ui::green(s)
                } else {
                    ui::yellow(s)
                }
            }),
            ui::dim(&format!(
                "{} Hz · {} · {}",
                c.sample_rate,
                match c.cloning {
                    Cloning::None => "single speaker",
                    Cloning::PrecomputedAsset => "voice cloning",
                },
                if c.streaming {
                    "streaming"
                } else {
                    "no streaming"
                }
            ))
        );
        println!("    {}", c.description);
        println!("    {} {}", ui::dim("weights  "), c.quantization.join(", "));
        match c.languages {
            Some(langs) => println!(
                "    {} {} {}",
                ui::dim("languages"),
                langs.join(", "),
                ui::yellow("(closed list)")
            ),
            None => println!("    {} unrestricted", ui::dim("languages")),
        }
        if let Some(reason) = c.reason {
            println!("    {} {reason}", ui::yellow("unavailable:"));
        }
    }
    println!("\n{} marks the default engine.", ui::green("*"));
    if let Some(caveat) = tts_engines::default_caveat() {
        println!("{} {caveat}", ui::yellow("note:"));
    }
    Ok(())
}

fn cmd_voice(path: &Path) -> Result<()> {
    let voice = Voice::load(path)?;
    ui::field("name", ui::bold(&voice.name));
    ui::field("engine", &voice.engine);
    if let Some(s) = voice.seconds {
        ui::field("length", format!("{s:.2} s of reference audio"));
    }
    ui::field("text", format!("{:?}", voice.text));
    ui::field("tensors", voice.keys().join(", "));
    let caps = tts_engines::catalogue();
    match caps.iter().find(|c| c.id == voice.engine) {
        Some(c) if c.available => {
            println!("\n{} usable with engine `{}`", ui::green("ok"), c.id)
        }
        Some(c) => println!(
            "\n{} engine `{}` is staged: {}",
            ui::yellow("note:"),
            c.id,
            c.reason.unwrap_or("")
        ),
        None => println!(
            "\n{} no engine in this build claims `{}`",
            ui::red("error:"),
            voice.engine
        ),
    }
    Ok(())
}

/// Print what the settings resolved to, and which file or variable said so.
///
/// A precedence chain is only usable if you can see through it. Without this, a config
/// file that is being ignored — wrong directory, `TTS_CONFIG` set in a shell profile — is
/// indistinguishable from one whose values happen to match the defaults.
fn cmd_config(cfg: &Config) -> Result<()> {
    ui::heading("paths");
    ui::field("root", cfg.root.display().to_string());
    ui::field_note(
        "data",
        cfg.data_dir.display().to_string(),
        cfg.data_dir_origin.label(),
    );
    match &cfg.source {
        Some(p) => ui::field("config", p.display().to_string()),
        None => {
            ui::field("config", ui::dim("none — built-in defaults"));
            ui::field(
                "",
                ui::dim(&format!(
                    "write {} to change them",
                    cfg.root.join(tts_core::config::PROJECT_CONFIG).display()
                )),
            );
        }
    }
    ui::field_note(
        "gpu lock",
        if cfg.gpu_lock() {
            ui::green("on")
        } else {
            ui::yellow("off")
        },
        &cfg.lock_path().display().to_string(),
    );

    ui::heading("resolved");
    let id = cfg
        .settings
        .engine
        .clone()
        .unwrap_or_else(|| tts_engines::default_id().to_string());
    ui::field_note(
        "engine",
        ui::bold(&id),
        if cfg.settings.engine.is_some() {
            Origin::ProjectConfig.label()
        } else {
            "registry default"
        },
    );
    let weights = cfg.data_path(tts_engines::default_root(&id));
    ui::field_note(
        "weights",
        weights.display().to_string(),
        &present(weights.is_dir(), "downloaded", "not downloaded"),
    );
    let voice = match &cfg.settings.voice {
        Some(v) => cfg.root_path(v),
        None => cfg.root_path(tts_engines::default_voice(&id)),
    };
    ui::field_note(
        "voice",
        voice.display().to_string(),
        &present(voice.is_dir(), "present", "missing"),
    );

    ui::heading("settings as read");
    let json = serde_json::to_string_pretty(&cfg.settings)?;
    if json == "{}" {
        println!("{}", ui::dim("(empty — nothing is overridden)"));
    } else {
        println!("{json}");
    }
    Ok(())
}

fn present(ok: bool, yes: &str, no: &str) -> String {
    if ok {
        yes.to_string()
    } else {
        no.to_string()
    }
}

/// Bytes under `path`, not following symlinks. Missing is zero rather than an error: most
/// of what this reports is legitimately absent.
fn dir_size(path: &Path) -> u64 {
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return 0;
    };
    if meta.is_symlink() {
        return 0;
    }
    if meta.is_file() {
        return meta.len();
    }
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    entries
        .filter_map(|e| e.ok())
        .map(|e| dir_size(&e.path()))
        .sum()
}

fn human(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

/// Directories a narration run wrote. `scripts/narrate-book.sh --out` names them, so the
/// only way to find them is to look.
fn narration_dirs(root: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n == "narration" || n.starts_with("narration-"))
        })
        .collect();
    found.sort();
    found
}

/// What is on disk and what removes it.
///
/// Reporting only. Deletion lives in `scripts/uninstall.sh`, which you can read before
/// running — a `--prune` flag on a synthesis tool is a footgun aimed at 13 GB that took an
/// hour to download.
fn cmd_storage(cfg: &Config) -> Result<()> {
    struct Row {
        what: String,
        path: PathBuf,
        removed_by: &'static str,
    }
    let mut rows: Vec<Row> = Vec::new();

    for id in tts_engines::ids() {
        rows.push(Row {
            what: format!("{id} weights"),
            path: cfg.data_path(tts_engines::default_root(id)),
            removed_by: "scripts/uninstall.sh --weights",
        });
        rows.push(Row {
            what: format!("{id} raw download"),
            path: cfg.data_path(format!("references/{id}/download")),
            removed_by: "safe to delete; conversion is done",
        });
        rows.push(Row {
            what: format!("{id} fixtures"),
            path: cfg.data_path(format!("fixtures/{id}")),
            removed_by: "scripts/fetch-assets.sh refetches",
        });
        rows.push(Row {
            what: format!("{id} python venv"),
            path: cfg.root_path(format!("references/{id}/.venv")),
            removed_by: "scripts/uninstall.sh --venvs",
        });
    }
    rows.push(Row {
        what: "voice assets".into(),
        path: cfg.root_path("voices"),
        removed_by: "shipped with the install",
    });
    rows.push(Row {
        what: "downloaded binaries".into(),
        path: cfg.root_path("bin"),
        removed_by: "scripts/fetch-prebuilt.sh refetches",
    });
    rows.push(Row {
        what: "build output".into(),
        path: cfg.root_path("target"),
        removed_by: "cargo clean",
    });
    // Narration runs are the largest thing this produces — WAV masters and delivery files,
    // gigabytes per book — and they are the whole reason a storage command is worth having.
    // Enumerated rather than named, because the directory is `--out`, chosen per run.
    for dir in narration_dirs(&cfg.root) {
        let name = dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        rows.push(Row {
            what: format!("narration: {name}"),
            path: dir,
            removed_by: "yours to keep or delete; reproducible from the markdown and a seed",
        });
    }

    // Sized first, then filtered: an absent directory is not a row, and the biggest thing
    // on disk is what someone opening this command came to find.
    let mut sized: Vec<(u64, &Row)> = rows
        .iter()
        .map(|r| (dir_size(&r.path), r))
        .filter(|(size, _)| *size > 0)
        .collect();
    sized.sort_by(|a, b| b.0.cmp(&a.0));

    if sized.is_empty() {
        println!(
            "{}",
            ui::dim("nothing downloaded yet — run ./scripts/bootstrap.sh")
        );
        return Ok(());
    }

    let total: u64 = sized.iter().map(|(s, _)| s).sum();
    let biggest = sized[0].0 as f64;
    const WHAT: usize = 24;
    for (size, r) in &sized {
        // Bars are relative to the largest row, not to the total: with one 4 GB checkpoint
        // dominating, share-of-total bars would all round to nothing.
        let cells = ((*size as f64 / biggest) * 12.0).round() as usize;
        println!(
            "{} {:>10}  {}",
            ui::cell(&r.what, WHAT, ui::bold),
            human(*size),
            ui::dim(&"█".repeat(cells.max(1)))
        );
        println!(
            "{:WHAT$} {:>10}  {}",
            "",
            "",
            ui::dim(&format!("{}  ·  {}", r.path.display(), r.removed_by))
        );
    }
    println!(
        "\n{} {:>10}",
        ui::cell("total", WHAT, ui::bold),
        ui::bold(&human(total))
    );
    Ok(())
}

fn cmd_speak(args: &Speak, cfg: &Config) -> Result<()> {
    // Precedence, once: flag, then config, then the registry. `chose` is what tells the
    // caveat below apart from a deliberate choice — a config file naming an engine is a
    // choice, so it must not warn either.
    let (id, chose) = match args.engine.clone().or_else(|| cfg.settings.engine.clone()) {
        Some(id) => (id, true),
        None => (tts_engines::default_id().to_string(), false),
    };
    if !chose {
        // Only on the defaulted path: a caller that named the engine accepted its limits,
        // and repeating them on every render would train them to be ignored.
        if let Some(caveat) = tts_engines::default_caveat() {
            eprintln!("note: {caveat}");
        }
    }
    let text = match (&args.text, &args.text_file) {
        (Some(t), _) => t.clone(),
        (None, Some(p)) => {
            let p = cfg.locate_or_err(p, "text file")?;
            documents::text_for_speaking(&p, args.raw)?
        }
        (None, None) => anyhow::bail!("pass --text or --text-file"),
    };

    let root = match &args.model_root {
        Some(p) => cfg.locate_or_err(p, "model root")?,
        None => cfg.data_path(tts_engines::default_root(&id)),
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

    // Voice assets load on the host; the engine pulls them onto its own device. See
    // `Voice::load` for why the CLI must not open a second device handle here.
    //
    // Resolution is against the install root, not the data directory: the assets ship with
    // the app and are there before anything is downloaded. And when nothing is named, the
    // engine's own shipped voice is used rather than failing — every engine here clones
    // from an asset, so `speak --text …` with no `--voice` would otherwise be an error, and
    // "it needs a voice" is not a decision the caller has any information to make.
    let named_voice = args
        .voice
        .clone()
        .or_else(|| cfg.settings.voice.clone())
        .map(|p| cfg.locate_or_err(&p, "voice asset"))
        .transpose()?;
    let clones = tts_engines::catalogue()
        .into_iter()
        .find(|c| c.id == id)
        .is_some_and(|c| c.cloning == Cloning::PrecomputedAsset);
    let voice_path = match named_voice {
        Some(p) => Some(p),
        None if clones => Some(
            cfg.locate_or_err(
                Path::new(tts_engines::default_voice(&id)),
                &format!("shipped voice for `{id}`"),
            )
            .with_context(|| {
                format!("engine `{id}` clones from a voice asset and none was given")
            })?,
        ),
        None => None,
    };
    let voice = match &voice_path {
        None => None,
        Some(p) => Some(Voice::load(p)?),
    };

    // Fail on an engine/voice mismatch before spending 1-2 s loading weights.
    if let Some(v) = &voice {
        if v.engine != id {
            anyhow::bail!(
                "voice `{}` was built for engine `{}`, but `{id}` was requested — voice \
                 assets are not interchangeable between engines",
                v.name,
                v.engine
            );
        }
    }

    // Before the weights, not after: a 4 GB load that then discovers it cannot run is the
    // most annoying possible ordering, and loading it is itself most of the memory pressure
    // the lock exists to prevent.
    let want_lock = !args.no_gpu_lock && cfg.gpu_lock();
    let _gpu = GpuLock::maybe(
        want_lock,
        &cfg.lock_path(),
        &format!("dream-tts speak --engine {id}"),
    )?;

    let load_started = std::time::Instant::now();
    let engine = tts_engines::load(&id, &config)?;
    let caps = engine.capabilities();
    ui::field_note(
        "engine",
        ui::bold(caps.id),
        &format!(
            "{} Hz · loaded in {}",
            caps.sample_rate,
            ui::duration(load_started.elapsed().as_secs_f64())
        ),
    );

    // The built-in defaults live in one place, `SynthesisRequest::new`, rather than being
    // restated as clap `default_value_t` where a config file could never reach them.
    let defaults = SynthesisRequest::new("");
    let request = SynthesisRequest {
        text,
        voice,
        sampling: Sampling {
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            seed: args.seed,
            greedy: args.greedy,
        },
        max_chars: args
            .max_chars
            .or(cfg.settings.max_chars)
            .unwrap_or(defaults.max_chars),
        max_new_tokens: args.max_new_tokens,
        gaps: Gaps {
            segment_ms: args
                .gap_ms
                .or(cfg.settings.gaps.and_then(|g| g.segment_ms))
                .unwrap_or(defaults.gaps.segment_ms),
            paragraph_ms: args
                .para_gap_ms
                .or(cfg.settings.gaps.and_then(|g| g.paragraph_ms))
                .unwrap_or(defaults.gaps.paragraph_ms),
        },
        progress: None,
        interrupt: None,
    };
    if let Some(v) = &request.voice {
        ui::field_note(
            "voice",
            &v.name,
            &v.seconds
                .map(|s| format!("{s:.2} s reference"))
                .unwrap_or_else(|| "no duration recorded".into()),
        );
    }
    ui::field(
        "text",
        format!("{} characters", request.text.chars().count()),
    );

    // The bar lives behind a mutex because the callback is `Fn` and shared: an engine is
    // free to report from whichever thread does the work, and one of them batches.
    let bar = std::sync::Arc::new(std::sync::Mutex::new(ui::Bar::new()));
    let request = {
        let bar = bar.clone();
        request.with_progress(std::sync::Arc::new(move |event| {
            if let Ok(mut b) = bar.lock() {
                match event {
                    tts_core::ProgressEvent::Planned { segments } => b.planned(segments),
                    tts_core::ProgressEvent::Advanced { stage, done, total } => {
                        b.advance(stage, done, total)
                    }
                }
            }
        }))
    };

    let out = engine.synthesize(&request);
    if let Ok(mut b) = bar.lock() {
        b.finish();
    }
    let out = out?;
    let seconds = out.audio.seconds();
    tts_core::wav::write_mono(&args.out, &out.audio.samples, out.audio.sample_rate)?;

    let s = &out.stats;
    ui::heading("done");
    println!(
        "{} of audio at {} Hz in {}  ·  {} {}",
        ui::bold(&ui::duration(seconds)),
        out.audio.sample_rate,
        ui::bold(&ui::duration(s.total_s)),
        ui::dim("RTF"),
        ui::green(&format!("{:.3}", s.rtf(seconds)))
    );
    println!(
        "{}",
        ui::dim(&format!(
            "{} segment{}, {} frames",
            s.segments,
            if s.segments == 1 { "" } else { "s" },
            s.frames
        ))
    );
    // The stage names come from the engine, so this prints a two-stage Audio8 split and
    // a three-stage CosyVoice one without knowing which it is talking to.
    let breakdown = s.breakdown(seconds);
    let widest = breakdown
        .iter()
        .map(|(n, ..)| n.len())
        .max()
        .unwrap_or(6)
        .max(6);
    for (name, secs, rtf, share) in breakdown {
        // A share bar makes the dominant stage findable without reading the numbers, which
        // is the whole reason this breakdown exists.
        let cells = ((share / 100.0) * 16.0).round() as usize;
        println!(
            "  {} {:>8} {:>6}  {} {}",
            ui::cell(&name, widest, ui::cyan),
            ui::duration(secs),
            format!("{share:.1}%"),
            ui::dim(&("█".repeat(cells.max(1)))),
            ui::dim(&format!("RTF {rtf:.3}")),
        );
    }
    println!(
        "\n{} {}",
        ui::green("wrote"),
        ui::bold(&args.out.display().to_string())
    );
    Ok(())
}

fn main() -> std::process::ExitCode {
    // Rust starts with SIGPIPE ignored, which turns `dream-tts storage | head` into a
    // panic on EPIPE — "failed printing to stdout: Broken pipe" and a backtrace note —
    // where every other Unix tool exits quietly. Whether it fires depends on how much
    // output fits the pipe buffer before the reader leaves, so it is the kind of bug that
    // shows up in someone else's terminal and not in yours. Restore the default.
    //
    // SAFETY: called before any thread is spawned, and only resets a handler to SIG_DFL.
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }

    let cli = Cli::parse();
    // Errors are rendered here rather than by `Termination`'s `Debug` formatting, which
    // prints the whole chain on one colon-joined line and buries the actionable end of it.
    let result = Config::load(cli.config.as_deref()).and_then(|cfg| match &cli.command {
        Command::Engines => cmd_engines(),
        Command::Voice { path } => cmd_voice(path),
        Command::Speak(args) => cmd_speak(args, &cfg),
        Command::Book(args) => book::cmd_book(args, &cfg),
        Command::Jobs(args) => book::cmd_jobs(args, &cfg),
        Command::Import(args) => documents::cmd_import(args),
        Command::Narrate(args) => documents::cmd_narrate(args),
        Command::Config => cmd_config(&cfg),
        Command::Storage => cmd_storage(&cfg),
    });
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            ui::report_error(&err);
            std::process::ExitCode::FAILURE
        }
    }
}
