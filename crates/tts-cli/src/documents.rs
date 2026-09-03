//! `dream-tts import` and `dream-tts narrate`: any document, into speakable text.
//!
//! Two commands rather than one because they are two decisions. `import` splits a document
//! into the `chapter-NNN.md` files the narration pipeline consumes, and stops — the split is
//! worth looking at before hours of synthesis. `narrate` is the markdown-to-speech step on
//! its own, which is what the book pipeline calls per chapter.

use crate::ui;
use anyhow::{Context, Result};
use clap::Args;
use std::path::{Path, PathBuf};
use tts_import::Format;
use tts_narrate::Options;

#[derive(Args)]
pub struct Import {
    /// The document. EPUB, DOCX, ODT, HTML, PDF, Markdown or plain text.
    pub path: PathBuf,
    /// Directory for the `chapter-NNN.md` files. Created if absent.
    #[arg(long, value_name = "DIR", required_unless_present = "dry_run")]
    pub out: Option<PathBuf>,
    /// Overwrite chapter files that already exist.
    #[arg(long)]
    pub force: bool,
    /// Report the split without writing anything. The split is worth looking at before
    /// hours of synthesis, and a 400-page PDF is exactly where it can be wrong.
    #[arg(long)]
    pub dry_run: bool,
}

#[derive(Args)]
pub struct Narrate {
    /// Markdown files, one per chapter.
    pub paths: Vec<PathBuf>,
    /// Write the narration to this file. One input only; stdout when omitted.
    #[arg(short, long, value_name = "PATH")]
    pub out: Option<PathBuf>,
    /// Write the page-word map the site's player consumes. One input only.
    #[arg(long, value_name = "PATH")]
    pub emit_map: Option<PathBuf>,
    /// Write `<stem>.txt`, and any map as `<stem>.map.json`, into this directory. The form
    /// to use for more than one input.
    #[arg(long, value_name = "DIR", conflicts_with_all = ["out", "emit_map"])]
    pub out_dir: Option<PathBuf>,
    /// Narrate fenced code blocks instead of dropping them.
    #[arg(long)]
    pub keep_code: bool,
    /// Omit figure captions instead of narrating them.
    #[arg(long)]
    pub no_captions: bool,
    /// With --out-dir, also write the page-word maps.
    #[arg(long, requires = "out_dir")]
    pub maps: bool,
    /// Report length, paragraph count and estimated duration.
    #[arg(long)]
    pub stats: bool,
}

impl Narrate {
    fn options(&self) -> Options {
        Options {
            keep_code: self.keep_code,
            keep_captions: !self.no_captions,
        }
    }
}

pub fn cmd_import(args: &Import) -> Result<()> {
    let document = tts_import::import(&args.path)?;
    ui::field("source", args.path.display().to_string());
    ui::field_note(
        "format",
        ui::bold(document.format.name()),
        &plural(document.chapters.len(), "chapter"),
    );
    if let Some(title) = &document.title {
        ui::field("title", title);
    }

    let out = match (&args.out, args.dry_run) {
        (Some(dir), false) => {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
            Some(dir)
        }
        _ => None,
    };

    ui::heading("chapters");
    let mut written = 0usize;
    for (index, chapter) in document.chapters.iter().enumerate() {
        let name = format!("chapter-{:03}.md", index + 1);
        let markdown = chapter.markdown();
        let words = markdown.split_whitespace().count();
        let title = ui::dim(chapter.title.as_deref().unwrap_or("(untitled)"));

        let Some(dir) = out else {
            println!(
                "  {} {words:>7} words  {title}",
                ui::cell(&name, 18, ui::dim)
            );
            continue;
        };
        let path = dir.join(&name);
        if path.exists() && !args.force {
            println!(
                "  {} {}  {}",
                ui::cell(&name, 18, ui::dim),
                ui::yellow("exists"),
                ui::dim("(--force to overwrite)")
            );
            continue;
        }
        std::fs::write(&path, &markdown).with_context(|| format!("writing {}", path.display()))?;
        written += 1;
        println!(
            "  {} {words:>7} words  {title}",
            ui::cell(&name, 18, ui::bold)
        );
    }

    match out {
        None => println!("\n{}", ui::dim("nothing written (--dry-run)")),
        Some(dir) => {
            println!(
                "\n{} {} into {}",
                ui::green("wrote"),
                plural(written, "chapter"),
                ui::bold(&dir.display().to_string())
            );
            println!(
                "{}",
                ui::dim(&format!(
                    "next:  scripts/narrate-book.sh --book {} --out narration",
                    dir.display()
                ))
            );
        }
    }
    Ok(())
}

pub fn cmd_narrate(args: &Narrate) -> Result<()> {
    anyhow::ensure!(!args.paths.is_empty(), "pass one or more markdown files");
    // `-o` and `--emit-map` name single files, as the Python they replace did, so
    // `narrate-book.sh` could keep calling this the same way. More than one input needs a
    // directory, or their narration would be interleaved on stdout.
    anyhow::ensure!(
        args.paths.len() == 1 || args.out_dir.is_some(),
        "more than one input needs --out-dir"
    );

    for path in &args.paths {
        let source =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let narration = tts_narrate::convert(&source, &args.options());

        // Always, not behind a flag: markup that reaches the voice is read aloud, and a
        // silent converter is how a chapter of "asterisk, asterisk, asterisk" shipped.
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        for finding in tts_narrate::lint::findings(&narration) {
            eprintln!("{} {name}: {finding}", ui::yellow("warning:"));
        }

        let (text_out, map_out) = match &args.out_dir {
            Some(dir) => {
                std::fs::create_dir_all(dir)
                    .with_context(|| format!("creating {}", dir.display()))?;
                let stem = path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned();
                (
                    Some(dir.join(format!("{stem}.txt"))),
                    args.maps.then(|| dir.join(format!("{stem}.map.json"))),
                )
            }
            None => (args.out.clone(), args.emit_map.clone()),
        };

        match &text_out {
            Some(out) => write_beside(out, &narration)?,
            None => print!("{narration}"),
        }
        if let Some(map) = &map_out {
            let json = tts_narrate::page::word_map(&source, &narration);
            // Compact: the player fetches this per chapter and it is mostly indices.
            write_beside(map, &serde_json::to_string(&json)?)?;
        }
        report(args, &name, &narration, text_out.as_deref());
    }
    Ok(())
}

/// Write, creating the parent directory. `narrate-book.sh` names outputs inside a run
/// directory that may not exist yet on a first pass.
fn write_beside(path: &Path, contents: &str) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(path, contents).with_context(|| format!("writing {}", path.display()))
}

fn report(args: &Narrate, name: &str, narration: &str, out: Option<&Path>) {
    if !args.stats {
        return;
    }
    let stats = tts_narrate::lint::stats(narration);
    eprintln!(
        "{name}: {} words, {} chars, {} paragraphs, ~{:.1} min of audio{}",
        stats.words,
        stats.chars,
        stats.paragraphs,
        stats.minutes,
        out.map(|p| format!(" -> {}", p.display()))
            .unwrap_or_default()
    );
    let spelled = tts_narrate::lint::spelled_out(narration);
    if !spelled.is_empty() {
        eprintln!("  spelled out as letters: {}", spelled.join(", "));
    }
}

/// Text for `--text-file`, importing and narrating anything that is not plain text.
///
/// Speaking raw markdown reads the syntax aloud, so a `.md` file goes through the converter
/// by default; `.txt` is taken literally, which is what the benchmark fixtures in
/// `examples/` depend on. `--raw` forces the literal reading for anything.
pub fn text_for_speaking(path: &Path, raw: bool) -> Result<String> {
    let format = Format::of(path);
    if raw || matches!(format, Some(Format::Text) | None) {
        return std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()));
    }
    let document = tts_import::import(path)?;
    let markdown = document
        .chapters
        .iter()
        .map(|c| c.markdown())
        .collect::<Vec<_>>()
        .join("\n\n");
    let narration = tts_narrate::convert(&markdown, &Options::default());
    if let Some(format) = format {
        ui::field_note(
            "input",
            path.display().to_string(),
            &format!(
                "{}, {} — narrated, not read literally (--raw to override)",
                format.name(),
                plural(document.chapters.len(), "chapter")
            ),
        );
    }
    Ok(narration)
}

/// A binary beside this one.
///
/// `current_exe`, not PATH: the shims and the release archive put `dream-tts` and
/// `dream-tts-serve` in the same directory, and a `dream-tts-serve` from somewhere else on
/// PATH could be a different version than the CLI that is about to talk to it.
pub fn sibling_binary(name: &str) -> anyhow::Result<PathBuf> {
    let exe = std::env::current_exe().context("finding this executable")?;
    let dir = exe.parent().context("this executable has no directory")?;
    let candidate = dir.join(name);
    anyhow::ensure!(
        candidate.is_file(),
        "no {name} beside {} — a release archive ships both in bin/",
        exe.display()
    );
    Ok(candidate)
}

fn plural(n: usize, noun: &str) -> String {
    if n == 1 {
        format!("1 {noun}")
    } else {
        format!("{n} {noun}s")
    }
}
