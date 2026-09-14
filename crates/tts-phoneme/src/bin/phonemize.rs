//! Phonemize text from the command line, for looking at what a change did.

use anyhow::{Context, Result};
use std::io::Read;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut dir = PathBuf::from("references/kokoro/weights/frontend");
    if let Some(i) = args.iter().position(|a| a == "--frontend") {
        dir = PathBuf::from(args.remove(i + 1));
        args.remove(i);
    }
    let strict = args.iter().any(|a| a == "--strict");
    args.retain(|a| a != "--strict");

    let g2p = tts_phoneme::g2p::G2P::load(&dir, false).context("loading frontend")?;
    let text = if args.is_empty() {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s)?;
        s
    } else {
        args.join(" ")
    };

    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        if strict {
            println!("{}", g2p.phonemize_strict(line));
        } else {
            let (ps, unknown) = g2p.phonemize_report(line);
            println!("{ps}");
            if !unknown.is_empty() {
                eprintln!("  unpronounceable: {}", unknown.join(", "));
            }
        }
    }
    Ok(())
}
