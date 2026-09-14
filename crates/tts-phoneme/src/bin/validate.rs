//! The frontend gate: tokens, then tags, then phonemes, against the pinned oracle.
//!
//! Staged on purpose. An end-to-end phoneme comparison says a line is wrong; it never says
//! whether the tokenizer split it differently, the tagger called a noun a verb, or the
//! lexicon rule is at fault — and those are three unrelated ports.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;
use tts_phoneme::tagger::Tagged;
use tts_phoneme::{Tagger, Tokenizer};

#[derive(Deserialize)]
struct SpacyToken {
    t: String,
    tag: String,
    ws: String,
}

#[derive(Deserialize)]
struct Record {
    text: String,
    spacy: Vec<SpacyToken>,
    phonemes: String,
}

struct Counts {
    lines: usize,
    tokens: usize,
    token_ok: usize,
    tag_ok: usize,
    line_token_ok: usize,
    phoneme_ok: usize,
    unk: usize,
    oracle_unk: usize,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let oracle = PathBuf::from(args.next().context("usage: phoneme-validate <oracle.jsonl> [frontend-dir]")?);
    let dir = PathBuf::from(
        args.next().unwrap_or_else(|| "references/kokoro/weights/frontend".into()),
    );

    let tok = Tokenizer::load(&dir)?;
    let tagger = Tagger::load(&dir)?;
    let g2p = tts_phoneme::g2p::G2P::load(&dir, false)?;

    let mut c = Counts {
        lines: 0,
        tokens: 0,
        token_ok: 0,
        tag_ok: 0,
        line_token_ok: 0,
        phoneme_ok: 0,
        unk: 0,
        oracle_unk: 0,
    };
    let mut shown = 0;
    let mut ps_shown = 0;
    let mut tag_shown = 0;
    for line in std::fs::read_to_string(&oracle)?.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let rec: Record = serde_json::from_str(line)?;
        let mine = tok.tokenize(&rec.text);
        c.lines += 1;

        let split_matches = mine.len() == rec.spacy.len()
            && mine.iter().zip(&rec.spacy).all(|(a, b)| a.text == b.t && a.whitespace == b.ws);
        c.line_token_ok += split_matches as usize;
        if !split_matches && shown < 8 {
            shown += 1;
            eprintln!("split mismatch: {}", rec.text);
            eprintln!("  ref  {:?}", rec.spacy.iter().map(|t| &t.t).collect::<Vec<_>>());
            eprintln!("  mine {:?}", mine.iter().map(|t| &t.text).collect::<Vec<_>>());
        }
        for (a, b) in mine.iter().zip(&rec.spacy) {
            c.tokens += 1;
            c.token_ok += (a.text == b.t && a.whitespace == b.ws) as usize;
        }

        // Identity is checked without the derivation stage; coverage is measured with it.
        let mine_ps = g2p.phonemize_strict(&rec.text);
        c.unk += g2p.phonemize(&rec.text).matches('\u{2753}').count();
        c.oracle_unk += rec.phonemes.matches('\u{2753}').count();
        if mine_ps == rec.phonemes {
            c.phoneme_ok += 1;
        } else if ps_shown < 10 {
            ps_shown += 1;
            eprintln!("phoneme mismatch: {}", rec.text);
            eprintln!("  ref  {}", rec.phonemes);
            eprintln!("  mine {mine_ps}");
        }

        // Tag the port's own tokens — the real path, which carries the NORM an exception
        // entry supplies. A line whose split already disagrees is skipped rather than
        // counted twice against the tagger.
        if !split_matches {
            continue;
        }
        let tagged: Vec<Tagged> = mine
            .iter()
            .map(|t| Tagged {
                text: &t.text,
                has_space: !t.whitespace.is_empty(),
                norm: t.norm.as_deref(),
            })
            .collect();
        for (got, want) in tagger.tag(&tagged, &tok.vocab).iter().zip(&rec.spacy) {
            let ok = *got == want.tag;
            c.tag_ok += ok as usize;
            if !ok && tag_shown < 12 {
                tag_shown += 1;
                eprintln!("tag mismatch: {:?} ref {} mine {}", want.t, want.tag, got);
            }
        }
    }

    let total_ref: usize = c.tokens;
    println!("{} lines, {} tokens", c.lines, total_ref);
    println!(
        "  tokenizer  {}/{} lines identical, {}/{} tokens",
        c.line_token_ok, c.lines, c.token_ok, c.tokens
    );
    println!("  tagger     {}/{} tags identical", c.tag_ok, total_ref);
    println!(
        "  phonemes   {}/{} lines identical to misaki; unknown {} -> {} after derivation",
        c.phoneme_ok, c.lines, c.oracle_unk, c.unk
    );

    let ok = c.line_token_ok == c.lines && c.tag_ok == total_ref && c.phoneme_ok == c.lines;
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
