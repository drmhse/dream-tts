//! Block structure: front matter, shortcodes, tables, headings, lists, footnotes.
//!
//! Paragraph assembly is where the engine's prosody is decided. A heading gets its own
//! paragraph because the engines insert a longer gap at a paragraph boundary than at a
//! sentence one (320 ms against 90 ms), and that gap is the audible section break.

use crate::inline::clean_inline;
use crate::math::drop_display_math;
use crate::re::{self, compile};
use crate::source::{
    caption_paragraph, ends_sentence, is_horizontal_rule, strip_front_matter, CODE_BLOCK,
    HTML_COMMENT, SHORTCODE,
};
use fancy_regex::Regex;
use once_cell::sync::Lazy;

static FOOTNOTE_DEF: Lazy<Regex> = Lazy::new(|| compile(r"^\s*\[\^([^\]]+)\]:\s*(.*)$"));
static BLOCKQUOTE: Lazy<Regex> = Lazy::new(|| compile(r"^\s*>\s?"));
static IS_BLOCKQUOTE: Lazy<Regex> = Lazy::new(|| compile(r"^\s*>"));
static BULLET: Lazy<Regex> = Lazy::new(|| compile(r"^\s*(?:[-*+]|\d+\.)\s+(.*)$"));
static TABLE_ROW: Lazy<Regex> = Lazy::new(|| compile(r"^\s*\|.*\|\s*$"));
static TABLE_RULE: Lazy<Regex> = Lazy::new(|| compile(r"^\s*\|[\s|:\-]+\|\s*$"));

/// How much of the source survives into speech.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Narrate fenced code blocks. Reading shell syntax aloud is noise, not content.
    pub keep_code: bool,
    /// Promote a figure's `caption=` to prose. The listener cannot see the figure, and the
    /// caption is usually the one sentence stating what it was for.
    pub keep_captions: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            keep_code: false,
            keep_captions: true,
        }
    }
}

fn resolve_shortcodes(text: &str, keep_captions: bool) -> String {
    re::sub(&SHORTCODE, text, |c| {
        if keep_captions {
            caption_paragraph(&re::g(c, 0))
        } else {
            String::new()
        }
    })
}

/// Markdown tables, rewritten as speakable sentences.
///
/// Without this a table becomes one run-on with no sentence boundaries, separator row
/// included. On one chapter that made the model degenerate into babble for the whole table
/// and it shipped: two independent recognisers transcribed it as "tampoligation,
/// tampolition, sambolition". Each row becomes its own short sentences instead.
fn render_tables(text: &str) -> String {
    let lines: Vec<&str> = text.split('\n').collect();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        if !TABLE_ROW.is_match(lines[i]).unwrap_or(false) {
            out.push(lines[i].to_string());
            i += 1;
            continue;
        }
        let mut block: Vec<&str> = Vec::new();
        while i < lines.len() && TABLE_ROW.is_match(lines[i]).unwrap_or(false) {
            block.push(lines[i]);
            i += 1;
        }
        let rows: Vec<Vec<String>> = block
            .iter()
            .filter(|r| !TABLE_RULE.is_match(r).unwrap_or(false))
            .map(|r| {
                r.trim()
                    .trim_matches('|')
                    .split('|')
                    .map(|c| c.trim().to_string())
                    .collect()
            })
            .collect();
        if rows.is_empty() {
            continue;
        }
        // Header-only degrades to its cells as sentences, which is still speakable.
        let (header, body) = if rows.len() == 1 {
            (Vec::new(), vec![rows[0].clone()])
        } else {
            (rows[0].clone(), rows[1..].to_vec())
        };
        out.push(String::new());
        for row in &body {
            let mut parts: Vec<String> = Vec::new();
            if let Some(label) = row.first().filter(|l| !l.is_empty()) {
                parts.push(sentence(label));
            }
            for (index, value) in row.iter().enumerate().skip(1) {
                if value.is_empty() {
                    continue;
                }
                let name = header.get(index).map_or("", String::as_str);
                let clause = if name.is_empty() {
                    value.clone()
                } else {
                    format!("{name}: {value}")
                };
                parts.push(sentence(&clause));
            }
            if !parts.is_empty() {
                out.push(parts.join(" "));
                out.push(String::new());
            }
        }
    }
    out.join("\n")
}

fn sentence(s: &str) -> String {
    if ends_sentence(s) {
        s.to_string()
    } else {
        format!("{s}.")
    }
}

/// The model's pronunciation is sensitive to case even though speech has none: a paragraph
/// beginning lower-case came out mangled — "founder intervention recorded" was spoken
/// "Sharpen intervention recorded", reproducibly across two seeds. Lower-case openings arise
/// naturally here, because snake_case identifiers become ordinary words.
fn capitalise_opening(p: &str) -> String {
    let mut chars = p.chars();
    match chars.next() {
        Some(first) if first.is_lowercase() => {
            first.to_uppercase().collect::<String>() + chars.as_str()
        }
        _ => p.to_string(),
    }
}

pub fn convert(text: &str, options: &Options) -> String {
    let mut text = strip_front_matter(text);
    text = re::sub_str(&HTML_COMMENT, &text, "");
    text = drop_display_math(&text);
    text = resolve_shortcodes(&text, options.keep_captions);
    // Before paragraph assembly, so table rows never reach the run-on buffer.
    text = render_tables(&text);
    if !options.keep_code {
        text = re::sub_str(&CODE_BLOCK, &text, "");
    }

    let mut paragraphs: Vec<String> = Vec::new();
    let mut buffer: Vec<String> = Vec::new();

    for raw in text.split('\n') {
        let line = raw.trim_end();
        if line.trim().is_empty() {
            flush(&mut buffer, &mut paragraphs);
            continue;
        }

        // A footnote definition. Its marker is stripped from the citing sentence, so the note
        // is spoken where it is written, named, and given its own paragraph gap.
        if let Ok(Some(caps)) = FOOTNOTE_DEF.captures(line) {
            flush(&mut buffer, &mut paragraphs);
            let body = clean_inline(&re::g(&caps, 2));
            if !body.is_empty() {
                paragraphs.push(format!("Footnote {}. {}", re::g(&caps, 1), sentence(&body)));
            }
            continue;
        }

        if line.trim_start().starts_with('#') {
            flush(&mut buffer, &mut paragraphs);
            let heading = clean_inline(line.trim_start().trim_start_matches('#').trim());
            if !heading.is_empty() {
                paragraphs.push(sentence(&heading));
            }
            continue;
        }

        let line = if IS_BLOCKQUOTE.is_match(line).unwrap_or(false) {
            re::sub_str(&BLOCKQUOTE, line, "")
        } else {
            line.to_string()
        };

        if let Ok(Some(caps)) = BULLET.captures(line.as_str()) {
            flush(&mut buffer, &mut paragraphs);
            let item = clean_inline(&re::g(&caps, 1));
            if !item.is_empty() {
                paragraphs.push(sentence(&item));
            }
            continue;
        }

        if is_horizontal_rule(&line) {
            continue;
        }

        let cleaned = clean_inline(&line);
        if !cleaned.is_empty() {
            buffer.push(cleaned);
        }
    }
    flush(&mut buffer, &mut paragraphs);

    let body: Vec<String> = paragraphs
        .iter()
        .map(|p| capitalise_opening(p))
        .filter(|p| !p.trim().is_empty())
        .collect();
    format!("{}\n", body.join("\n\n"))
}

fn flush(buffer: &mut Vec<String>, paragraphs: &mut Vec<String>) {
    if !buffer.is_empty() {
        paragraphs.push(buffer.join(" "));
        buffer.clear();
    }
}
