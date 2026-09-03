//! The prose a browser would show, in DOM order, and the tokeniser both sides share.
//!
//! Same removals as narration except the ones that only affect *speech*: no comma for em
//! dashes, no full stop appended to headings. The player discovers page words by walking the
//! DOM, so a manifest index has to point into rendered-prose word order — and any difference
//! between these two tokenisers shows up as unmapped words rather than as a warning.

use crate::re::{self, compile};
use crate::source::{
    caption_paragraph, is_horizontal_rule, strip_front_matter, CODE_BLOCK, HTML_COMMENT, SHORTCODE,
};
use fancy_regex::Regex;
use once_cell::sync::Lazy;

static HEADING: Lazy<Regex> = Lazy::new(|| compile(r"^#+\s*"));
static BLOCKQUOTE: Lazy<Regex> = Lazy::new(|| compile(r"^\s*>\s?"));
static BULLET: Lazy<Regex> = Lazy::new(|| compile(r"^\s*(?:[-*+]|\d+\.)\s+"));
static CHECKBOX: Lazy<Regex> = Lazy::new(|| compile(r"\[[ xX]?\]\s*"));
static BRACKETS: Lazy<Regex> = Lazy::new(|| compile(r"\[([^\[\]]*)\]"));
static HTML_TAG: Lazy<Regex> = Lazy::new(|| {
    compile(r"(?i)</?(?:br|em|strong|span|div|a|img|sup|sub|code|pre|p|ul|ol|li|hr)\b[^>]*/?>")
});
static PLACEHOLDER: Lazy<Regex> = Lazy::new(|| compile(r"<([A-Za-z][\w -]*)>"));
static CODE_SPAN: Lazy<Regex> = Lazy::new(|| compile(r"`([^`]*)`"));
static IMAGE: Lazy<Regex> = Lazy::new(|| compile(r"!\[[^\]]*\]\([^)]*\)"));
static LINK: Lazy<Regex> = Lazy::new(|| compile(r"\[([^\]]*)\]\([^)]*\)"));
static BOLD: Lazy<Regex> = Lazy::new(|| compile(r"\*\*([^*]+)\*\*"));
static ITALIC: Lazy<Regex> = Lazy::new(|| compile(r"(?<!\*)\*([^*]+)\*(?!\*)"));
static UNDERSCORE_ITALIC: Lazy<Regex> =
    Lazy::new(|| compile(r"(?<![A-Za-z0-9])_([^\s_][^_]*?)_(?![A-Za-z0-9])"));
static SNAKE_CASE: Lazy<Regex> = Lazy::new(|| compile(r"(?<=[A-Za-z0-9])_(?=[A-Za-z0-9])"));
static ARROW: Lazy<Regex> = Lazy::new(|| compile(r"\s*(?:->|=>|\u{2192}|\u{21d2})\s*"));

/// Matches `wordPattern` in the site's `audio-player.js`, so both sides tokenise identically.
static WORD: Lazy<Regex> = Lazy::new(|| compile(r"[A-Za-z0-9]+(?:['\u{2019}][A-Za-z0-9]+)?"));
static NON_WORD: Lazy<Regex> = Lazy::new(|| compile(r"[^a-z0-9']"));

pub fn words(text: &str) -> Vec<String> {
    WORD.find_iter(text)
        .filter_map(Result::ok)
        .map(|m| m.as_str().to_string())
        .collect()
}

pub fn normalize_word(w: &str) -> String {
    let lowered = w.to_lowercase().replace('\u{2019}', "'");
    re::sub_str(&NON_WORD, &lowered, "")
}

pub fn page_text(md: &str) -> String {
    let mut md = strip_front_matter(md);
    md = re::sub_str(&HTML_COMMENT, &md, "");
    // A figure contributes only its caption, which is what actually renders as text.
    md = re::sub(&SHORTCODE, &md, |c| caption_paragraph(&re::g(c, 0)));
    md = re::sub_str(&CODE_BLOCK, &md, "");

    let mut out: Vec<String> = Vec::new();
    for raw in md.split('\n') {
        let mut line = raw.trim().to_string();
        if line.is_empty() {
            continue;
        }
        line = re::sub_str(&HEADING, &line, "");
        line = re::sub_str(&BLOCKQUOTE, &line, "");
        line = re::sub_str(&BULLET, &line, "");
        line = re::sub_str(&CHECKBOX, &line, "");
        line = re::sub_str(&BRACKETS, &line, "${1}");
        line = re::sub_str(&HTML_TAG, &line, "");
        line = re::sub_str(&PLACEHOLDER, &line, "${1}");
        if is_horizontal_rule(&line) {
            continue;
        }
        line = re::sub_str(&CODE_SPAN, &line, "${1}");
        line = re::sub_str(&IMAGE, &line, "");
        line = re::sub_str(&LINK, &line, "${1}");
        line = re::sub_str(&BOLD, &line, "${1}");
        line = re::sub_str(&ITALIC, &line, "${1}");
        line = re::sub_str(&UNDERSCORE_ITALIC, &line, "${1}");
        line = re::sub_str(&SNAKE_CASE, &line, " ");
        line = re::sub_str(&ARROW, &line, ", then ");
        out.push(line);
    }
    out.join(" ")
}

/// The word map the site's player consumes: page words in rendered order, spoken words, and
/// the index from each spoken word into the page.
pub fn word_map(md: &str, narration: &str) -> serde_json::Value {
    let page_tokens = words(&page_text(md));
    let spoken_tokens = words(narration);
    let page_norm: Vec<String> = page_tokens.iter().map(|w| normalize_word(w)).collect();
    let spoken_norm: Vec<String> = spoken_tokens.iter().map(|w| normalize_word(w)).collect();
    let mapping = crate::align::align_tokens(&spoken_norm, &page_norm);
    let mapped = mapping.iter().filter(|m| **m >= 0).count();
    let total = spoken_tokens.len().max(1);

    let pair = |t: &[String], n: &[String]| -> Vec<serde_json::Value> {
        t.iter()
            .zip(n)
            .map(|(text, normalized)| serde_json::json!({"text": text, "normalized": normalized}))
            .collect()
    };
    serde_json::json!({
        "pageWords": pair(&page_tokens, &page_norm),
        "spokenWords": pair(&spoken_tokens, &spoken_norm),
        "spokenToPageWord": mapping,
        "pageMapping": {
            // Named for what the player expects, which is the key it looks up rather than a
            // description of the algorithm; the matcher replaced the walk this names.
            "algorithm": "two-pointer-lookahead-24",
            "spokenWordCount": spoken_tokens.len(),
            "pageWordCount": page_tokens.len(),
            "mappedSpokenWordCount": mapped,
            "unmappedSpokenWordCount": spoken_tokens.len() - mapped,
            "mappedShare": (mapped as f64 / total as f64 * 1e6).round() / 1e6,
        },
    })
}
