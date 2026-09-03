//! What to warn about after converting.
//!
//! Markup that reaches the voice is not cosmetic. A `***` that survived cleaning made one
//! chapter read "asterisk, asterisk, asterisk" and then degenerate into a repetition loop,
//! destroying the passage — and it shipped, because the converter was silent. These run
//! always, not behind a flag.

use crate::re::compile;
use crate::tables::PROSE_CAPS_AS_WORDS;
use fancy_regex::Regex;
use once_cell::sync::Lazy;

/// Anything that still looks like markup, with the name a reader would use for it.
static RESIDUE: &[(&str, &str)] = &[
    ("asterisk", r"\*"),
    ("pipe (table)", r"\|"),
    ("backtick", "`"),
    ("bracket", r"[\[\]]"),
    ("heading hash", r"(?m)(?:^|\s)#{1,6}\s"),
    ("blockquote", r"(?:^|\n)\s*>"),
    ("html tag", r"<[a-zA-Z/][^>]*>"),
    ("shortcode", r"\{\{"),
];

static COMPILED: Lazy<Vec<(&'static str, Regex)>> = Lazy::new(|| {
    RESIDUE
        .iter()
        .map(|(label, p)| (*label, compile(p)))
        .collect()
});
static CAPS: Lazy<Regex> = Lazy::new(|| compile(r"\b[A-Z]{3,}\b"));
static SENTENCE_SPLIT: Lazy<Regex> = Lazy::new(|| compile(r"(?<=[.!?])\s+"));

/// One thing worth telling the operator about the narration text.
#[derive(Debug, PartialEq, Eq)]
pub enum Finding {
    /// Markup the voice will read aloud.
    Residue {
        label: &'static str,
        count: usize,
        context: String,
    },
    /// A sentence the AR loop may not finish. The engine splits and retries now, but naming
    /// the sentence points at the cause rather than the symptom.
    LongSentence {
        count: usize,
        longest: usize,
        sample: String,
    },
}

impl std::fmt::Display for Finding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Residue { label, count, context } => write!(
                f,
                "{count} surviving {label} in narration text — the voice will read it: ...{context}..."
            ),
            Self::LongSentence { count, longest, sample } => write!(
                f,
                "{count} sentence(s) over 400 chars; longest {longest}: {sample}..."
            ),
        }
    }
}

pub fn findings(narration: &str) -> Vec<Finding> {
    let mut out = Vec::new();
    for (label, regex) in COMPILED.iter() {
        let count = regex.find_iter(narration).filter(|m| m.is_ok()).count();
        if count == 0 {
            continue;
        }
        let context = match regex.find(narration) {
            Ok(Some(m)) => {
                let start = narration[..m.start()]
                    .char_indices()
                    .rev()
                    .nth(39)
                    .map_or(0, |(i, _)| i);
                let end = narration[m.start()..]
                    .char_indices()
                    .nth(40)
                    .map_or(narration.len(), |(i, _)| m.start() + i);
                narration[start..end].replace('\n', " ")
            }
            _ => String::new(),
        };
        out.push(Finding::Residue {
            label,
            count,
            context,
        });
    }

    let sentences: Vec<&str> = split_sentences(narration);
    let long: Vec<&&str> = sentences.iter().filter(|s| s.len() > 400).collect();
    if let Some(first) = long.first() {
        let longest = long.iter().map(|s| s.len()).max().unwrap_or(0);
        let sample: String = first.chars().take(70).collect();
        out.push(Finding::LongSentence {
            count: long.len(),
            longest,
            sample,
        });
    }
    out
}

fn split_sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut last = 0usize;
    for m in SENTENCE_SPLIT.find_iter(text) {
        let Ok(m) = m else { break };
        out.push(&text[last..m.start()]);
        last = m.end();
    }
    out.push(&text[last..]);
    out
}

/// All-caps tokens the voice will spell rather than read, minus the ones already known to be
/// words. Most are acronyms and spelling them is right; the ones that are English words are
/// not, and nothing in the token itself tells them apart — so this reports and lets the next
/// audit decide. Much cheaper than a listener finding it.
pub fn spelled_out(narration: &str) -> Vec<String> {
    let mut found: Vec<String> = CAPS
        .find_iter(narration)
        .filter_map(Result::ok)
        .map(|m| m.as_str().to_string())
        .filter(|w| !PROSE_CAPS_AS_WORDS.contains(&w.as_str()))
        .collect();
    found.sort();
    found.dedup();
    found
}

/// Length, paragraph count and the duration those imply. ~155 wpm is what these engines
/// produce at speed 1.0 on this material.
pub struct Stats {
    pub words: usize,
    pub chars: usize,
    pub paragraphs: usize,
    pub minutes: f64,
}

pub fn stats(narration: &str) -> Stats {
    let words = narration.split_whitespace().count();
    Stats {
        words,
        chars: narration.len(),
        paragraphs: narration.matches("\n\n").count() + 1,
        minutes: words as f64 / 155.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_surviving_asterisk_is_reported() {
        let found = findings("a * b");
        assert!(matches!(
            found.first(),
            Some(Finding::Residue {
                label: "asterisk",
                count: 1,
                ..
            })
        ));
    }

    #[test]
    fn clean_text_reports_nothing() {
        assert!(findings("An ordinary sentence, with a comma.\n\nAnd another.\n").is_empty());
    }

    #[test]
    fn a_long_sentence_is_named() {
        let long = "word ".repeat(120);
        let found = findings(&long);
        assert!(
            found
                .iter()
                .any(|f| matches!(f, Finding::LongSentence { .. })),
            "{found:?}"
        );
    }

    #[test]
    fn known_words_are_not_reported_as_spelled_out() {
        assert_eq!(spelled_out("The API MUST work"), vec!["API".to_string()]);
    }

    #[test]
    fn stats_count_paragraphs_and_estimate_duration() {
        let s = stats("one two three\n\nfour five\n");
        assert_eq!(s.words, 5);
        assert_eq!(s.paragraphs, 2);
        assert!((s.minutes - 5.0 / 155.0).abs() < 1e-9);
    }
}
