//! spaCy's tokenizer, driven by the rules exported from the pinned model.
//!
//! The patterns and the exception table are data (`tokenizer.json`); only the affix loop is
//! code. Retyping the rules would give something that tokenizes English plausibly and
//! disagrees with the tagger's training distribution in ways no test names.

use crate::vocab::Vocab;
use anyhow::{Context, Result};
use fancy_regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Deserialize, Clone)]
pub struct Special {
    pub orth: String,
    pub norm: Option<String>,
}

#[derive(Deserialize)]
struct Rules {
    prefix_search: String,
    suffix_search: String,
    infix_finditer: String,
    url_match: Option<String>,
    exceptions: HashMap<String, Vec<Special>>,
    lexeme_norm: HashMap<String, String>,
    symbols: HashMap<String, u64>,
    base_norms: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub text: String,
    /// " " when the token is followed by a single space, "" otherwise.
    pub whitespace: &'static str,
    /// Carried from the exception table; `None` means derive it from the lexeme norms.
    pub norm: Option<String>,
}

pub struct Tokenizer {
    prefix: Regex,
    suffix: Regex,
    infix: Regex,
    url: Option<Regex>,
    specials: HashMap<String, Vec<Special>>,
    /// Special keys indexed by the token sequence affix-splitting alone produces for them,
    /// bucketed on the first token. This is the second pass: affix splitting turns "id."
    /// into `i d .` (because "id" is the contraction rule), and only a match over the
    /// *result* puts "d." back together the way spaCy does.
    rematch: HashMap<String, Vec<(Vec<String>, String)>>,
    pub vocab: Vocab,
}

/// Python's `re` spells a code point `\uXXXX`; the Rust engine spells it `\x{XXXX}`. The
/// character-class patterns are full of them, and an untranslated one compiles to a literal
/// `u` — a pattern that still matches things, just the wrong things.
fn translate(pattern: &str) -> String {
    let mut out = String::with_capacity(pattern.len());
    let b = pattern.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\\'
            && i + 5 < b.len()
            && b[i + 1] == b'u'
            && b[i + 2..i + 6].iter().all(u8::is_ascii_hexdigit)
        {
            out.push_str("\\x{");
            out.push_str(&pattern[i + 2..i + 6]);
            out.push('}');
            i += 6;
        } else {
            let ch = pattern[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

impl Tokenizer {
    pub fn load(dir: &Path) -> Result<Self> {
        let raw = std::fs::read(dir.join("tokenizer.json"))
            .with_context(|| format!("reading tokenizer.json from {}", dir.display()))?;
        let r: Rules = serde_json::from_slice(&raw)?;
        let mut tk = Self {
            prefix: Regex::new(&translate(&r.prefix_search))?,
            suffix: Regex::new(&translate(&r.suffix_search))?,
            infix: Regex::new(&translate(&r.infix_finditer))?,
            url: r
                .url_match
                .as_deref()
                .map(|p| Regex::new(&translate(p)))
                .transpose()?,
            specials: r.exceptions,
            vocab: Vocab {
                symbols: r.symbols,
                base_norms: r.base_norms,
                lexeme_norm: r
                    .lexeme_norm
                    .into_iter()
                    .map(|(k, v)| Ok((k.parse::<u64>()?, v)))
                    .collect::<Result<_>>()?,
            },
            rematch: HashMap::new(),
        };
        tk.build_rematch();
        Ok(tk)
    }

    /// Register the keys spaCy registers. `faster_heuristics` keeps out any key that
    /// affix splitting would leave intact ("gonna"), because re-matching those can only
    /// re-find what the first pass already produced.
    fn build_rematch(&mut self) {
        let keys: Vec<String> = self.specials.keys().cloned().collect();
        for key in keys {
            let registered = self.find_prefix(&key) != 0
                || self.find_suffix(&key) != 0
                || self.infix.find(&key).ok().flatten().is_some()
                || key.contains(' ');
            if !registered {
                continue;
            }
            let mut seq = Vec::new();
            self.tokenize_span(&mut seq, &key, false);
            let orths: Vec<String> = seq.into_iter().map(|t| t.text).collect();
            if orths.len() < 2 {
                continue;
            }
            self.rematch
                .entry(orths[0].clone())
                .or_default()
                .push((orths, key));
        }
    }

    fn find_prefix(&self, s: &str) -> usize {
        self.prefix
            .find(s)
            .ok()
            .flatten()
            .map(|m| m.end())
            .unwrap_or(0)
    }

    fn find_suffix(&self, s: &str) -> usize {
        self.suffix
            .find(s)
            .ok()
            .flatten()
            .map(|m| s.len() - m.start())
            .unwrap_or(0)
    }

    fn is_url(&self, s: &str) -> bool {
        self.url
            .as_ref()
            .map(|r| r.is_match(s).unwrap_or(false))
            .unwrap_or(false)
    }

    fn push_special(&self, out: &mut Vec<Token>, entries: &[Special]) {
        for e in entries {
            out.push(Token {
                text: e.orth.clone(),
                whitespace: "",
                norm: e.norm.clone(),
            });
        }
    }

    /// One whitespace-delimited span, minus its affixes.
    fn tokenize_span(&self, out: &mut Vec<Token>, span: &str, with_specials: bool) {
        let mut prefixes: Vec<String> = Vec::new();
        let mut suffixes: Vec<String> = Vec::new();
        let mut s = span.to_string();
        let mut last = usize::MAX;
        while !s.is_empty() && s.len() != last {
            if with_specials && self.specials.contains_key(&s) {
                break;
            }
            last = s.len();
            let pre = self.find_prefix(&s);
            let mut minus_pre = String::new();
            if pre != 0 {
                minus_pre = s[pre..].to_string();
                if with_specials && !minus_pre.is_empty() && self.specials.contains_key(&minus_pre)
                {
                    prefixes.push(s[..pre].to_string());
                    s = minus_pre;
                    break;
                }
            }
            let suf = self.find_suffix(&s[pre..]);
            let mut minus_suf = String::new();
            if suf != 0 {
                minus_suf = s[..s.len() - suf].to_string();
                if with_specials && !minus_suf.is_empty() && self.specials.contains_key(&minus_suf)
                {
                    suffixes.push(s[s.len() - suf..].to_string());
                    s = minus_suf;
                    break;
                }
            }
            if pre != 0 && suf != 0 && pre + suf <= s.len() {
                prefixes.push(s[..pre].to_string());
                suffixes.push(s[s.len() - suf..].to_string());
                s = s[pre..s.len() - suf].to_string();
            } else if pre != 0 {
                prefixes.push(s[..pre].to_string());
                s = minus_pre;
            } else if suf != 0 {
                suffixes.push(s[s.len() - suf..].to_string());
                s = minus_suf;
            }
        }

        for p in &prefixes {
            out.push(Token {
                text: p.clone(),
                whitespace: "",
                norm: None,
            });
        }
        if !s.is_empty() {
            if let Some(entries) = self.specials.get(&s).filter(|_| with_specials) {
                self.push_special(out, entries);
            } else if self.is_url(&s) {
                out.push(Token {
                    text: s.clone(),
                    whitespace: "",
                    norm: None,
                });
            } else {
                let matches: Vec<(usize, usize)> = self
                    .infix
                    .find_iter(&s)
                    .filter_map(|m| m.ok())
                    .map(|m| (m.start(), m.end()))
                    .collect();
                if matches.is_empty() {
                    out.push(Token {
                        text: s.clone(),
                        whitespace: "",
                        norm: None,
                    });
                } else {
                    let mut start = 0usize;
                    for (a, b) in matches {
                        if a == 0 {
                            continue;
                        }
                        if a != start {
                            out.push(Token {
                                text: s[start..a].into(),
                                whitespace: "",
                                norm: None,
                            });
                        }
                        if a != b {
                            out.push(Token {
                                text: s[a..b].into(),
                                whitespace: "",
                                norm: None,
                            });
                        }
                        start = b;
                    }
                    if start < s.len() {
                        out.push(Token {
                            text: s[start..].into(),
                            whitespace: "",
                            norm: None,
                        });
                    }
                }
            }
        }
        for suffix in suffixes.iter().rev() {
            out.push(Token {
                text: suffix.clone(),
                whitespace: "",
                norm: None,
            });
        }
    }

    fn emit(&self, out: &mut Vec<Token>, span: &str) {
        match self.specials.get(span) {
            Some(entries) => self.push_special(out, entries),
            None => self.tokenize_span(out, span, true),
        }
    }

    /// spaCy's second pass: find special keys in the already-split token stream and put
    /// them back. Longest match wins, then leftmost, and a span is dropped if either end
    /// is already inside a kept one.
    fn apply_special_cases(&self, tokens: Vec<Token>) -> Vec<Token> {
        let mut spans: Vec<(usize, usize, &str)> = Vec::new();
        for i in 0..tokens.len() {
            let Some(cands) = self.rematch.get(&tokens[i].text) else {
                continue;
            };
            for (seq, key) in cands {
                if i + seq.len() <= tokens.len()
                    && seq.iter().zip(&tokens[i..]).all(|(a, b)| *a == b.text)
                {
                    spans.push((i, i + seq.len(), key.as_str()));
                }
            }
        }
        if spans.is_empty() {
            return tokens;
        }
        spans.sort_by(|a, b| (b.1 - b.0).cmp(&(a.1 - a.0)).then(a.0.cmp(&b.0)));
        let mut seen = vec![false; tokens.len()];
        let mut kept: Vec<(usize, usize, &str)> = Vec::new();
        for s in spans {
            if !seen[s.0] && !seen[s.1 - 1] {
                kept.push(s);
            }
            seen[s.0..s.1].iter_mut().for_each(|v| *v = true);
        }
        kept.sort_by_key(|s| s.0);

        let mut out: Vec<Token> = Vec::with_capacity(tokens.len());
        let mut i = 0;
        let mut next = 0;
        while i < tokens.len() {
            if next < kept.len() && kept[next].0 == i {
                let (start, end, key) = kept[next];
                next += 1;
                match self.specials.get(key) {
                    Some(entries) => {
                        let tail = tokens[end - 1].whitespace;
                        self.push_special(&mut out, entries);
                        if let Some(t) = out.last_mut() {
                            t.whitespace = tail;
                        }
                    }
                    None => out.extend_from_slice(&tokens[start..end]),
                }
                i = end;
            } else {
                out.push(tokens[i].clone());
                i += 1;
            }
        }
        out
    }

    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut out: Vec<Token> = Vec::new();
        if text.is_empty() {
            return out;
        }
        // Spans of whitespace and non-whitespace alternately, with single spaces dropped
        // into the preceding token's trailing-space flag rather than emitted. A run of two
        // spaces still produces a token for the second one, which the tagger sees as _SP.
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let mut in_ws = chars[0].1.is_whitespace();
        let mut start = 0usize;
        for &(i, c) in &chars {
            if c.is_whitespace() != in_ws {
                if start < i {
                    self.emit(&mut out, &text[start..i]);
                }
                if c == ' ' {
                    if let Some(t) = out.last_mut() {
                        t.whitespace = " ";
                    }
                    start = i + 1;
                } else {
                    start = i;
                }
                in_ws = !in_ws;
            }
        }
        if start < text.len() {
            self.emit(&mut out, &text[start..]);
            if let Some(t) = out.last_mut() {
                t.whitespace = if text.ends_with(' ') && !in_ws {
                    " "
                } else {
                    ""
                };
            }
        }
        self.apply_special_cases(out)
    }
}

pub fn load(dir: &Path) -> Result<Tokenizer> {
    Tokenizer::load(dir)
}
