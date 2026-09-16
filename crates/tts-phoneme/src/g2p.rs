//! misaki's driver: tokens in, one phoneme string out.
//!
//! The sentence is walked **right to left** because several lexicon rules depend on what
//! comes next — "the" before a vowel, "to" before a consonant, "used to". Sub-tokens of a
//! hyphenated or apostrophised word are then resolved by trying the longest merge first and
//! backing off, which is why they travel as a group rather than as tokens.

use crate::lexicon::{
    self, apply_stress, is_vowel, non_quote_punct, stress_weight, Hit, Lexicon, TokenContext,
    CONSONANTS, PRIMARY_STRESS, PUNCTS, SUBTOKEN_JUNKS,
};
use crate::tagger::Tagged;
use crate::{Tagger, Tokenizer};
use anyhow::Result;
use fancy_regex::Regex;
use std::path::Path;
use unicode_normalization::UnicodeNormalization;

const UNK: &str = "❓";
const PUNCT_TAGS: [&str; 11] =
    [".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP"];

fn punct_tag_phoneme(tag: &str) -> Option<&'static str> {
    match tag {
        "-LRB-" => Some("("),
        "-RRB-" => Some(")"),
        "``" => Some("\u{201c}"),
        "\"\"" | "''" => Some("\u{201d}"),
        _ => None,
    }
}

/// One source word, and the bytes of the phoneme string it produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WordSpan {
    pub text: String,
    pub phonemes: std::ops::Range<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct MToken {
    pub text: String,
    pub tag: String,
    pub whitespace: String,
    pub phonemes: Option<String>,
    pub is_head: bool,
    pub alias: Option<String>,
    pub stress: Option<f64>,
    pub currency: Option<String>,
    pub num_flags: String,
    pub prespace: bool,
    pub rating: Option<u8>,
}

enum Word {
    One(MToken),
    Many(Vec<MToken>),
}

/// Text, as the merged token spans it, including the whitespace between its parts.
fn span_text(tokens: &[MToken]) -> String {
    let mut s = String::new();
    for (i, tk) in tokens.iter().enumerate() {
        s.push_str(&tk.text);
        if i + 1 < tokens.len() {
            s.push_str(&tk.whitespace);
        }
    }
    s
}

/// `unk` present means the merge also concatenates phonemes — the final assembly — while
/// `None` means it is only a candidate being offered to the lexicon.
fn merge_tokens(tokens: &[MToken], unk: Option<&str>) -> MToken {
    let mut stresses: Vec<f64> = Vec::new();
    for tk in tokens {
        if let Some(s) = tk.stress {
            if !stresses.contains(&s) {
                stresses.push(s);
            }
        }
    }
    let currency = tokens.iter().filter_map(|t| t.currency.clone()).max();
    let any_missing_rating = tokens.iter().any(|t| t.rating.is_none());
    let rating = if any_missing_rating { None } else { tokens.iter().filter_map(|t| t.rating).min() };

    let phonemes = unk.map(|unk| {
        let mut out = String::new();
        for tk in tokens {
            if tk.prespace
                && !out.is_empty()
                && !out.chars().last().unwrap().is_whitespace()
                && tk.phonemes.as_deref().map(|p| !p.is_empty()).unwrap_or(false)
            {
                out.push(' ');
            }
            match &tk.phonemes {
                Some(p) => out.push_str(p),
                None => out.push_str(unk),
            }
        }
        out
    });

    // The tag of whichever part carries the most capital letters, first one winning ties.
    let tag = tokens
        .iter()
        .map(|tk| {
            let score: usize =
                tk.text.chars().map(|c| if c.is_lowercase() || !c.is_alphabetic() { 1 } else { 2 }).sum();
            (score, tk.tag.clone())
        })
        .enumerate()
        .max_by(|a, b| a.1 .0.cmp(&b.1 .0).then(b.0.cmp(&a.0)))
        .map(|(_, (_, tag))| tag)
        .unwrap_or_default();

    let mut num_flags: Vec<char> = tokens.iter().flat_map(|t| t.num_flags.chars()).collect();
    num_flags.sort_unstable();
    num_flags.dedup();

    MToken {
        text: span_text(tokens),
        tag,
        whitespace: tokens.last().map(|t| t.whitespace.clone()).unwrap_or_default(),
        phonemes,
        is_head: tokens[0].is_head,
        alias: None,
        stress: if stresses.len() == 1 { Some(stresses[0]) } else { None },
        currency,
        num_flags: num_flags.into_iter().collect(),
        prespace: tokens[0].prespace,
        rating,
    }
}

pub struct G2P {
    tokenizer: Tokenizer,
    tagger: Tagger,
    lexicon: Lexicon,
    subtoken: Regex,
}

impl G2P {
    pub fn load(dir: &Path, british: bool) -> Result<Self> {
        Ok(Self {
            tokenizer: Tokenizer::load(dir)?,
            tagger: Tagger::load(dir)?,
            lexicon: Lexicon::load(dir, british)?,
            subtoken: Regex::new(
                r"^['\u{2018}\u{2019}]+|\p{Lu}(?=\p{Lu}\p{Ll})|(?:^-)?(?:\d?[,.]?\d)+|[-_]+|['\u{2018}\u{2019}]{2,}|\p{L}*?(?:['\u{2018}\u{2019}]\p{L})*?\p{Ll}(?=\p{Lu})|\p{L}+(?:['\u{2018}\u{2019}]\p{L})*|[^-_\p{L}'\u{2018}\u{2019}\d]|['\u{2018}\u{2019}]+$",
            )?,
        })
    }

    fn subtokenize(&self, word: &str) -> Vec<String> {
        self.subtoken
            .find_iter(word)
            .filter_map(|m| m.ok())
            .map(|m| m.as_str().to_string())
            .collect()
    }

    /// Tokenize and tag. Everything downstream reads only `text`, `tag` and `whitespace`.
    fn tokenize(&self, text: &str) -> Vec<MToken> {
        let tokens = self.tokenizer.tokenize(text);
        let tagged: Vec<Tagged> = tokens
            .iter()
            .map(|t| Tagged {
                text: &t.text,
                has_space: !t.whitespace.is_empty(),
                norm: t.norm.as_deref(),
            })
            .collect();
        let tags = self.tagger.tag(&tagged, &self.tokenizer.vocab);
        tokens
            .iter()
            .zip(tags)
            .map(|(t, tag)| MToken {
                text: t.text.clone(),
                tag: tag.to_string(),
                whitespace: t.whitespace.to_string(),
                is_head: true,
                ..Default::default()
            })
            .collect()
    }

    /// Split words into sub-tokens, mark punctuation and currency, and group the pieces
    /// that belong to one written word so the resolver can try merges over them.
    fn retokenize(&self, tokens: Vec<MToken>) -> Vec<Word> {
        let mut words: Vec<Word> = Vec::new();
        let mut currency: Option<String> = None;
        let n = tokens.len();
        for (i, token) in tokens.iter().enumerate() {
            let mut tks: Vec<MToken> = if token.alias.is_none() && token.phonemes.is_none() {
                self.subtokenize(&token.text)
                    .into_iter()
                    .map(|t| MToken {
                        text: t,
                        tag: token.tag.clone(),
                        whitespace: String::new(),
                        is_head: true,
                        num_flags: token.num_flags.clone(),
                        stress: token.stress,
                        ..Default::default()
                    })
                    .collect()
            } else {
                vec![token.clone()]
            };
            if tks.is_empty() {
                continue;
            }
            let last = tks.len() - 1;
            tks[last].whitespace = token.whitespace.clone();

            for j in 0..tks.len() {
                let tag = tks[j].tag.clone();
                let text = tks[j].text.clone();
                if tks[j].alias.is_some() || tks[j].phonemes.is_some() {
                } else if tag == "$" && lexicon::currency_symbol(&text) {
                    currency = Some(text.clone());
                    tks[j].phonemes = Some(String::new());
                    tks[j].rating = Some(4);
                } else if tag == ":" && (text == "-" || text == "–") {
                    tks[j].phonemes = Some("—".into());
                    tks[j].rating = Some(3);
                } else if PUNCT_TAGS.contains(&tag.as_str())
                    && !text.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_uppercase())
                {
                    tks[j].phonemes = Some(match punct_tag_phoneme(&tag) {
                        Some(p) => p.to_string(),
                        None => text.chars().filter(|c| PUNCTS.contains(*c)).collect(),
                    });
                    tks[j].rating = Some(4);
                } else if currency.is_some() {
                    if tag != "CD" {
                        currency = None;
                    } else if j + 1 == tks.len()
                        && (i + 1 == n || tokens[i + 1].tag != "CD")
                    {
                        tks[j].currency = currency.clone();
                    }
                } else if j > 0
                    && j + 1 < tks.len()
                    && text == "2"
                    && tks[j - 1].text.chars().last().map(|c| c.is_alphabetic()).unwrap_or(false)
                    && tks[j + 1].text.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false)
                {
                    tks[j].alias = Some("to".into());
                }

                let tk = tks[j].clone();
                if tk.alias.is_some() || tk.phonemes.is_some() {
                    words.push(Word::One(tk));
                } else if matches!(words.last(), Some(Word::Many(g)) if g.last().unwrap().whitespace.is_empty())
                {
                    let Some(Word::Many(g)) = words.last_mut() else { unreachable!() };
                    let mut tk = tk;
                    tk.is_head = false;
                    g.push(tk);
                } else if tk.whitespace.is_empty() {
                    words.push(Word::Many(vec![tk]));
                } else {
                    words.push(Word::One(tk));
                }
            }
        }
        words
            .into_iter()
            .map(|w| match w {
                Word::Many(mut g) if g.len() == 1 => Word::One(g.pop().unwrap()),
                other => other,
            })
            .collect()
    }

    /// What the token to the left needs to know: whether the next spoken sound is a vowel,
    /// and whether the next word is "to".
    fn token_context(ctx: TokenContext, ps: Option<&str>, token: &MToken) -> TokenContext {
        let mut future_vowel = ctx.future_vowel;
        if let Some(ps) = ps.filter(|p| !p.is_empty()) {
            for c in ps.chars() {
                if is_vowel(c) || CONSONANTS.contains(c) || non_quote_punct(c) {
                    future_vowel = if non_quote_punct(c) { None } else { Some(is_vowel(c)) };
                    break;
                }
            }
        }
        let future_to = token.text == "to"
            || token.text == "To"
            || (token.text == "TO" && (token.tag == "TO" || token.tag == "IN"));
        TokenContext { future_vowel, future_to }
    }

    fn lexicon_lookup(&self, tk: &MToken, ctx: &TokenContext) -> Option<Hit> {
        let source = tk.alias.as_deref().unwrap_or(&tk.text);
        let word: String = source
            .replace('\u{2018}', "'")
            .replace('\u{2019}', "'")
            .nfkc()
            .map(numeric_if_needed)
            .collect();
        let lower = word.to_lowercase();
        let stress = if word == lower {
            None
        } else if word == word.to_uppercase() {
            Some(2.0)
        } else {
            Some(0.5)
        };
        if let Some((ps, rating)) = self.lexicon.get_word(&word, &tk.tag, stress, ctx) {
            let ps = self.lexicon.with_currency(&ps, tk.currency.as_deref());
            return Some((apply_stress(&ps, tk.stress), rating));
        }
        if Lexicon::is_number(&word, tk.is_head) {
            let (ps, rating) =
                self.lexicon.get_number(&word, tk.currency.as_deref(), tk.is_head, &tk.num_flags)?;
            return Some((apply_stress(&ps, tk.stress), rating));
        }
        None
    }

    /// Stress within one written word: when several parts carry primary stress, demote the
    /// lighter half, so "co-worker" is not two stressed words.
    fn resolve_tokens(tokens: &mut [MToken]) {
        let text = span_text(tokens);
        let mut classes: Vec<u8> = text
            .chars()
            .filter(|c| !SUBTOKEN_JUNKS.contains(*c))
            .map(|c| if c.is_alphabetic() { 0 } else if c.is_ascii_digit() { 1 } else { 2 })
            .collect();
        classes.sort_unstable();
        classes.dedup();
        let prespace = text.contains(' ') || text.contains('/') || classes.len() > 1;

        let n = tokens.len();
        for i in 0..n {
            if tokens[i].phonemes.is_none() {
                if i == n - 1 && tokens[i].text.chars().count() == 1
                    && tokens[i].text.chars().all(non_quote_punct)
                {
                    tokens[i].phonemes = Some(tokens[i].text.clone());
                    tokens[i].rating = Some(3);
                } else if !tokens[i].text.is_empty()
                    && tokens[i].text.chars().all(|c| SUBTOKEN_JUNKS.contains(c))
                {
                    tokens[i].phonemes = Some(String::new());
                    tokens[i].rating = Some(3);
                }
            } else if i > 0 {
                tokens[i].prespace = prespace;
            }
        }
        if prespace {
            return;
        }
        let mut indices: Vec<(bool, usize, usize)> = tokens
            .iter()
            .enumerate()
            .filter_map(|(i, tk)| {
                tk.phonemes.as_deref().filter(|p| !p.is_empty()).map(|p| {
                    (p.contains(PRIMARY_STRESS), stress_weight(p), i)
                })
            })
            .collect();
        if indices.len() == 2 && tokens[indices[0].2].text.chars().count() == 1 {
            let i = indices[1].2;
            let ps = tokens[i].phonemes.clone().unwrap();
            tokens[i].phonemes = Some(apply_stress(&ps, Some(-0.5)));
            return;
        }
        let stressed = indices.iter().filter(|(b, _, _)| *b).count();
        if indices.len() < 2 || stressed <= (indices.len() + 1) / 2 {
            return;
        }
        indices.sort();
        for (_, _, i) in &indices[..indices.len() / 2] {
            let ps = tokens[*i].phonemes.clone().unwrap();
            tokens[*i].phonemes = Some(apply_stress(&ps, Some(-0.5)));
        }
    }

    pub fn phonemize(&self, text: &str) -> String {
        self.phonemize_report(text).0
    }

    /// The phonemes, and which slice of them each source word produced.
    ///
    /// The spans are what makes alignment free. A duration-predicting model already computes
    /// a length for every phoneme on the way to the audio; knowing which word each phoneme
    /// belongs to turns that into a word clock without a second model, a second pass over the
    /// audio, or a recogniser that has to be told what it is listening to.
    pub fn phonemize_spans(&self, text: &str) -> (String, Vec<WordSpan>) {
        let (phonemes, _, spans) = self.run(text, true);
        (phonemes, spans)
    }

    /// misaki exactly, with nothing derived — what the byte-identity gate compares.
    pub fn phonemize_strict(&self, text: &str) -> String {
        self.run(text, false).0
    }

    /// Also returns the words nothing could pronounce — the lint, rather than a guess.
    pub fn phonemize_report(&self, text: &str) -> (String, Vec<String>) {
        let (phonemes, unknown, _) = self.run(text, true);
        (phonemes, unknown)
    }

    fn run(&self, text: &str, derive_unknown: bool) -> (String, Vec<String>, Vec<WordSpan>) {
        let mut unknown: Vec<String> = Vec::new();
        let mut words = self.retokenize(self.tokenize(text.trim_start()));
        let mut ctx = TokenContext::default();
        for w in words.iter_mut().rev() {
            match w {
                Word::One(tk) => {
                    if tk.phonemes.is_none() {
                        let hit = self
                            .lexicon_lookup(tk, &ctx)
                            .or_else(|| self.derive(&tk.text, derive_unknown, &mut unknown));
                        if let Some((ps, rating)) = hit {
                            tk.phonemes = Some(ps);
                            tk.rating = Some(rating);
                        }
                    }
                    ctx = Self::token_context(ctx, tk.phonemes.as_deref(), tk);
                }
                Word::Many(group) => {
                    // Longest merge first, backing off from the left, then dropping the
                    // rightmost part — the order misaki uses, and the reason a hyphenated
                    // compound resolves as a word when the lexicon has it and as parts
                    // when it does not.
                    let (mut left, mut right) = (0usize, group.len());
                    while left < right {
                        let blocked = group[left..right]
                            .iter()
                            .any(|tk| tk.alias.is_some() || tk.phonemes.is_some());
                        let candidate =
                            if blocked { None } else { Some(merge_tokens(&group[left..right], None)) };
                        let hit = candidate.as_ref().and_then(|tk| self.lexicon_lookup(tk, &ctx));
                        if let Some((ps, rating)) = hit {
                            group[left].phonemes = Some(ps.clone());
                            group[left].rating = Some(rating);
                            for x in &mut group[left + 1..right] {
                                x.phonemes = Some(String::new());
                            }
                            ctx = Self::token_context(ctx, Some(&ps), candidate.as_ref().unwrap());
                            right = left;
                            left = 0;
                        } else if left + 1 < right {
                            left += 1;
                        } else {
                            right -= 1;
                            let derived = {
                                let tk = &group[right];
                                if tk.phonemes.is_none()
                                    && !(!tk.text.is_empty()
                                        && tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)))
                                {
                                    self.derive(&tk.text, derive_unknown, &mut unknown)
                                } else {
                                    None
                                }
                            };
                            let tk = &mut group[right];
                            if tk.phonemes.is_none() {
                                if !tk.text.is_empty()
                                    && tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c))
                                {
                                    tk.phonemes = Some(String::new());
                                    tk.rating = Some(3);
                                } else if let Some((ps, rating)) = derived {
                                    tk.phonemes = Some(ps);
                                    tk.rating = Some(rating);
                                }
                            }
                            left = 0;
                        }
                    }
                    Self::resolve_tokens(group);
                }
            }
        }

        let merged: Vec<MToken> = words
            .into_iter()
            .map(|w| match w {
                Word::One(tk) => tk,
                Word::Many(g) => merge_tokens(&g, Some(UNK)),
            })
            .collect();
        let mut out = String::new();
        let mut spans: Vec<WordSpan> = Vec::with_capacity(merged.len());
        for tk in &merged {
            let start = out.len();
            match &tk.phonemes {
                // The 1.0 vocabulary has no flap or glottal stop; both are spelled with
                // the letters the model was trained on.
                Some(p) => out.push_str(&p.replace('ɾ', "T").replace('ʔ', "t")),
                None => out.push_str(UNK),
            }
            // A part of a merged compound carries no phonemes of its own — the head took
            // them all — so it joins the head rather than becoming a word that is said in no
            // time at all.
            match (out.len() == start, spans.last_mut()) {
                (true, Some(last)) => {
                    last.text.push_str(&tk.text);
                }
                _ => spans.push(WordSpan {
                    text: tk.text.clone(),
                    phonemes: start..out.len(),
                }),
            }
            out.push_str(&tk.whitespace);
        }
        (out, unknown, spans)
    }

    /// The derivation stage, plus the lint. A word reported here is one the voice cannot
    /// say; naming it beats inventing a pronunciation for it.
    fn derive(&self, text: &str, derive_unknown: bool, unknown: &mut Vec<String>) -> Option<Hit> {
        if text.is_empty() {
            return None;
        }
        if !derive_unknown {
            if !unknown.iter().any(|w| w == text) {
                unknown.push(text.to_string());
            }
            return None;
        }
        if !text.chars().any(char::is_alphabetic) {
            return None;
        }
        match self.lexicon.derive(text) {
            Some(hit) => Some(hit),
            None => {
                if !unknown.iter().any(|w| w == text) {
                    unknown.push(text.to_string());
                }
                None
            }
        }
    }
}

/// Fold a Unicode digit onto its ASCII value; leave anything else alone.
fn numeric_if_needed(c: char) -> char {
    match c.to_digit(10) {
        Some(d) if !c.is_ascii_digit() => char::from_digit(d, 10).unwrap_or(c),
        _ => c,
    }
}
