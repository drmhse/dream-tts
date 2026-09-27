//! Text as the pieces a fine-tuned talker was trained on, each BPE-encoded alone, reference
//! transcript included. Whole-word BPE hands the talker English-shaped pieces (Swahili "bunifu"
//! is `b` + `unifu`, spoken "bimoni").
//!
//! [`Scheme`] is a language's orthography, read from the adapter's `meta::scheme` — the same
//! JSON as `references/qwen3tts/langs/<language>.json`, and the same rules as its
//! `orthography.py`. [`pieces`] is the older plain open-syllable cut (`meta::syllables`).

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;

fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U')
}

/// Each word's first piece keeps its leading whitespace, so it still encodes as a word start.
pub fn pieces(text: &str) -> Vec<String> {
    cut(text, is_vowel)
}

/// Cut after every vowel a letter follows.
fn cut(text: &str, vowel: impl Fn(char) -> bool) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut cur = String::new();
    for (i, &c) in chars.iter().enumerate() {
        if c.is_whitespace() && !cur.trim().is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        cur.push(c);
        if vowel(c) && chars.get(i + 1).is_some_and(|n| n.is_alphabetic()) {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

#[derive(Debug, Deserialize)]
struct Spec {
    language: String,
    row: u32,
    vowels: String,
    #[serde(default)]
    apostrophes: String,
    #[serde(default)]
    map: BTreeMap<String, String>,
    #[serde(default)]
    onsets: Vec<String>,
    #[serde(default)]
    syllabic: String,
    #[serde(default)]
    glides: String,
    #[serde(default)]
    isolate_map: bool,
}

/// A language's orthography. Per open syllable, in order: an onset in `map` becomes its own
/// piece (text streams a token a frame, so Swahili ng' is ŋ, or the talker has committed to
/// [ŋg] on "ng" before the apostrophe arrives); an onset in `onsets` is split off (" mbi" is
/// ' mb'+'i', one Qwen token, not ' m'+'bi', the shape of syllabic m, read "mibili"); a
/// `syllabic` nasal before a consonant stands alone.
#[derive(Debug)]
pub struct Scheme {
    pub language: String,
    pub row: u32,
    vowels: Vec<char>,
    apostrophes: Vec<char>,
    map: Vec<(String, String)>,
    onsets: Vec<String>,
    syllabic: Vec<char>,
    glides: Vec<char>,
    /// A mapped onset is always one lone lowercase token, its leading space a piece of its own:
    /// 'Ŋ', ' ŋ' and 'ŋ' are three tokens, and sentence-initial 'Ŋ', the rarest, was dropped.
    isolate_map: bool,
}

impl Scheme {
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let s: Spec = serde_json::from_slice(bytes).context("parsing an orthography scheme")?;
        let mut map: Vec<_> = s.map.into_iter().collect();
        map.sort_by_key(|(k, _)| std::cmp::Reverse(k.chars().count()));
        let mut onsets = s.onsets;
        onsets.sort_by_key(|o| std::cmp::Reverse(o.chars().count()));
        Ok(Self {
            language: s.language.to_lowercase(),
            row: s.row,
            vowels: s.vowels.chars().flat_map(|c| [c].into_iter().chain(c.to_uppercase())).collect(),
            apostrophes: s.apostrophes.chars().collect(),
            map,
            onsets,
            syllabic: s.syllabic.chars().collect(),
            glides: s.glides.chars().collect(),
            isolate_map: s.isolate_map,
        })
    }

    pub fn pieces(&self, text: &str) -> Vec<String> {
        let folded: String = text
            .chars()
            .map(|c| if self.apostrophes.contains(&c) { '\'' } else { c })
            .collect();
        cut(&folded, |c| self.vowels.contains(&c))
            .iter()
            .flat_map(|p| self.split(p))
            .collect()
    }

    fn split(&self, piece: &str) -> Vec<String> {
        let body = piece.trim_start();
        let ws = &piece[..piece.len() - body.len()];
        let low = body.to_lowercase();
        let chars: Vec<char> = body.chars().collect();
        let rest = |n: usize| chars[n..].iter().collect::<String>();
        for (k, v) in &self.map {
            if low.starts_with(k.as_str()) {
                let n = k.chars().count();
                let mut out = if self.isolate_map {
                    [ws.to_string(), v.clone()].into_iter().filter(|p| !p.is_empty()).collect()
                } else {
                    let v = if chars[0].is_uppercase() { v.to_uppercase() } else { v.clone() };
                    vec![format!("{ws}{v}")]
                };
                if chars.len() > n {
                    out.push(rest(n));
                }
                return out;
            }
        }
        for o in &self.onsets {
            let n = o.chars().count();
            if low.starts_with(o.as_str()) && chars.len() > n {
                return vec![format!("{ws}{}", chars[..n].iter().collect::<String>()), rest(n)];
            }
        }
        let lc: Vec<char> = low.chars().collect();
        if lc.len() > 1
            && self.syllabic.contains(&lc[0])
            && lc[1].is_alphabetic()
            && !self.vowels.contains(&lc[1])
            && !self.glides.contains(&lc[1])
        {
            return vec![format!("{ws}{}", chars[0]), rest(1)];
        }
        vec![piece.to_string()]
    }
}

#[cfg(test)]
mod tests {
    use super::{pieces, Scheme};

    const SWAHILI: &str = include_str!("../../../references/qwen3tts/langs/swahili.json");

    #[test]
    fn open_syllables() {
        assert_eq!(pieces("ubunifu"), ["u", "bu", "ni", "fu"]);
        assert_eq!(pieces("Ng'ombe wa maarifa."), ["Ng'o", "mbe", " wa", " ma", "a", "ri", "fa."]);
        assert_eq!(pieces("mtu, nchi"), ["mtu,", " nchi"]);
        assert_eq!(pieces("a  b"), ["a", "  b"]);
    }

    #[test]
    fn swahili_scheme() {
        let s = Scheme::from_json(SWAHILI.as_bytes()).unwrap();
        assert_eq!((s.language.as_str(), s.row), ("swahili", 2074));
        let p = |t: &str| s.pieces(t);
        assert_eq!(p("Ng'ombe wa mbili ndani."), ["ŋ", "o", "mb", "e", " wa", " mb", "i", "li", " nd", "a", "ni."]);
        assert_eq!(
            p("mtoto nchi mwana nyumba mvua"),
            ["m", "to", "to", " n", "chi", " mwa", "na", " nyu", "mb", "a", " mv", "u", "a"]
        );
        assert_eq!(p("Nzuri, wangu mbwa"), ["Nz", "u", "ri,", " wa", "ng", "u", " mb", "wa"]);
        assert_eq!(p("ng’ambo kung'ata"), ["ŋ", "a", "mb", "o", " ku", "ŋ", "a", "ta"]);
        assert_eq!(p("Nina ng'ombe"), ["Ni", "na", " ", "ŋ", "o", "mb", "e"]);
        assert_eq!(p("Jana kijiji njia"), ["J", "a", "na", " ki", "j", "i", "j", "i", " nj", "i", "a"]);
    }

    /// `PIECES_PARITY=file.jsonl` of {scheme, text, pieces} from `orthography.py parity`.
    #[test]
    fn parity_with_python() {
        let Ok(path) = std::env::var("PIECES_PARITY") else {
            return;
        };
        let mut n = 0;
        for line in std::fs::read_to_string(path).unwrap().lines() {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            let s = Scheme::from_json(v["scheme"].to_string().as_bytes()).unwrap();
            let want: Vec<String> = serde_json::from_value(v["pieces"].clone()).unwrap();
            assert_eq!(s.pieces(v["text"].as_str().unwrap()), want, "{}", v["text"]);
            n += 1;
        }
        eprintln!("{n} texts agree");
    }

    #[test]
    fn vowels_beyond_ascii() {
        let s = Scheme::from_json(r#"{"language":"Kikuyu","row":2075,"vowels":"aeiouĩũ"}"#.as_bytes()).unwrap();
        assert_eq!(s.pieces("mũtĩ"), ["mũ", "tĩ"]);
    }
}
