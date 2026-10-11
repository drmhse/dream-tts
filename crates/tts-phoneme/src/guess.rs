//! A pronunciation for a word no lexicon has and no rule derives.
//!
//! Kyubyong Park's g2p_en (Apache-2.0): a GRU encoder over the letters and a GRU decoder over
//! ARPAbet, trained on CMUdict. misaki asks espeak-ng here, which is GPL; this is the same
//! question asked of a 3 MB model whose licence the release can carry.
//!
//! ARPAbet is then written the way misaki writes American English. The rules below are the
//! ones that moved exact agreement with misaki's gold lexicon, over the 33,322 words both
//! it and CMUdict have, from 55.7% to 63.7%: the flap, the glottal stop, `-ed`/`-est`, and
//! syllabic consonants. What still differs is mostly secondary stress and the colour of an
//! unstressed vowel.

use anyhow::{Context, Result};
use safetensors::SafeTensors;
use std::path::Path;

const HIDDEN: usize = 256;
const MAX_PHONEMES: usize = 20;
/// The decoder's start and end symbols.
const START: usize = 2;
const END: usize = 3;

const PHONEMES: [&str; 74] = [
    "<pad>", "<unk>", "<s>", "</s>", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2",
    "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "EH0",
    "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F", "G", "HH", "IH0", "IH1", "IH2",
    "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2",
    "P", "R", "S", "SH", "T", "TH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V", "W", "Y",
    "Z", "ZH",
];

struct Gru {
    w_ih: Vec<f32>,
    w_hh: Vec<f32>,
    b_ih: Vec<f32>,
    b_hh: Vec<f32>,
}

impl Gru {
    /// PyTorch's cell: gates stacked reset, update, new.
    fn step(&self, x: &[f32], h: &[f32]) -> Vec<f32> {
        let gi = affine(&self.w_ih, &self.b_ih, x);
        let gh = affine(&self.w_hh, &self.b_hh, h);
        (0..HIDDEN)
            .map(|i| {
                let r = sigmoid(gi[i] + gh[i]);
                let z = sigmoid(gi[HIDDEN + i] + gh[HIDDEN + i]);
                let n = (gi[2 * HIDDEN + i] + r * gh[2 * HIDDEN + i]).tanh();
                (1.0 - z) * n + z * h[i]
            })
            .collect()
    }
}

fn affine(w: &[f32], b: &[f32], x: &[f32]) -> Vec<f32> {
    b.iter()
        .zip(w.chunks_exact(x.len()))
        .map(|(b, row)| b + row.iter().zip(x).map(|(w, x)| w * x).sum::<f32>())
        .collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub struct Guesser {
    enc_emb: Vec<f32>,
    enc: Gru,
    dec_emb: Vec<f32>,
    dec: Gru,
    fc_w: Vec<f32>,
    fc_b: Vec<f32>,
}

fn f32s(t: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let v = t
        .tensor(name)
        .with_context(|| format!("missing tensor `{name}`"))?;
    Ok(v.data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

impl Guesser {
    /// `None` when the frontend was exported without one: the word is then named, not said.
    pub fn load(dir: &Path) -> Result<Option<Self>> {
        let path = dir.join("g2p.safetensors");
        if !path.exists() {
            return Ok(None);
        }
        let buf = std::fs::read(&path)?;
        let t = SafeTensors::deserialize(&buf)?;
        let gru = |p: &str| -> Result<Gru> {
            Ok(Gru {
                w_ih: f32s(&t, &format!("{p}_w_ih"))?,
                w_hh: f32s(&t, &format!("{p}_w_hh"))?,
                b_ih: f32s(&t, &format!("{p}_b_ih"))?,
                b_hh: f32s(&t, &format!("{p}_b_hh"))?,
            })
        };
        Ok(Some(Self {
            enc_emb: f32s(&t, "enc_emb")?,
            enc: gru("enc")?,
            dec_emb: f32s(&t, "dec_emb")?,
            dec: gru("dec")?,
            fc_w: f32s(&t, "fc_w")?,
            fc_b: f32s(&t, "fc_b")?,
        }))
    }

    /// The model's reading of a lowercase word, greedily decoded.
    pub fn arpabet(&self, word: &str) -> Vec<&'static str> {
        // Letters are 3.., unknown 1, end 2.
        let ids = word
            .chars()
            .map(|c| match c {
                'a'..='z' => c as usize - 'a' as usize + 3,
                _ => 1,
            })
            .chain([2]);
        let mut h = vec![0.0; HIDDEN];
        for id in ids {
            h = self.enc.step(&self.enc_emb[id * HIDDEN..][..HIDDEN], &h);
        }
        let mut out = Vec::new();
        let mut last = START;
        for _ in 0..MAX_PHONEMES {
            h = self.dec.step(&self.dec_emb[last * HIDDEN..][..HIDDEN], &h);
            let logits = affine(&self.fc_w, &self.fc_b, &h);
            let best = (0..logits.len())
                .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                .unwrap_or(END);
            if best == END {
                break;
            }
            if best > END {
                out.push(PHONEMES[best]);
            }
            last = best;
        }
        out
    }

    /// The word in misaki's American phonemes, or `None` for anything but letters.
    pub fn phonemes(&self, word: &str) -> Option<String> {
        let lower = word.to_ascii_lowercase();
        if lower.is_empty() || !lower.chars().all(|c| c.is_ascii_lowercase()) {
            return None;
        }
        let arpabet = self.arpabet(&lower);
        (!arpabet.is_empty()).then(|| misaki(&arpabet, &lower))
    }
}

fn stress(p: &str) -> Option<char> {
    p.chars().last().filter(char::is_ascii_digit)
}

fn vowel(p: &str) -> bool {
    stress(p).is_some() || p == "UW"
}

/// ARPAbet as misaki writes American English: one letter per diphthong, the stress mark
/// before the vowel it falls on. `word` is the spelling, which ARPAbet alone does not keep:
/// misaki reduces "-est" and not "-ist".
pub fn misaki(ps: &[&str], word: &str) -> String {
    let at = |i: usize| ps.get(i).copied();
    let mut out = String::new();
    let mut i = 0;
    while i < ps.len() {
        let p = ps[i];
        let prev = i.checked_sub(1).and_then(at);
        let (next, after) = (at(i + 1), at(i + 2));
        // button, batten: the t closes the throat and the n carries the syllable.
        if p == "T"
            && prev.is_some_and(vowel)
            && next == Some("AH0")
            && after == Some("N")
            && !at(i + 3).is_some_and(vowel)
        {
            out.push_str("ʔn");
            i += 3;
            continue;
        }
        if vowel(p) {
            let s = stress(p);
            out.push_str(match s {
                Some('1') => "ˈ",
                Some('2') => "ˌ",
                _ => "",
            });
            let base = p.trim_end_matches(|c: char| c.is_ascii_digit());
            let unstressed = s == Some('0') || s.is_none();
            out.push_str(match base {
                "AH" if unstressed && syllabic(prev, next, after) => "ᵊ",
                "AH" if unstressed => "ə",
                "AH" => "ʌ",
                "ER" if unstressed => "əɹ",
                "ER" => "ɜɹ",
                "IH" if unstressed
                    && reduced(
                        prev,
                        next,
                        after,
                        i + 3 == ps.len() && word.ends_with("est"),
                    ) =>
                {
                    "ᵻ"
                }
                "AA" => "ɑ",
                "AE" => "æ",
                "AO" => "ɔ",
                "AW" => "W",
                "AY" => "I",
                "EH" => "ɛ",
                "EY" => "A",
                "IH" => "ɪ",
                "IY" => "i",
                "OW" => "O",
                "OY" => "Y",
                "UH" => "ʊ",
                _ => "u",
            });
        } else if p == "T"
            && prev.is_some_and(|q| vowel(q) || q == "R")
            && next.is_some_and(|q| stress(q) == Some('0'))
        {
            // water, party: the American flap before an unstressed vowel.
            out.push('ɾ');
        } else {
            out.push_str(match p {
                "B" => "b",
                "CH" => "ʧ",
                "D" => "d",
                "DH" => "ð",
                "F" => "f",
                "G" => "ɡ",
                "HH" => "h",
                "JH" => "ʤ",
                "K" => "k",
                "L" => "l",
                "M" => "m",
                "N" => "n",
                "NG" => "ŋ",
                "P" => "p",
                "R" => "ɹ",
                "S" => "s",
                "SH" => "ʃ",
                "T" => "t",
                "TH" => "θ",
                "V" => "v",
                "W" => "w",
                "Y" => "j",
                "Z" => "z",
                _ => "ʒ",
            });
        }
        i += 1;
    }
    out
}

/// bottle, lesson: an unstressed vowel that the l or n swallows. Only at the end after a
/// consonant for l, and only after t, d, s or z for n, which is where misaki does it.
fn syllabic(prev: Option<&str>, next: Option<&str>, after: Option<&str>) -> bool {
    let Some(prev) = prev else { return false };
    match next {
        Some("L") => {
            after.is_none()
                && !vowel(prev)
                && !matches!(prev, "L" | "R" | "Y" | "W" | "SH" | "ZH" | "CH" | "JH")
        }
        Some("N") => matches!(prev, "T" | "D" | "S" | "Z") && !after.is_some_and(vowel),
        _ => false,
    }
}

/// wanted, biggest, buses: misaki's reduced vowel in the endings.
fn reduced(prev: Option<&str>, next: Option<&str>, after: Option<&str>, est: bool) -> bool {
    match (next, after) {
        (Some("D"), None) => matches!(prev, Some("T" | "D")),
        (Some("S"), Some("T")) => est,
        (Some("Z"), None) => matches!(prev, Some("S" | "Z" | "SH" | "ZH" | "CH" | "JH")),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn says(arpabet: &str) -> String {
        misaki(&arpabet.split_whitespace().collect::<Vec<_>>(), "")
    }

    #[test]
    fn arpabet_is_written_as_misaki_writes_it() {
        assert_eq!(says("W AO1 T ER0"), "wˈɔɾəɹ");
        assert_eq!(says("B AH1 T AH0 N"), "bˈʌʔn");
        assert_eq!(says("B AA1 T AH0 L"), "bˈɑɾᵊl");
        assert_eq!(says("W AA1 N T IH0 D"), "wˈɑntᵻd");
        assert_eq!(says("N EY1 SH AH0 N"), "nˈAʃən");
        assert_eq!(says("HH AH0 L OW1"), "həlˈO");
    }
}
