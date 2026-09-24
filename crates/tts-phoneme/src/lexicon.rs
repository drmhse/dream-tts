//! misaki's English lexicon and its stress, affix and number rules.
//!
//! Ported against the pinned misaki: `scripts/check-phonemes.sh` requires the phoneme
//! string to be identical, because a phoneme that is merely plausible is a fluent
//! mispronunciation and nothing downstream can hear the difference.

use crate::num2words;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

pub const PRIMARY_STRESS: char = 'ˈ';
pub const SECONDARY_STRESS: char = 'ˌ';
const STRESSES: [char; 2] = ['ˌ', 'ˈ'];
const DIPHTHONGS: &str = "AIOQWYʤʧ";
pub const VOWELS: &str = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
pub const CONSONANTS: &str = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
const US_TAUS: &str = "AIOWYiuæɑəɛɪɹʊʌ";
pub const SUBTOKEN_JUNKS: &str = "',-._‘’/";
pub const PUNCTS: &str = ";:,.!?—…\"“”";
const ORDINAL_SUFFIXES: [&str; 4] = ["st", "nd", "rd", "th"];

pub fn is_vowel(c: char) -> bool {
    VOWELS.contains(c)
}

pub fn non_quote_punct(c: char) -> bool {
    PUNCTS.contains(c) && !"\"“”".contains(c)
}

fn currency_units(c: &str) -> Option<(&'static str, &'static str)> {
    match c {
        "$" => Some(("dollar", "cent")),
        "£" => Some(("pound", "pence")),
        "€" => Some(("euro", "cent")),
        _ => None,
    }
}

fn add_symbol(w: &str) -> Option<&'static str> {
    match w {
        "." => Some("dot"),
        "/" => Some("slash"),
        _ => None,
    }
}

pub fn symbol(w: &str) -> Option<&'static str> {
    match w {
        "%" => Some("percent"),
        "&" => Some("and"),
        "+" => Some("plus"),
        "@" => Some("at"),
        _ => None,
    }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(untagged)]
enum Entry {
    Word(String),
    /// A homograph: phonemes per coarse POS, with DEFAULT as the fallback. 790 of 90,201.
    Tagged(HashMap<String, Option<String>>),
}

#[derive(Deserialize)]
struct RawLexicon {
    gold: HashMap<String, Entry>,
    silver: HashMap<String, Entry>,
}

/// What the caller knows about the token to the right, which two rules depend on.
#[derive(Clone, Copy, Default, Debug)]
pub struct TokenContext {
    pub future_vowel: Option<bool>,
    pub future_to: bool,
}

pub struct Lexicon {
    british: bool,
    golds: HashMap<String, Entry>,
    silvers: HashMap<String, Entry>,
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f
            .to_uppercase()
            .chain(c.flat_map(char::to_lowercase))
            .collect(),
        None => String::new(),
    }
}

/// Materialise the capitalised and lowercased variants misaki derives at load. Exported
/// ungrown because this is deterministic and doubles the asset.
fn grow(d: HashMap<String, Entry>) -> HashMap<String, Entry> {
    let mut out: HashMap<String, Entry> = HashMap::with_capacity(d.len() * 2);
    for (k, v) in &d {
        if k.chars().count() < 2 {
            continue;
        }
        let lower = k.to_lowercase();
        if *k == lower {
            let cap = capitalize(k);
            if *k != cap {
                out.insert(cap, v.clone());
            }
        } else if *k == capitalize(&lower) {
            out.insert(lower, v.clone());
        }
    }
    out.extend(d);
    out
}

/// Move each stress mark to just before the vowel it belongs to.
fn restress(ps: &str) -> String {
    let chars: Vec<char> = ps.chars().collect();
    let mut keyed: Vec<(f64, char)> = Vec::with_capacity(chars.len());
    for (i, c) in chars.iter().enumerate() {
        if STRESSES.contains(c) {
            match chars[i..].iter().position(|v| is_vowel(*v)) {
                Some(off) => keyed.push(((i + off) as f64 - 0.5, *c)),
                None => keyed.push((i as f64, *c)),
            }
        } else {
            keyed.push((i as f64, *c));
        }
    }
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    keyed.into_iter().map(|(_, c)| c).collect()
}

pub fn apply_stress(ps: &str, stress: Option<f64>) -> String {
    let Some(stress) = stress else {
        return ps.to_string();
    };
    let has_primary = ps.contains(PRIMARY_STRESS);
    let has_secondary = ps.contains(SECONDARY_STRESS);
    let has_any = has_primary || has_secondary;
    let has_vowel = ps.chars().any(is_vowel);

    if stress < -1.0 {
        return ps.chars().filter(|c| !STRESSES.contains(c)).collect();
    }
    if stress == -1.0 || ((stress == 0.0 || stress == -0.5) && has_primary) {
        return ps
            .chars()
            .filter(|c| *c != SECONDARY_STRESS)
            .map(|c| {
                if c == PRIMARY_STRESS {
                    SECONDARY_STRESS
                } else {
                    c
                }
            })
            .collect();
    }
    if (stress == 0.0 || stress == 0.5 || stress == 1.0) && !has_any {
        if !has_vowel {
            return ps.to_string();
        }
        return restress(&format!("{SECONDARY_STRESS}{ps}"));
    }
    if stress >= 1.0 && !has_primary && has_secondary {
        return ps.replace(SECONDARY_STRESS, &PRIMARY_STRESS.to_string());
    }
    if stress > 1.0 && !has_any {
        if !has_vowel {
            return ps.to_string();
        }
        return restress(&format!("{PRIMARY_STRESS}{ps}"));
    }
    ps.to_string()
}

pub fn stress_weight(ps: &str) -> usize {
    ps.chars()
        .map(|c| if DIPHTHONGS.contains(c) { 2 } else { 1 })
        .sum()
}

fn is_digits(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}

fn lexicon_ord(c: char) -> bool {
    c == '\'' || c == '-' || c.is_ascii_alphabetic()
}

/// A phoneme string and how much misaki trusts it (4 gold, 3 derived).
pub type Hit = (String, u8);

impl Lexicon {
    pub fn load(dir: &Path, british: bool) -> Result<Self> {
        let name = if british {
            "lexicon-gb.json"
        } else {
            "lexicon-us.json"
        };
        let raw = std::fs::read(dir.join(name))
            .with_context(|| format!("reading {name} from {}", dir.display()))?;
        let r: RawLexicon = serde_json::from_slice(&raw)?;
        Ok(Self {
            british,
            golds: grow(r.gold),
            silvers: grow(r.silver),
        })
    }

    fn parent_tag(tag: &str) -> &str {
        if tag.starts_with("VB") {
            "VERB"
        } else if tag.starts_with("NN") {
            "NOUN"
        } else if tag.starts_with("ADV") || tag.starts_with("RB") {
            "ADV"
        } else if tag.starts_with("ADJ") || tag.starts_with("JJ") {
            "ADJ"
        } else {
            tag
        }
    }

    fn gold_str(&self, word: &str) -> Option<&str> {
        match self.golds.get(word) {
            Some(Entry::Word(w)) => Some(w),
            _ => None,
        }
    }

    /// Spell a word out letter by letter, primary stress on the last.
    fn get_nnp(&self, word: &str) -> Option<Hit> {
        let mut ps = String::new();
        for c in word.chars().filter(|c| c.is_alphabetic()) {
            let up: String = c.to_uppercase().collect();
            ps.push_str(self.gold_str(&up)?);
        }
        let ps = apply_stress(&ps, Some(0.0));
        Some((
            match ps.rfind(SECONDARY_STRESS) {
                Some(i) => format!(
                    "{}{PRIMARY_STRESS}{}",
                    &ps[..i],
                    &ps[i + SECONDARY_STRESS.len_utf8()..]
                ),
                None => ps,
            },
            3,
        ))
    }

    fn is_known(&self, word: &str, _tag: &str) -> bool {
        if self.golds.contains_key(word)
            || symbol(word).is_some()
            || self.silvers.contains_key(word)
        {
            return true;
        }
        if !word.chars().all(|c| c.is_alphabetic()) || !word.chars().all(lexicon_ord) {
            return false;
        }
        if word.chars().count() == 1 {
            return true;
        }
        if word == word.to_uppercase() && self.golds.contains_key(&word.to_lowercase()) {
            return true;
        }
        let tail: String = word.chars().skip(1).collect();
        tail == tail.to_uppercase()
    }

    pub fn lookup(
        &self,
        word: &str,
        tag: &str,
        stress: Option<f64>,
        ctx: Option<&TokenContext>,
    ) -> Option<Hit> {
        let mut word = word.to_string();
        let mut is_nnp = None;
        if word == word.to_uppercase() && !self.golds.contains_key(&word) {
            word = word.to_lowercase();
            is_nnp = Some(tag == "NNP");
        }
        let mut rating = 4u8;
        let mut entry = self.golds.get(&word).cloned();
        if entry.is_none() && is_nnp != Some(true) {
            entry = self.silvers.get(&word).cloned();
            if entry.is_some() {
                rating = 3;
            }
        }
        let ps: Option<String> = match entry {
            Some(Entry::Word(w)) => Some(w),
            Some(Entry::Tagged(map)) => {
                let mut key = tag.to_string();
                if ctx.map(|c| c.future_vowel.is_none()).unwrap_or(false)
                    && map.contains_key("None")
                {
                    key = "None".into();
                } else if !map.contains_key(&key) {
                    key = Self::parent_tag(tag).to_string();
                }
                map.get(&key)
                    .cloned()
                    .unwrap_or_else(|| map["DEFAULT"].clone())
            }
            None => None,
        };
        let needs_nnp = ps.is_none()
            || (is_nnp == Some(true) && !ps.as_deref().unwrap().contains(PRIMARY_STRESS));
        if needs_nnp {
            if let Some(hit) = self.get_nnp(&word) {
                return Some(hit);
            }
        }
        Some((apply_stress(ps.as_deref()?, stress), rating))
    }

    /// https://en.wiktionary.org/wiki/-s
    fn suffix_s(&self, stem: &str) -> Option<String> {
        let last = stem.chars().last()?;
        Some(if "ptkfθ".contains(last) {
            format!("{stem}s")
        } else if "szʃʒʧʤ".contains(last) {
            format!("{stem}{}z", if self.british { 'ɪ' } else { 'ᵻ' })
        } else {
            format!("{stem}z")
        })
    }

    /// https://en.wiktionary.org/wiki/-ed
    fn suffix_ed(&self, stem: &str) -> Option<String> {
        let chars: Vec<char> = stem.chars().collect();
        let last = *chars.last()?;
        Some(if "pkfθʃsʧ".contains(last) {
            format!("{stem}t")
        } else if last == 'd' {
            format!("{stem}{}d", if self.british { 'ɪ' } else { 'ᵻ' })
        } else if last != 't' {
            format!("{stem}d")
        } else if self.british || chars.len() < 2 {
            format!("{stem}ɪd")
        } else if US_TAUS.contains(chars[chars.len() - 2]) {
            let head: String = chars[..chars.len() - 1].iter().collect();
            format!("{head}ɾᵻd")
        } else {
            format!("{stem}ᵻd")
        })
    }

    /// https://en.wiktionary.org/wiki/-ing
    fn suffix_ing(&self, stem: &str) -> Option<String> {
        let chars: Vec<char> = stem.chars().collect();
        let last = *chars.last()?;
        if self.british {
            if last == 'ə' || last == 'ː' {
                return None;
            }
        } else if chars.len() > 1 && last == 't' && US_TAUS.contains(chars[chars.len() - 2]) {
            let head: String = chars[..chars.len() - 1].iter().collect();
            return Some(format!("{head}ɾɪŋ"));
        }
        Some(format!("{stem}ɪŋ"))
    }

    fn stem_s(
        &self,
        word: &str,
        tag: &str,
        stress: Option<f64>,
        ctx: Option<&TokenContext>,
    ) -> Option<Hit> {
        if word.chars().count() < 3 || !word.ends_with('s') {
            return None;
        }
        let n = word.len();
        let stem = if !word.ends_with("ss") && self.is_known(&word[..n - 1], tag) {
            word[..n - 1].to_string()
        } else if (word.ends_with("'s")
            || (word.chars().count() > 4 && word.ends_with("es") && !word.ends_with("ies")))
            && self.is_known(&word[..n - 2], tag)
        {
            word[..n - 2].to_string()
        } else if word.chars().count() > 4
            && word.ends_with("ies")
            && self.is_known(&format!("{}y", &word[..n - 3]), tag)
        {
            format!("{}y", &word[..n - 3])
        } else {
            return None;
        };
        let (stem, rating) = self.lookup(&stem, tag, stress, ctx)?;
        Some((self.suffix_s(&stem)?, rating))
    }

    fn stem_ed(
        &self,
        word: &str,
        tag: &str,
        stress: Option<f64>,
        ctx: Option<&TokenContext>,
    ) -> Option<Hit> {
        if word.chars().count() < 4 || !word.ends_with('d') {
            return None;
        }
        let n = word.len();
        let stem = if !word.ends_with("dd") && self.is_known(&word[..n - 1], tag) {
            word[..n - 1].to_string()
        } else if word.chars().count() > 4
            && word.ends_with("ed")
            && !word.ends_with("eed")
            && self.is_known(&word[..n - 2], tag)
        {
            word[..n - 2].to_string()
        } else {
            return None;
        };
        let (stem, rating) = self.lookup(&stem, tag, stress, ctx)?;
        Some((self.suffix_ed(&stem)?, rating))
    }

    fn stem_ing(
        &self,
        word: &str,
        tag: &str,
        stress: Option<f64>,
        ctx: Option<&TokenContext>,
    ) -> Option<Hit> {
        let count = word.chars().count();
        if count < 5 || !word.ends_with("ing") {
            return None;
        }
        let n = word.len();
        let stem = if count > 5 && self.is_known(&word[..n - 3], tag) {
            word[..n - 3].to_string()
        } else if self.is_known(&format!("{}e", &word[..n - 3]), tag) {
            format!("{}e", &word[..n - 3])
        } else if count > 5 && doubled_before_ing(word) && self.is_known(&word[..n - 4], tag) {
            word[..n - 4].to_string()
        } else {
            return None;
        };
        let (stem, rating) = self.lookup(&stem, tag, stress, ctx)?;
        Some((self.suffix_ing(&stem)?, rating))
    }
}

/// `([bcdgklmnprstvxz])\1ing$|cking$` without paying for a regex on every word.
fn doubled_before_ing(word: &str) -> bool {
    let c: Vec<char> = word.chars().collect();
    if c.len() < 5 {
        return false;
    }
    let n = c.len();
    if c[n - 5..] == ['c', 'k', 'i', 'n', 'g'] {
        return true;
    }
    c[n - 4] == c[n - 5] && "bcdgklmnprstvxz".contains(c[n - 5])
}

pub fn is_ordinal_suffix(s: &str) -> bool {
    ORDINAL_SUFFIXES.contains(&s)
}

impl Lexicon {
    /// The hand-written exceptions, in misaki's order — several depend on the token to the
    /// right, which is why the driver walks the sentence backwards.
    fn get_special_case(
        &self,
        word: &str,
        tag: &str,
        stress: Option<f64>,
        ctx: &TokenContext,
    ) -> Option<Hit> {
        if tag == "ADD" {
            if let Some(name) = add_symbol(word) {
                return self.lookup(name, "", Some(-0.5), Some(ctx));
            }
        }
        if let Some(name) = symbol(word) {
            return self.lookup(name, "", None, Some(ctx));
        }
        let stripped = word.trim_matches('.');
        if stripped.contains('.')
            && word.replace('.', "").chars().all(char::is_alphabetic)
            && !word.replace('.', "").is_empty()
            && word
                .split('.')
                .map(|p| p.chars().count())
                .max()
                .unwrap_or(0)
                < 3
        {
            return self.get_nnp(word);
        }
        match word {
            "a" | "A" => return Some(((if tag == "DT" { "ɐ" } else { "ˈA" }).into(), 4)),
            "am" | "Am" | "AM" => {
                if tag.starts_with("NN") {
                    return self.get_nnp(word);
                }
                if ctx.future_vowel.is_none()
                    || word != "am"
                    || stress.map(|s| s > 0.0).unwrap_or(false)
                {
                    return Some((self.gold_str("am")?.to_string(), 4));
                }
                return Some(("ɐm".into(), 4));
            }
            "an" | "An" | "AN" => {
                if word == "AN" && tag.starts_with("NN") {
                    return self.get_nnp(word);
                }
                return Some(("ɐn".into(), 4));
            }
            "I" if tag == "PRP" => return Some((format!("{SECONDARY_STRESS}I"), 4)),
            "by" | "By" | "BY" if Self::parent_tag(tag) == "ADV" => return Some(("bˈI".into(), 4)),
            "to" | "To" => return Some((self.to_phonemes(ctx)?, 4)),
            "TO" if tag == "TO" || tag == "IN" => return Some((self.to_phonemes(ctx)?, 4)),
            "in" | "In" => return Some((self.in_phonemes(tag, ctx), 4)),
            "IN" if tag != "NNP" => return Some((self.in_phonemes(tag, ctx), 4)),
            "the" | "The" => {
                return Some((
                    (if ctx.future_vowel == Some(true) {
                        "ði"
                    } else {
                        "ðə"
                    })
                    .into(),
                    4,
                ))
            }
            "THE" if tag == "DT" => {
                return Some((
                    (if ctx.future_vowel == Some(true) {
                        "ði"
                    } else {
                        "ðə"
                    })
                    .into(),
                    4,
                ))
            }
            "used" | "Used" | "USED" => {
                let key = if (tag == "VBD" || tag == "JJ") && ctx.future_to {
                    "VBD"
                } else {
                    "DEFAULT"
                };
                if let Some(Entry::Tagged(map)) = self.golds.get("used") {
                    return Some((map.get(key)?.clone()?, 4));
                }
                return None;
            }
            _ => {}
        }
        if tag == "IN" {
            let lower = word.to_lowercase();
            if lower == "vs" || lower == "vs." {
                return self.lookup("versus", "", None, Some(ctx));
            }
        }
        None
    }

    fn to_phonemes(&self, ctx: &TokenContext) -> Option<String> {
        Some(match ctx.future_vowel {
            None => self.gold_str("to")?.to_string(),
            Some(false) => "tə".into(),
            Some(true) => "tʊ".into(),
        })
    }

    fn in_phonemes(&self, tag: &str, ctx: &TokenContext) -> String {
        let lead = if ctx.future_vowel.is_none() || tag != "IN" {
            PRIMARY_STRESS.to_string()
        } else {
            String::new()
        };
        format!("{lead}ɪn")
    }

    pub fn get_word(
        &self,
        word: &str,
        tag: &str,
        stress: Option<f64>,
        ctx: &TokenContext,
    ) -> Option<Hit> {
        if let Some(hit) = self.get_special_case(word, tag, stress, ctx) {
            return Some(hit);
        }
        let lower = word.to_lowercase();
        let mut word = word.to_string();
        let tail: String = word.chars().skip(1).collect();
        let alpha_apostrophe = word.replace('\'', "").chars().all(char::is_alphabetic)
            && !word.replace('\'', "").is_empty();
        if word.chars().count() > 1
            && alpha_apostrophe
            && word != lower
            && (tag != "NNP" || word.chars().count() > 7)
            && !self.golds.contains_key(&word)
            && !self.silvers.contains_key(&word)
            && (word == word.to_uppercase() || tail == tail.to_lowercase())
            && (self.golds.contains_key(&lower)
                || self.silvers.contains_key(&lower)
                || self.stem_s(&lower, tag, stress, Some(ctx)).is_some()
                || self.stem_ed(&lower, tag, stress, Some(ctx)).is_some()
                || self.stem_ing(&lower, tag, stress, Some(ctx)).is_some())
        {
            word = lower;
        }
        if self.is_known(&word, tag) {
            return self.lookup(&word, tag, stress, Some(ctx));
        }
        if word.ends_with("s'") {
            let alt = format!("{}'s", &word[..word.len() - 2]);
            if self.is_known(&alt, tag) {
                return self.lookup(&alt, tag, stress, Some(ctx));
            }
        }
        if word.ends_with('\'') {
            let alt = &word[..word.len() - 1];
            if self.is_known(alt, tag) {
                return self.lookup(alt, tag, stress, Some(ctx));
            }
        }
        self.stem_s(&word, tag, stress, Some(ctx))
            .or_else(|| self.stem_ed(&word, tag, stress, Some(ctx)))
            .or_else(|| self.stem_ing(&word, tag, stress.or(Some(0.5)), Some(ctx)))
    }

    fn is_currency_amount(word: &str) -> bool {
        if !word.contains('.') {
            return true;
        }
        if word.matches('.').count() > 1 {
            return false;
        }
        let cents = word.split('.').nth(1).unwrap_or("");
        cents.chars().count() < 3 || cents.chars().all(|c| c == '0')
    }

    fn extend_num(&self, result: &mut Vec<Hit>, words: &str, first: bool, num_flags: &str) {
        let splits: Vec<&str> = split_non_lower(words);
        for (i, w) in splits.iter().enumerate() {
            if *w != "and" || num_flags.contains('&') {
                if first && i == 0 && splits.len() > 1 && *w == "one" && num_flags.contains('a') {
                    result.push(("ə".into(), 4));
                } else {
                    let stress = if *w == "point" { Some(-2.0) } else { None };
                    if let Some(hit) = self.lookup(w, "", stress, None) {
                        result.push(hit);
                    }
                }
            } else if *w == "and" && num_flags.contains('n') && !result.is_empty() {
                let last = result.last_mut().unwrap();
                last.0.push_str("ən");
            }
        }
    }

    fn extend_int(&self, result: &mut Vec<Hit>, num: &str, first: bool, num_flags: &str) {
        let Ok(v) = num.parse::<i64>() else { return };
        self.extend_num(result, &num2words::cardinal(v), first, num_flags);
    }

    pub fn get_number(
        &self,
        word: &str,
        currency: Option<&str>,
        is_head: bool,
        num_flags: &str,
    ) -> Option<Hit> {
        let suffix: Option<String> = {
            let tail: String = word
                .chars()
                .rev()
                .take_while(|c| c.is_ascii_lowercase() || *c == '\'')
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            if tail.is_empty() {
                None
            } else {
                Some(tail)
            }
        };
        let mut word = match &suffix {
            Some(s) => word[..word.len() - s.len()].to_string(),
            None => word.to_string(),
        };
        let mut result: Vec<Hit> = Vec::new();
        if word.starts_with('-') {
            if let Some(hit) = self.lookup("minus", "", None, None) {
                result.push(hit);
            }
            word = word[1..].to_string();
        }

        let is_ordinal = suffix.as_deref().map(is_ordinal_suffix).unwrap_or(false);
        let plain = word.replace(',', "");
        if is_digits(&word) && is_ordinal {
            let v: i64 = word.parse().ok()?;
            self.extend_num(&mut result, &num2words::ordinal(v), true, num_flags);
        } else if result.is_empty()
            && word.chars().count() == 4
            && currency.and_then(currency_units).is_none()
            && is_digits(&word)
        {
            let v: i64 = word.parse().ok()?;
            self.extend_num(&mut result, &num2words::year(v), true, num_flags);
        } else if !is_head && !word.contains('.') {
            let num = plain.clone();
            let chars: Vec<char> = num.chars().collect();
            if chars.first() == Some(&'0') || chars.len() > 3 {
                for c in &chars {
                    self.extend_int(&mut result, &c.to_string(), false, num_flags);
                }
            } else if chars.len() == 3 && !num.ends_with("00") {
                self.extend_int(&mut result, &chars[0].to_string(), true, num_flags);
                if chars[1] == '0' {
                    if let Some(hit) = self.lookup("O", "", Some(-2.0), None) {
                        result.push(hit);
                    }
                    self.extend_int(&mut result, &chars[2].to_string(), false, num_flags);
                } else {
                    self.extend_int(&mut result, &num[1..], false, num_flags);
                }
            } else {
                self.extend_int(&mut result, &num, true, num_flags);
            }
        } else if word.matches('.').count() > 1 || !is_head {
            let mut first = true;
            for num in plain.split('.') {
                if num.is_empty() {
                } else {
                    let chars: Vec<char> = num.chars().collect();
                    if chars[0] == '0' || (chars.len() != 2 && chars[1..].iter().any(|c| *c != '0'))
                    {
                        for c in &chars {
                            self.extend_int(&mut result, &c.to_string(), false, num_flags);
                        }
                    } else {
                        self.extend_int(&mut result, num, first, num_flags);
                    }
                }
                first = false;
            }
        } else if let Some((major, minor)) = currency
            .and_then(currency_units)
            .filter(|_| Self::is_currency_amount(&word))
        {
            let parts: Vec<&str> = plain.split('.').collect();
            let mut pairs: Vec<(i64, &str)> = parts
                .iter()
                .zip([major, minor])
                .map(|(n, unit)| (n.parse::<i64>().unwrap_or(0), unit))
                .collect();
            if pairs.len() > 1 {
                if pairs[1].0 == 0 {
                    pairs.truncate(1);
                } else if pairs[0].0 == 0 {
                    pairs.remove(0);
                }
            }
            for (i, (num, unit)) in pairs.iter().enumerate() {
                if i > 0 {
                    if let Some(hit) = self.lookup("and", "", None, None) {
                        result.push(hit);
                    }
                }
                self.extend_int(&mut result, &num.to_string(), i == 0, num_flags);
                let hit = if num.abs() != 1 && *unit != "pence" {
                    self.stem_s(&format!("{unit}s"), "", None, None)
                } else {
                    self.lookup(unit, "", None, None)
                };
                if let Some(hit) = hit {
                    result.push(hit);
                }
            }
        } else {
            let words = if is_digits(&word) {
                num2words::cardinal(word.parse().ok()?)
            } else if !word.contains('.') {
                let v: i64 = plain.parse().ok()?;
                if is_ordinal {
                    num2words::ordinal(v)
                } else {
                    num2words::cardinal(v)
                }
            } else if plain.starts_with('.') {
                let mut s = String::from("point");
                for c in plain[1..].chars() {
                    s.push(' ');
                    s.push_str(&num2words::cardinal(c.to_digit(10)? as i64));
                }
                s
            } else {
                num2words::decimal(&plain)?
            };
            self.extend_num(&mut result, &words, true, num_flags);
        }

        if result.is_empty() {
            return None;
        }
        let rating = result.iter().map(|(_, r)| *r).min().unwrap();
        let joined = result
            .iter()
            .map(|(p, _)| p.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Some((
            match suffix.as_deref() {
                Some("s") | Some("'s") => self.suffix_s(&joined)?,
                Some("ed") | Some("'d") => self.suffix_ed(&joined)?,
                Some("ing") => self.suffix_ing(&joined)?,
                _ => joined,
            },
            rating,
        ))
    }

    fn append_currency(&self, ps: &str, currency: Option<&str>) -> String {
        let Some((major, _)) = currency.and_then(currency_units) else {
            return ps.to_string();
        };
        match self.stem_s(&format!("{major}s"), "", None, None) {
            Some((c, _)) => format!("{ps} {c}"),
            None => ps.to_string(),
        }
    }

    /// Does this look like a number to misaki? Suffixes are stripped first, so "1980s" and
    /// "42nd" both qualify.
    pub fn is_number(word: &str, is_head: bool) -> bool {
        if !word.chars().any(|c| c.is_ascii_digit()) {
            return false;
        }
        let mut w = word;
        for s in ["ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"] {
            if w.ends_with(s) {
                w = &w[..w.len() - s.len()];
                break;
            }
        }
        w.chars().enumerate().all(|(i, c)| {
            c.is_ascii_digit() || c == ',' || c == '.' || (is_head && i == 0 && c == '-')
        })
    }
}

/// Python's `re.split(r'[^a-z]+', s)`, which keeps the empty edges.
fn split_non_lower(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let bytes = s.as_bytes();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i].is_ascii_lowercase() {
            i += 1;
            continue;
        }
        out.push(&s[start..i]);
        while i < bytes.len() && !bytes[i].is_ascii_lowercase() {
            i += 1;
        }
        start = i;
    }
    out.push(&s[start..]);
    out
}

pub fn currency_symbol(c: &str) -> bool {
    currency_units(c).is_some()
}

impl Lexicon {
    pub fn with_currency(&self, ps: &str, currency: Option<&str>) -> String {
        self.append_currency(ps, currency)
    }
}

/// Derivations misaki does not attempt, for words it returns nothing for.
///
/// Deliberately a separate stage rather than extra rules inside `get_word`: the core is
/// gated on being byte-identical to misaki, and rules mixed into it would make that gate
/// meaningless. This only ever runs where misaki gave up, and `phoneme-validate` checks
/// exactly that.
///
/// Sized on a technical corpus, the worst case for a lexicon: 1,967 unknown occurrences in
/// 522,542 tokens, of which these recover about two thirds. What is left needs
/// letter-to-sound, not more rules, and is reported rather than guessed.
impl Lexicon {
    /// Productive prefixes, already carrying the secondary stress they take in a
    /// derivative, spelled out rather than looked up: the lexicon's "re" is the musical
    /// note (ɹˌA) and its "inter" is the verb (ɪntˈɜɹ), so a lookup here yields a word that
    /// is confidently and audibly wrong.
    fn prefix_phonemes(word: &str) -> Option<&'static str> {
        Some(match word {
            "un" => "ˌʌn",
            "re" => "ɹˌi",
            "non" => "nˌɑn",
            "multi" => "mˌʌlti",
            "over" => "ˌOvəɹ",
            "under" => "ˌʌndəɹ",
            "pre" => "pɹˌi",
            "de" => "dˌi",
            "mis" => "mˌɪs",
            "sub" => "sˌʌb",
            "inter" => "ˌɪntəɹ",
            "anti" => "ˌæntI",
            "auto" => "ˌɔɾO",
            "co" => "kˌO",
            "micro" => "mˌIkɹO",
            "mono" => "mˌɑnO",
            "semi" => "sˌɛmI",
            "pseudo" => "sˌudO",
            "quasi" => "kwˌAzI",
            "intra" => "ˌɪntɹə",
            _ => return None,
        })
    }

    /// The same particles standing alone, which happens because "non-trivial" reaches the
    /// lexicon as three sub-tokens.
    fn particle(word: &str) -> Option<String> {
        match word {
            "non" | "multi" | "quasi" | "pseudo" | "semi" | "intra" => {
                Self::prefix_phonemes(word).map(|p| apply_stress(p, Some(2.0)))
            }
            _ => None,
        }
    }

    fn known_phonemes(&self, word: &str) -> Option<String> {
        let (ps, _) = self.lookup(word, "", None, None)?;
        if ps.is_empty() {
            return None;
        }
        Some(ps)
    }

    fn known_word(&self, word: &str) -> bool {
        self.golds.contains_key(word) || self.silvers.contains_key(word)
    }

    /// Join two halves of a compound: the head keeps primary stress, the tail is demoted.
    fn join_compound(head: &str, tail: &str) -> String {
        format!("{head}{}", apply_stress(tail, Some(-0.5)))
    }

    pub fn derive(&self, word: &str) -> Option<Hit> {
        let lower = word.to_lowercase();
        if let Some(ps) = Self::particle(&lower) {
            return Some((ps, 2));
        }

        // copied, verified, denied — misaki's -ies/-y rule exists for plurals only.
        if let Some(stem) = lower.strip_suffix("ied") {
            if self.known_word(&format!("{stem}y")) {
                let ps = self.known_phonemes(&format!("{stem}y"))?;
                return Some((self.suffix_ed(&ps)?, 2));
            }
        }
        if let Some(stem) = lower
            .strip_suffix("ier")
            .or_else(|| lower.strip_suffix("iest"))
        {
            if self.known_word(&format!("{stem}y")) {
                let ps = self.known_phonemes(&format!("{stem}y"))?;
                let tail = if lower.ends_with("iest") {
                    "ᵻst"
                } else {
                    "əɹ"
                };
                return Some((format!("{ps}{tail}"), 2));
            }
        }
        for (suffix, tail) in [("est", "ᵻst"), ("er", "əɹ")] {
            if let Some(stem) = lower.strip_suffix(suffix) {
                if stem.chars().count() >= 3 && self.known_word(stem) {
                    let ps = self.known_phonemes(stem)?;
                    return Some((format!("{ps}{tail}"), 2));
                }
            }
        }

        // Productive prefixes the lexicon lists only on some of their derivatives.
        for prefix in [
            "un", "re", "non", "multi", "over", "under", "pre", "de", "mis", "sub", "inter",
            "anti", "auto", "co", "micro", "mono",
        ] {
            let Some(rest) = lower.strip_prefix(prefix) else {
                continue;
            };
            if rest.chars().count() < 3 || !self.known_word(rest) {
                continue;
            }
            let head = Self::prefix_phonemes(prefix)?;
            // Only the prefix is demoted — the stem keeps its primary stress, unlike a
            // compound where the *tail* is the one that gives way.
            return Some((format!("{head}{}", self.known_phonemes(rest)?), 2));
        }

        // Inflection sits outside the compound, not inside it: "namespaces" is
        // name+space+s, and splitting the inflected form directly finds names+paces, which
        // is two real words and the wrong two.
        for (suffix, inflect) in [("s", 0u8), ("es", 0), ("ed", 1), ("ing", 2)] {
            let Some(stem) = lower.strip_suffix(suffix) else {
                continue;
            };
            if stem.chars().count() < 4 || lower.ends_with("ss") {
                continue;
            }
            for candidate in [stem.to_string(), format!("{stem}e")] {
                if candidate == lower {
                    continue;
                }
                let Some((ps, _)) = self.derive(&candidate) else {
                    continue;
                };
                let out = match inflect {
                    0 => self.suffix_s(&ps),
                    1 => self.suffix_ed(&ps),
                    _ => self.suffix_ing(&ps),
                };
                if let Some(out) = out {
                    return Some((out, 2));
                }
            }
        }

        // runbook, webhook, lifecycle: two known words written as one. The most balanced
        // split wins, then the shorter head — "namespace" is name+space, not names+pace,
        // and both of those are real words, so neither longest-head nor first-hit works.
        let chars: Vec<char> = lower.chars().collect();
        let best = (3..=chars.len().saturating_sub(2))
            .filter(|split| {
                let head: String = chars[..*split].iter().collect();
                let tail: String = chars[*split..].iter().collect();
                self.known_word(&head) && self.known_word(&tail)
            })
            .max_by_key(|split| {
                (
                    split.min(&(chars.len() - split)).to_owned(),
                    chars.len() - split,
                )
            })?;
        let head: String = chars[..best].iter().collect();
        let tail: String = chars[best..].iter().collect();
        Some((
            Self::join_compound(&self.known_phonemes(&head)?, &self.known_phonemes(&tail)?),
            2,
        ))
    }
}
