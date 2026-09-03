//! Abbreviations, expanded rather than left to the voice.
//!
//! Two failures, not one: the period is a sentence boundary the segmenter believes, so
//! "Fig. 2" is spoken with a full stop's fall in the middle of a noun phrase; and the
//! abbreviation itself is read as letters ("cf." as "see eff"). A paper fires several per
//! paragraph, which is enough to make the narration sound like a list.

use crate::re::{self, compile};
use fancy_regex::Regex;
use once_cell::sync::Lazy;

/// Longest form first, or `e.g.` is consumed by the `g.` half of a shorter rule and `Ph.D.`
/// by `D.`. The order of this list *is* the rule order.
static PATTERNS: &[(&str, &str)] = &[
    (r"\be\.\s*g\.(?=\s|$)", "for example"),
    (r"\bi\.\s*e\.(?=\s|$)", "that is"),
    (r"\ba\.\s*k\.\s*a\.(?=\s|$)", "also known as"),
    (r"\bw\.\s*r\.\s*t\.(?=\s|$)", "with respect to"),
    (r"\bet\s+al\.(?=[\s,;:)\]]|$)", "and colleagues"),
    (r"\betc\.(?=[\s,;:)\]]|$)", "and so on"),
    (r"\bcf\.(?=\s|$)", "compare"),
    (r"\bvs\.?(?=\s|$)", "versus"),
    (r"\bresp\.(?=\s|$)", "respectively"),
    (r"\bapprox\.(?=\s|$)", "approximately"),
    (r"\bca\.(?=\s+\d)", "circa"),
    (r"\bFigs?\.(?=\s*\d)", "Figure"),
    (r"\bEqs?\.(?=\s*\(?\d)", "Equation"),
    (r"\bTabs?\.(?=\s*\d)", "Table"),
    (r"\bSecs?\.(?=\s*\d)", "Section"),
    (r"\bChs?\.(?=\s*\d)", "Chapter"),
    (r"\bApp\.(?=\s*[A-Z\d])", "Appendix"),
    (r"\bRefs?\.(?=\s*\[?\d)", "Reference"),
    (r"\bAlg\.(?=\s*\d)", "Algorithm"),
    (r"\b[Nn]os?\.(?=\s*\d)", "number"),
    (r"\b[Vv]ol\.(?=\s*\d)", "volume"),
    (r"\bpp\.(?=\s*\d)", "pages"),
    (r"\bp\.(?=\s*\d)", "page"),
    (r"\bPh\.\s*D\.", "PhD"),
    (r"\bM\.\s*Sc\.", "MSc"),
    (r"\bB\.\s*Sc\.", "BSc"),
    (r"\bU\.\s*S\.\s*A?\.", "US"),
    (r"\bDr\.(?=\s|$)", "Doctor"),
    (r"\bProf\.(?=\s|$)", "Professor"),
    (r"\bMr\.(?=\s|$)", "Mister"),
    (r"\bMrs\.(?=\s|$)", "Missus"),
    (r"\bMs\.(?=\s|$)", "Miz"),
    (r"\bSt\.(?=\s+[A-Z])", "Saint"),
];

/// Abbreviated month names, which expand to the month rather than to a fixed string.
static MONTH_ABBREVIATIONS: &[(&str, &str)] = &[
    ("Jan", "January"),
    ("Feb", "February"),
    ("Mar", "March"),
    ("Apr", "April"),
    ("Jun", "June"),
    ("Jul", "July"),
    ("Aug", "August"),
    ("Sep", "September"),
    ("Sept", "September"),
    ("Oct", "October"),
    ("Nov", "November"),
    ("Dec", "December"),
];

static COMPILED: Lazy<Vec<(Regex, &'static str)>> =
    Lazy::new(|| PATTERNS.iter().map(|(p, w)| (compile(p), *w)).collect());
static MONTH_LONG: Lazy<Regex> =
    Lazy::new(|| compile(r"\b(Jan|Feb|Aug|Sept?|Oct|Nov|Dec)\.(?=\s*\d)"));
static MONTH_SHORT: Lazy<Regex> = Lazy::new(|| compile(r"\b(Mar|Apr|Jun|Jul)\.(?=\s*\d)"));

/// A *run* of initials. "J. R. R. Tolkien" is three sentence boundaries to the segmenter and
/// three falling cadences to the listener. A run is required, because a lone capital before a
/// full stop is far more often a label ending a sentence — "demand at A. The model then..." —
/// and stripping that period welds two sentences together.
static INITIALS: Lazy<Regex> = Lazy::new(|| compile(r"\b(?:[A-Z]\.\s*){2,}"));
static INITIAL_DOT: Lazy<Regex> = Lazy::new(|| compile(r"\.\s*"));

fn month(abbr: &str) -> &'static str {
    MONTH_ABBREVIATIONS
        .iter()
        .find(|(k, _)| *k == abbr)
        .map_or("", |(_, v)| *v)
}

pub fn expand(line: &str) -> String {
    let mut line = line.to_string();
    for (regex, word) in COMPILED.iter() {
        line = re::sub_str(regex, &line, word);
    }
    line = re::sub(&MONTH_LONG, &line, |c| month(&re::g(c, 1)).to_string());
    re::sub(&MONTH_SHORT, &line, |c| month(&re::g(c, 1)).to_string())
}

pub fn collapse_initials(line: &str) -> String {
    re::sub(&INITIALS, line, |c| {
        re::sub_str(&INITIAL_DOT, &re::g(c, 0), " ")
    })
}
