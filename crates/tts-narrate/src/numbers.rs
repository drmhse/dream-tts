//! Numeric notation: dates, ranges, exponents, rates, magnitudes, units, comparisons.
//!
//! Every rule here runs before the general prose rules, because each one depends on
//! punctuation those rules remove: the range rule needs the en dash the em-dash rule is
//! about to turn into a comma, and the rate rule needs the slash the alternation rule is
//! about to turn into "or".

use crate::re::{self, compile};
use crate::tables::{
    alternation_longest_first, map_of, BARE_UNITS, INVARIANT_UNITS, MAGNITUDES, MONTHS, NOT_A_NOUN,
    RATE_UNITS, UNITS,
};
use fancy_regex::Regex;
use once_cell::sync::Lazy;
use std::collections::HashMap;

static ISO_DATE: Lazy<Regex> = Lazy::new(|| compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"));
static SCI_NEG: Lazy<Regex> = Lazy::new(|| compile(r"\b(\d+(?:\.\d+)?)[eE]-(\d+)\b"));
static SCI_POS: Lazy<Regex> = Lazy::new(|| compile(r"\b(\d+(?:\.\d+)?)[eE]\+?(\d+)\b"));
static TEN_NEG: Lazy<Regex> = Lazy::new(|| compile(r"\b10\^\{?-\s*(\d+)\}?"));
static TEN_POS: Lazy<Regex> = Lazy::new(|| compile(r"\b10\^\{?(\d+)\}?"));
static POW_NEG: Lazy<Regex> = Lazy::new(|| compile(r"(?<=[A-Za-z0-9])\^\{?-\s*(\d+)\}?"));
static POW_ANY: Lazy<Regex> = Lazy::new(|| compile(r"(?<=[A-Za-z0-9])\^\{?(\w+)\}?"));
static PERCENT_RANGE: Lazy<Regex> = Lazy::new(|| {
    compile(r"\b(\d+(?:\.\d+)?)\s*(?:%|percent)\s*[-–—]\s*(\d+(?:\.\d+)?)\s*(?:%|percent)")
});
static DASH_RANGE: Lazy<Regex> = Lazy::new(|| compile(r"(\d)\s*[–—]\s*(?=\d)"));
static HYPHEN_RANGE: Lazy<Regex> =
    Lazy::new(|| compile(r"(?<![\w.])(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)(?![\w.])"));
static TWO_DIGIT_YEAR: Lazy<Regex> =
    Lazy::new(|| compile(r"(?<!\d)(\d{2})(\d{2}) to (\d{2})(?!\d)"));
static PERCENTAGE_POINTS: Lazy<Regex> = Lazy::new(|| compile(r"(?<=\d)\s+pp\b"));
static AT_MOST: Lazy<Regex> = Lazy::new(|| compile(r"\s*<=\s*"));
static AT_LEAST: Lazy<Regex> = Lazy::new(|| compile(r"\s*>=\s*"));
static NOT_EQUAL: Lazy<Regex> = Lazy::new(|| compile(r"\s*!=\s*"));
static LESS_NUM: Lazy<Regex> = Lazy::new(|| compile(r"(?<=[\w)])\s*<\s*(?=-?\.?\d)"));
static LESS_WORD: Lazy<Regex> = Lazy::new(|| compile(r"(?<=\d)\s*<\s*(?=[\w(])"));
static GREATER_NUM: Lazy<Regex> = Lazy::new(|| compile(r"(?<=[\w)])\s*>\s*(?=-?\.?\d)"));
static GREATER_WORD: Lazy<Regex> = Lazy::new(|| compile(r"(?<=\d)\s*>\s*(?=[\w(])"));
static EQUALS: Lazy<Regex> = Lazy::new(|| compile(r"(?<=[\w)])\s*=\s*(?=[\w(.-])"));
static PLUS_BETWEEN: Lazy<Regex> = Lazy::new(|| compile(r"(?<=[\w)])\s+\+\s+(?=[\w(])"));
static PLUS_TRAILING: Lazy<Regex> = Lazy::new(|| compile(r"(?<=[\w)])\s+\+\s*$"));
static DEGREES_C: Lazy<Regex> = Lazy::new(|| compile(r"\s*°\s*C\b"));
static DEGREES_F: Lazy<Regex> = Lazy::new(|| compile(r"\s*°\s*F\b"));
static DEGREES: Lazy<Regex> = Lazy::new(|| compile(r"\s*°"));
static SECTION: Lazy<Regex> = Lazy::new(|| compile(r"§\s*"));
static PARAGRAPH: Lazy<Regex> = Lazy::new(|| compile(r"¶\s*"));
static MAGNITUDE: Lazy<Regex> =
    Lazy::new(|| compile(r"(?<![\w.-])(\d+(?:\.\d+)?)\s?([KMBT])\b(?![\w.-])"));
static FOLLOWING_LOWER: Lazy<Regex> = Lazy::new(|| compile(r"^\s+([a-z]+)"));
static FOLLOWING_WORD: Lazy<Regex> = Lazy::new(|| compile(r"^\s+([A-Za-z]+)"));
static ATTRIBUTIVE: Lazy<Regex> =
    Lazy::new(|| compile(r"(?i)\b(?:a|an|the|[A-Za-z]+'s)\s+(?:[a-z]+\s+){0,2}\(?$"));

/// A bracketed pair of numbers is a range, not a list. Read before `clean_inline` strips the
/// brackets and leaves "95 percent CI 1.2, 3.4" — two unrelated figures.
pub static NUMERIC_INTERVAL: Lazy<Regex> =
    Lazy::new(|| compile(r"[\[(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\])]"));

static RATE: Lazy<Regex> = Lazy::new(|| {
    compile(&format!(
        r"\b(\d+(?:\.\d+)?\s*)?([A-Za-z%]+)\s*/\s*({})\b",
        alternation_longest_first(RATE_UNITS.iter().map(|(k, _)| *k))
    ))
});
static UNIT: Lazy<Regex> = Lazy::new(|| {
    compile(&format!(
        r"\b(\d+(?:\.\d+)?)\s*({})\b(?![\w-]|\.\d)",
        alternation_longest_first(UNITS.iter().map(|(k, _)| *k))
    ))
});
static PER_BARE_UNIT: Lazy<Regex> = Lazy::new(|| {
    compile(&format!(
        r"\bper\s+({})\b",
        alternation_longest_first(BARE_UNITS.iter().copied())
    ))
});
static BARE_UNIT: Lazy<Regex> = Lazy::new(|| {
    compile(&format!(
        r"(?<![\w.(])({})\b(?![\w)-])",
        alternation_longest_first(BARE_UNITS.iter().copied())
    ))
});

static UNIT_WORDS: Lazy<HashMap<&str, &str>> = Lazy::new(|| map_of(UNITS));
static RATE_WORDS: Lazy<HashMap<&str, &str>> = Lazy::new(|| map_of(RATE_UNITS));
static MAGNITUDE_WORDS: Lazy<HashMap<&str, &str>> = Lazy::new(|| map_of(MAGNITUDES));

/// A unit follows its quantity's number — "1 MW" is one megawatt — and "per" takes the
/// singular whatever the quantity was.
pub fn unit_word(token: &str, singular: bool) -> String {
    let word = UNIT_WORDS[token];
    if singular || INVARIANT_UNITS.contains(&token) {
        word.to_string()
    } else {
        format!("{word}s")
    }
}

pub fn magnitude_word(suffix: &str) -> Option<&'static str> {
    MAGNITUDE_WORDS.get(suffix).copied()
}

/// Two cases take the singular: a quantity of exactly one, and attributive use — "a 4.4 GW
/// plant", the same construction as "a $12 platform fee".
fn speak_unit(count: &str, token: &str, before: &str, after: &str) -> String {
    let mut singular = count == "1";
    if !singular && ATTRIBUTIVE.is_match(before).unwrap_or(false) {
        // The noun may be capitalised — "a 16 GB M4" — so the test is on its lower-case form.
        if let Ok(Some(caps)) = FOLLOWING_WORD.captures(after) {
            let word = re::g(&caps, 1).to_lowercase();
            if !NOT_A_NOUN.contains(&word.as_str()) {
                singular = true;
            }
        }
    }
    format!("{count} {}", unit_word(token, singular))
}

pub fn speak_numbers(line: &str) -> String {
    // ISO dates, before any rule that reads a hyphen as a range or a minus.
    let mut line = re::sub(&ISO_DATE, line, |c| {
        let (y, mo, d) = (re::g(c, 1), re::g(c, 2), re::g(c, 3));
        let (mo_n, d_n) = (
            mo.parse::<usize>().unwrap_or(0),
            d.parse::<usize>().unwrap_or(0),
        );
        if !(1..=12).contains(&mo_n) || !(1..=31).contains(&d_n) {
            return re::g(c, 0);
        }
        format!(
            "{} {}, {}",
            MONTHS[mo_n - 1],
            d_n,
            y.parse::<u32>().unwrap_or(0)
        )
    });

    // `1.5e-3` and `10^-3` are the same quantity written two ways; both were read as chars.
    line = re::sub_str(&SCI_NEG, &line, "${1} times ten to the minus ${2}");
    line = re::sub_str(&SCI_POS, &line, "${1} times ten to the ${2}");
    line = re::sub_str(&TEN_NEG, &line, "ten to the minus ${1}");
    line = re::sub_str(&TEN_POS, &line, "ten to the ${1}");
    line = re::sub_str(&POW_NEG, &line, " to the minus ${1}");
    line = re::sub_str(&POW_ANY, &line, " to the power ${1}");

    // An en dash between numbers is "to"; as a comma it becomes a list, and "20, 25 degrees"
    // is a different claim than "20 to 25 degrees".
    line = re::sub_str(&PERCENT_RANGE, &line, "${1} to ${2} percent");
    line = re::sub_str(&DASH_RANGE, &line, "${1} to ");
    // Both sides must be bare numbers, or `x86-64` becomes "x86 to 64".
    line = re::sub_str(&HYPHEN_RANGE, &line, "${1} to ${2}");

    // A quantity or a unit must sit left of the slash, or "every bus/node on the grid"
    // becomes "every bus per node".
    line = re::sub(&RATE, &line, |c| {
        let count = re::opt(c, 1).unwrap_or("");
        let (numerator, denominator) = (re::g(c, 2), re::g(c, 3));
        let named = UNIT_WORDS.contains_key(numerator.as_str())
            || numerator == "percent"
            || numerator == "dollars";
        if count.is_empty() && !named {
            return re::g(c, 0);
        }
        format!(
            "{count}{numerator} per {}",
            RATE_WORDS[denominator.as_str()]
        )
    });

    // A magnitude suffix, but only where it is a quantity of something: `v2.0.1`, `p99` and
    // `T5` must survive, and "Qwen3.8 9B on a 16 GB M4" names a model where "7B parameters"
    // counts them — so the noun behind the suffix licenses the reading.
    // `src` is the haystack the closure inspects. Snapshotted because the rule reads the
    // text *around* its match and the assignment target is the same variable.
    let src = line.clone();
    line = re::replace(&MAGNITUDE, &src, |c, _, end| {
        match FOLLOWING_LOWER.captures(&src[end..]) {
            Ok(Some(caps)) if !NOT_A_NOUN.contains(&re::g(&caps, 1).as_str()) => {
                format!("{} {}", re::g(c, 1), MAGNITUDE_WORDS[re::g(c, 2).as_str()])
            }
            _ => re::g(c, 0),
        }
    });

    // Units, after the magnitude rule so `1.5M` is a magnitude rather than a stray metre.
    let src = line.clone();
    line = re::replace(&UNIT, &src, |c, start, end| {
        speak_unit(&re::g(c, 1), &re::g(c, 2), &src[..start], &src[end..])
    });
    line = re::sub(&PER_BARE_UNIT, &line, |c| {
        format!("per {}", unit_word(&re::g(c, 1), true))
    });
    // ...but never the gloss that introduces it: "megawatt hours (MWh)" defines the
    // abbreviation, and expanding it makes the sentence define a term by repeating it.
    line = re::sub(&BARE_UNIT, &line, |c| unit_word(&re::g(c, 1), false));

    // "FY2024 to 25" is 2024 to 2025.
    line = re::sub(&TWO_DIGIT_YEAR, &line, |c| {
        format!(
            "{}{} to {}{}",
            re::g(c, 1),
            re::g(c, 2),
            re::g(c, 1),
            re::g(c, 3)
        )
    });
    line = re::sub_str(&PERCENTAGE_POINTS, &line, " percentage points");

    // `p < 0.05` is the whole claim of a result sentence and the voice says nothing for the
    // operator. Only where a number is on one side: "Open Networking > Tunnels" is a
    // breadcrumb, and reading that as a comparison is worse than leaving it silent.
    line = re::sub_str(&AT_MOST, &line, " at most ");
    line = re::sub_str(&AT_LEAST, &line, " at least ");
    line = re::sub_str(&NOT_EQUAL, &line, " not equal to ");
    line = re::sub_str(&LESS_NUM, &line, " less than ");
    line = re::sub_str(&LESS_WORD, &line, " less than ");
    line = re::sub_str(&GREATER_NUM, &line, " greater than ");
    line = re::sub_str(&GREATER_WORD, &line, " greater than ");
    line = re::sub_str(&EQUALS, &line, " equals ");
    // Now that the equals sign beside it is spoken: "Supply equals Demand + Net Exports"
    // left the addition silent, so the equation lost an operand.
    line = re::sub_str(&PLUS_BETWEEN, &line, " plus ");
    line = re::sub_str(&PLUS_TRAILING, &line, " plus");

    // `°C` is a unit; a bare `°` is an angle.
    line = re::sub_str(&DEGREES_C, &line, " degrees Celsius");
    line = re::sub_str(&DEGREES_F, &line, " degrees Fahrenheit");
    line = re::sub_str(&DEGREES, &line, " degrees");
    line = re::sub_str(&SECTION, &line, "section ");
    re::sub_str(&PARAGRAPH, &line, "paragraph ")
}
