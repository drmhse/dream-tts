//! Written forms a listener cannot see and a voice gets wrong: clock times and durations,
//! units with their prefixes and powers, superscripts and subscripts.
//!
//! Run on a block's source before the markdown rules, which would otherwise see `m/s²` as a
//! slash between two words and `14:00` as two numbers. Code spans are left as written. Each
//! rule turns a form into words the front end already says well — "14 hundred", "9 30 p m",
//! "metres per second squared" — rather than spelling numbers out itself.
//!
//! Measured on 2026-11-07, before this: "14:00" was "fourteen zero zero", "2:03:45" "two zero
//! three forty-five", "9.30pm" "nine point three p m", "kPa" "kappa"; µm, nm, cm and ly were
//! dropped as unsayable; "10⁹" was "one hundred nine"; a footnote's "here¹" was "here one".

use fancy_regex::{Captures, Regex};
use once_cell::sync::Lazy;

/// The pass, outside code spans.
pub fn spoken(source: &str) -> String {
    let mut out = String::with_capacity(source.len() + 16);
    // Backtick runs open and close code spans of the same length; inside one nothing changes.
    let mut rest = source;
    while let Some(open) = rest.find('`') {
        let ticks = rest[open..].chars().take_while(|&c| c == '`').count();
        let fence = &rest[open..open + ticks];
        out.push_str(&prose(&rest[..open]));
        let body = &rest[open + ticks..];
        match body.find(fence) {
            Some(close) => {
                out.push_str(fence);
                out.push_str(&body[..close + ticks]);
                rest = &body[close + ticks..];
            }
            None => {
                out.push_str(fence);
                rest = body;
            }
        }
    }
    out.push_str(&prose(rest));
    out
}

fn prose(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let text = ordinals(text);
    let text = footnotes(&text);
    let text = durations(&text);
    let text = times(&text);
    let text = units(&text);
    let text = powers(&text);
    subscripts(&text)
}

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).expect("a valid pattern")
}

fn sub(re: &Regex, text: &str, f: impl Fn(&Captures<'_, str>) -> String) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for caps in re.captures_iter(text).flatten() {
        let m = caps.get(0).expect("a match");
        out.push_str(&text[last..m.start()]);
        out.push_str(&f(&caps));
        last = m.end();
    }
    out.push_str(&text[last..]);
    out
}

fn g<'a>(c: &'a Captures<'_, str>, i: usize) -> &'a str {
    c.get(i).map_or("", |m| m.as_str())
}

// ---------------------------------------------------------------- superscripts

const SUPERSCRIPT: &str = "⁰¹²³⁴⁵⁶⁷⁸⁹";

/// `²³` to `23`, `⁻²⁷` to `-27`.
fn plain(sup: &str) -> String {
    sup.chars()
        .map(|c| match c {
            '⁻' => '-',
            '⁺' => '+',
            c => SUPERSCRIPT
                .chars()
                .position(|s| s == c)
                .map_or(c, |d| char::from(b'0' + d as u8)),
        })
        .collect()
}

static ORDINAL_MARK: Lazy<Regex> = Lazy::new(|| re(r"(\d)(ˢᵗ|ⁿᵈ|ʳᵈ|ᵗʰ)"));

/// `1ˢᵗ` is `1st`.
fn ordinals(text: &str) -> String {
    sub(&ORDINAL_MARK, text, |c| {
        let suffix = match g(c, 2) {
            "ˢᵗ" => "st",
            "ⁿᵈ" => "nd",
            "ʳᵈ" => "rd",
            _ => "th",
        };
        format!("{}{suffix}", g(c, 1))
    })
}

/// A superscript number straight after a word of three letters or more, or after the
/// punctuation that ends one, marks a footnote: nothing a listener needs said. After a single
/// letter or two (`x²`, `mc²`) or a digit it is a power, and after a unit the units rule has it.
static FOOTNOTE: Lazy<Regex> = Lazy::new(|| {
    re(
        r"(?:(?<=[A-Za-z]{3})|(?<=[.,;:!?)\]’”'\x22]))[⁰¹²³⁴⁵⁶⁷⁸⁹]+(?:[,⁻][⁰¹²³⁴⁵⁶⁷⁸⁹]+)*(?![⁰¹²³⁴⁵⁶⁷⁸⁹])",
    )
});

fn footnotes(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for m in FOOTNOTE.find_iter(text).flatten() {
        let word: String = text[..m.start()]
            .chars()
            .rev()
            .take_while(|c| c.is_alphabetic())
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        // A unit with a power (`cm³`, `km²`) is not a footnote.
        if is_unit(&word) {
            continue;
        }
        out.push_str(&text[last..m.start()]);
        last = m.end();
    }
    out.push_str(&text[last..]);
    out
}

static CARET: Lazy<Regex> = Lazy::new(|| re(r"(?<![A-Za-z0-9])([A-Za-z])\^\{?(-?\d+)\}?(?![\d}])"));
static POWER: Lazy<Regex> = Lazy::new(|| {
    re(r"(?:(?<![A-Za-z])([b-df-hj-np-tv-z]{2})|(?<=[A-Za-z0-9)]))([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)")
});

/// What is left after the footnotes and units are powers: `10⁹` "10 to the 9th", `x²`
/// "x squared", `10⁻²⁷` "10 to the minus 27th". Two consonants before one are two symbols,
/// spelled: "mc squared", which said as a word was dropped.
fn powers(text: &str) -> String {
    // A caret on a letter is the same power typed plainly: "x^2 + y^2" was "x to the power 2 or
    // higher", the plus taken for "2+". `10^9` is left to the number rules, which have it.
    let text = sub(&CARET, text, |c| {
        format!("{} {}", g(c, 1), power_words(g(c, 2)))
    });
    sub(&POWER, &text, |c| {
        let base = g(c, 1);
        let spelled = if base.is_empty() {
            String::new()
        } else {
            base.chars().map(String::from).collect::<Vec<_>>().join(" ")
        };
        format!("{spelled} {}", power_words(&plain(g(c, 2))))
    })
}

fn power_words(n: &str) -> String {
    match n {
        "2" => "squared".into(),
        "3" => "cubed".into(),
        _ => match n.strip_prefix('-') {
            Some(m) => format!("to the minus {}", ordinal(m)),
            None => format!("to the {}", ordinal(n.trim_start_matches('+'))),
        },
    }
}

/// `9` to `9th`, which the front end says as an ordinal.
fn ordinal(n: &str) -> String {
    let suffix = if n.ends_with("11") || n.ends_with("12") || n.ends_with("13") {
        "th"
    } else if n.ends_with('1') {
        "st"
    } else if n.ends_with('2') {
        "nd"
    } else if n.ends_with('3') {
        "rd"
    } else {
        "th"
    };
    format!("{n}{suffix}")
}

// ---------------------------------------------------------------- subscripts

static SUBSCRIPT_DIGITS: Lazy<Regex> = Lazy::new(|| re(r"([A-Za-z])([₀₁₂₃₄₅₆₇₈₉]+)"));
/// A formula of element symbols and subscript digits: `H₂O`, `CO₂`, `C₆H₁₂O₆`.
static FORMULA: Lazy<Regex> = Lazy::new(|| re(r"^(?:[A-Z][a-z]?[₀₁₂₃₄₅₆₇₈₉]*)+$"));
static ELEMENT: Lazy<Regex> = Lazy::new(|| re(r"([A-Z][a-z]?)([₀₁₂₃₄₅₆₇₈₉]*)"));
/// `x_1`, `a_i`, `b_{ij}`: one letter, then what it is subscripted by. A longer name before
/// the underscore is an identifier (`max_len`), not maths.
static UNDERSCORE: Lazy<Regex> = Lazy::new(|| {
    re(r"(?<![A-Za-z0-9_\\])([A-Za-z])_(?:\{([A-Za-z0-9,]+)\}|([A-Za-z0-9]))(?![A-Za-z0-9_])")
});

fn subscripts(text: &str) -> String {
    let text = sub(&UNDERSCORE, text, |c| {
        let index = if g(c, 2).is_empty() { g(c, 3) } else { g(c, 2) };
        let spaced: Vec<String> = index
            .split(',')
            .map(|p| {
                if p.chars().all(|ch| ch.is_ascii_digit()) {
                    p.to_string()
                } else {
                    p.chars().map(String::from).collect::<Vec<_>>().join(" ")
                }
            })
            .collect();
        format!("{} sub {}", g(c, 1), spaced.join(" "))
    });
    // In a formula the digit is said after its element ("H 2 O"); elsewhere it is an index.
    let mut out = String::with_capacity(text.len());
    for (i, word) in text.split(' ').enumerate() {
        if i > 0 {
            out.push(' ');
        }
        let core = word.trim_matches(|c: char| !c.is_alphanumeric() && !"₀₁₂₃₄₅₆₇₈₉".contains(c));
        let formula = FORMULA.is_match(core).unwrap_or(false)
            && core.chars().any(|c| "₀₁₂₃₄₅₆₇₈₉".contains(c));
        if formula {
            // Element by element, or "CO" is read as a word.
            let spelled = sub(&ELEMENT, word, |c| {
                let digits: String = g(c, 2)
                    .chars()
                    .map(|d| {
                        char::from(
                            b'0' + "₀₁₂₃₄₅₆₇₈₉".chars().position(|s| s == d).unwrap_or(0) as u8,
                        )
                    })
                    .collect();
                if digits.is_empty() {
                    format!("{} ", g(c, 1))
                } else {
                    format!("{} {digits} ", g(c, 1))
                }
            });
            out.push_str(spelled.trim_end());
            continue;
        }
        out.push_str(
            &sub(&SUBSCRIPT_DIGITS, word, |c| {
                let digits: String = g(c, 2)
                    .chars()
                    .map(|d| {
                        char::from(
                            b'0' + "₀₁₂₃₄₅₆₇₈₉".chars().position(|s| s == d).unwrap_or(0) as u8,
                        )
                    })
                    .collect();
                if formula {
                    format!("{} {digits} ", g(c, 1))
                } else {
                    format!("{} sub {digits} ", g(c, 1))
                }
            })
            .trim_end()
            .replace("  ", " "),
        );
    }
    out
}

// ---------------------------------------------------------------- times and durations

static DURATION_HMS: Lazy<Regex> =
    Lazy::new(|| re(r"(?<![\d:.])(\d{1,3}):([0-5]\d):([0-5]\d)(?![\d:]|\s?[AaPp]\.?\s?[Mm]\b)"));
static DURATION_MS: Lazy<Regex> = Lazy::new(|| {
    re(
        r"(?i)\b(in|of|took|takes|finished|time|runtime|lasted|lasts|lasting|duration|length|about|under|over)\s(\d{1,2}):([0-5]\d)(?![\d:])(?!\s?[ap]\.?\s?m\b)",
    )
});
static COMPOUND: Lazy<Regex> =
    Lazy::new(|| re(r"\b(\d+)\s?h(?:rs?)?\s?(\d+)\s?m(?:in)?(?:\s?(\d+)\s?s(?:ec)?)?\b"));

/// `2:03:45` and, after a word that introduces one, `4:30`: a length of time, not a clock.
fn durations(text: &str) -> String {
    let text = sub(&COMPOUND, text, |c| {
        let mut parts = vec![count(g(c, 1), "hour"), count(g(c, 2), "minute")];
        if !g(c, 3).is_empty() {
            parts.push(count(g(c, 3), "second"));
        }
        and_list(&parts)
    });
    let text = sub(&DURATION_HMS, &text, |c| {
        let mut parts = Vec::new();
        let hours = g(c, 1).trim_start_matches('0');
        if !hours.is_empty() {
            parts.push(count(hours, "hour"));
        }
        parts.push(count(g(c, 2).trim_start_matches('0').max("0"), "minute"));
        parts.push(count(g(c, 3).trim_start_matches('0').max("0"), "second"));
        and_list(&parts)
    });
    sub(&DURATION_MS, &text, |c| {
        format!(
            "{} {}",
            g(c, 1),
            and_list(&[
                count(g(c, 2).trim_start_matches('0').max("0"), "minute"),
                count(g(c, 3).trim_start_matches('0').max("0"), "second"),
            ])
        )
    })
}

fn count(n: &str, unit: &str) -> String {
    let n = if n.is_empty() { "0" } else { n };
    format!("{n} {unit}{}", if n == "1" { "" } else { "s" })
}

fn and_list(parts: &[String]) -> String {
    match parts {
        [] => String::new(),
        [one] => one.clone(),
        [init @ .., last] => format!("{} and {last}", init.join(", ")),
    }
}

/// A clock time, with am or pm in any of its spellings, or a dotted British one that has it.
/// The dot after "a.m." is left to `ampm_dot`: it may also end the sentence.
static CLOCK_AMPM: Lazy<Regex> =
    Lazy::new(|| re(r"(?<![\d:.])(\d{1,2})[:.]([0-5]\d)\s?([AaPp])\.?\s?[Mm](?=[\s,;:!?.)\]]|$)"));
static CLOCK: Lazy<Regex> =
    Lazy::new(|| re(r"(?<![\d:.$£€])([01]?\d|2[0-3]):([0-5]\d)(?![\d:]|\.\d)"));
/// Two numbers about a colon that are no clock: `3:2`, `16:9`.
static RATIO: Lazy<Regex> = Lazy::new(|| re(r"(?<![\d:.])(\d{1,3}):(\d{1,3})(?![\d:])"));
/// `0800` and `0930 hours` after "at": the military clock.
static MILITARY: Lazy<Regex> =
    Lazy::new(|| re(r"(?i)(?<=\bat\s)([01]\d|2[0-3])([0-5]\d)\b(?:\s?hours\b)?"));
static HOUR_AMPM: Lazy<Regex> =
    Lazy::new(|| re(r"(?<![\d:.])(\d{1,2})\s?([AaPp])\.?\s?[Mm](?=[\s,;:!?.)\]]|$)"));
/// "a m." left by the clock rules: the dot ends the sentence only when one starts after it.
static AMPM_DOT: Lazy<Regex> = Lazy::new(|| re(r"\b([ap] m)\.(?=\s+[a-z0-9])"));

fn times(text: &str) -> String {
    let text = sub(&CLOCK_AMPM, text, |c| {
        let hour = g(c, 1).trim_start_matches('0').max("0").to_string();
        format!("{} {}", clock(&hour, g(c, 2), false), ampm(g(c, 3)))
    });
    let text = sub(&HOUR_AMPM, &text, |c| {
        format!(
            "{} {}",
            g(c, 1).trim_start_matches('0').max("0"),
            ampm(g(c, 2))
        )
    });
    let text = sub(&MILITARY, &text, |c| {
        let hour = g(c, 1).trim_start_matches('0');
        if g(c, 2) == "00" {
            format!("{} hundred", if hour.is_empty() { "0" } else { hour })
        } else {
            format!(
                "{} {}",
                if hour.is_empty() { "0" } else { hour },
                minutes(g(c, 2))
            )
        }
    });
    let text = sub(&CLOCK, &text, |c| {
        let raw = g(c, 1);
        let hour = raw.trim_start_matches('0').max("0");
        let twenty_four =
            raw.len() == 2 && (raw.starts_with('0') || hour.parse::<u32>().unwrap_or(0) >= 13);
        clock(hour, g(c, 2), twenty_four)
    });
    let text = sub(&AMPM_DOT, &text, |c| g(c, 1).to_string());
    sub(&RATIO, &text, |c| format!("{} to {}", g(c, 1), g(c, 2)))
}

/// Hours and minutes as they are said: "9 30", "12 oh 5", "7 o'clock", "14 hundred".
fn clock(hour: &str, mins: &str, twenty_four: bool) -> String {
    match (mins, twenty_four) {
        ("00", true) => format!("{hour} hundred"),
        ("00", false) => format!("{hour} o'clock"),
        _ => format!("{hour} {}", minutes(mins)),
    }
}

fn minutes(mins: &str) -> String {
    match mins.strip_prefix('0') {
        Some(d) => format!("oh {d}"),
        None => mins.to_string(),
    }
}

fn ampm(letter: &str) -> &'static str {
    if letter.eq_ignore_ascii_case("a") {
        "a m"
    } else {
        "p m"
    }
}

// ---------------------------------------------------------------- units

/// Prefixes as written and said.
const PREFIXES: &[(&str, &str)] = &[
    ("p", "pico"),
    ("n", "nano"),
    ("µ", "micro"),
    ("μ", "micro"),
    ("u", "micro"),
    ("m", "milli"),
    ("c", "centi"),
    ("d", "deci"),
    ("k", "kilo"),
    ("K", "kilo"),
    ("M", "mega"),
    ("G", "giga"),
    ("T", "tera"),
    ("P", "peta"),
];

/// A base unit, how it is said, its plural, and the prefixes it takes. One letter bases are
/// units only after a number: on their own they are letters and words.
const BASES: &[(&str, &str, &str, &str)] = &[
    ("m", "metre", "metres", "pnµμumcdk"),
    ("g", "gram", "grams", "nµμumk"),
    ("s", "second", "seconds", "pnµμum"),
    ("A", "ampere", "amperes", "nµμumk"),
    ("V", "volt", "volts", "µμumkM"),
    ("W", "watt", "watts", "µμumkMGT"),
    ("Wh", "watt hour", "watt hours", "kMGT"),
    ("Hz", "hertz", "hertz", "kMGT"),
    ("Pa", "pascal", "pascals", "hkMG"),
    ("J", "joule", "joules", "mkMG"),
    ("N", "newton", "newtons", "mk"),
    ("Ω", "ohm", "ohms", "mkM"),
    ("F", "farad", "farads", "pnµμum"),
    ("T", "tesla", "teslas", "µμnm"),
    ("mol", "mole", "moles", "nµμum"),
    ("L", "litre", "litres", "µμumcd"),
    ("eV", "electronvolt", "electronvolts", "kMGT"),
    ("B", "byte", "bytes", "kKMGTP"),
    ("b", "bit", "bits", "kKMGT"),
    ("bps", "bit per second", "bits per second", "kKMGT"),
    ("cal", "calorie", "calories", "k"),
    ("Bq", "becquerel", "becquerels", "kMG"),
    ("Gy", "gray", "grays", "m"),
    ("Sv", "sievert", "sieverts", "mµμ"),
    ("lm", "lumen", "lumens", ""),
    ("lx", "lux", "lux", ""),
    ("K", "kelvin", "kelvins", ""),
    ("min", "minute", "minutes", ""),
    ("mins", "minute", "minutes", ""),
    ("h", "hour", "hours", ""),
    ("hr", "hour", "hours", ""),
    ("hrs", "hour", "hours", ""),
    ("sec", "second", "seconds", ""),
    ("secs", "second", "seconds", ""),
    ("AU", "astronomical unit", "astronomical units", ""),
    ("ly", "light year", "light years", ""),
    ("kpc", "kiloparsec", "kiloparsecs", ""),
    ("Mpc", "megaparsec", "megaparsecs", ""),
    ("atm", "atmosphere", "atmospheres", ""),
    ("bar", "bar", "bars", "m"),
    ("px", "pixel", "pixels", ""),
    ("dpi", "dot per inch", "dots per inch", ""),
    ("M", "molar", "molar", "mµμun"),
];

/// Units that are also words, letters or names when written alone: said as units only right
/// after a number with a space between, and never after a word that names a grade or a model.
const AMBIGUOUS: &[&str] = &[
    "A", "T", "N", "J", "K", "V", "W", "L", "g", "s", "m", "h", "bar",
];
/// Units only with a prefix: alone they are far more often a letter — "1.5 M users", "45 F",
/// "8 B parameters".
const PREFIXED_ONLY: &[&str] = &["M", "B", "b", "F"];
const NAMING: &[&str] = &[
    "grade", "class", "level", "plan", "type", "model", "section", "part", "row", "seat", "gate",
    "platform", "vitamin", "size", "chapter", "figure", "fig", "table", "page", "group", "team",
    "series", "version", "phase", "round", "stage", "unit", "box", "line", "room", "floor",
    "route", "highway", "street", "avenue", "exit", "block", "lot", "bay", "hall", "wing",
];

/// The unit a symbol names, as (singular, plural), with its prefix.
fn unit_named(symbol: &str) -> Option<(String, String)> {
    for (base, one, many, takes) in BASES {
        if symbol == *base {
            if PREFIXED_ONLY.contains(base) {
                continue;
            }
            return Some((one.to_string(), many.to_string()));
        }
        if let Some(prefix) = symbol.strip_suffix(base) {
            if prefix.is_empty() {
                continue;
            }
            if let Some((p, said)) = PREFIXES.iter().find(|(p, _)| *p == prefix) {
                if takes.contains(p) {
                    return Some((format!("{said}{one}"), format!("{said}{many}")));
                }
            }
        }
    }
    None
}

fn is_unit(symbol: &str) -> bool {
    unit_named(symbol).is_some()
}

/// A quantity and its unit: a number (maybe `× 10ⁿ`), then a unit with an optional power, then
/// any further units after `/`, `·` or `⋅`, each with its own power.
static QUANTITY: Lazy<Regex> = Lazy::new(|| {
    let unit = r"[A-Za-zµμΩ]{1,4}(?:\^\{?-?\d+\}?|[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)?";
    re(&format!(
        r"(?<![\w.$£€])(\d+(?:[.,]\d+)?(?:\s?[×x]\s?10(?:\^\{{?-?\d+\}}?|[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+))?)(\s?)({unit}(?:\s?[/·⋅]\s?{unit})*)(?![\w/])"
    ))
});

fn units(text: &str) -> String {
    let src = text.to_string();
    let mut out = String::with_capacity(src.len());
    let mut last = 0;
    for caps in QUANTITY.captures_iter(src.as_str()).flatten() {
        let m = caps.get(0).expect("a match");
        let (number, gap, written) = (g(&caps, 1), g(&caps, 2), g(&caps, 3));
        let Some(said) = said_units(number, written) else {
            continue;
        };
        // One-letter units are letters until a number and a single space vouch for them.
        let first = written.split(['/', '·', '⋅']).next().unwrap_or("").trim();
        let bare: String = first
            .chars()
            .take_while(|c| !"^⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺".contains(*c))
            .collect();
        if AMBIGUOUS.contains(&bare.as_str()) {
            let before = src[..m.start()].trim_end();
            let previous = before
                .rsplit(|c: char| !c.is_alphanumeric())
                .next()
                .unwrap_or("")
                .to_lowercase();
            let compound_unit = written.contains(['/', '·', '⋅']) || bare != first;
            if (gap.is_empty() && !compound_unit && bare.chars().count() == 1)
                || NAMING.contains(&previous.as_str())
            {
                continue;
            }
        }
        // "a 10 µm particle": a quantity before its noun is said in the singular.
        let attributive = ATTRIBUTIVE.is_match(&src[..m.start()]).unwrap_or(false)
            && NOUN_AFTER.is_match(&src[m.end()..]).unwrap_or(false);
        let said = if attributive {
            said_units("1", written).unwrap_or(said)
        } else {
            said
        };
        out.push_str(&src[last..m.start()]);
        out.push_str(&format!("{number} {said}"));
        last = m.end();
    }
    out.push_str(&src[last..]);
    out
}

static ATTRIBUTIVE: Lazy<Regex> = Lazy::new(|| re(r"(?i)\b(?:a|an|the|each|every|one)\s+$"));
static NOUN_AFTER: Lazy<Regex> = Lazy::new(|| {
    re(
        r"^\s+(?!(?:to|from|and|or|for|in|on|of|at|than|per|by|is|was|were|with|under|over|but|apart|away|long|wide|high|deep|tall|thick|ago)\b)[a-z]{3,}",
    )
});

/// Lengths take "square" and "cubic" before them wherever they stand; anything else takes
/// "squared" after it when it divides: "per cubic centimetre", "per second squared".
fn is_length(symbol: &str) -> bool {
    symbol == "m"
        || PREFIXES
            .iter()
            .any(|(p, _)| symbol.strip_prefix(p) == Some("m"))
}

/// `m/s²` as "metres per second squared"; None when any part is not a unit.
fn said_units(number: &str, written: &str) -> Option<String> {
    let parts: Vec<&str> = written.split(['/', '·', '⋅']).map(str::trim).collect();
    let dividers: Vec<char> = written.chars().filter(|c| "/·⋅".contains(*c)).collect();
    let singular = number == "1";
    let mut said = String::new();
    for (i, part) in parts.iter().enumerate() {
        let (symbol, power) = split_power(part);
        let (one, many) = unit_named(symbol)?;
        let per = i > 0 && dividers.get(i - 1) == Some(&'/');
        // The first unit follows the number's count; after "per" a unit is singular. A unit
        // multiplied in (`N·m`) is said as one compound noun with the count on the last.
        let last_of_product = i + 1 == parts.len() || dividers.get(i) == Some(&'/');
        let name = if per || singular || !last_of_product {
            one.clone()
        } else {
            many
        };
        if i > 0 {
            said.push_str(if per { " per " } else { " " });
        }
        let negative = power.as_deref().is_some_and(|p| p.starts_with('-'));
        match power.as_deref() {
            Some("2") if i == 0 || !per || is_length(symbol) => {
                said.push_str(&format!("square {name}"))
            }
            Some("3") if i == 0 || !per || is_length(symbol) => {
                said.push_str(&format!("cubic {name}"))
            }
            Some("2") => said.push_str(&format!("{name} squared")),
            Some("3") => said.push_str(&format!("{name} cubed")),
            // `s⁻¹` is "per second".
            Some("-1") if i > 0 || parts.len() == 1 => said.push_str(&format!("per {one}")),
            Some(p) if negative => {
                said.push_str(&format!("{name} to the minus {}", ordinal(&p[1..])))
            }
            Some(p) => said.push_str(&format!("{name} to the {}", ordinal(p))),
            None => said.push_str(&name),
        }
    }
    Some(said.replace("  ", " "))
}

/// `cm³` to (`cm`, `3`), `s^-1` to (`s`, `-1`).
fn split_power(part: &str) -> (&str, Option<String>) {
    if let Some(at) = part.find('^') {
        let p = part[at + 1..].trim_matches(|c| c == '{' || c == '}');
        return (&part[..at], Some(p.to_string()));
    }
    match part.find(|c: char| "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺".contains(c)) {
        Some(at) => (
            &part[..at],
            Some(plain(&part[at..]).trim_start_matches('+').to_string()),
        ),
        None => (part, None),
    }
}

#[cfg(test)]
mod tests {
    use super::spoken;

    fn says(cases: &[(&str, &str)]) {
        let wrong: Vec<String> = cases
            .iter()
            .filter(|(text, want)| spoken(text) != *want)
            .map(|(text, want)| format!("{text:?}\n   said {:?}\n   want {want:?}", spoken(text)))
            .collect();
        assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
    }

    #[test]
    fn times() {
        says(&[
            (
                "The meeting is at 9:30 a.m. tomorrow.",
                "The meeting is at 9 30 a m tomorrow.",
            ),
            (
                "At 14:00 the market closed.",
                "At 14 hundred the market closed.",
            ),
            ("Updated 10:41 AM EST.", "Updated 10 41 a m EST."),
            ("It ends at 12:05.", "It ends at 12 oh 5."),
            ("Dinner at 7:00 tonight.", "Dinner at 7 o'clock tonight."),
            ("At 9.30pm we left.", "At 9 30 p m we left."),
            ("Doors open at 7pm.", "Doors open at 7 p m."),
            ("It starts at 7 PM sharp.", "It starts at 7 p m sharp."),
            ("Between 9am and 5pm.", "Between 9 a m and 5 p m."),
            ("She came back at 0800.", "She came back at 8 hundred."),
            ("The 3:2 ratio held.", "The 3 to 2 ratio held."),
            ("A 16:9 screen.", "A 16 to 9 screen."),
        ]);
    }

    #[test]
    fn durations() {
        says(&[
            (
                "The race finished in 2:03:45.",
                "The race finished in 2 hours, 3 minutes and 45 seconds.",
            ),
            (
                "It took 4:30 to boil.",
                "It took 4 minutes and 30 seconds to boil.",
            ),
            ("Runtime 1h 30m.", "Runtime 1 hour and 30 minutes."),
            ("Posted 5 min ago.", "Posted 5 minutes ago."),
            ("Wait 1 min.", "Wait 1 minute."),
            ("A 0:00:05 clip.", "A 0 minutes and 5 seconds clip."),
        ]);
    }

    #[test]
    fn units() {
        says(&[
            ("Pressure 101.3 kPa.", "Pressure 101.3 kilopascals."),
            ("A 10 µm particle.", "A 10 micrometre particle."),
            ("Wavelength 532 nm.", "Wavelength 532 nanometres."),
            ("Distance 4.2 ly.", "Distance 4.2 light years."),
            ("Field strength 3 T.", "Field strength 3 teslas."),
            ("It drew 2.5 A at 12 V.", "It drew 2.5 amperes at 12 volts."),
            ("Energy 13.6 eV.", "Energy 13.6 electronvolts."),
            (
                "Acceleration 9.81 m/s².",
                "Acceleration 9.81 metres per second squared.",
            ),
            (
                "Density 1.2 g/cm³.",
                "Density 1.2 grams per cubic centimetre.",
            ),
            ("Area of 30 m².", "Area of 30 square metres."),
            ("Bandwidth 100 Mbps.", "Bandwidth 100 megabits per second."),
            (
                "Concentration 0.1 mol/L.",
                "Concentration 0.1 moles per litre.",
            ),
            ("A rate of 5 s⁻¹.", "A rate of 5 per second."),
            ("Torque 40 N·m.", "Torque 40 newton metres."),
            (
                "Mass 1.67×10⁻²⁷ kg.",
                "Mass 1.67×10 to the minus 27th kilograms.",
            ),
            ("It weighs 1 kg.", "It weighs 1 kilogram."),
            ("A 5 Ω resistor.", "A 5 ohm resistor."),
            ("It is 300 K.", "It is 300 kelvins."),
        ]);
    }

    #[test]
    fn letters_and_words_stay_themselves() {
        says(&[
            ("Plan A was better.", "Plan A was better."),
            ("A T-shirt and a Model T.", "A T-shirt and a Model T."),
            ("V for Vendetta.", "V for Vendetta."),
            ("He got Grade 5 A.", "He got Grade 5 A."),
            (
                "Vitamin B12 and CO2 emissions.",
                "Vitamin B12 and CO2 emissions.",
            ),
            ("The 100m sprint.", "The 100m sprint."),
            ("It cost $30m.", "It cost $30m."),
            ("x86-64 and v2.0.1", "x86-64 and v2.0.1"),
            ("John 3:16 says", "John 3 16 says"),
            ("A 4K video and a 5G phone.", "A 4K video and a 5G phone."),
            (
                "Some 1.5 M users, 45 F outside, 8 B parameters.",
                "Some 1.5 M users, 45 F outside, 8 B parameters.",
            ),
            (
                "We meet at 9 a.m. Then lunch.",
                "We meet at 9 a m. Then lunch.",
            ),
            ("Take Route 5 W north.", "Take Route 5 W north."),
            ("`a_i` in code and `14:00`", "`a_i` in code and `14:00`"),
            ("max_len and snake_case", "max_len and snake_case"),
        ]);
    }

    #[test]
    fn superscripts_and_subscripts() {
        says(&[
            ("10⁹ parameters.", "10 to the 9th parameters."),
            ("E = mc²", "E = m c squared"),
            ("x^2 + y^{3}", "x squared + y cubed"),
            ("x² + y² = r²", "x squared + y squared = r squared"),
            ("Mass 10⁻²⁷", "Mass 10 to the minus 27th"),
            ("A marker here¹ and there²³.", "A marker here and there."),
            ("as shown.¹² Next", "as shown. Next"),
            ("H₂O and CO₂ levels rose.", "H 2 O and C O 2 levels rose."),
            ("C₆H₁₂O₆ is glucose.", "C 6 H 12 O 6 is glucose."),
            ("x₁ plus x₂", "x sub 1 plus x sub 2"),
            ("a_i and b_{ij}", "a sub i and b sub i j"),
            ("The 1ˢᵗ and 2ⁿᵈ.", "The 1st and 2nd."),
        ]);
    }
}
