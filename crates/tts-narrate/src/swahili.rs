//! Swahili verbalisation, run ahead of the English inline rules so none of them sees a digit.
//! Swahili is head-first ("shilingi mia tano", "asilimia kumi"): units move before the number.

use crate::re::{self, compile};
use fancy_regex::Regex;
use once_cell::sync::Lazy;

const UNITS: [&str; 10] = [
    "sifuri", "moja", "mbili", "tatu", "nne", "tano", "sita", "saba", "nane", "tisa",
];
const TENS: [&str; 10] = [
    "", "kumi", "ishirini", "thelathini", "arobaini", "hamsini", "sitini", "sabini",
    "themanini", "tisini",
];
const MONTHS: [&str; 12] = [
    "Januari", "Februari", "Machi", "Aprili", "Mei", "Juni", "Julai", "Agosti", "Septemba",
    "Oktoba", "Novemba", "Desemba",
];

/// Titles precede a name, so their dot is never a full stop.
const TITLES: &[(&str, &str)] = &[
    ("Dkt.", "Daktari"),
    ("Dk.", "Daktari"),
    ("Bi.", "Bibi"),
    ("Bw.", "Bwana"),
    ("Mhe.", "Mheshimiwa"),
    ("Mh.", "Mheshimiwa"),
    ("Prof.", "Profesa"),
    ("Mt.", "Mtakatifu"),
    ("Kif.", "Kifungu"),
    ("uk.", "ukurasa"),
    ("S.L.P.", "sanduku la posta"),
];
/// These can end a sentence, and then their dot is the full stop too.
const PHRASES: &[(&str, &str)] = &[
    ("n.k.", "na kadhalika"),
    ("k.m.", "kwa mfano"),
    ("B.K.", "baada ya Kristo"),
    ("K.K.", "kabla ya Kristo"),
];
const MEASURES: &[(&str, &str)] = &[
    ("km", "kilomita"),
    ("cm", "sentimita"),
    ("mm", "milimita"),
    ("kg", "kilo"),
    ("ml", "mililita"),
    ("min", "dakika"),
    ("m", "mita"),
    ("g", "gramu"),
    ("l", "lita"),
];

const NUM: &str = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?";

static SHILLING: Lazy<Regex> = Lazy::new(|| {
    compile(&format!(
        r"\b(?:KSh|Ksh|KES|Sh)\.?\s?({NUM})(\s?(?:milioni|bilioni))?"
    ))
});
static TZ_SHILLING: Lazy<Regex> = Lazy::new(|| compile(&format!(r"\bTSh\.?\s?({NUM})")));
static DOLLAR: Lazy<Regex> = Lazy::new(|| compile(&format!(r"\$\s?({NUM})")));
static EURO: Lazy<Regex> = Lazy::new(|| compile(&format!(r"€\s?({NUM})")));
static POUND: Lazy<Regex> = Lazy::new(|| compile(&format!(r"£\s?({NUM})")));
static PERCENT: Lazy<Regex> = Lazy::new(|| compile(&format!(r"({NUM})\s?%")));
static DATE: Lazy<Regex> = Lazy::new(|| compile(r"(?:tarehe\s+)?\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b"));
static DAY_MONTH: Lazy<Regex> = Lazy::new(|| {
    compile(&format!(r"(?:tarehe\s+)?\b(\d{{1,2}})\s+({})\b", MONTHS.join("|")))
});
static RANGE: Lazy<Regex> = Lazy::new(|| compile(&format!(r"\b({NUM})\s?[-–]\s?({NUM})\b")));
static MEASURE: Lazy<Regex> = Lazy::new(|| {
    let names: Vec<&str> = MEASURES.iter().map(|(k, _)| *k).collect();
    compile(&format!(r"\b({NUM})\s?({})\b", names.join("|")))
});
static ORDINAL_SUFFIX: Lazy<Regex> = Lazy::new(|| compile(r"\b(\d+)(?:st|nd|rd|th)\b"));
static NUMBER: Lazy<Regex> = Lazy::new(|| compile(NUM));
static AMPERSAND: Lazy<Regex> = Lazy::new(|| compile(r"\s*&\s*"));
static ARROW: Lazy<Regex> = Lazy::new(|| compile(r"\s*(?:->|=>|\u{2192}|\u{21d2})\s*"));
static TILDE: Lazy<Regex> = Lazy::new(|| compile(r"~\s*(?=\d)"));
static WORD_SLASH: Lazy<Regex> = Lazy::new(|| compile(r"(?<=[A-Za-z])/(?=[A-Za-z])"));
static WORD_PLUS: Lazy<Regex> = Lazy::new(|| compile(r"(?<=[A-Za-z0-9])\s*\+\s*(?=[A-Za-z0-9])"));

fn below_100(n: u64) -> String {
    let (t, u) = (n / 10, n % 10);
    match (t, u) {
        (0, u) => UNITS[u as usize].to_string(),
        (t, 0) => TENS[t as usize].to_string(),
        (t, u) => format!("{} na {}", TENS[t as usize], UNITS[u as usize]),
    }
}

fn below_1000(n: u64) -> String {
    let (h, r) = (n / 100, n % 100);
    match (h, r) {
        (0, r) => below_100(r),
        (h, 0) => format!("mia {}", UNITS[h as usize]),
        (h, r) => format!("mia {} na {}", UNITS[h as usize], below_100(r)),
    }
}

/// Kenyan usage: `laki` for hundreds of thousands, `na` before a trailing part under 100.
pub fn cardinal(n: u64) -> String {
    if n == 0 {
        return UNITS[0].to_string();
    }
    let mut parts: Vec<String> = Vec::new();
    let mut n = n;
    for (size, name) in [
        (1_000_000_000_000u64, "trilioni"),
        (1_000_000_000, "bilioni"),
        (1_000_000, "milioni"),
    ] {
        if n >= size {
            parts.push(format!("{name} {}", cardinal(n / size)));
            n %= size;
        }
    }
    let mut thousands = n / 1000;
    n %= 1000;
    if thousands >= 100 {
        parts.push(format!("laki {}", cardinal(thousands / 100)));
        thousands %= 100;
    }
    if thousands > 0 {
        parts.push(format!("elfu {}", cardinal(thousands)));
    }
    if n > 0 {
        let rest = below_1000(n);
        if !parts.is_empty() && n < 100 {
            parts.push(format!("na {rest}"));
        } else {
            parts.push(rest);
        }
    }
    parts.join(" ")
}

/// `3.25` is "tatu nukta mbili tano".
fn number(raw: &str) -> String {
    let clean = raw.replace(',', "");
    let (whole, frac) = clean.split_once('.').unwrap_or((&clean, ""));
    // A figure too long for u64 is an identifier, not a quantity; read it digit by digit.
    let Ok(w) = whole.parse::<u64>() else {
        return digits(whole);
    };
    let mut out = cardinal(w);
    if !frac.is_empty() {
        out.push_str(" nukta ");
        out.push_str(&digits(frac));
    }
    out
}

/// Two decimals on money are cents: "mia moja na themanini na senti hamsini".
fn money(raw: &str) -> String {
    match raw.split_once('.') {
        Some((whole, cents)) if cents.len() == 2 => {
            let whole = number(whole);
            match cents.parse::<u64>().unwrap_or(0) {
                0 => whole,
                c => format!("{whole} na senti {}", cardinal(c)),
            }
        }
        _ => number(raw),
    }
}

fn digits(s: &str) -> String {
    s.chars()
        .filter_map(|c| c.to_digit(10))
        .map(|d| UNITS[d as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

fn expand_abbreviations(line: &str) -> String {
    let mut out = line.to_string();
    for (k, v) in TITLES {
        let pat = compile(&format!(r"(?<!\w){}", fancy_regex::escape(k)));
        out = re::sub_str(&pat, &out, v);
    }
    for (k, v) in PHRASES {
        let pat = compile(&format!(
            r"(?<!\w){}(?=(\s*$|\s+[A-Z])?)",
            fancy_regex::escape(k)
        ));
        out = re::sub(&pat, &out, |c| {
            if c.get(1).is_some() {
                format!("{v}.")
            } else {
                v.to_string()
            }
        });
    }
    out
}

/// Money, first: `$5` must be read before the `$...$` maths rule sees a delimiter.
pub fn speak_currency(line: &str) -> String {
    let mut s = re::sub(&SHILLING, line, |c| {
        format!("shilingi {}{}", money(&re::g(c, 1)), re::g(c, 2))
    });
    s = re::sub(&TZ_SHILLING, &s, |c| {
        format!("shilingi za Tanzania {}", number(&re::g(c, 1)))
    });
    s = re::sub(&DOLLAR, &s, |c| format!("dola {}", money(&re::g(c, 1))));
    s = re::sub(&EURO, &s, |c| format!("yuro {}", number(&re::g(c, 1))));
    re::sub(&POUND, &s, |c| format!("pauni {}", number(&re::g(c, 1))))
}

/// Everything but money. Links and URLs must already be gone, or their digits are read.
pub fn speak(line: &str) -> String {
    let mut s = expand_abbreviations(line);
    s = re::sub_str(&TILDE, &s, "takriban ");
    s = re::sub(&PERCENT, &s, |c| format!("asilimia {}", number(&re::g(c, 1))));
    s = re::sub(&DATE, &s, |c| {
        let month: usize = re::g(c, 2).parse().unwrap_or(0);
        if !(1..=12).contains(&month) {
            return re::g(c, 0);
        }
        format!(
            "tarehe {} {} {}",
            number(&re::g(c, 1)),
            MONTHS[month - 1],
            number(&re::g(c, 3))
        )
    });
    s = re::sub(&DAY_MONTH, &s, |c| {
        format!("tarehe {} {}", number(&re::g(c, 1)), re::g(c, 2))
    });
    s = re::sub(&RANGE, &s, |c| {
        format!("{} hadi {}", number(&re::g(c, 1)), number(&re::g(c, 2)))
    });
    s = re::sub(&MEASURE, &s, |c| {
        let unit = re::g(c, 2);
        let word = MEASURES
            .iter()
            .find(|(k, _)| *k == unit)
            .map_or(unit.as_str(), |(_, v)| v);
        format!("{word} {}", number(&re::g(c, 1)))
    });
    s = re::sub(&ORDINAL_SUFFIX, &s, |c| number(&re::g(c, 1)));
    s = re::sub(&NUMBER, &s, |c| number(&re::g(c, 0)));
    s = re::sub_str(&AMPERSAND, &s, " na ");
    s = re::sub_str(&ARROW, &s, ", kisha ");
    s = re::sub_str(&WORD_SLASH, &s, " au ");
    re::sub_str(&WORD_PLUS, &s, " na ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cardinals() {
        for (n, want) in [
            (0, "sifuri"),
            (11, "kumi na moja"),
            (40, "arobaini"),
            (105, "mia moja na tano"),
            (125, "mia moja na ishirini na tano"),
            (1000, "elfu moja"),
            (1960, "elfu moja mia tisa na sitini"),
            (2026, "elfu mbili na ishirini na sita"),
            (350_000, "laki tatu elfu hamsini"),
            (1_500_000, "milioni moja laki tano"),
        ] {
            assert_eq!(cardinal(n), want, "{n}");
        }
    }

    #[test]
    fn money_leads_with_the_unit() {
        assert_eq!(speak_currency("KSh 1,500"), "shilingi elfu moja mia tano");
        assert_eq!(speak_currency("$12.50"), "dola kumi na mbili na senti hamsini");
        assert_eq!(speak_currency("Sh200 milioni"), "shilingi mia mbili milioni");
        assert_eq!(speak_currency("KSh 180.50"), "shilingi mia moja na themanini na senti hamsini");
        assert_eq!(speak_currency("$3.00"), "dola tatu");
    }

    #[test]
    fn prose() {
        for (src, want) in [
            ("Mwaka 1960, watu 1,250.", "Mwaka elfu moja mia tisa na sitini, watu elfu moja mia mbili na hamsini."),
            ("punguzo la 15%", "punguzo la asilimia kumi na tano"),
            ("tarehe 25/09/2026", "tarehe ishirini na tano Septemba elfu mbili na ishirini na sita"),
            ("walifika 12 Machi", "walifika tarehe kumi na mbili Machi"),
            ("tarehe 12 Machi", "tarehe kumi na mbili Machi"),
            ("umbali wa 42 km", "umbali wa kilomita arobaini na mbili"),
            ("uzito 3.5 kg", "uzito kilo tatu nukta tano"),
            ("wanafunzi 5-10", "wanafunzi tano hadi kumi"),
            ("vitabu, kalamu, n.k.", "vitabu, kalamu, na kadhalika."),
            ("Dkt. Omari na Bi. Amina", "Daktari Omari na Bibi Amina"),
            ("chai & mkate", "chai na mkate"),
            ("~20 watu", "takriban ishirini watu"),
        ] {
            assert_eq!(speak(src), want, "{src}");
        }
    }
}
