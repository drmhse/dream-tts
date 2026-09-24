//! English number words, matching the `num2words` output misaki feeds to its lexicon.
//!
//! The words are only an intermediate: misaki splits them on `[^a-z]+` and looks each up,
//! so hyphens and commas are irrelevant but "and" is not — it is dropped unless the token
//! carries the `&` flag. `scripts/check-phonemes.sh` compares against the Python over a
//! swept range; the placement of "and" versus "," is the part that is easy to get subtly
//! wrong and impossible to hear.

const ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];
const SCALES: [&str; 7] = [
    "",
    "thousand",
    "million",
    "billion",
    "trillion",
    "quadrillion",
    "quintillion",
];

fn under_100(v: u64) -> String {
    if v < 20 {
        return ONES[v as usize].to_string();
    }
    let (t, o) = (v / 10, v % 10);
    if o == 0 {
        TENS[t as usize].to_string()
    } else {
        format!("{}-{}", TENS[t as usize], ONES[o as usize])
    }
}

fn under_1000(v: u64) -> String {
    let (h, r) = (v / 100, v % 100);
    if h == 0 {
        return under_100(r);
    }
    let head = format!("{} hundred", ONES[h as usize]);
    if r == 0 {
        head
    } else {
        format!("{head} and {}", under_100(r))
    }
}

pub fn cardinal(n: i64) -> String {
    if n < 0 {
        return format!("minus {}", cardinal(n.unsigned_abs() as i64));
    }
    let mut v = n as u64;
    if v == 0 {
        return "zero".into();
    }
    let mut groups: Vec<(u64, usize)> = Vec::new();
    let mut scale = 0;
    while v > 0 {
        let g = v % 1000;
        if g != 0 {
            groups.push((g, scale));
        }
        v /= 1000;
        scale += 1;
    }
    groups.reverse();
    let parts: Vec<String> = groups
        .iter()
        .map(|(g, s)| {
            if *s == 0 {
                under_1000(*g)
            } else {
                format!("{} {}", under_1000(*g), SCALES[*s])
            }
        })
        .collect();
    if parts.len() == 1 {
        return parts.into_iter().next().unwrap();
    }
    // The final group joins with "and" when it is a bare tens-or-units — "one thousand and
    // one" — and with a comma once it reaches a hundred.
    let last = groups.last().unwrap();
    let sep = if last.1 == 0 && last.0 < 100 {
        " and "
    } else {
        ", "
    };
    let head = parts[..parts.len() - 1].join(", ");
    format!("{head}{sep}{}", parts[parts.len() - 1])
}

pub fn ordinal(n: i64) -> String {
    let cardinal = cardinal(n);
    let (head, tail) = match cardinal.rfind(' ') {
        Some(i) => (&cardinal[..=i], &cardinal[i + 1..]),
        None => ("", cardinal.as_str()),
    };
    let (hyphen_head, word) = match tail.rfind('-') {
        Some(i) => (&tail[..=i], &tail[i + 1..]),
        None => ("", tail),
    };
    let last = match word {
        "one" => "first".into(),
        "two" => "second".into(),
        "three" => "third".into(),
        "five" => "fifth".into(),
        "eight" => "eighth".into(),
        "nine" => "ninth".into(),
        "twelve" => "twelfth".into(),
        w if w.ends_with('y') => format!("{}ieth", &w[..w.len() - 1]),
        w => format!("{w}th"),
    };
    format!("{head}{hyphen_head}{last}")
}

pub fn year(n: i64) -> String {
    let v = n.unsigned_abs();
    let (high, low) = (v / 100, v % 100);
    // 00XX, X00X and anything past 9999 are read as plain numbers.
    if high == 0 || (high % 10 == 0 && low < 10) || high >= 100 {
        return cardinal(n);
    }
    let tail = if low == 0 {
        "hundred".to_string()
    } else if low < 10 {
        format!("oh-{}", cardinal(low as i64))
    } else {
        cardinal(low as i64)
    };
    format!("{} {tail}", cardinal(high as i64))
}

/// A decimal string, not a float: "12.30" is "twelve point three", and the fraction is
/// read digit by digit.
pub fn decimal(s: &str) -> Option<String> {
    let (int_part, frac) = match s.split_once('.') {
        Some((a, b)) => (a, b.trim_end_matches('0')),
        None => (s, ""),
    };
    let negative = int_part.starts_with('-');
    let digits = int_part.trim_start_matches('-');
    let whole: i64 = if digits.is_empty() {
        0
    } else {
        digits.parse().ok()?
    };
    let head = cardinal(if negative { -whole } else { whole });
    if frac.is_empty() {
        return Some(head);
    }
    let mut out = format!("{head} point");
    for c in frac.chars() {
        out.push(' ');
        out.push_str(ONES[c.to_digit(10)? as usize]);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every value in the committed sweep, against num2words itself.
    #[test]
    fn matches_fixture() {
        // The committed fixture samples the sweep so this runs without the venv;
        // `scripts/check-phonemes.sh` points this at the exhaustive one.
        let path = std::env::var("DREAM_TTS_NUMBERS").unwrap_or_else(|_| {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../fixtures/kokoro/numbers.json"
            )
            .into()
        });
        let text = std::fs::read_to_string(&path).expect("numbers fixture");
        let records: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
        let mut checked = 0;
        for v in records {
            if let Some(s) = v.get("s").and_then(|s| s.as_str()) {
                assert_eq!(
                    decimal(s).unwrap(),
                    v["dec"].as_str().unwrap(),
                    "decimal {s}"
                );
                checked += 1;
                continue;
            }
            let n = v["n"].as_i64().unwrap();
            assert_eq!(cardinal(n), v["card"].as_str().unwrap(), "cardinal {n}");
            if let Some(o) = v.get("ord").and_then(|o| o.as_str()) {
                assert_eq!(ordinal(n), o, "ordinal {n}");
            }
            if let Some(y) = v.get("year").and_then(|y| y.as_str()) {
                assert_eq!(year(n), y, "year {n}");
            }
            checked += 1;
        }
        assert!(checked > 1000, "fixture looks truncated: {checked}");
    }

    /// The joins that are easy to get subtly wrong.
    #[test]
    fn matches_num2words() {
        assert_eq!(cardinal(1001), "one thousand and one");
        assert_eq!(cardinal(1100), "one thousand, one hundred");
        assert_eq!(cardinal(1234), "one thousand, two hundred and thirty-four");
        assert_eq!(cardinal(1020003), "one million, twenty thousand and three");
        assert_eq!(cardinal(101000), "one hundred and one thousand");
        assert_eq!(ordinal(1234), "one thousand, two hundred and thirty-fourth");
        assert_eq!(ordinal(40), "fortieth");
        assert_eq!(year(905), "nine oh-five");
        assert_eq!(year(2100), "twenty-one hundred");
        assert_eq!(year(2000), "two thousand");
        assert_eq!(decimal("12.30").unwrap(), "twelve point three");
    }
}
