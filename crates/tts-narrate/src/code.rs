//! One inline code span, as a narrator would read it aloud.
//!
//! Dropping the span is not an option — it is a sentence constituent, and removing it leaves
//! a hole the listener hears as a mistake. Reading it literally is worse: the raw characters
//! give "seats set remaining equals sign remaining minus sign one".

use crate::re::{self, compile, squeeze, strip_spaces_and_commas};
use crate::tables::SQL_KEYWORDS;
use fancy_regex::Regex;
use once_cell::sync::Lazy;

static WORDS: Lazy<Regex> = Lazy::new(|| compile(r"[A-Za-z][A-Za-z0-9]*"));
static ALL_CAPS_WORD: Lazy<Regex> = Lazy::new(|| compile(r"\b[A-Z]+\b"));
static CAPS_2: Lazy<Regex> = Lazy::new(|| compile(r"[A-Z]{2,}"));
static CAPS_4: Lazy<Regex> = Lazy::new(|| compile(r"[A-Z]{4,}"));
static OPEN_PAREN: Lazy<Regex> = Lazy::new(|| compile(r"\s*\(\s*"));
static CLOSE_PAREN: Lazy<Regex> = Lazy::new(|| compile(r"\s*\)\s*"));
static BRACES: Lazy<Regex> = Lazy::new(|| compile(r"[{}]"));
static COLON: Lazy<Regex> = Lazy::new(|| compile(r"\s*:\s*"));
static SPACED_HYPHEN: Lazy<Regex> = Lazy::new(|| compile(r"(?<=\s)-(?=\s)"));
static PARAMETER: Lazy<Regex> = Lazy::new(|| compile(r"\s*\?"));
static SLASH: Lazy<Regex> = Lazy::new(|| compile(r"\s*/\s*"));
static DESC: Lazy<Regex> = Lazy::new(|| compile(r"(?i)\bDESC\b"));
static ASC: Lazy<Regex> = Lazy::new(|| compile(r"(?i)\bASC\b"));

/// Two-character operators first, or `>=` becomes "greater than equals".
static OPERATORS: &[(&str, &str)] = &[
    (r">=", " at least "),
    (r"<=", " at most "),
    (r"!=", " not equal to "),
    (r"=", " equals "),
    (r">", " greater than "),
    (r"<", " less than "),
    (r"\+", " plus "),
];

static OPERATOR_RES: Lazy<Vec<(Regex, &'static str)>> = Lazy::new(|| {
    OPERATORS
        .iter()
        .map(|(pat, word)| (compile(&format!(r"\s*{pat}\s*")), *word))
        .collect()
});

/// `OrderCancellationAccepted` -> "Order cancellation accepted".
///
/// Lowercasing the tail is what keeps the voice from mangling it: "Order Cancellation
/// Accepted" came back as "order, scancelation accepted".
fn split_camel(token: &str) -> String {
    static BOUNDARY: Lazy<Regex> = Lazy::new(|| compile(r"(?<=[a-z0-9])(?=[A-Z])"));
    let mut parts: Vec<&str> = Vec::new();
    let mut last = 0usize;
    let mut pos = 0usize;
    while let Ok(Some(m)) = BOUNDARY.find_from_pos(token, pos) {
        parts.push(&token[last..m.start()]);
        last = m.start();
        pos = m.start() + 1;
    }
    parts.push(&token[last..]);
    if parts.len() == 1 {
        return token.to_string();
    }
    let mut out = parts[0].to_string();
    for p in &parts[1..] {
        out.push(' ');
        out.push_str(&p.to_lowercase());
    }
    out
}

pub fn speak_code(inner: &str) -> String {
    let s = inner.trim();
    if s.is_empty() {
        return String::new();
    }
    // CamelCase, but not an all-caps run: `PaymentCaptured` splits, `SLO` and `ID` do not.
    let mut s = re::sub(&WORDS, s, |c| split_camel(&re::g(c, 0)));
    s = re::sub(&ALL_CAPS_WORD, &s, |c| {
        let w = re::g(c, 0);
        if SQL_KEYWORDS.contains(&w.as_str()) {
            w.to_lowercase()
        } else {
            w
        }
    });
    // SCREAMING_SNAKE identifiers: `OBSERVED_AT_IP, USED_DEVICE` came back as "OBS or VAT.
    // ATIP, USE device". An underscore is what distinguishes one from an acronym like `API`,
    // which must keep its capitals to be spelled on purpose. Two triggers, because one label
    // in the same list has no underscore at all (`REFERRED`).
    if inner.contains('_') {
        s = re::sub(&CAPS_2, &s, |c| re::g(c, 0).to_lowercase());
    }
    s = re::sub(&CAPS_4, &s, |c| re::g(c, 0).to_lowercase());
    // Call and tuple syntax: `PaymentCaptured(provider_transaction_id)` is an event and the
    // field it carries, and a comma is the pause that says so.
    s = re::sub_str(&OPEN_PAREN, &s, ", ");
    s = re::sub_str(&CLOSE_PAREN, &s, " ");
    // `product:{id}` is one key template, not two things.
    s = re::sub_str(&BRACES, &s, "");
    s = re::sub_str(&COLON, &s, " ");
    for (regex, word) in OPERATOR_RES.iter() {
        s = re::sub_str(regex, &s, word);
    }
    // Only a spaced hyphen is arithmetic; an unspaced one belongs to `B-tree` or `us-east-2a`.
    s = re::sub_str(&SPACED_HYPHEN, &s, "minus");
    // A bound parameter. Naming it is what a narrator does; "question mark" is not.
    s = re::sub_str(&PARAMETER, &s, " a parameter");
    s = re::sub_str(&SLASH, &s, " ");
    s = s.replace(',', ", ");
    s = re::sub_str(&DESC, &s, "descending");
    s = re::sub_str(&ASC, &s, "ascending");
    strip_spaces_and_commas(&squeeze(&s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camel_case_splits_and_lowercases_the_tail() {
        assert_eq!(
            split_camel("OrderCancellationAccepted"),
            "Order cancellation accepted"
        );
        assert_eq!(split_camel("SLO"), "SLO");
        assert_eq!(split_camel("id"), "id");
    }

    #[test]
    fn an_event_and_its_field() {
        assert_eq!(
            speak_code("PaymentCaptured(provider_transaction_id)"),
            "Payment captured, provider_transaction_id"
        );
    }

    #[test]
    fn operators_carry_the_clause() {
        assert_eq!(speak_code("remaining > 0"), "remaining greater than 0");
        assert_eq!(speak_code("remaining >= 0"), "remaining at least 0");
        assert_eq!(speak_code("WHERE id = ?"), "where id equals a parameter");
    }
}
