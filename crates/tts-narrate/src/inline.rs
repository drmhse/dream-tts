//! One line of prose, cleaned for speech.
//!
//! Order is the whole design here. Almost every rule depends on punctuation a later rule
//! removes, and the comments record only the cases where getting the order wrong shipped
//! audible damage.

use crate::re::{self, compile};
use crate::tables::{
    alternation_longest_first, map_of, GREEK_GLYPHS, MAGNITUDES, MANGLED_COMPOUNDS, NOT_A_NOUN,
    PROSE_CAPS_AS_WORDS, PROSE_SYMBOLS,
};
use crate::{abbrev, code, math, numbers};
use fancy_regex::Regex;
use once_cell::sync::Lazy;
use std::collections::HashMap;

static CODE_SPAN: Lazy<Regex> = Lazy::new(|| compile(r"`([^`]*)`"));
static FLOOR: Lazy<Regex> = Lazy::new(|| compile(r"(\d[KMBT]?)\+(?=\W|$)"));
static PRICE_RANGE: Lazy<Regex> = Lazy::new(|| compile(r"\$(\d[\d,]*(?:\.\d+)?)\s*[-–—]\s*(?=\d)"));
static DISPLAY_SPAN: Lazy<Regex> = Lazy::new(|| compile(r"\$\$([^$]+)\$\$"));
static INLINE_SPAN: Lazy<Regex> = Lazy::new(|| compile(r"\$([^$\n]+)\$"));
static IMAGE: Lazy<Regex> = Lazy::new(|| compile(r"!\[[^\]]*\]\([^)]*\)"));
static LINK: Lazy<Regex> = Lazy::new(|| compile(r"\[([^\]]*)\]\([^)]*\)"));
static FOOTNOTE_REF: Lazy<Regex> = Lazy::new(|| compile(r"\[\^[^\]]+\]"));
static ANGLE_URL: Lazy<Regex> = Lazy::new(|| compile(r"<(https?://[^>]*)>"));
static BARE_URL: Lazy<Regex> = Lazy::new(|| compile(r"(?<![\w<])https?://\S+"));
static BARE_DOI: Lazy<Regex> = Lazy::new(|| compile(r"(?i)\bdoi:\s*10\.\d{4,9}/\S+"));
static CHECKBOX_START: Lazy<Regex> = Lazy::new(|| compile(r"^\s*\[[ xX]?\]\s*"));
static CHECKBOX_MID: Lazy<Regex> = Lazy::new(|| compile(r"(?<=\s)\[[ xX]?\]\s*"));
static BRACKETS: Lazy<Regex> = Lazy::new(|| compile(r"\[([^\[\]]*)\]"));
static HTML_TAG: Lazy<Regex> = Lazy::new(|| {
    compile(r"(?i)</?(?:br|em|strong|span|div|a|img|sup|sub|code|pre|p|ul|ol|li|hr)\b[^>]*/?>")
});
static PLACEHOLDER: Lazy<Regex> = Lazy::new(|| compile(r"<([A-Za-z][\w -]*)>"));
static BOLD_ITALIC: Lazy<Regex> = Lazy::new(|| compile(r"\*\*\*([^*]+)\*\*\*"));
static BOLD: Lazy<Regex> = Lazy::new(|| compile(r"\*\*([^*]+)\*\*"));
static ITALIC: Lazy<Regex> = Lazy::new(|| compile(r"(?<!\*)\*([^*]+)\*(?!\*)"));
static UNDERSCORE_ITALIC: Lazy<Regex> =
    Lazy::new(|| compile(r"(?<![A-Za-z0-9])_([^\s_][^_]*?)_(?![A-Za-z0-9])"));
static SNAKE_CASE: Lazy<Regex> = Lazy::new(|| compile(r"(?<=[A-Za-z0-9])_(?=[A-Za-z0-9])"));
static ARROW: Lazy<Regex> = Lazy::new(|| compile(r"\s*(?:->|=>|\u{2192}|\u{21d2})\s*"));
static ATTACK: Lazy<Regex> = Lazy::new(|| compile(r"\bATT&CK\b"));
static AMP_ENTITY: Lazy<Regex> = Lazy::new(|| compile(r"&amp;"));
static SPACE_ENTITY: Lazy<Regex> = Lazy::new(|| compile(r"&(?:nbsp|thinsp|ensp|emsp);"));
static ANY_ENTITY: Lazy<Regex> = Lazy::new(|| compile(r"&[a-z]+;|&#\d+;"));
static AMPERSAND: Lazy<Regex> = Lazy::new(|| compile(r"(?<=[\w)])\s*&\s*(?=[\w(])"));
static AUTHZ: Lazy<Regex> = Lazy::new(|| compile(r"(?i)\bAUTHZ\b"));
static AUTHN: Lazy<Regex> = Lazy::new(|| compile(r"(?i)\bAUTHN\b"));
static CAPS_WORD: Lazy<Regex> = Lazy::new(|| compile(r"\b[A-Z]{3,}\b"));
static FLOOR_DIGIT: Lazy<Regex> = Lazy::new(|| compile(r"(\d)\s*\+(?=\W|$)"));
static AT_SIGN: Lazy<Regex> = Lazy::new(|| compile(r"(?<=[A-Za-z0-9])@(?=[A-Za-z0-9])"));
static BLANK: Lazy<Regex> = Lazy::new(|| compile(r"_{2,}"));
static PERCENT: Lazy<Regex> = Lazy::new(|| compile(r"(\d)\s*%"));
static LETTER_NUMBER: Lazy<Regex> = Lazy::new(|| compile(r"\b([A-Za-z])-(\d)"));
static SLASH_PAIRS: Lazy<Regex> =
    Lazy::new(|| compile(r"(?i)\b(I/O|pub/sub|read/write|write/read|and/or)\b"));
static TRAILING_SLASH: Lazy<Regex> = Lazy::new(|| compile(r"\s*/\s*(?=[.,;:!?]|$)"));
static SPACED_SLASH: Lazy<Regex> = Lazy::new(|| compile(r"\s+/\s+"));
static LETTER_SLASH: Lazy<Regex> = Lazy::new(|| compile(r"(?<=[A-Za-z])/(?=[A-Za-z])"));
static WORD_PLUS: Lazy<Regex> = Lazy::new(|| compile(r"(?<=[A-Za-z0-9])\+(?=[A-Za-z0-9])"));
static APPROX_TILDE: Lazy<Regex> =
    Lazy::new(|| compile(r"(?i)\b(around|about|roughly|approximately)\s+~\s*(?=[\d$])"));
static TILDE_QUANTITY: Lazy<Regex> = Lazy::new(|| compile(r"~\s*(?=[\d$])"));
static TILDE_SPACED: Lazy<Regex> = Lazy::new(|| compile(r"(?<=\s)~(?=\s)"));
static DOT_BEFORE_COLON: Lazy<Regex> = Lazy::new(|| compile(r"\.(?=\s*:)"));
static PARENTHETICAL: Lazy<Regex> = Lazy::new(|| compile(r"\(([^()]*)\)"));
static SEMICOLON_CLAUSE: Lazy<Regex> =
    Lazy::new(|| compile(r#"\s*;\s+(["'\u{201c}\u{2018}]?)(\w)"#));
static SEMICOLON_END: Lazy<Regex> = Lazy::new(|| compile(r"\s*;\s*$"));
static GREEK_GLYPH_RE: Lazy<Regex> = Lazy::new(|| {
    compile(&format!(
        "[{}]",
        GREEK_GLYPHS.iter().map(|(g, _)| *g).collect::<String>()
    ))
});
static LOWER_HYPHEN: Lazy<Regex> = Lazy::new(|| compile(r"(?<=[a-z])-(?=[a-z])"));
static EM_DASH: Lazy<Regex> = Lazy::new(|| compile(r"\s*[—–]\s*"));
static MANGLED: Lazy<Regex> = Lazy::new(|| {
    compile(&format!(
        r"(?i)\b(?:{})\b",
        alternation_longest_first(MANGLED_COMPOUNDS.iter().map(|(k, _)| *k))
    ))
});
static SPACES: Lazy<Regex> = Lazy::new(|| compile(r"[ \t]+"));

/// A currency amount and the scale word that may follow. Both are needed together: the unit
/// is spoken after a scale ("4.76 million dollars") but before the fraction ("4 dollars 82
/// cents"), and no rule reading only the digits can tell those apart.
static CURRENCY: Lazy<Regex> = Lazy::new(|| {
    // Assembled with `concat!`, not a multi-line raw string: a raw string keeps the
    // backslash and the newline, which silently produced a pattern that never matched and
    // left every `$` in the text for the inline-maths rule to pair up.
    compile(concat!(
        r"(?i)(\b(?:a|an|the|[A-Za-z]+'s)\s+(?:[a-z]+\s+){0,2})?",
        r"\$(\d+(?:,\d{3})*)(?:\.(\d+))?",
        r"(\s*(?:thousand|million|billion|trillion)\b|[KMBT]\b)?",
        r"(\s+[a-z]+\b)?",
    ))
});

static MANGLED_MAP: Lazy<HashMap<&str, &str>> = Lazy::new(|| map_of(MANGLED_COMPOUNDS));
static MAGNITUDE_MAP: Lazy<HashMap<&str, &str>> = Lazy::new(|| map_of(MAGNITUDES));

/// A bare link, as a narrator reads one: the site, not the address. Forty characters of
/// protocol and path are untranscribable, and its dots are sentence boundaries besides.
fn speak_url(url: &str) -> String {
    static SCHEME: Lazy<Regex> = Lazy::new(|| compile(r"^https?://(?:www\.)?"));
    let host = re::sub_str(&SCHEME, url, "");
    host.split('/')
        .next()
        .unwrap_or("")
        .trim_end_matches(['.', ',', ';', ':'])
        .to_string()
}

fn speak_currency(
    det: Option<&str>,
    whole: &str,
    frac: Option<&str>,
    scale: Option<&str>,
    tail: Option<&str>,
) -> String {
    let tail = tail.unwrap_or("");
    let scale_raw = scale.unwrap_or("").trim().to_string();
    let scale = if scale_raw.is_empty() {
        String::new()
    } else {
        let upper = scale_raw.to_uppercase();
        format!(
            " {}",
            MAGNITUDE_MAP
                .get(upper.as_str())
                .map_or_else(|| scale_raw.to_lowercase(), |w| (*w).to_string())
        )
    };
    // Attributive use: "a $12 platform fee" is "a 12 dollar platform fee", singular. An
    // article ahead and a noun behind are the two cheap signals; either alone is wrong often
    // enough to matter.
    if let Some(det) = det {
        if scale.is_empty()
            && frac.is_none()
            && !tail.trim().is_empty()
            && !NOT_A_NOUN.contains(&tail.trim())
        {
            return format!("{det}{whole} dollar{tail}");
        }
    }
    let det = det.unwrap_or("");
    if !scale.is_empty() {
        // Cents make no sense at this magnitude; the decimal is part of the quantity.
        let amount = frac.map_or_else(|| whole.to_string(), |f| format!("{whole}.{f}"));
        return format!("{det}{amount}{scale} dollars{tail}");
    }
    match frac {
        Some(f) if f.len() == 2 => format!("{det}{whole} dollars {f} cents{tail}"),
        Some(f) => format!("{det}{whole} point {f} dollars{tail}"),
        None => format!("{det}{whole} dollars{tail}"),
    }
}

/// A glyph is padded only where it would otherwise fuse into the word beside it.
fn speak_greek(glyph: char, before: Option<char>, after: Option<char>) -> String {
    let word = GREEK_GLYPHS
        .iter()
        .find(|(g, _)| *g == glyph)
        .map_or("", |(_, w)| *w);
    let pad = |c: Option<char>| match c {
        None => "",
        Some(c) if c.is_whitespace() || !c.is_alphanumeric() => "",
        Some(_) => " ",
    };
    format!("{}{}{}", pad(before), word, pad(after))
}

fn split_mangled(line: &str) -> String {
    re::sub(&MANGLED, line, |c| {
        let word = re::g(c, 0);
        let fixed = MANGLED_MAP[word.to_lowercase().as_str()];
        if word.chars().next().is_some_and(char::is_uppercase) {
            let mut chars = fixed.chars();
            match chars.next() {
                Some(f) => f.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        } else {
            fixed.to_string()
        }
    })
}

pub fn clean_inline(line: &str) -> String {
    let mut line = re::sub(&CODE_SPAN, line, |c| code::speak_code(&re::g(c, 1)));
    // Currency before maths, and both before anything else: a paragraph quoting two prices
    // looks exactly like one `$...$` span, so the prose between them would be swallowed.
    // "Level 2+" and "$100M+" are the same construction — a floor — and the plus has to be
    // read before the currency rule consumes the number in front of it.
    line = re::sub_str(&FLOOR, &line, "${1} or higher");
    // A price range carries the unit at both ends, or the currency rule reads only the first
    // half and leaves "10 dollars-150".
    // `$$` is an escaped dollar in the replacement syntax, so a dollar followed by group 1
    // is `$$` then `${1}`.
    line = re::sub_str(&PRICE_RANGE, &line, "$$${1} to $$");
    line = re::sub(&CURRENCY, &line, |c| {
        speak_currency(
            re::opt(c, 1),
            &re::g(c, 2),
            re::opt(c, 3),
            re::opt(c, 4),
            re::opt(c, 5),
        )
    });
    line = re::sub(&DISPLAY_SPAN, &line, |c| math::speak_math(&re::g(c, 1)));
    line = re::sub(&INLINE_SPAN, &line, |c| math::speak_math(&re::g(c, 1)));
    line = re::sub_str(&IMAGE, &line, "");
    line = re::sub_str(&LINK, &line, "${1}");
    // A footnote marker is a superscript link on the page; spoken it is a caret and a digit
    // mid-clause. The note is narrated where it is defined.
    line = re::sub_str(&FOOTNOTE_REF, &line, "");
    line = re::sub_str(&numbers::NUMERIC_INTERVAL, &line, "${1} to ${2}");
    line = re::sub(&ANGLE_URL, &line, |c| speak_url(&re::g(c, 1)));
    line = re::sub(&BARE_URL, &line, |c| speak_url(&re::g(c, 0)));
    // A bare DOI, which the rule above never sees because it has no scheme. Twenty-six
    // characters of digits, dots and slashes is the input most likely to send the AR loop
    // into babble, and on a real paper's abstract it did. A narrator says "DOI" and moves on.
    line = re::sub_str(&BARE_DOI, &line, "DOI");
    line = re::sub_str(&CHECKBOX_START, &line, "");
    line = re::sub_str(&CHECKBOX_MID, &line, "");
    // Bare square brackets are template notation — "move from [painful current state]". A
    // narrator reads the words and drops the brackets, which is what the page's own
    // tokeniser does too. Links were unwrapped above, so nothing here can be one.
    line = re::sub_str(&BRACKETS, &line, "${1}");
    line = re::sub_str(&HTML_TAG, &line, "");
    line = re::sub_str(&PLACEHOLDER, &line, "${1}");
    // Bold-italic first: `***x***` defeats both rules below, and the markers survived into
    // the narration — one chapter said "asterisk, asterisk, asterisk" and then degenerated.
    line = re::sub_str(&BOLD_ITALIC, &line, "${1}");
    line = re::sub_str(&BOLD, &line, "${1}");
    line = re::sub_str(&ITALIC, &line, "${1}");
    // Only when the underscores delimit a word. Unanchored, the rule paired the underscore in
    // `agency_account` with the one in `source_export` and produced "agencyaccount".
    line = re::sub_str(&UNDERSCORE_ITALIC, &line, "${1}");
    line = re::sub_str(&SNAKE_CASE, &line, " ");
    // Arrows carry the meaning of a chain and were being dropped, leaving a list of nouns.
    line = re::sub_str(&ARROW, &line, ", then ");
    // `ATT&CK` is pronounced "attack" and the ampersand is silent in it.
    line = re::sub_str(&ATTACK, &line, "attack");
    line = re::sub_str(&AMP_ENTITY, &line, " and ");
    line = re::sub_str(&SPACE_ENTITY, &line, " ");
    line = re::sub_str(&ANY_ENTITY, &line, " ");
    line = re::sub_str(&AMPERSAND, &line, " and ");
    line = re::sub_str(&AUTHZ, &line, "auth Z");
    line = re::sub_str(&AUTHN, &line, "auth N");
    // Before the all-caps rules: `Ph.D.` and `U.S.` are units here rather than capital runs.
    line = abbrev::expand(&line);
    line = abbrev::collapse_initials(&line);
    line = re::sub(&CAPS_WORD, &line, |c| {
        let w = re::g(c, 0);
        if PROSE_CAPS_AS_WORDS.contains(&w.as_str()) {
            w.to_lowercase()
        } else {
            w
        }
    });
    line = re::sub_str(&FLOOR_DIGIT, &line, "${1} or higher");
    line = re::sub_str(&AT_SIGN, &line, " at ");
    // A run of identical characters is the input most likely to send the model into a loop.
    // "Blank" is what a narrator reading a worksheet aloud says.
    line = re::sub_str(&BLANK, &line, "blank");
    line = re::sub_str(&PERCENT, &line, "${1} percent");
    line = numbers::speak_numbers(&line);
    // `I-2048` — an illustrative invoice number. The space keeps the letter from being
    // swallowed into the number.
    line = re::sub_str(&LETTER_NUMBER, &line, "${1} ${2}");
    // A spaced slash comes from a table cell listing alternatives, where a comma is the
    // reading; an unspaced one joins alternatives in prose, where "or" is.
    line = re::sub(&SLASH_PAIRS, &line, |c| re::g(c, 1).replace('/', " "));
    line = re::sub_str(&TRAILING_SLASH, &line, "");
    line = re::sub_str(&SPACED_SLASH, &line, ", ");
    line = re::sub_str(&LETTER_SLASH, &line, " or ");
    line = re::sub_str(&WORD_PLUS, &line, " plus ");
    line = line.replace("~~", "");
    // A tilde before a quantity is "about"; left in it is silent, so an approximation reads
    // as an exact figure. An approximator already says it.
    line = re::sub_str(&APPROX_TILDE, &line, "${1} ");
    line = re::sub_str(&TILDE_QUANTITY, &line, "about ");
    line = re::sub_str(&TILDE_SPACED, &line, "about");
    line = re::sub_str(&DOT_BEFORE_COLON, &line, "");
    // Any asterisk still here is unmatched or nested unexpectedly, and there is no reading of
    // one that belongs in speech. Removing it unconditionally is safer than another special
    // case, because the failure mode is a degenerate loop rather than a blemish.
    line = line.replace('*', "");
    // A semicolon inside parentheses is a citation separator — "(Smith, 2021; Zhou, 2019)" is
    // one parenthetical — and turning that into a full stop closed a sentence inside the
    // brackets and opened one that never closed.
    line = re::sub(&PARENTHETICAL, &line, |c| {
        format!("({})", re::g(c, 1).replace(';', ","))
    });
    // Between clauses it splits: a listener has no way to hear the join, and it is also what
    // keeps segments under the engine's 220-character budget.
    line = re::sub(&SEMICOLON_CLAUSE, &line, |c| {
        format!(". {}{}", re::g(c, 1), re::g(c, 2).to_uppercase())
    });
    line = re::sub_str(&SEMICOLON_END, &line, ".");
    // Glyphs typed rather than written as commands. Before the hyphen rules, so "α-β" becomes
    // "alpha-beta" and is then spaced like any other compound.
    for (glyph, word) in PROSE_SYMBOLS {
        if line.contains(glyph) {
            line = line.replace(glyph, word);
        }
    }
    let src = line.clone();
    line = re::replace(&GREEK_GLYPH_RE, &src, |c, start, end| {
        let glyph = re::g(c, 0).chars().next().unwrap_or(' ');
        speak_greek(
            glyph,
            src[..start].chars().next_back(),
            src[end..].chars().next(),
        )
    });
    // A hyphen between two lower-case words is a compound modifier: `paid-but-unfulfilled`
    // was rendered "paid button fulfilled".
    line = re::sub_str(&LOWER_HYPHEN, &line, " ");
    line = re::sub_str(&EM_DASH, &line, ", ");
    line = line.replace('…', "...");
    for (a, b) in [
        ("\u{201c}", "\""),
        ("\u{201d}", "\""),
        ("\u{2018}", "'"),
        ("\u{2019}", "'"),
    ] {
        line = line.replace(a, b);
    }
    line = split_mangled(&line);
    re::sub_str(&SPACES, &line, " ").trim().to_string()
}
