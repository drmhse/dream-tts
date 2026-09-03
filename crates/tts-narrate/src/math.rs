//! Inline maths, as a narrator reads it.
//!
//! A paper's inline maths is a sentence constituent — "the loss is $\mathcal{L}$" — so it
//! can no more be dropped than a code span can, and read literally it is worse than
//! dropped: backslashes and braces are both meaningless and the kind of nonsense token that
//! sends the model into a repetition loop.
//!
//! Structure before symbols, largest structures first: a fraction has to become "over" while
//! its braces still say which operand is which, and `\sum_{i=1}^{N}` has to be read as a sum
//! before the generic superscript rule turns its bound into "to the power N".

use crate::re::{self, compile, squeeze, strip_spaces_and_commas};
use crate::tables::{GREEK, GREEK_GLYPHS, MATH_SYMBOLS};
use fancy_regex::Regex;
use once_cell::sync::Lazy;

static INTERVAL: Lazy<Regex> = Lazy::new(|| compile(r"\[\s*([^\[\],]+?)\s*,\s*([^\[\],]+?)\s*\]"));
static FRAC: Lazy<Regex> = Lazy::new(|| compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}"));
static SQRT: Lazy<Regex> = Lazy::new(|| compile(r"\\sqrt\s*\{([^{}]*)\}"));
static FONT: Lazy<Regex> = Lazy::new(|| {
    compile(
        r"\\(?:textrm|text|mathrm|mathbf|mathbb|mathcal|boldsymbol|operatorname)\s*\{([^{}]*)\}",
    )
});
static ARG_MINMAX_SUB: Lazy<Regex> =
    Lazy::new(|| compile(r"\\arg\s*\\(max|min)_\s*\{?\\?(\w+)\}?"));
static ARG_MINMAX: Lazy<Regex> = Lazy::new(|| compile(r"\\arg\s*\\(max|min)"));
static POW_NEG: Lazy<Regex> = Lazy::new(|| compile(r"\^\s*\{?-\s*(\d+)\}?"));
static POW_2: Lazy<Regex> = Lazy::new(|| compile(r"\^\s*\{?2\}?(?!\d)"));
static POW_3: Lazy<Regex> = Lazy::new(|| compile(r"\^\s*\{?3\}?(?!\d)"));
static POW_BRACED: Lazy<Regex> = Lazy::new(|| compile(r"\^\s*\{([^{}]*)\}"));
static POW_BARE: Lazy<Regex> = Lazy::new(|| compile(r"\^\s*([^{}\s]+)"));
static SUB_BRACED: Lazy<Regex> = Lazy::new(|| compile(r"_\s*\{([^{}]*)\}"));
static SUB_CMD: Lazy<Regex> = Lazy::new(|| compile(r"_\s*\\(\w+)"));
static SUB_CHAR: Lazy<Regex> = Lazy::new(|| compile(r"_\s*([A-Za-z0-9])"));
static ANY_COMMAND: Lazy<Regex> = Lazy::new(|| compile(r"\\[a-zA-Z]+"));
static ESCAPED: Lazy<Regex> = Lazy::new(|| compile(r"\\."));
static LETTER_HYPHEN: Lazy<Regex> = Lazy::new(|| compile(r"(?<=[A-Za-z])-(?=[A-Za-z])"));

/// Accents are spoken after the symbol, which is how they are named aloud: "theta hat".
static ACCENTS: &[(&str, &str)] = &[
    ("hat", "hat"),
    ("bar", "bar"),
    ("tilde", "tilde"),
    ("vec", "vector"),
];

/// Big operators, whose bounds are read before the generic superscript rule sees them.
static BIG_OPS: &[(&str, &str)] = &[
    ("sum", "the sum"),
    ("prod", "the product"),
    ("int", "the integral"),
    ("bigcup", "the union"),
    ("bigcap", "the intersection"),
];

static COMMANDS: &[(&str, &str)] = &[
    ("times", " times "),
    ("cdot", " times "),
    ("div", " divided by "),
    ("leq", " at most "),
    ("le", " at most "),
    ("geq", " at least "),
    ("ge", " at least "),
    ("neq", " not equal to "),
    ("ne", " not equal to "),
    ("approx", " about "),
    ("pm", " plus or minus "),
    ("mid", " given "),
    ("in", " in "),
    ("notin", " not in "),
    ("infty", " infinity "),
    ("to", " to "),
    ("rightarrow", " to "),
    ("propto", " proportional to "),
    ("equiv", " identical to "),
    ("sim", " about "),
    ("ll", " much less than "),
    ("gg", " much greater than "),
    ("forall", " for all "),
    ("exists", " there exists "),
    ("log", " log "),
    ("exp", " exp "),
    ("ln", " natural log "),
    ("max", " max "),
    ("min", " min "),
    ("partial", " partial "),
    ("nabla", " gradient "),
    ("%", " percent "),
];

struct Compiled {
    accents_braced: Vec<(Regex, String)>,
    accents_bare: Vec<(Regex, String)>,
    ops_bounded: Vec<(Regex, String)>,
    ops_braced: Vec<(Regex, String)>,
    ops_word: Vec<(Regex, String)>,
    ops_bare: Vec<(Regex, String)>,
    commands: Vec<(Regex, &'static str)>,
    greek: Regex,
}

static RULES: Lazy<Compiled> = Lazy::new(|| Compiled {
    accents_braced: ACCENTS
        .iter()
        .map(|(cmd, word)| {
            (
                compile(&format!(r"\\{cmd}\s*\{{([^{{}}]*)\}}")),
                format!("${{1}} {word}"),
            )
        })
        .collect(),
    accents_bare: ACCENTS
        .iter()
        .map(|(cmd, word)| {
            (
                compile(&format!(r"\\{cmd}\s+(\w)")),
                format!("${{1}} {word}"),
            )
        })
        .collect(),
    ops_bounded: BIG_OPS
        .iter()
        .map(|(cmd, word)| {
            (
                compile(&format!(
                    r"\\{cmd}_\s*\{{([^{{}}]*)\}}\s*\^\s*\{{([^{{}}]*)\}}"
                )),
                format!(" {word} from ${{1}} to ${{2}} of "),
            )
        })
        .collect(),
    ops_braced: BIG_OPS
        .iter()
        .map(|(cmd, word)| {
            (
                compile(&format!(r"\\{cmd}_\s*\{{([^{{}}]*)\}}")),
                format!(" {word} over ${{1}} of "),
            )
        })
        .collect(),
    ops_word: BIG_OPS
        .iter()
        .map(|(cmd, word)| {
            (
                compile(&format!(r"\\{cmd}_\s*(\w+)")),
                format!(" {word} over ${{1}} of "),
            )
        })
        .collect(),
    ops_bare: BIG_OPS
        .iter()
        .map(|(cmd, word)| (compile(&format!(r"\\{cmd}\b")), format!(" {word} of ")))
        .collect(),
    commands: COMMANDS
        .iter()
        .map(|(cmd, word)| {
            // `%` is not a word character, so `(?![a-zA-Z])` after it is the same guard.
            let escaped = fancy_regex::escape(cmd);
            (compile(&format!(r"\\{escaped}(?![a-zA-Z])")), *word)
        })
        .collect(),
    greek: compile(&format!(r"\\({})(?![a-zA-Z])", GREEK.join("|"))),
});

pub fn speak_math(inner: &str) -> String {
    // `[0, 1]` is an interval, and its comma reads as a list unless the bounds are named.
    let mut s = re::sub_str(&INTERVAL, inner, " the range ${1} to ${2} ");
    // Nested fractions, innermost first.
    for _ in 0..3 {
        s = re::sub_str(&FRAC, &s, "${1} over ${2}");
    }
    s = re::sub_str(&SQRT, &s, " the square root of ${1} ");
    s = re::sub_str(&FONT, &s, "${1}");
    for (regex, replacement) in &RULES.accents_braced {
        s = re::sub_str(regex, &s, replacement);
    }
    for (regex, replacement) in &RULES.accents_bare {
        s = re::sub_str(regex, &s, replacement);
    }
    for group in [
        &RULES.ops_bounded,
        &RULES.ops_braced,
        &RULES.ops_word,
        &RULES.ops_bare,
    ] {
        for (regex, replacement) in group {
            s = re::sub_str(regex, &s, replacement);
        }
    }
    s = re::sub_str(&ARG_MINMAX_SUB, &s, " arg ${1} over ${2} ");
    s = re::sub_str(&ARG_MINMAX, &s, " arg ${1} ");
    s = re::sub_str(&POW_NEG, &s, " to the minus ${1} ");
    s = re::sub_str(&POW_2, &s, " squared ");
    s = re::sub_str(&POW_3, &s, " cubed ");
    s = re::sub_str(&POW_BRACED, &s, " to the power ${1} ");
    s = re::sub_str(&POW_BARE, &s, " to the power ${1} ");
    // `x_i` is a named quantity, so the subscript is its own letter. The command form is
    // stripped first so `_\theta` keeps its name.
    s = re::sub_str(&SUB_BRACED, &s, " ${1}");
    s = re::sub_str(&SUB_CMD, &s, " ${1}");
    s = re::sub_str(&SUB_CHAR, &s, " ${1}");
    for (regex, word) in &RULES.commands {
        s = re::sub_str(regex, &s, word);
    }
    s = re::sub(&RULES.greek, &s, |c| format!(" {} ", re::g(c, 1)));
    s = re::sub_str(&ANY_COMMAND, &s, " "); // \left, \right, \quad and anything unmodelled
    s = re::sub_str(&ESCAPED, &s, " ");
    s = s.replace(['{', '}'], " ");
    // A hyphen between letters joins a compound name — `Volt-Amps`. Only a hyphen beside a
    // number or a space is arithmetic.
    s = re::sub_str(&LETTER_HYPHEN, &s, " ");
    for (symbol, word) in MATH_SYMBOLS {
        s = s.replace(symbol, word);
    }
    s = s
        .chars()
        .map(|ch| {
            GREEK_GLYPHS
                .iter()
                .find(|(g, _)| *g == ch)
                .map_or_else(|| ch.to_string(), |(_, w)| (*w).to_string())
        })
        .collect();
    strip_spaces_and_commas(&squeeze(&s))
}

/// Display equations and LaTeX environments, removed.
///
/// A displayed equation is a block, and this pipeline's rule for block notation is the site's:
/// blocks are shown, not spoken. Verbalising one gives a minute of "the sum from i equals 1 to
/// N of" with no way to see the expression, and the surrounding prose almost always says the
/// same thing in words.
pub fn drop_display_math(text: &str) -> String {
    static ENV: Lazy<Regex> = Lazy::new(|| {
        compile(r"(?s)\\begin\{(equation|align|gather|multline|eqnarray)\*?\}.*?\\end\{\1\*?\}")
    });
    static DOLLARS: Lazy<Regex> = Lazy::new(|| compile(r"(?sm)^[ \t]*\$\$.*?\$\$[ \t]*$"));
    static BRACKETS: Lazy<Regex> = Lazy::new(|| compile(r"(?sm)^[ \t]*\\\[.*?\\\][ \t]*$"));
    let s = re::sub_str(&ENV, text, "");
    let s = re::sub_str(&DOLLARS, &s, "");
    re::sub_str(&BRACKETS, &s, "")
}
