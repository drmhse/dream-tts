//! The curated word lists. Each one is a judgement about how a voice reads a token, arrived
//! at by listening; `scripts/md-to-narration.py` carries the long-form reasoning for every
//! entry and this is the machine-readable half.

use std::collections::HashMap;

/// Upper case puts this voice into spelling mode: `WHERE id = ?` was spoken "WHAE IED".
/// Lower-cased inside a code span, where the token is a keyword rather than an initialism.
pub const SQL_KEYWORDS: &[&str] = &[
    "SELECT", "INSERT", "UPDATE", "DELETE", "FROM", "WHERE", "SET", "ORDER", "BY", "GROUP",
    "HAVING", "LIMIT", "OFFSET", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "ON", "AND", "OR",
    "NOT", "NULL", "VALUES", "INTO", "AS", "DISTINCT", "COUNT", "SUM", "EXISTS", "BETWEEN", "LIKE",
    "IN", "IS", "CREATE", "TABLE", "INDEX", "UNIQUE", "PRIMARY", "KEY",
];

/// All-caps tokens in *prose* that are English words. A shouted RFC 2119 "MUST" took its
/// neighbours with it and started a repetition loop. Acronyms that do not collide with a
/// word — SBOM, PKCE, OIDC, SAML — are deliberately absent: spelling those is correct.
pub const PROSE_CAPS_AS_WORDS: &[&str] = &[
    "MUST",
    "SHOULD",
    "SHALL",
    "MAY",
    "REQUIRED",
    "RECOMMENDED",
    "OPTIONAL",
    "NOT",
    "HEAD",
    "STORE",
    "SECRETS",
    "STRIDE",
    "REST",
    "MIME",
    "PASTA",
    "SANS",
    "SOAP",
    "ACID",
    "DOM",
    "GET",
    "LOG",
    "MAC",
    "NET",
    "RAG",
    "RUN",
    "SEC",
    "WAF",
    "WEB",
];

/// Following an amount, these mean the amount is the noun rather than modifying one:
/// "credit $12 to revenue" keeps the plural, "a $12 platform fee" does not.
pub const NOT_A_NOUN: &[&str] = &[
    "to", "from", "and", "or", "for", "in", "on", "of", "at", "than", "while", "unless", "per",
    "by", "is", "was", "were", "after", "before", "with", "under", "over", "but",
];

pub const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

/// A suffix a paper writes for a quantity: `7B parameters` is seven billion of them.
pub const MAGNITUDES: &[(&str, &str)] = &[
    ("K", "thousand"),
    ("M", "million"),
    ("B", "billion"),
    ("T", "trillion"),
];

/// Rate denominators, so `3.2 GB/s` is a rate rather than "3.2 GB or s".
pub const RATE_UNITS: &[(&str, &str)] = &[
    ("s", "second"),
    ("sec", "second"),
    ("min", "minute"),
    ("h", "hour"),
    ("hr", "hour"),
    ("day", "day"),
    ("wk", "week"),
    ("mo", "month"),
    ("yr", "year"),
    ("year", "year"),
    ("kg", "kilogram"),
    ("km", "kilometre"),
    ("m", "metre"),
    ("L", "litre"),
    ("W", "watt"),
    ("core", "core"),
    ("node", "node"),
    ("user", "user"),
    ("seat", "seat"),
    ("request", "request"),
    ("req", "request"),
    ("query", "query"),
    ("op", "operation"),
    ("token", "token"),
    ("call", "call"),
    ("capita", "head"),
    ("GB", "gigabyte"),
    ("MB", "megabyte"),
    ("TB", "terabyte"),
];

/// Expanded because the voice spells an unknown short token: "16 GB" read as "one six gee
/// bee". Only the units a technical corpus actually writes; an unlisted one is left alone.
pub const UNITS: &[(&str, &str)] = &[
    ("GB", "gigabyte"),
    ("MB", "megabyte"),
    ("KB", "kilobyte"),
    ("kB", "kilobyte"),
    ("TB", "terabyte"),
    ("PB", "petabyte"),
    ("Gbps", "gigabit per second"),
    ("Mbps", "megabit per second"),
    ("kbps", "kilobit per second"),
    ("GHz", "gigahertz"),
    ("MHz", "megahertz"),
    ("kHz", "kilohertz"),
    ("Hz", "hertz"),
    ("ms", "millisecond"),
    ("\u{b5}s", "microsecond"),
    ("ns", "nanosecond"),
    ("GW", "gigawatt"),
    ("MW", "megawatt"),
    ("kW", "kilowatt"),
    ("W", "watt"),
    ("TWh", "terawatt hour"),
    ("GWh", "gigawatt hour"),
    ("MWh", "megawatt hour"),
    ("kWh", "kilowatt hour"),
    ("km", "kilometre"),
    ("cm", "centimetre"),
    ("mm", "millimetre"),
    ("kg", "kilogram"),
    ("mg", "milligram"),
    ("t", "tonne"),
    ("mL", "millilitre"),
    ("L", "litre"),
    ("vCPU", "virtual CPU"),
];

/// Hertz is its own plural; everything else here takes an "s".
pub const INVARIANT_UNITS: &[&str] = &["Hz", "kHz", "MHz", "GHz"];

/// Spoken without a number in front: "50 dollars per MWh" was spelled "em double-you aitch".
/// A bare `t` is as likely to be a variable as a mass, so the licence stops where a
/// collision starts.
pub const BARE_UNITS: &[&str] = &[
    "GW", "MW", "kW", "TWh", "GWh", "MWh", "kWh", "GB", "TB", "PB",
];

/// LaTeX command names for Greek letters.
pub const GREEK: &[&str] = &[
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau", "phi", "chi", "psi", "omega",
];

/// The glyphs themselves, for prose that types the letter. Capitals are named where the case
/// carries meaning — a paper's Δ is a change and its δ is not.
pub const GREEK_GLYPHS: &[(char, &str)] = &[
    ('α', "alpha"),
    ('β', "beta"),
    ('γ', "gamma"),
    ('δ', "delta"),
    ('ε', "epsilon"),
    ('ζ', "zeta"),
    ('η', "eta"),
    ('θ', "theta"),
    ('ι', "iota"),
    ('κ', "kappa"),
    ('λ', "lambda"),
    ('μ', "mu"),
    ('ν', "nu"),
    ('ξ', "xi"),
    ('ρ', "rho"),
    ('σ', "sigma"),
    ('τ', "tau"),
    ('φ', "phi"),
    ('χ', "chi"),
    ('ψ', "psi"),
    ('ω', "omega"),
    ('Δ', "delta"),
    ('Σ', "sigma"),
    ('Π', "product"),
    ('Ω', "omega"),
    ('Φ', "phi"),
    ('Θ', "theta"),
    ('Λ', "lambda"),
    ('Γ', "gamma"),
    ('Ψ', "psi"),
    ('Ξ', "xi"),
];

/// Symbols in prose rather than inside a maths span. Every one is silent or mispronounced
/// when passed through: `d ≈ 0.42` was read "d 0.42", which asserts equality.
///
/// Insertion order is the substitution order and it matters: the multi-character forms have
/// no overlap here, but `≃`/`≅` must not be reached by a rule that already replaced `≈`.
pub const PROSE_SYMBOLS: &[(&str, &str)] = &[
    ("±", " plus or minus "),
    ("×", " times "),
    ("÷", " divided by "),
    ("≈", " about "),
    ("≃", " about "),
    ("≅", " about "),
    ("≤", " at most "),
    ("≥", " at least "),
    ("≠", " not equal to "),
    ("≡", " identical to "),
    ("∝", " proportional to "),
    ("∞", " infinity "),
    ("∈", " in "),
    ("∉", " not in "),
    ("∀", " for all "),
    ("∃", " there exists "),
    ("∑", " the sum of "),
    ("∏", " the product of "),
    ("∫", " the integral of "),
    ("√", " the square root of "),
    ("∂", " partial "),
    ("∇", " gradient "),
    ("⊂", " a subset of "),
    ("∪", " union "),
    ("∩", " intersection "),
    ("→", ", then "),
    ("←", " from "),
    ("↔", " and "),
    ("⇒", ", then "),
    ("‰", " per mille "),
    ("†", ""),
    ("‡", ""),
    ("″", " seconds "),
    ("′", " minutes "),
];

pub const MATH_SYMBOLS: &[(&str, &str)] = &[
    ("=", " equals "),
    ("+", " plus "),
    ("-", " minus "),
    ("*", " times "),
    ("/", " over "),
    ("<", " less than "),
    (">", " greater than "),
];

/// Compounds this voice cannot pronounce as one token. Found by aggregating alignment
/// manifests — a mangled word is never recognised — then listened to individually:
/// "timezone" was spoken "Heideheb" and "signup" "SignGen".
pub const MANGLED_COMPOUNDS: &[(&str, &str)] = &[
    ("timezone", "time zone"),
    ("timezones", "time zones"),
    ("signup", "sign up"),
    ("signups", "sign ups"),
];

pub fn map_of(pairs: &[(&'static str, &'static str)]) -> HashMap<&'static str, &'static str> {
    pairs.iter().copied().collect()
}

/// Alternation body sorted longest-first, so `TWh` is tried before `W`.
pub fn alternation_longest_first(keys: impl IntoIterator<Item = &'static str>) -> String {
    let mut keys: Vec<&str> = keys.into_iter().collect();
    keys.sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(b)));
    keys.join("|")
}
