//! Regex plumbing shared by the rule modules.
//!
//! One thing the `fancy_regex` API does not give directly and that the ported rules need: a
//! replacement closure that can see the whole haystack around its match. Three rules decide
//! by looking at what precedes or follows — the unit and magnitude rules read the noun
//! behind the match for number agreement, and the Greek rule pads only where a glyph would
//! fuse into a neighbouring word.

use fancy_regex::{Captures, Regex};

/// Compile or panic. Every pattern is a literal in this crate, so a failure is a bug here
/// rather than anything a caller can cause.
pub fn compile(pattern: &str) -> Regex {
    Regex::new(pattern).unwrap_or_else(|e| panic!("bad pattern {pattern:?}: {e}"))
}

/// `replace_all` with a closure over the captures.
pub fn sub<F>(re: &Regex, input: &str, mut f: F) -> String
where
    F: FnMut(&Captures<'_, str>) -> String,
{
    replace(re, input, |caps, _, _| f(caps))
}

/// Static replacement; `${1}` references work as in `Regex::replace_all`.
pub fn sub_str(re: &Regex, input: &str, replacement: &str) -> String {
    re.replace_all(input, replacement).into_owned()
}

/// `replace_all` where the closure also receives the haystack and the match bounds.
pub fn replace<F>(re: &Regex, input: &str, mut f: F) -> String
where
    F: FnMut(&Captures<'_, str>, usize, usize) -> String,
{
    let mut out = String::with_capacity(input.len());
    let mut last = 0usize;
    for caps in re.captures_iter(input) {
        // A backtracking limit on pathological input: stop substituting rather than corrupt
        // the line. Everything from `last` is copied through untouched below.
        let Ok(caps) = caps else { break };
        let Some(whole) = caps.get(0) else { continue };
        let (start, end) = (whole.start(), whole.end());
        if start < last {
            continue;
        }
        out.push_str(&input[last..start]);
        out.push_str(&f(&caps, start, end));
        last = end;
    }
    out.push_str(&input[last..]);
    out
}

/// The capture group as a string, or empty when it did not participate.
pub fn g(caps: &Captures<'_, str>, i: usize) -> String {
    caps.get(i)
        .map_or_else(String::new, |m| m.as_str().to_string())
}

/// `Some` only when the group participated. Several rules branch on the difference between
/// an absent optional group and an empty one.
pub fn opt<'t>(caps: &Captures<'t, str>, i: usize) -> Option<&'t str> {
    caps.get(i).map(|m| m.as_str())
}

/// Collapse runs of whitespace to one space.
pub fn squeeze(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_space = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !in_space {
                out.push(' ');
            }
            in_space = true;
        } else {
            out.push(c);
            in_space = false;
        }
    }
    out
}

/// Python's `str.strip(" ,")`.
pub fn strip_spaces_and_commas(s: &str) -> String {
    s.trim_matches(|c| c == ' ' || c == ',').to_string()
}
