//! Fenced code, lifted out before anything else looks at a line.
//!
//! `page_text` deletes fenced blocks with `(?sm)^```.*?^``` `, so their contents never reach
//! the word stream. The reader still has to *show* them, which means lifting them out here
//! rather than deleting them — and matching that regex's behaviour exactly, including the one
//! case that matters: an unterminated fence is not a match, so its lines stay ordinary text.
//! Deleting to end of file would eat the rest of the document.

pub struct Fence {
    /// Index of the opening fence line in the input.
    pub at: usize,
    /// Lines the fence spans, opener and closer included.
    pub len: usize,
    pub lang: Option<String>,
    pub body: String,
}

fn is_fence(line: &str) -> bool {
    line.starts_with("```")
}

/// Every terminated fence, in order.
pub fn find(lines: &[&str]) -> Vec<Fence> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        if !is_fence(lines[i]) {
            i += 1;
            continue;
        }
        let Some(close) = (i + 1..lines.len()).find(|&j| is_fence(lines[j])) else {
            // Unterminated: the regex does not match, so neither do we, and the remaining
            // lines are read as ordinary markdown.
            break;
        };
        let lang = lines[i].trim_start_matches('`').trim();
        out.push(Fence {
            at: i,
            len: close - i + 1,
            lang: (!lang.is_empty()).then(|| lang.to_string()),
            body: lines[i + 1..close].join("\n"),
        });
        i = close + 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(s: &str) -> Vec<&str> {
        s.split('\n').collect()
    }

    #[test]
    fn a_fence_is_found_with_its_language_and_body() {
        let src = lines("before\n```rust\nlet a = 1;\n```\nafter");
        let f = find(&src);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].at, 1);
        assert_eq!(f[0].len, 3);
        assert_eq!(f[0].lang.as_deref(), Some("rust"));
        assert_eq!(f[0].body, "let a = 1;");
    }

    #[test]
    fn a_fence_without_a_language_has_none() {
        assert_eq!(find(&lines("```\nx\n```"))[0].lang, None);
    }

    /// The case that would otherwise eat the document.
    #[test]
    fn an_unterminated_fence_is_not_a_fence() {
        assert!(find(&lines("```\nstill prose\nand more")).is_empty());
    }

    #[test]
    fn two_fences_do_not_pair_across_each_other() {
        let f = find(&lines("```\na\n```\ntext\n```\nb\n```"));
        assert_eq!(f.len(), 2);
        assert_eq!(f[1].at, 4);
    }
}
