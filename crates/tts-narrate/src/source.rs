//! Markdown primitives both readers of the source need.
//!
//! `blocks` produces narration and `page` produces the prose a browser shows, and they must
//! agree about what the source *is* — front matter, comments, code fences, a figure's
//! caption. Where they differ is policy, not parsing: `blocks` honours `keep_captions` and
//! `keep_code`, `page` always keeps the caption and never the code. So the primitives live
//! here and the policy stays with each caller. Two copies of this drifted once already,
//! and a difference between the two tokenisers shows up as unmapped words rather than as an
//! error.

use crate::re::{self, compile};
use fancy_regex::Regex;
use once_cell::sync::Lazy;

pub static CODE_BLOCK: Lazy<Regex> = Lazy::new(|| compile(r"(?sm)^```.*?^```"));
pub static HTML_COMMENT: Lazy<Regex> = Lazy::new(|| compile(r"(?s)<!--.*?-->"));
pub static SHORTCODE: Lazy<Regex> = Lazy::new(|| compile(r"(?s)\{\{[<%].*?[>%]\}\}"));
static CAPTION: Lazy<Regex> = Lazy::new(|| compile(r#"caption\s*=\s*"([^"]*)""#));

/// TOML (`+++`) and YAML (`---`) front matter, removed.
pub fn strip_front_matter(text: &str) -> String {
    for fence in ["+++", "---"] {
        let body = text.trim_start();
        if let Some(rest) = body.strip_prefix(fence) {
            let needle = format!("\n{fence}");
            if let Some(end) = rest.find(&needle) {
                return rest[end + needle.len()..].to_string();
            }
        }
    }
    text.to_string()
}

/// A shortcode's `caption=`, as its own paragraph.
///
/// `{{< chapter-figure >}}` marks an image the listener cannot see, and its caption is
/// usually the one sentence stating what the figure was for — so it is the only part of a
/// shortcode that becomes prose.
pub fn caption_paragraph(shortcode: &str) -> String {
    let Ok(Some(caps)) = CAPTION.captures(shortcode) else {
        return String::new();
    };
    let caption = re::g(&caps, 1).trim().to_string();
    if caption.is_empty() {
        String::new()
    } else if ends_sentence(&caption) {
        format!("\n\n{caption}\n\n")
    } else {
        format!("\n\n{caption}.\n\n")
    }
}

/// A colon and a semicolon count: a table label that already ends in one needs no full stop
/// added, and neither does a closing quote.
pub fn ends_sentence(s: &str) -> bool {
    s.ends_with(['.', '!', '?', ':', ';', '"'])
}

/// A line of only `-`, `=`, `*` or `_` is a horizontal rule, which is shown and not spoken.
pub fn is_horizontal_rule(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.len() >= 3 && trimmed.chars().all(|c| matches!(c, '-' | '=' | '*' | '_'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_front_matter_fences_are_stripped() {
        assert_eq!(
            strip_front_matter("+++\na = 1\n+++\n\nBody\n"),
            "\n\nBody\n"
        );
        assert_eq!(strip_front_matter("---\na: 1\n---\n\nBody\n"), "\n\nBody\n");
    }

    #[test]
    fn text_without_front_matter_is_untouched() {
        assert_eq!(strip_front_matter("Body\n"), "Body\n");
        // An unterminated fence is not front matter; removing to end of file would eat the
        // whole document.
        assert_eq!(strip_front_matter("+++\nno end\n"), "+++\nno end\n");
    }

    #[test]
    fn a_caption_becomes_a_sentence_of_its_own() {
        assert_eq!(
            caption_paragraph(r#"{{< figure caption="The two paths" >}}"#),
            "\n\nThe two paths.\n\n"
        );
        assert_eq!(
            caption_paragraph(r#"{{< figure caption="Ends already." >}}"#),
            "\n\nEnds already.\n\n"
        );
        assert_eq!(caption_paragraph("{{< figure >}}"), "");
        assert_eq!(caption_paragraph(r#"{{< figure caption="  " >}}"#), "");
    }

    #[test]
    fn horizontal_rules_are_recognised() {
        assert!(is_horizontal_rule("---"));
        assert!(is_horizontal_rule("  ***  "));
        assert!(!is_horizontal_rule("--"));
        assert!(!is_horizontal_rule("- item"));
    }
}
