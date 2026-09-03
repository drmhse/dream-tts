//! Emitting the markdown every importer produces.
//!
//! Four readers arrive at the same three-way decision — heading, list item, or plain
//! paragraph — from four different signals: a DOCX style name, an ODT outline level, an
//! HTML tag, a line of hashes. The signals differ and belong to their formats; what to write
//! once the level is known does not, and three copies of it drifted apart on whether an
//! empty paragraph should be emitted.

/// One block, or `None` when there is nothing to say.
///
/// A heading is written as hashes rather than as its text alone because the level is what
/// `split_markdown` splits on and what produces the 320 ms paragraph gap downstream.
pub fn block(level: Option<usize>, in_list: bool, text: &str) -> Option<String> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    Some(match level {
        Some(level) => format!("{} {text}", "#".repeat(level.clamp(1, 6))),
        None if in_list => format!("- {text}"),
        None => text.to_string(),
    })
}

/// Blocks as a document: one blank line between, one newline at the end.
pub fn document(blocks: &[String]) -> String {
    format!("{}\n", blocks.join("\n\n").trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_level_becomes_hashes() {
        assert_eq!(block(Some(1), false, "Title").as_deref(), Some("# Title"));
        assert_eq!(block(Some(3), false, "Deep").as_deref(), Some("### Deep"));
    }

    #[test]
    fn a_level_outside_markdown_is_clamped_rather_than_dropped() {
        // ODT allows outline levels past 6 and DOCX styles are whatever a template says.
        // Seven hashes is not a heading to any reader, so it would silently become prose.
        assert_eq!(
            block(Some(9), false, "Deep").as_deref(),
            Some("###### Deep")
        );
        assert_eq!(block(Some(0), false, "Zero").as_deref(), Some("# Zero"));
    }

    #[test]
    fn a_list_item_gets_a_bullet_only_when_it_is_not_a_heading() {
        assert_eq!(block(None, true, "item").as_deref(), Some("- item"));
        assert_eq!(
            block(Some(2), true, "heading in a list").as_deref(),
            Some("## heading in a list")
        );
    }

    #[test]
    fn empty_and_whitespace_blocks_are_nothing() {
        assert_eq!(block(None, false, "   "), None);
        assert_eq!(block(Some(1), false, ""), None);
    }

    #[test]
    fn a_document_is_blank_line_separated_and_newline_terminated() {
        assert_eq!(
            document(&["# One".into(), "prose".into()]),
            "# One\n\nprose\n"
        );
        assert_eq!(document(&[]), "\n");
    }
}
