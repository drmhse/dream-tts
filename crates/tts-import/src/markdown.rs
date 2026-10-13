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

/// A table as blocks: a pipe table when it tabulates something, its cells as paragraphs when
/// it only lays out a page.
pub fn table(rows: Vec<Vec<String>>) -> Vec<String> {
    let columns = rows.iter().map(Vec::len).max().unwrap_or(0);
    let long = rows.iter().flatten().any(|c| c.chars().count() > 300);
    if rows.len() < 2 || columns < 2 || long {
        return rows
            .into_iter()
            .flatten()
            .filter_map(|cell| block(None, false, &cell))
            .collect();
    }
    let row = |cells: &[String]| {
        let cells: Vec<String> = (0..columns)
            .map(|i| {
                cells
                    .get(i)
                    .map_or("", String::as_str)
                    .replace('|', "\u{2223}")
            })
            .collect();
        format!("| {} |", cells.join(" | "))
    };
    let mut lines = vec![row(&rows[0])];
    lines.push(format!("|{}", " --- |".repeat(columns)));
    lines.extend(rows[1..].iter().map(|r| row(r)));
    vec![lines.join("\n")]
}

/// A table read from XML whose cells hold paragraphs, as DOCX's and ODT's do. A table inside a
/// cell is read as that cell's text.
#[derive(Default)]
pub struct Grid {
    depth: usize,
    rows: Vec<Vec<String>>,
    row: Vec<String>,
    cell: Option<String>,
}

impl Grid {
    pub fn open(&mut self) {
        self.depth += 1;
    }

    /// The finished table's blocks, once the outermost one closes.
    pub fn close(&mut self) -> Vec<String> {
        self.depth = self.depth.saturating_sub(1);
        if self.depth > 0 {
            return Vec::new();
        }
        self.row.clear();
        self.cell = None;
        table(std::mem::take(&mut self.rows))
    }

    pub fn open_row(&mut self) {
        if self.depth == 1 {
            self.row.clear();
        }
    }

    pub fn close_row(&mut self) {
        if self.depth == 1 && !self.row.is_empty() {
            self.rows.push(std::mem::take(&mut self.row));
        }
    }

    pub fn open_cell(&mut self) {
        if self.depth == 1 {
            self.cell = Some(String::new());
        }
    }

    pub fn close_cell(&mut self) {
        if self.depth == 1 {
            if let Some(cell) = self.cell.take() {
                self.row.push(cell.trim().to_string());
            }
        }
    }

    /// Takes a paragraph that belongs to a cell; `false` when no table is open.
    pub fn paragraph(&mut self, text: &str) -> bool {
        if self.depth == 0 {
            return false;
        }
        let text = text.trim();
        if let Some(cell) = self.cell.as_mut().filter(|_| !text.is_empty()) {
            if !cell.is_empty() {
                cell.push(' ');
            }
            cell.push_str(text);
        }
        true
    }
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
