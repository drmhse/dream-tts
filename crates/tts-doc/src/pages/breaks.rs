//! Where a PDF's paragraphs end, from the shape of its lines.
//!
//! The import cannot tell us: it rejoins Quartz's hard-wrapped lines and its paragraph test is
//! applied to the wrong string, so a whole page arrives as one block (`docs/upstream-gaps.md`).
//! The page itself can. A paragraph's last line stops short of the margin, and this project
//! already knows where every word sits.
//!
//! Geometry alone is not enough — a line before a figure, or the last line of a page, is also
//! short — so the text has to agree: the line closes a sentence *and* stops short. Either
//! signal alone is common mid-paragraph; together they are not.
//!
//! A heading closes nothing, and run into the paragraph under it the engine reads the two as
//! one sentence. So a short line with no closing mark also ends a block when it stands alone —
//! after another short line, at the top of a page, or set larger than the text — is a few
//! words long, and the next line on the same page opens with a capital.

use crate::geometry::WordBox;
use std::collections::BTreeSet;

/// How far short of the margin a line must stop. A justified line reaches the margin exactly
/// and a ragged one lands within a word of it, so this only has to clear measurement noise.
const SHORT: f64 = 0.02;

/// The margin is the 95th percentile of line ends rather than the maximum, so one line that
/// runs into the gutter does not move it for the whole document.
const MARGIN: usize = 95;

/// Canonical word indices at which a new paragraph begins.
pub fn paragraphs(boxes: &[WordBox]) -> BTreeSet<usize> {
    let lines = lines(boxes);
    if lines.is_empty() {
        return BTreeSet::new();
    }
    let mut ends: Vec<f64> = lines.iter().map(|l| l.right).collect();
    ends.sort_by(f64::total_cmp);
    let margin = ends[(ends.len() - 1) * MARGIN / 100];
    let cut = margin * (1.0 - SHORT);

    let mut heights: Vec<f64> = lines.iter().map(|l| l.height).collect();
    heights.sort_by(f64::total_cmp);
    let body = heights[heights.len() / 2];

    let short = |line: &Line| line.right < cut;
    let mut out = BTreeSet::new();
    for (i, line) in lines.iter().enumerate() {
        if !short(line) {
            continue;
        }
        if closes_a_sentence(&line.last) {
            out.insert(line.next_word);
            continue;
        }
        let Some(next) = lines.get(i + 1).filter(|n| n.page == line.page) else {
            continue;
        };
        let alone = match i.checked_sub(1).map(|p| &lines[p]) {
            Some(previous) => previous.page != line.page || short(previous),
            None => true,
        } || line.height > body * HEADING;
        // Past an opening quote or bracket: `“Why` opens a sentence as `Why` does.
        let capital = next
            .first
            .chars()
            .find(|c| c.is_alphanumeric())
            .is_some_and(char::is_uppercase);
        if alone && capital && line.words <= HEADING_WORDS {
            out.insert(line.next_word);
        }
    }
    out
}

/// A line this much taller than the median is set in a larger size.
const HEADING: f64 = 1.15;

/// Past this a short line is the tail of a paragraph, not a heading.
const HEADING_WORDS: usize = 12;

struct Line {
    right: f64,
    last: String,
    first: String,
    page: usize,
    words: usize,
    height: f64,
    /// The canonical index of the first word after this line.
    next_word: usize,
}

/// Boxes gathered into the lines they sit on. They arrive in reading order, so a line ends when
/// the next box is on another page or another row.
fn lines(boxes: &[WordBox]) -> Vec<Line> {
    let mut out: Vec<Line> = Vec::new();
    let mut page = usize::MAX;
    let mut y = f64::MIN;
    for b in boxes {
        let same = b.page == page && (b.rect.y - y).abs() <= b.rect.h.max(4.0);
        match out.last_mut() {
            Some(line) if same => {
                line.right = line.right.max(b.rect.x + b.rect.w);
                line.last.clone_from(&b.text);
                line.next_word = b.words.end;
                line.words += 1;
                line.height = line.height.max(b.rect.h);
            }
            _ => out.push(Line {
                right: b.rect.x + b.rect.w,
                last: b.text.clone(),
                first: b.text.clone(),
                page: b.page,
                words: 1,
                height: b.rect.h,
                next_word: b.words.end,
            }),
        }
        page = b.page;
        y = b.rect.y;
    }
    out
}

/// The closing punctuation is checked after any quote or bracket that followed it, because
/// `end.”` and `end.)` close a sentence exactly as `end.` does.
fn closes_a_sentence(word: &str) -> bool {
    word.trim_end_matches(['"', '\u{201d}', '\u{2019}', '\'', ')', ']'])
        .ends_with(['.', '!', '?', ':'])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Rect;

    fn word(text: &str, page: usize, x: f64, y: f64, w: f64, at: usize) -> WordBox {
        WordBox {
            page,
            block: 0,
            text: text.to_string(),
            rect: Rect { x, y, w, h: 10.0 },
            words: at..at + 1,
        }
    }

    /// A line that stops short *and* closes a sentence ends a paragraph. One that does only
    /// one of the two does not — which is the case the import got wrong.
    #[test]
    fn both_signals_are_needed() {
        let mut boxes = Vec::new();
        // Four full-width lines, so the margin is 500.
        for (i, y) in [0.0, 20.0, 40.0, 60.0].into_iter().enumerate() {
            boxes.push(word("word", 0, 100.0, y, 400.0, i));
        }
        // A short line that does not close a sentence: a figure caption is about to follow.
        boxes.push(word("and", 0, 100.0, 80.0, 80.0, 4));
        // A full-width line that closes one: mid-paragraph, which is the common case.
        boxes.push(word("stop.", 0, 100.0, 100.0, 400.0, 5));
        // Short and closing: a paragraph really ended.
        boxes.push(word("done.", 0, 100.0, 120.0, 80.0, 6));

        assert_eq!(paragraphs(&boxes), BTreeSet::from([7]));
    }

    fn sized(text: &str, y: f64, w: f64, h: f64, at: usize) -> WordBox {
        let mut b = word(text, 0, 100.0, y, w, at);
        b.rect.h = h;
        b
    }

    /// A heading ends where it stops, though it closes no sentence; a line that stops short
    /// before a figure mid-sentence does not, because a full line came before it.
    #[test]
    fn a_heading_ends_its_block_and_a_line_before_a_figure_does_not() {
        let boxes = vec![
            sized("full", 0.0, 400.0, 10.0, 0),
            sized("full", 20.0, 400.0, 10.0, 1),
            sized("ends.", 40.0, 150.0, 10.0, 2),
            // A heading after a paragraph's short last line.
            sized("Heading", 60.0, 120.0, 10.0, 3),
            sized("Text", 80.0, 400.0, 10.0, 4),
            sized("full", 100.0, 400.0, 10.0, 5),
            // Short, after a full line: the text continues after a figure.
            sized("mobile", 120.0, 150.0, 10.0, 6),
            sized("Figure", 140.0, 400.0, 10.0, 7),
            sized("full", 160.0, 400.0, 10.0, 8),
            // Larger type stands alone even after a full line.
            sized("Big", 180.0, 150.0, 14.0, 9),
            sized("Text", 200.0, 400.0, 10.0, 10),
            sized("lower", 220.0, 400.0, 10.0, 11),
            sized("Short", 240.0, 150.0, 10.0, 12),
            // Lower case after a short line is the same sentence carrying on.
            sized("carrying", 260.0, 400.0, 10.0, 13),
        ];
        assert_eq!(paragraphs(&boxes), BTreeSet::from([3, 4, 10]));
    }

    #[test]
    fn a_closing_quote_does_not_hide_the_full_stop() {
        assert!(closes_a_sentence("end.\u{201d}"));
        assert!(closes_a_sentence("end.)"));
        assert!(!closes_a_sentence("end"));
    }

    #[test]
    fn nothing_to_read_is_no_breaks() {
        assert!(paragraphs(&[]).is_empty());
    }
}
