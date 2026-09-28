//! Splitting a block where the document says a paragraph ended.
//!
//! Needed because a PDF's paragraphs do not survive import. `tts-import` rejoins Quartz's
//! hard-wrapped lines and means to end a paragraph on a short line that closes a sentence, but
//! the length test is applied to the paragraph accumulated so far rather than to the line — so
//! nothing longer than sixty characters can ever end one, and a whole page arrives as a single
//! block. Recorded in `docs/upstream-gaps.md`; this is the adapter that stands in until it is
//! fixed upstream.
//!
//! The breaks come from the page's own geometry, so this module only has to cut, and it cuts
//! only prose: a heading, a list item or a table is a shape the layout already knows.

use crate::{Block, Kind};
use std::collections::BTreeSet;

/// Split blocks at the canonical word indices in `breaks`.
///
/// A break is the index of the first word of the *next* paragraph, and one that falls on a
/// block's own boundary is already a break and does nothing.
pub fn split(blocks: Vec<Block>, breaks: &BTreeSet<usize>) -> Vec<Block> {
    let mut out = Vec::with_capacity(blocks.len());
    for block in blocks {
        if !matches!(block.kind, Kind::Paragraph) {
            out.push(block);
            continue;
        }
        let inside: Vec<usize> = breaks
            .range(block.words.start + 1..block.words.end)
            .copied()
            .collect();
        let mut rest = block;
        for at in inside {
            let n = at - rest.words.start;
            let (Some(text_cut), Some(source_cut)) =
                (after_word(&rest.text, n), after_word(&rest.source, n))
            else {
                // The two halves disagree about where the word is, which means the source
                // carries markup the prose does not. Leave the block whole rather than cut it
                // in two different places.
                break;
            };
            let head = Block {
                kind: Kind::Paragraph,
                text: rest.text[..text_cut].trim_end().to_string(),
                source: rest.source[..source_cut].trim_end().to_string(),
                words: rest.words.start..at,
            };
            rest = Block {
                kind: Kind::Paragraph,
                text: rest.text[text_cut..].trim_start().to_string(),
                source: rest.source[source_cut..].trim_start().to_string(),
                words: at..rest.words.end,
            };
            out.push(head);
        }
        out.push(rest);
    }
    out
}

/// Where to cut so that `n` words fall on the left: the start of word `n + 1`, not the end of
/// word `n`, so the full stop that closed the sentence stays with the sentence.
///
/// It has to be counted here rather than asked for, because `page::words` returns the words and
/// not where they were. `a_cut_agrees_with_the_canonical_tokeniser` is what keeps the two from
/// drifting: it checks that cutting here really does put `n` words on one side.
fn after_word(text: &str, n: usize) -> Option<usize> {
    if n == 0 {
        return None;
    }
    let bytes = text.as_bytes();
    let mut count = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if !bytes[i].is_ascii_alphanumeric() {
            i += 1;
            continue;
        }
        let mut end = run_end(bytes, i);
        // `[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?` — one apostrophe may join two runs, so `don't`
        // is one word and not two.
        for apostrophe in ["'", "\u{2019}"] {
            let after = end + apostrophe.len();
            if text[end..].starts_with(apostrophe)
                && bytes.get(after).is_some_and(u8::is_ascii_alphanumeric)
            {
                end = run_end(bytes, after);
                break;
            }
        }
        count += 1;
        if count == n {
            let next = bytes[end..].iter().position(u8::is_ascii_alphanumeric)?;
            return Some(end + next);
        }
        i = end;
    }
    None
}

fn run_end(bytes: &[u8], from: usize) -> usize {
    let mut at = from;
    while bytes.get(at).is_some_and(u8::is_ascii_alphanumeric) {
        at += 1;
    }
    at
}

#[cfg(test)]
mod tests {
    use super::*;
    use tts_narrate::page;

    fn paragraph(text: &str, words: std::ops::Range<usize>) -> Block {
        Block {
            kind: Kind::Paragraph,
            text: text.to_string(),
            source: text.to_string(),
            words,
        }
    }

    /// The tokeniser stays the authority. This checks the cut agrees with it rather than
    /// asserting a byte offset, which would pass while meaning something different.
    #[test]
    fn a_cut_agrees_with_the_canonical_tokeniser() {
        let text = "The 1.7 billion checkpoint doesn't fit. It won\u{2019}t ever fit — 4096 rows.";
        let all = page::words(text);
        for n in 1..all.len() {
            let cut = after_word(text, n).expect("a cut for every word");
            assert_eq!(
                page::words(&text[..cut]).len(),
                n,
                "cutting after word {n} of {:?} put the wrong number on the left",
                text
            );
            let mut halves = page::words(&text[..cut]);
            halves.extend(page::words(&text[cut..]));
            assert_eq!(halves, all, "cutting after word {n} lost or gained a word");
        }
    }

    #[test]
    fn a_paragraph_splits_where_the_document_says_it_ended() {
        let blocks = vec![paragraph("One two. Three four five. Six seven.", 0..7)];
        let breaks = BTreeSet::from([2, 5]);
        let out = split(blocks, &breaks);
        let shapes: Vec<(&str, std::ops::Range<usize>)> = out
            .iter()
            .map(|b| (b.text.as_str(), b.words.clone()))
            .collect();
        assert_eq!(
            shapes,
            vec![
                ("One two.", 0..2),
                ("Three four five.", 2..5),
                ("Six seven.", 5..7),
            ]
        );
    }

    #[test]
    fn only_prose_is_cut() {
        let heading = Block {
            kind: Kind::Heading(2),
            text: "One two three".into(),
            source: "## One two three".into(),
            words: 0..3,
        };
        let out = split(vec![heading], &BTreeSet::from([1, 2]));
        assert_eq!(out.len(), 1, "a heading was cut into pieces");
    }

    #[test]
    fn a_break_on_a_boundary_changes_nothing() {
        let blocks = vec![paragraph("One two.", 0..2), paragraph("Three four.", 2..4)];
        let out = split(blocks, &BTreeSet::from([0, 2, 4]));
        assert_eq!(out.len(), 2);
    }
}
