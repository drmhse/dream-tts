//! What gets synthesised, and what it covers.
//!
//! A chunk is a run of whole sentences inside one block, long enough that the engine keeps its
//! prosody and short enough that the highlight keeps moving.
//!
//! It was a whole block until a PDF proved that wrong. A document imported from PDF loses its
//! paragraph structure — `fixtures/prose.md` is twelve blocks as markdown and three when the
//! same document is read back from a typeset PDF — so a chunk-per-block put one highlight on
//! screen, motionless, for two and a half minutes. The same defect is there in markdown
//! whenever a paragraph is long; the PDF only made it impossible to miss.
//!
//! Sentences are grouped rather than taken one at a time because each chunk is a separate
//! request, and an engine cannot carry prosody across one. `MIN_WORDS` is the floor.
//!
//! A block that owns no words is not a chunk. Code is shown and never spoken.

use crate::sentence;
use crate::text::{flowing, operators, unshouted};
use std::ops::Range;
use tts_doc::Block;
use tts_narrate::page::{normalize_word, words as tokenise};
use tts_narrate::{align, blocks, Options};

/// Words below which a chunk is joined to the next sentence.
///
/// About five seconds at 155 words a minute. Short enough that a highlight moves often, long
/// enough that "Yes." is not sent to an engine on its own — a one-word request comes back with
/// no sentence prosody at all, and a paragraph read that way sounds like a list.
const MIN_WORDS: usize = 13;

/// Words above which a group is cut down to size, on word boundaries.
///
/// About twenty-three seconds at 155 words a minute: a highlight that does not move for
/// longer reads as stuck, and past this size a chunk is one engine timeout away from
/// parking and vetoing the export. The cut is mid-sentence when the sentence is the
/// problem — a whole page with no full stop — and the prosody across it is the price.
const MAX_WORDS: usize = 60;

#[derive(Clone, Debug, PartialEq)]
pub struct Chunk {
    /// Index into the block list.
    pub block: usize,
    /// Canonical word indices this chunk covers, so a highlight can be placed from it.
    pub words: Range<usize>,
    /// What the engine is asked to say.
    pub text: String,
    /// The chunk's own words in page order: `page::words` of the block's text over `words`.
    ///
    /// The engine says `text`, but its words land on `page`. The two streams are the same
    /// length only for plain prose — narration expands (`API` to three letters, `12.4` to
    /// three words) and drops (URLs the page counts but nothing says) — so indexing
    /// narration positions into page space drifts every word after the first divergence and
    /// overruns the range entirely on expansions. `page` is what the clock is matched
    /// against; it is exactly `words.len()` long, so a match cannot point off the chunk.
    pub page: Vec<String>,
}

/// The chunks of a document, in reading order.
pub fn plan(blocks: &[Block]) -> Vec<Chunk> {
    let mut out = Vec::new();
    for (i, block) in blocks.iter().enumerate() {
        if block.words.is_empty() || !block.kind.is_narrated() {
            continue;
        }
        // The sibling's own markdown-to-speech rules, not a second copy of them: they carry a
        // chapter's worth of failures — a table that became babble, a lower-case opening that
        // came out as a different word.
        let narration = blocks::convert(&operators(&block.source), &Options::default());
        let narration = flowing(&unshouted(narration.trim()));
        if narration.is_empty() {
            continue;
        }
        out.extend(split_block(i, block, &narration));
    }
    out
}

/// One block into chunks, each carrying the canonical words it covers.
fn split_block(index: usize, block: &Block, narration: &str) -> Vec<Chunk> {
    let pieces = grouped(narration);
    // The block's page words once, sliced per chunk below. Retokenising the stored text
    // reproduces exactly the stream the ranges were counted from — `scan::emit` counted
    // `words` of this same string — so the slice over a chunk's range is that chunk's words
    // in page order, whatever the narration did to them.
    let page_all = tokenise(&block.text);
    debug_assert_eq!(page_all.len(), block.words.len());
    if pieces.len() < 2 {
        return vec![Chunk {
            block: index,
            words: block.words.clone(),
            text: narration.to_string(),
            page: page_all,
        }];
    }

    // Spoken words are not canonical words — narration expands numbers, drops images, rewrites
    // tables — so the two streams are matched rather than assumed equal. Locally, within one
    // block, which bounds what a mis-anchor can cost to a word.
    let spoken: Vec<String> = pieces
        .iter()
        .flat_map(|p| tokenise(p))
        .map(|w| normalize_word(&w))
        .collect();
    let canonical: Vec<String> = tokenise(&block.text)
        .iter()
        .map(|w| normalize_word(w))
        .collect();
    let mapping = align::align_tokens(&spoken, &canonical);

    let base = block.words.start;
    let mut chunks: Vec<Chunk> = Vec::with_capacity(pieces.len());
    let mut at = 0usize;
    let mut previous_end = base;

    for piece in &pieces {
        let count = tokenise(piece).len();
        let mapped: Vec<usize> = mapping[at..(at + count).min(mapping.len())]
            .iter()
            .filter(|m| **m >= 0)
            .map(|m| base + *m as usize)
            .collect();
        at += count;

        // A piece whose every word failed to match — narration with no counterpart on the page
        // — still has to occupy time, so it takes the boundary it sits on rather than an empty
        // range that would leave a gap in the highlight.
        let end = match mapped.last() {
            Some(last) => (*last + 1).max(previous_end),
            None => previous_end,
        };
        chunks.push(Chunk {
            block: index,
            words: previous_end..end,
            text: piece.to_string(),
            page: page_slice(&page_all, base, previous_end..end),
        });
        previous_end = end;
    }

    // The last chunk owns whatever is left, so the block's words are covered exactly.
    if let Some(last) = chunks.last_mut() {
        last.words.end = block.words.end;
        last.page = page_slice(&page_all, base, last.words.clone());
    }
    chunks
}

/// The page words for a chunk's range: `page_all` is the block's whole stream in order and
/// `range` is absolute, so the local slice is the chunk's words in page order. Clamped rather
/// than panicking — a mismatch here means the ranges drifted from the text, and a short page
/// is an honest subset while an index past the end would invent words.
fn page_slice(page_all: &[String], base: usize, range: Range<usize>) -> Vec<String> {
    let start = range.start.saturating_sub(base).min(page_all.len());
    let end = range.end.saturating_sub(base).min(page_all.len());
    page_all[start.min(end)..end].to_vec()
}

/// Sentences, joined until each group carries its weight.
fn grouped(narration: &str) -> Vec<String> {
    let sentences = sentence::split(narration);
    let mut out: Vec<String> = Vec::new();
    let mut buffer = String::new();
    let mut count = 0usize;

    for piece in sentences {
        if !buffer.is_empty() {
            buffer.push(' ');
        }
        buffer.push_str(piece);
        count += tokenise(piece).len();
        if count >= MIN_WORDS {
            out.push(std::mem::take(&mut buffer));
            count = 0;
        }
    }
    // A short tail joins the group before it rather than being sent on its own.
    if !buffer.is_empty() {
        match out.last_mut() {
            Some(last) => {
                last.push(' ');
                last.push_str(&buffer);
            }
            None => out.push(buffer),
        }
    }
    // ...and a group that grew past bearing is cut down: a block with no sentence end —
    // a whole PDF page of run-on prose, a pasted URL salad — would otherwise go to the
    // engine as one request of unbounded size, time out past the 600 s limit, burn its
    // retries, get parked, and veto the entire export. Split on word boundaries; the
    // prosody across the cut is cheaper than a wedged book.
    out.into_iter()
        .flat_map(|group| cap_group(&group))
        .collect()
}

/// One group into pieces of at most `MAX_WORDS` words and `MAX_CHARS` chars, split on
/// whitespace — or, for a single token past bearing (a spaceless CJK run, a pasted URL), on
/// char boundaries. Intra-token pieces join back with no space: they were one token, and a
/// space inside a URL or a CJK run would be spoken, or misread, as content.
fn cap_group(group: &str) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut run: Vec<&str> = Vec::new();
    for token in group.split_whitespace() {
        if token.chars().count() > MAX_CHARS {
            pieces.extend(balanced(&std::mem::take(&mut run)));
            let chars: Vec<char> = token.chars().collect();
            pieces.extend(
                chars
                    .chunks(MAX_CHARS)
                    .map(|part| part.iter().collect::<String>()),
            );
        } else {
            run.push(token);
        }
    }
    pieces.extend(balanced(&run));
    if pieces.is_empty() {
        pieces.push(group.to_string());
    }
    pieces
}

const MAX_CHARS: usize = 400;

/// Tokens cut into as few pieces as the limits allow, of even size, each ending at a clause
/// mark near its share where there is one. Greedy filling left "the current ones." as a
/// three-word request of its own, with no sentence prosody at all.
fn balanced(tokens: &[&str]) -> Vec<String> {
    let chars = |t: &[&str]| t.iter().map(|w| w.len()).sum::<usize>() + t.len().saturating_sub(1);
    let mut out = Vec::new();
    let mut rest = tokens;
    while !rest.is_empty() {
        let total = chars(rest);
        let left = rest
            .len()
            .div_ceil(MAX_WORDS)
            .max(total.div_ceil(MAX_CHARS));
        if left <= 1 {
            out.push(rest.join(" "));
            break;
        }
        let target = total / left;
        let mut fit = 0;
        let mut size = 0;
        let mut cuts = Vec::new();
        for (i, w) in rest.iter().enumerate() {
            size += w.len() + usize::from(i > 0);
            if i + 1 > MAX_WORDS || size > MAX_CHARS {
                break;
            }
            fit = i + 1;
            cuts.push(size);
        }
        let fit = fit.max(1);
        let distance = |n: usize| cuts.get(n - 1).map_or(usize::MAX, |c| c.abs_diff(target));
        let clause = (1..=fit)
            .filter(|&n| {
                rest[n - 1].ends_with([',', ';', ':'])
                    && cuts.get(n - 1).is_some_and(|c| c * 2 >= target)
            })
            .min_by_key(|&n| distance(n));
        let even = (1..=fit).min_by_key(|&n| distance(n)).unwrap_or(fit);
        let at = clause
            .filter(|&n| distance(n) <= target / 3)
            .unwrap_or(even);
        out.push(rest[..at].join(" "));
        rest = &rest[at..];
    }
    out
}

/// Rough speech duration from a word count, for a document nothing has synthesised yet.
///
/// Only ever an estimate, and every caller has to say so: it is what lets the reader open,
/// paginate and show a document with no engine running, which is the degradation the plan
/// requires. It must never reach a highlight — see `Timing::measured`.
pub fn estimate_seconds(words: usize) -> f64 {
    // 155 words per minute, the rate the four engines land between on the fixtures.
    words as f64 * 60.0 / 155.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use tts_doc::layout;

    #[test]
    fn a_long_sentence_is_cut_evenly_at_a_clause() {
        let sentence = format!(
            "{}, {} end.",
            "alpha ".repeat(34).trim(),
            "beta ".repeat(30).trim()
        );
        let pieces = balanced(&sentence.split_whitespace().collect::<Vec<_>>());
        assert_eq!(pieces.len(), 2, "{pieces:?}");
        assert!(pieces[0].ends_with("alpha,"), "{pieces:?}");
        let tail = format!("{} tail words here.", "word ".repeat(59).trim());
        let pieces = balanced(&tail.split_whitespace().collect::<Vec<_>>());
        assert!(
            pieces.iter().all(|p| p.split(' ').count() >= 20),
            "{pieces:?}"
        );
    }

    #[test]
    fn a_chunk_is_a_block_that_owns_words() {
        let md = "# Title\n\nA paragraph.\n\n```sh\ncargo build\n```\n\nAnother.\n";
        let blocks = layout(md);
        let chunks = plan(&blocks);
        assert_eq!(chunks.len(), 3, "the code block is not a chunk");
        assert_eq!(chunks[0].words, blocks[0].words);
        assert!(chunks.iter().all(|c| !c.text.is_empty()));
    }

    #[test]
    fn chunks_cover_the_document_in_order() {
        let md = "# One\n\nTwo words here.\n\n- A list item\n";
        let chunks = plan(&layout(md));
        let mut last = 0;
        for c in &chunks {
            assert!(c.words.start >= last);
            last = c.words.start;
        }
    }

    /// Narration is the sibling's, so markdown syntax never reaches the voice.
    #[test]
    fn narration_text_is_speakable_rather_than_markdown() {
        let chunks = plan(&layout("A **bold** claim with `code` in it.\n"));
        assert_eq!(chunks[0].text, "A bold claim with code in it.");
    }

    /// A short paragraph is one chunk: splitting it would send a fragment to the engine and
    /// lose the prosody across it.
    #[test]
    fn a_short_paragraph_stays_whole() {
        let chunks = plan(&layout("One thing. Then another.\n"));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "One thing. Then another.");
    }

    /// The defect this exists for: a long block held one highlight motionless for minutes.
    #[test]
    fn a_long_paragraph_becomes_several_chunks() {
        let sentence = "The configuration file shipped with the checkpoint declares a semantic \
                        codebook of four thousand entries and every tensor is deep. ";
        let md = format!("{}\n", sentence.repeat(8));
        let blocks = layout(&md);
        let chunks = plan(&blocks);
        assert!(chunks.len() >= 6, "only {} chunks", chunks.len());
        assert!(chunks.iter().all(|c| c.block == 0));
    }

    /// Whatever the split, the chunks of a block must cover its canonical words exactly and in
    /// order — that range is what places the highlight.
    #[test]
    fn chunks_cover_their_blocks_words_without_gap_or_overlap() {
        let md = "# A title\n\nOne sentence here. A second sentence follows it. A third one \
                  closes the paragraph off. And a fourth for good measure now.\n\n\
                  - A list item that is short\n";
        let blocks = layout(md);
        let chunks = plan(&blocks);

        for (i, block) in blocks.iter().enumerate() {
            let mine: Vec<&Chunk> = chunks.iter().filter(|c| c.block == i).collect();
            if mine.is_empty() {
                continue;
            }
            assert_eq!(
                mine[0].words.start, block.words.start,
                "block {i} starts late"
            );
            assert_eq!(
                mine[mine.len() - 1].words.end,
                block.words.end,
                "block {i} ends early"
            );
            for pair in mine.windows(2) {
                assert_eq!(
                    pair[0].words.end, pair[1].words.start,
                    "gap inside block {i}"
                );
            }
        }
    }

    /// Every word of the narration still reaches the engine.
    #[test]
    fn splitting_loses_no_words() {
        let md = "One sentence here. A second sentence follows it. A third one closes it off.\n";
        let blocks = layout(md);
        let spoken: String = plan(&blocks)
            .iter()
            .map(|c| c.text.clone())
            .collect::<Vec<_>>()
            .join(" ");
        let whole = tts_narrate::blocks::convert(&blocks[0].source, &Options::default());
        assert_eq!(
            spoken.split_whitespace().collect::<Vec<_>>(),
            whole.split_whitespace().collect::<Vec<_>>()
        );
    }

    /// A block with no sentence end — a whole PDF page of run-on prose — used to go to the
    /// engine as one unbounded request: past the timeout, through its retries, parked, and
    /// then vetoing the entire export. Now it is cut to size on word boundaries.
    #[test]
    fn a_run_on_block_is_cut_to_size() {
        let md = format!("{}\n", "word ".repeat(3000));
        let chunks = plan(&layout(&md));
        assert!(chunks.len() > 10, "only {} chunk(s)", chunks.len());
        for c in &chunks {
            let words = c.text.split_whitespace().count();
            assert!(words <= 60, "{words} words in one chunk");
        }
        let spoken: String = chunks
            .iter()
            .map(|c| c.text.clone())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(
            spoken.split_whitespace().count(),
            3000,
            "words were lost in the cut"
        );
    }

    /// A single spaceless token — a pasted URL, a CJK run — cannot split on whitespace, so
    /// it splits on char boundaries instead of going out whole.
    #[test]
    fn a_giant_token_is_cut_on_char_boundaries() {
        let md = format!("{}\n", "x".repeat(1000));
        let chunks = plan(&layout(&md));
        assert!(chunks.len() > 1, "one chunk");
        for c in &chunks {
            assert!(
                c.text.chars().count() <= 400,
                "{} chars in one chunk",
                c.text.len()
            );
        }
        let spoken: String = chunks
            .iter()
            .map(|c| c.text.clone())
            .collect::<Vec<_>>()
            .join("");
        // Less the full stop `flowing` closes the block with.
        let spoken = spoken.strip_suffix('.').unwrap_or(&spoken);
        assert_eq!(spoken.chars().count(), 1000, "chars were lost in the cut");
    }
}

#[cfg(test)]
mod page_tests {
    use super::*;
    use tts_doc::layout;

    /// What `placed` relies on: a chunk's page stream is exactly its word range, so a match
    /// index can never point off the chunk. Checked over prose, a link (narration shorter
    /// than page), and an expansion (narration longer than page).
    #[test]
    fn every_chunk_carries_its_own_page_words() {
        let md = "# A title\n\nOne sentence here. A second sentence follows it. A third one \
                  closes the paragraph off. And a fourth for good measure now.\n\n\
                  See the [porting notes](https://example.com/notes) and use the API today.\n";
        let blocks = layout(md);
        for c in plan(&blocks) {
            let block = &blocks[c.block];
            let page: Vec<String> = tokenise(&block.text)
                [c.words.start - block.words.start..c.words.end - block.words.start]
                .to_vec();
            assert_eq!(c.page, page, "chunk {} disagrees with its range", c.text);
            assert_eq!(c.page.len(), c.words.len());
        }
    }
}
