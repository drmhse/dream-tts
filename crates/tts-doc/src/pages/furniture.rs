//! Running heads, folios and print footers: on every page, and never part of the text.
//!
//! Read aloud they interrupt a sentence that runs over a page break with "12/18/25, 2:24 PM
//! Comprehensive Rust google.github.io 135/675". They are found by repetition: a line in the
//! top or bottom band of the page whose text, digits masked, recurs on a share of the pages.

use crate::geometry::WordBox;
use std::collections::{BTreeSet, HashMap};

/// How near an edge a line must sit, as a share of the page height.
const BAND: f64 = 0.09;

/// Lines from each edge that are candidates: a head and a subhead, a folio under a footer.
const EDGE_LINES: usize = 2;

/// What a page's furniture covers: the canonical words to leave unspoken, and the word
/// indices at which a block has to be cut so those words stand in blocks of their own.
#[derive(Debug, Default, PartialEq)]
pub struct Furniture {
    pub words: BTreeSet<usize>,
    pub breaks: BTreeSet<usize>,
}

pub fn find(boxes: &[WordBox], heights: &[f64]) -> Furniture {
    let lines = lines(boxes);
    let pages = heights.len().max(1);
    let mut by_page: HashMap<usize, Vec<&Line>> = HashMap::new();
    for line in &lines {
        by_page.entry(line.page).or_default().push(line);
    }

    let mut candidates: Vec<&Line> = Vec::new();
    for (page, mut on_page) in by_page {
        let Some(&height) = heights.get(page) else {
            continue;
        };
        on_page.sort_by(|a, b| a.top.total_cmp(&b.top));
        let n = on_page.len();
        for (i, line) in on_page.into_iter().enumerate() {
            let near_top = i < EDGE_LINES && line.bottom < height * BAND;
            let near_foot = i + EDGE_LINES >= n && line.top > height * (1.0 - BAND);
            if near_top || near_foot {
                candidates.push(line);
            }
        }
    }

    let mut seen: HashMap<&str, BTreeSet<usize>> = HashMap::new();
    for line in &candidates {
        seen.entry(line.key.as_str()).or_default().insert(line.page);
    }
    let needed = (pages / 4).max(3);
    let mut out = Furniture::default();
    for line in candidates {
        if seen[line.key.as_str()].len() < needed {
            continue;
        }
        out.words.extend(line.words.clone());
        out.breaks.insert(line.words.start);
        out.breaks.insert(line.words.end);
    }
    out
}

struct Line {
    page: usize,
    top: f64,
    bottom: f64,
    /// The text with every run of digits one `#`, so "135/675" and "136/675" agree.
    key: String,
    words: std::ops::Range<usize>,
}

fn lines(boxes: &[WordBox]) -> Vec<Line> {
    let mut out: Vec<Line> = Vec::new();
    let mut y = f64::MIN;
    for b in boxes {
        let same = out
            .last()
            .is_some_and(|l| l.page == b.page && (b.rect.y - y).abs() <= b.rect.h.max(4.0));
        let text = mask(&b.text);
        match out.last_mut() {
            Some(line) if same => {
                line.top = line.top.min(b.rect.y);
                line.bottom = line.bottom.max(b.rect.y + b.rect.h);
                line.key.push(' ');
                line.key.push_str(&text);
                line.words.end = line.words.end.max(b.words.end);
            }
            _ => out.push(Line {
                page: b.page,
                top: b.rect.y,
                bottom: b.rect.y + b.rect.h,
                key: text,
                words: b.words.clone(),
            }),
        }
        y = b.rect.y;
    }
    out
}

fn mask(word: &str) -> String {
    let mut out = String::with_capacity(word.len());
    for c in word.to_lowercase().chars() {
        if c.is_ascii_digit() {
            if !out.ends_with('#') {
                out.push('#');
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Rect;

    fn word(text: &str, page: usize, y: f64, at: usize) -> WordBox {
        WordBox {
            page,
            block: 0,
            text: text.to_string(),
            rect: Rect {
                x: 50.0,
                y,
                w: 40.0,
                h: 10.0,
            },
            words: at..at + 1,
        }
    }

    #[test]
    fn a_folio_on_every_page_is_furniture_and_the_text_is_not() {
        let mut boxes = Vec::new();
        let mut at = 0;
        for page in 0..4 {
            boxes.push(word("Chapter", page, 20.0, at));
            boxes.push(word("Body", page, 400.0, at + 1));
            boxes.push(word(&format!("{}/675", page + 7), page, 770.0, at + 2));
            at += 3;
        }
        let found = find(&boxes, &[792.0; 4]);
        assert_eq!(found.words, BTreeSet::from([0, 2, 3, 5, 6, 8, 9, 11]));
        assert!(found.breaks.contains(&1) && found.breaks.contains(&3));
    }

    /// A line near an edge that does not repeat is the page's own text, however near the top.
    #[test]
    fn a_line_that_does_not_recur_is_text() {
        let boxes: Vec<WordBox> = ["Intro", "Scaling", "Caching", "Storage"]
            .iter()
            .enumerate()
            .map(|(p, w)| word(w, p, 20.0, p))
            .collect();
        assert!(find(&boxes, &[792.0; 4]).words.is_empty());
    }
}
