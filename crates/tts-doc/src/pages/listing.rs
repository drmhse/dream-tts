//! Code on a PDF's page: shown, never said, as a fenced block is in markdown.
//!
//! A PDF names no fonts here, but monospace gives itself away in the geometry: every word is
//! as wide as its character count says. Proportional type varies with the letters — `ill`
//! against `mmm` — and a prose line never holds that ratio across its words.

use super::furniture::Furniture;
use crate::geometry::WordBox;

/// Largest spread of width-per-character across a line that is still one advance. Measured on
/// alphanumeric words only: extraction drops `_` and punctuation from a box's text but not from
/// its width. At 0.04, 11% of a code-heavy book's lines and 0.4% of a prose book's.
const SPREAD: f64 = 0.04;

/// Words of three letters or more a line needs before its spread means anything. Three passed
/// short prose lines by chance: "IDs must be unique." went silent.
const SAMPLE: usize = 4;

/// Lines that must pass on their own before a page is taken to hold code at all.
const LINES: usize = 2;

struct Line {
    range: std::ops::Range<usize>,
    page: usize,
    /// Width per character, and the character count, of each alphanumeric word.
    ratios: Vec<(f64, usize)>,
}

impl Line {
    fn strict(&self) -> bool {
        self.ratios.iter().filter(|r| r.1 >= 3).count() >= SAMPLE
            && spread(&self.ratios.iter().map(|r| r.0).collect::<Vec<_>>()) <= SPREAD
    }
}

pub fn find(boxes: &[WordBox]) -> Furniture {
    let lines = lines(boxes);
    let mut strict_on: std::collections::HashMap<usize, Vec<f64>> = Default::default();
    for line in lines.iter().filter(|l| l.strict()) {
        let ratios: Vec<f64> = line.ratios.iter().map(|r| r.0).collect();
        strict_on
            .entry(line.page)
            .or_default()
            .push(median(&ratios));
    }
    let mut code: Vec<bool> = lines
        .iter()
        .map(|l| l.strict() && strict_on[&l.page].len() >= LINES)
        .collect();
    // Outward from lines that passed alone, through neighbours too short to judge — `let x`,
    // a lone `}` — whose words keep the page's advance or that have none to measure.
    let keeps = |line: &Line| {
        line.ratios.is_empty()
            || strict_on.get(&line.page).is_some_and(|a| {
                a.iter().any(|adv| {
                    line.ratios
                        .iter()
                        .all(|r| (r.0 - adv).abs() / adv <= SPREAD)
                })
            })
    };
    loop {
        let mut grew = false;
        for i in 0..lines.len() {
            if code[i] || !keeps(&lines[i]) {
                continue;
            }
            let beside = |j: usize| code[j] && lines[j].page == lines[i].page;
            if (i > 0 && beside(i - 1)) || (i + 1 < lines.len() && beside(i + 1)) {
                code[i] = true;
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }

    let mut out = Furniture::default();
    for (line, is_code) in lines.iter().zip(code) {
        if !is_code {
            continue;
        }
        for b in &boxes[line.range.clone()] {
            out.words.extend(b.words.clone());
        }
        out.breaks.insert(boxes[line.range.start].words.start);
        out.breaks.insert(boxes[line.range.end - 1].words.end);
    }
    out
}

fn lines(boxes: &[WordBox]) -> Vec<Line> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < boxes.len() {
        let mut end = start + 1;
        while end < boxes.len()
            && boxes[end].page == boxes[start].page
            && (boxes[end].rect.y - boxes[start].rect.y).abs() <= boxes[start].rect.h.max(4.0)
        {
            end += 1;
        }
        let ratios = boxes[start..end]
            .iter()
            .map(|b| (b, b.text.chars().count()))
            .filter(|(b, n)| *n >= 2 && b.text.chars().all(char::is_alphanumeric))
            .map(|(b, n)| (b.rect.w / n as f64, n))
            .collect();
        out.push(Line {
            range: start..end,
            page: boxes[start].page,
            ratios,
        });
        start = end;
    }
    out
}

fn spread(ratios: &[f64]) -> f64 {
    let lo = ratios.iter().copied().fold(f64::MAX, f64::min);
    let hi = ratios.iter().copied().fold(f64::MIN, f64::max);
    if lo > 0.0 {
        (hi - lo) / lo
    } else {
        f64::MAX
    }
}

fn median(ratios: &[f64]) -> f64 {
    let mut sorted = ratios.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Rect;

    fn line(words: &[(&str, f64)], y: f64, from: usize) -> Vec<WordBox> {
        let mut x = 50.0;
        words
            .iter()
            .enumerate()
            .map(|(i, (text, w))| {
                let b = WordBox {
                    page: 0,
                    block: 0,
                    text: text.to_string(),
                    rect: Rect {
                        x,
                        y,
                        w: *w,
                        h: 10.0,
                    },
                    words: from + i..from + i + 1,
                };
                x += w + 5.0;
                b
            })
            .collect()
    }

    #[test]
    fn a_line_of_one_advance_is_code_and_prose_is_not() {
        let code = [("let", 18.0), ("opt", 18.0), ("Some", 24.0), ("None", 24.0)];
        let mut boxes = line(&code, 0.0, 0);
        // Too few words to judge alone, and it keeps the advance: code.
        boxes.extend(line(&[("fn", 12.0)], 20.0, 4));
        // Nothing to judge, between two lines of code: code.
        boxes.extend(line(&[("}", 6.0)], 40.0, 5));
        boxes.extend(line(&code, 60.0, 6));
        boxes.extend(line(
            &[("the", 16.0), ("minimum", 44.0), ("will", 15.0)],
            80.0,
            10,
        ));
        // One prose word, off the advance: prose.
        boxes.extend(line(&[("wrote", 27.0)], 100.0, 13));
        let found = find(&boxes);
        assert_eq!(found.words, (0..10).collect());
    }

    /// One line passing by chance is not a listing.
    #[test]
    fn a_single_even_line_is_prose() {
        let boxes = line(
            &[
                ("IDs", 18.0),
                ("must", 24.0),
                ("unique", 36.0),
                ("keys", 24.0),
            ],
            0.0,
            0,
        );
        assert!(find(&boxes).words.is_empty());
    }
}
