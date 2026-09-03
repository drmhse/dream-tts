//! Python's `difflib.SequenceMatcher.get_matching_blocks`, reproduced.
//!
//! Not a generic LCS diff: the two algorithms disagree on real input, and the word map this
//! feeds is consumed by a player that highlights the wrong word when they do. So this is a
//! deliberate transliteration — same longest-match search, same tie-breaking (earliest `i`,
//! then earliest `j`), same recursive split, same adjacent-block merge.
//!
//! `autojunk` is off, as it must be: it discards the most common words in a long chapter as
//! noise, and those are exactly the ones that anchor the alignment.

use std::collections::HashMap;

/// `(i, j, size)` — `a[i..i+size] == b[j..j+size]`.
type Block = (usize, usize, usize);

pub struct Matcher<'a, T: Eq + std::hash::Hash> {
    a: &'a [T],
    b: &'a [T],
    /// Element -> every index it occupies in `b`.
    b2j: HashMap<&'a T, Vec<usize>>,
}

impl<'a, T: Eq + std::hash::Hash> Matcher<'a, T> {
    pub fn new(a: &'a [T], b: &'a [T]) -> Self {
        let mut b2j: HashMap<&T, Vec<usize>> = HashMap::new();
        for (j, item) in b.iter().enumerate() {
            b2j.entry(item).or_default().push(j);
        }
        Self { a, b, b2j }
    }

    /// The longest matching block in `a[alo..ahi]` against `b[blo..bhi]`, earliest first.
    fn longest_match(&self, alo: usize, ahi: usize, blo: usize, bhi: usize) -> Block {
        let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0usize);
        // j2len[j] = length of the longest match ending at a[i], b[j]. Rebuilt per i, which
        // is what keeps this linear in the number of (element, position) pairs.
        let mut j2len: HashMap<usize, usize> = HashMap::new();
        for i in alo..ahi {
            let mut newj2len: HashMap<usize, usize> = HashMap::new();
            if let Some(indices) = self.b2j.get(&self.a[i]) {
                for &j in indices {
                    if j < blo {
                        continue;
                    }
                    if j >= bhi {
                        break;
                    }
                    let k = j2len.get(&j.wrapping_sub(1)).copied().unwrap_or(0) + 1;
                    newj2len.insert(j, k);
                    if k > bestsize {
                        besti = i + 1 - k;
                        bestj = j + 1 - k;
                        bestsize = k;
                    }
                }
            }
            j2len = newj2len;
        }
        // Extend across elements that b2j never indexed. With junk disabled this cannot
        // extend the match, but the walk is kept so the block bounds match Python's for the
        // degenerate cases (empty inputs, zero-length matches).
        while besti > alo && bestj > blo && self.a[besti - 1] == self.b[bestj - 1] {
            besti -= 1;
            bestj -= 1;
            bestsize += 1;
        }
        while besti + bestsize < ahi
            && bestj + bestsize < bhi
            && self.a[besti + bestsize] == self.b[bestj + bestsize]
        {
            bestsize += 1;
        }
        (besti, bestj, bestsize)
    }

    pub fn matching_blocks(&self) -> Vec<Block> {
        let (la, lb) = (self.a.len(), self.b.len());
        let mut queue = vec![(0usize, la, 0usize, lb)];
        let mut blocks: Vec<Block> = Vec::new();
        while let Some((alo, ahi, blo, bhi)) = queue.pop() {
            let (i, j, k) = self.longest_match(alo, ahi, blo, bhi);
            if k == 0 {
                continue;
            }
            blocks.push((i, j, k));
            if alo < i && blo < j {
                queue.push((alo, i, blo, j));
            }
            if i + k < ahi && j + k < bhi {
                queue.push((i + k, ahi, j + k, bhi));
            }
        }
        blocks.sort_unstable();

        // Merge adjacent blocks, then terminate with the sentinel Python appends.
        let mut merged: Vec<Block> = Vec::with_capacity(blocks.len());
        let (mut i1, mut j1, mut k1) = (0usize, 0usize, 0usize);
        for (i2, j2, k2) in blocks {
            if i1 + k1 == i2 && j1 + k1 == j2 {
                k1 += k2;
            } else {
                if k1 > 0 {
                    merged.push((i1, j1, k1));
                }
                (i1, j1, k1) = (i2, j2, k2);
            }
        }
        if k1 > 0 {
            merged.push((i1, j1, k1));
        }
        merged.push((la, lb, 0));
        merged
    }
}

/// `spoken[i]` -> index into `page`, or -1.
pub fn align_tokens(spoken: &[String], page: &[String]) -> Vec<i64> {
    let matcher = Matcher::new(spoken, page);
    let mut out = vec![-1i64; spoken.len()];
    for (si, pi, size) in matcher.matching_blocks() {
        for k in 0..size {
            out[si + k] = (pi + k) as i64;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn identical_sequences_map_one_to_one() {
        let a = words("the quick brown fox");
        assert_eq!(align_tokens(&a, &a), vec![0, 1, 2, 3]);
    }

    #[test]
    fn an_insertion_does_not_shift_what_follows() {
        let spoken = words("a b spoken only c d");
        let page = words("a b c d");
        let m = align_tokens(&spoken, &page);
        assert_eq!(m[0], 0);
        assert_eq!(m[1], 1);
        assert_eq!(m[4], 2, "c must still map to c");
        assert_eq!(m[5], 3);
    }

    #[test]
    fn a_word_absent_from_the_page_is_unmapped() {
        let m = align_tokens(&words("x y"), &words("x"));
        assert_eq!(m, vec![0, -1]);
    }

    #[test]
    fn empty_inputs_are_not_a_panic() {
        assert!(align_tokens(&[], &words("a")).is_empty());
        assert_eq!(align_tokens(&words("a"), &[]), vec![-1]);
    }

    /// Repeated column headers are the case that broke a two-pointer walk with a 24-token
    /// lookahead: chapter 9 fell from 100% mapped to 47.8%.
    #[test]
    fn repeated_runs_stay_synchronised() {
        let page = words("alpha beta gamma delta epsilon");
        let spoken = words("name alpha value beta name gamma value delta epsilon");
        let m = align_tokens(&spoken, &page);
        assert_eq!(m[8], 4, "the tail must still anchor");
    }
}
