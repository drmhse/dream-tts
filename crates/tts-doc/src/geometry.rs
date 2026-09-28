//! Where a word sits on a page: the one shape every page source — a typesetter, a PDF, an OCR
//! pass — answers in, so nothing downstream learns which one it came from.

use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

#[derive(Clone, Debug)]
pub struct WordBox {
    /// Zero-based page, which is also the slide index.
    pub page: usize,
    /// Index into the blocks the document was emitted from.
    pub block: usize,
    /// The word as laid out.
    pub text: String,
    pub rect: Rect,
    /// The canonical words this box covers, as indices into
    /// `page::words(page_text(md))` — the same index a manifest's `pageWordIndex` counts in.
    ///
    /// A *range*, not an index, because one laid-out word is not one canonical word: the
    /// canonical tokeniser matches `[A-Za-z0-9]+`, so `1.7` on the page is `1` and `7` in the
    /// stream. Empty means the box is on the page and can never be highlighted — a link's URL
    /// has the opposite problem and shows up as canonical words with no box.
    pub words: Range<usize>,
}
