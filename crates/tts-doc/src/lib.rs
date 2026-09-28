//! The document model both DreamTTS and DreamReader read: blocks that carry word ranges,
//! and the geometry that places those words on a page.
//!
//! Every importer ends here, whatever the format, and every consumer — narration, the
//! reader's typesetter, a PDF shown as itself — starts here. Markdown is one way in.
//!
//! `dream-tts` flattens a document to one line of prose and counts words across the whole of
//! it: `manifest.transcript.words[].pageWordIndex` is an index into
//! `page::words(page_text(md))`. That index is what carries a word's *time*. To also carry a
//! word's *place* — which paragraph it is in, and later which rectangle on a typeset page —
//! something has to say which block each index falls in.
//!
//! That is all this crate does, and the whole of its correctness is one invariant:
//!
//! ```text
//! layout(md).iter().flat_map(|b| words(&b.text))  ==  words(&page_text(md))
//! ```
//!
//! The cleaning rules are not reimplemented here. Each block's text comes from calling
//! `page_text` on that block's own source lines, which is exact because every rule in
//! `page_text` is per-line with no lookbehind across lines, and lines are joined with a space
//! that cannot merge two tokens. Reimplementing them would drift — and a tokeniser that
//! drifts by one rule shifts every highlight in a book with no statistic showing it.

pub mod geometry;
pub mod pages;
pub mod split;

use std::ops::Range;

mod fence;
mod scan;

pub use scan::layout;

/// What a block is, for the typesetter and for how it is highlighted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The document's opening `#`. Set apart because it is the only heading a title page uses.
    Title,
    /// `##` … `######`, level 2-6.
    Heading(u8),
    Paragraph,
    ListItem {
        ordered: bool,
        depth: u8,
    },
    Quote,
    /// Shown, never spoken: `page_text` deletes fenced code, so this owns no words.
    Code {
        lang: Option<String>,
    },
    /// One block per contiguous run of `|` rows: a table narrates as one shape, and the
    /// highlight moves through it chunk by chunk like any other block. It may flow across
    /// pages — the only block that may — and the plan pages its highlights individually.
    Table,
    /// `![alt](src)` alone on a line. Also wordless — `page_text` deletes the image.
    Figure {
        alt: String,
        src: String,
    },
    /// `---`. Kept so the typesetter can draw it; contributes nothing to speech or to words.
    Rule,
    Footnote {
        label: String,
    },
    /// A running head, folio or print footer on a PDF's page: shown where it is, never said.
    Furniture,
    /// Code on a PDF's page, found by its monospace: shown, never said, like `Code`.
    Listing,
}

impl Kind {
    /// Whether this block can ever be spoken. A wordless block is displayed and rides along
    /// with its neighbour's slide rather than owning one.
    pub fn is_narrated(&self) -> bool {
        !matches!(
            self,
            Kind::Code { .. } | Kind::Figure { .. } | Kind::Rule | Kind::Furniture | Kind::Listing
        )
    }
}

#[derive(Clone, Debug)]
pub struct Block {
    pub kind: Kind,
    /// Rendered prose, cleaned exactly as `page_text` cleans it.
    pub text: String,
    /// The block's own markdown, for the typesetter. Kept verbatim: the typesetter wants the
    /// emphasis and the code spans that `text` has thrown away.
    pub source: String,
    /// Into `page::words(page_text(md))`. Empty for a block that owns no words.
    pub words: Range<usize>,
}

/// A whole document in blocks: what an importer produces and every consumer reads.
#[derive(Clone, Debug, Default)]
pub struct Document {
    pub title: Option<String>,
    /// BCP 47, when the source says. Narration and the lexicon choose by it.
    pub language: Option<String>,
    pub blocks: Vec<Block>,
    /// The text the blocks' word ranges count in: `words(page_text(markdown))` is the stream
    /// every range indexes, so an importer that builds blocks any other way still writes this.
    pub markdown: String,
}

impl Document {
    pub fn from_markdown(markdown: &str) -> Self {
        Self {
            title: markdown
                .lines()
                .find_map(|l| l.strip_prefix("# ").map(str::trim))
                .filter(|t| !t.is_empty())
                .map(str::to_string),
            language: None,
            blocks: layout(markdown),
            markdown: markdown.to_string(),
        }
    }
}
