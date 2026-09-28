//! Line classification, and the word ranges that come out of it.
//!
//! The order of the tests below mirrors `page_text`'s order of removals, because the two
//! disagree otherwise on lines that satisfy more than one rule. `- - -` is the example worth
//! keeping in mind: `page_text` strips the bullet first and only then asks whether what is
//! left is a horizontal rule, so it is a list item holding no words — not a rule.

use crate::fence;
use crate::{Block, Kind};
use fancy_regex::{Captures, Regex};
use std::ops::Range;
use std::sync::OnceLock;
use tts_narrate::page::{page_text, words};
use tts_narrate::source::{caption_paragraph, is_horizontal_rule, strip_front_matter};
use tts_narrate::source::{HTML_COMMENT, SHORTCODE};

fn re(cell: &'static OnceLock<Regex>, pattern: &'static str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("literal pattern"))
}

macro_rules! pattern {
    ($name:ident, $p:expr) => {
        fn $name() -> &'static Regex {
            static CELL: OnceLock<Regex> = OnceLock::new();
            re(&CELL, $p)
        }
    };
}

pattern!(heading, r"^(#{1,6})\s*");
pattern!(bullet, r"^(\s*)(?:([-*+])|(\d+)\.)\s+");
pattern!(blockquote, r"^\s*>");
pattern!(table_row, r"^\s*\|.*\|\s*$");
pattern!(footnote_def, r"^\s*\[\^([^\]]+)\]:");
pattern!(image_only, r"^!\[([^\]]*)\]\(([^)]*)\)$");

fn matches(r: &Regex, line: &str) -> bool {
    r.is_match(line).unwrap_or(false)
}

/// Everything `page_text` does before it looks at a line, minus the deletion of fenced code —
/// which is lifted out instead, so the reader can show what the narration never speaks.
fn preprocess(md: &str) -> String {
    let text = strip_front_matter(md);
    let text = HTML_COMMENT.replace_all(&text, "").into_owned();
    SHORTCODE
        .replace_all(&text, |caps: &Captures<'_, str>| {
            caption_paragraph(caps.get(0).map_or("", |m| m.as_str()))
        })
        .into_owned()
}

/// What a line opens, when it is not a continuation of what came before.
enum Open {
    Heading(u8),
    Footnote(String),
    Figure(String, String),
    Rule,
    Item { ordered: bool, depth: u8 },
    Table,
    Quote,
    Prose,
}

fn classify(line: &str) -> Open {
    if let Ok(Some(c)) = heading().captures(line) {
        return Open::Heading(c.get(1).map_or(1, |m| m.as_str().len()) as u8);
    }
    if let Ok(Some(c)) = footnote_def().captures(line) {
        return Open::Footnote(c.get(1).map_or("", |m| m.as_str()).to_string());
    }
    if matches(table_row(), line) {
        return Open::Table;
    }
    if let Ok(Some(c)) = bullet().captures(line) {
        let indent = c.get(1).map_or(0, |m| m.as_str().len());
        return Open::Item {
            ordered: c.get(3).is_some(),
            // Two spaces per level is the common convention; a deeper guess would only be
            // wrong more confidently.
            depth: (indent / 2) as u8,
        };
    }
    if is_horizontal_rule(line) {
        return Open::Rule;
    }
    if matches(blockquote(), line) {
        return Open::Quote;
    }
    if let Ok(Some(c)) = image_only().captures(line) {
        return Open::Figure(
            c.get(1).map_or("", |m| m.as_str()).to_string(),
            c.get(2).map_or("", |m| m.as_str()).to_string(),
        );
    }
    Open::Prose
}

/// A block under construction: the lines it owns, and what it will become.
struct Pending {
    kind: Kind,
    lines: Vec<String>,
    /// Whether a following line of the same shape joins this block or starts a new one. A
    /// paragraph, a quote and a table run on; a heading and a list item do not.
    runs_on: bool,
}

struct Builder {
    blocks: Vec<Block>,
    pending: Option<Pending>,
    next_word: usize,
    /// Set once the document's opening `#` has been seen, so only the first is the title.
    titled: bool,
}

impl Builder {
    fn new() -> Self {
        Self {
            blocks: Vec::new(),
            pending: None,
            next_word: 0,
            titled: false,
        }
    }

    fn flush(&mut self) {
        let Some(p) = self.pending.take() else { return };
        let source = p.lines.join("\n");
        self.emit(p.kind, page_text(&source), source);
    }

    /// The one place a word range is assigned, so they cannot fail to be contiguous.
    fn emit(&mut self, kind: Kind, text: String, source: String) {
        let count = words(&text).len();
        let words_range: Range<usize> = self.next_word..self.next_word + count;
        self.next_word += count;
        self.blocks.push(Block {
            kind,
            text: text.trim().to_string(),
            source,
            words: words_range,
        });
    }

    fn push(&mut self, kind: Kind, line: &str, runs_on: bool) {
        match &mut self.pending {
            Some(p) if p.runs_on && p.kind == kind => {
                p.lines.push(line.to_string());
                return;
            }
            _ => {}
        }
        self.flush();
        self.pending = Some(Pending {
            kind,
            lines: vec![line.to_string()],
            runs_on,
        });
    }

    fn standalone(&mut self, kind: Kind, line: &str) {
        self.flush();
        let source = line.to_string();
        self.emit(kind, page_text(&source), source);
    }
}

/// Markdown to blocks, each carrying its range in `page::words(page_text(md))`.
pub fn layout(md: &str) -> Vec<Block> {
    let pre = preprocess(md);
    let raw: Vec<&str> = pre.split('\n').collect();
    let fences = fence::find(&raw);

    let mut b = Builder::new();
    let mut i = 0usize;
    let mut fence_at = 0usize;

    while i < raw.len() {
        if let Some(f) = fences.get(fence_at).filter(|f| f.at == i) {
            b.flush();
            // Deliberately wordless: `page_text` deletes fenced code, so a code block is shown
            // and never spoken, and it rides on its neighbour's slide rather than owning one.
            b.emit(
                Kind::Code {
                    lang: f.lang.clone(),
                },
                String::new(),
                f.body.clone(),
            );
            i += f.len;
            fence_at += 1;
            continue;
        }

        let line = raw[i].trim();
        i += 1;
        if line.is_empty() {
            b.flush();
            continue;
        }

        match classify(line) {
            Open::Heading(level) => {
                let kind = if level == 1 && !b.titled {
                    b.titled = true;
                    Kind::Title
                } else {
                    Kind::Heading(level)
                };
                b.standalone(kind, line);
            }
            Open::Footnote(label) => b.standalone(Kind::Footnote { label }, line),
            Open::Figure(alt, src) => b.standalone(Kind::Figure { alt, src }, line),
            Open::Rule => b.standalone(Kind::Rule, line),
            // Each item is its own block because narration gives each its own paragraph gap,
            // and a slide that splits a list must split it between items.
            Open::Item { ordered, depth } => b.standalone(Kind::ListItem { ordered, depth }, line),
            Open::Table => b.push(Kind::Table, line, true),
            Open::Quote => b.push(Kind::Quote, line, true),
            Open::Prose => b.push(Kind::Paragraph, line, true),
        }
    }
    b.flush();
    b.blocks
}
