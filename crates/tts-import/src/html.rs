//! HTML to markdown, keeping only what a narrator needs.
//!
//! Not a general converter: the output feeds `tts-narrate`, which already handles emphasis,
//! links and entities. What matters here is the *structure* — headings, paragraphs, list
//! items and block boundaries — because that is what the chapter split and the paragraph
//! gaps are built on. Everything else becomes plain text.

use crate::markdown;
use crate::xml::{local, reader};
use crate::Chapter;
use anyhow::Result;
use quick_xml::events::{BytesStart, Event};

/// Elements whose content is never prose.
const SKIPPED: &[&str] = &["script", "style", "head", "noscript", "svg", "template"];

/// Elements that end the current block.
const BLOCKS: &[&str] = &[
    "p",
    "div",
    "section",
    "article",
    "br",
    "hr",
    "blockquote",
    "figcaption",
    "td",
    "th",
    "tr",
    "table",
    "dd",
    "dt",
    "dl",
    "main",
    "header",
    "footer",
    "aside",
    "nav",
];

pub fn to_markdown(html: &str) -> Result<String> {
    to_markdown_with(html, None)
}

/// As `to_markdown`, keeping the document's own images: a picture standing on its own becomes
/// a figure whose path is `images` joined to its `src`. `None` drops them, as a page whose
/// images live elsewhere must.
pub fn to_markdown_with(html: &str, images: Option<&str>) -> Result<String> {
    let html = bogus_comments(html);
    let mut rest: &str = &html;
    let mut reader = reader(rest);
    let mut c = Converter::new(images);

    // Broken markup is common in the wild, and a browser renders past it: on a parse error
    // reading resumes at the next tag, so one bad construct costs itself, not the rest of the
    // page. Each restart moves forward, so this ends.
    loop {
        match reader.read_event() {
            Ok(Event::Eof) => break,
            Err(_) => {
                let at = (reader.buffer_position() as usize).clamp(1, rest.len());
                let at = (at..rest.len())
                    .find(|&i| rest.is_char_boundary(i))
                    .unwrap_or(rest.len());
                match rest[at..].find('<') {
                    Some(next) => {
                        rest = &rest[at + next..];
                        reader = crate::xml::reader(rest);
                    }
                    None => break,
                }
            }
            Ok(Event::Start(e)) => c.start(&e, false),
            Ok(Event::Empty(e)) => c.start(&e, true),
            Ok(Event::End(e)) => c.end(&local(e.name().as_ref())),
            Ok(Event::Text(t)) => c.text(&decoded(&t)),
            Ok(Event::CData(t)) => c.text(&String::from_utf8_lossy(&t)),
            Ok(_) => {}
        }
    }
    Ok(c.finish())
}

/// The inline styles kept, written into the text as private-use markers while a block is
/// read, and resolved to markdown once it is whole: only then is it known whether a run
/// starts and ends on word boundaries, which is the only place markdown's `*` can sit
/// without splitting a word in two.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Mark {
    Emphasis,
    Strong,
    Code,
}

impl Mark {
    fn of(name: &str) -> Option<Self> {
        match name {
            "em" | "i" | "cite" | "dfn" | "var" => Some(Self::Emphasis),
            "strong" | "b" => Some(Self::Strong),
            "code" | "kbd" | "samp" | "tt" => Some(Self::Code),
            _ => None,
        }
    }

    pub(crate) fn open(self) -> char {
        match self {
            Self::Emphasis => '\u{E000}',
            Self::Strong => '\u{E002}',
            Self::Code => '\u{E004}',
        }
    }

    pub(crate) fn close(self) -> char {
        char::from_u32(self.open() as u32 + 1).expect("private use")
    }

    fn opened_by(c: char) -> Option<Self> {
        [Self::Emphasis, Self::Strong, Self::Code]
            .into_iter()
            .find(|m| m.open() == c)
    }

    fn delimiter(self) -> &'static str {
        match self {
            Self::Emphasis => "*",
            Self::Strong => "**",
            Self::Code => "`",
        }
    }
}

fn is_marker(c: char) -> bool {
    ('\u{E000}'..='\u{E005}').contains(&c)
}

/// A table read whole before it is written, since only then is it known to be one.
struct Table {
    rows: Vec<Vec<String>>,
    row: Vec<String>,
    cell: Option<String>,
    /// Tables inside this one, read as text in its cells.
    nested: usize,
}

/// Where a run of text goes while it is not part of the block.
enum Capture {
    /// A footnote reference, written `[^label]`: the id it points at, and its text.
    NoteRef { target: String, text: String },
    /// A link inside a note, dropped if it only repeats the note's own number.
    Backlink(String),
}

struct Note {
    element: String,
    depth: usize,
    label: Option<String>,
    written: bool,
}

struct Converter<'a> {
    images: Option<&'a str>,
    out: Vec<String>,
    line: String,
    skip_depth: usize,
    /// Inside an SVG, which is skipped but for the picture it wraps: a cover is often one.
    svg_depth: usize,
    heading: Option<usize>,
    list_item: bool,
    // Preformatted text keeps its lines and becomes a fence, which is shown and never spoken:
    // flattened into prose, a README's Rust was read out as "Fn main ( ), then Result <".
    pre: Option<(usize, String)>,
    /// Inline elements open, and the mark each began, if any: one style at a time is kept.
    inline: Vec<(String, Option<Mark>)>,
    quote_depth: usize,
    table: Option<Table>,
    capture: Option<(Capture, usize)>,
    /// Footnote labels by the id their reference points at.
    notes: std::collections::HashMap<String, String>,
    note: Option<Note>,
    /// A drop cap waiting for the rest of its word, which often sits in the next block.
    initial: Option<char>,
}

impl<'a> Converter<'a> {
    fn new(images: Option<&'a str>) -> Self {
        Self {
            images,
            out: Vec::new(),
            line: String::new(),
            skip_depth: 0,
            svg_depth: 0,
            heading: None,
            list_item: false,
            pre: None,
            inline: Vec::new(),
            quote_depth: 0,
            table: None,
            capture: None,
            notes: std::collections::HashMap::new(),
            note: None,
            initial: None,
        }
    }

    fn start(&mut self, e: &BytesStart, empty: bool) {
        let name = local(e.name().as_ref());
        if let Some((depth, code)) = self.pre.as_mut() {
            if name == "pre" && !empty {
                *depth += 1;
            } else if name == "br" {
                code.push('\n');
            }
            return;
        }
        if self.skip_depth > 0 {
            if self.svg_depth > 0 && name == "image" {
                if let Some(src) = attr(e, "href") {
                    self.figure(&src, "");
                }
            }
            if !empty && SKIPPED.contains(&name.as_str()) {
                self.skip_depth += 1;
                self.svg_depth += usize::from(name == "svg");
            }
            return;
        }
        if let Some(note) = self.note.as_mut() {
            if note.element == name && !empty {
                note.depth += 1;
            }
        }
        if let Some((_, depth)) = self.capture.as_mut() {
            if name == "a" && !empty {
                *depth += 1;
            }
            return;
        }
        if name == "pre" && !empty {
            self.flush();
            self.heading = None;
            self.list_item = false;
            self.pre = Some((1, String::new()));
        } else if SKIPPED.contains(&name.as_str()) {
            if !empty {
                self.skip_depth += 1;
                self.svg_depth += usize::from(name == "svg");
            }
        } else if name == "img" {
            let alt = attr(e, "alt").unwrap_or_default();
            let mut letters = alt.trim().chars();
            // A drop cap drawn as a picture is the first letter of its word.
            if let (Some(letter), None) = (letters.next(), letters.next()) {
                if letter.is_alphabetic() {
                    self.initial = Some(letter);
                    return;
                }
            }
            if let Some(src) = attr(e, "src") {
                let small = ["width", "height"].iter().all(|k| {
                    attr(e, k)
                        .and_then(|v| v.trim_end_matches("px").parse::<f64>().ok())
                        .is_some_and(|v| v < 40.0)
                });
                if !small {
                    self.figure(&src, &alt);
                }
            }
        } else if let Some(table) = self.table.as_mut() {
            match name.as_str() {
                "table" if !empty => table.nested += 1,
                _ if table.nested > 0 => {}
                "tr" => table.row.clear(),
                "td" | "th" if !empty => table.cell = Some(String::new()),
                _ if BLOCKS.contains(&name.as_str()) || heading_level(&name).is_some() => {
                    if let Some(cell) = table.cell.as_mut() {
                        push_text(cell, " ");
                    }
                }
                _ => self.inline_start(&name, empty),
            }
        } else if name == "table" && !empty {
            self.flush();
            self.table = Some(Table {
                rows: Vec::new(),
                row: Vec::new(),
                cell: None,
                nested: 0,
            });
        } else if name == "a" && !empty && has_type(e, "noteref") {
            let target = attr(e, "href")
                .and_then(|h| h.rsplit_once('#').map(|(_, id)| id.to_string()))
                .unwrap_or_default();
            self.capture = Some((
                Capture::NoteRef {
                    target,
                    text: String::new(),
                },
                1,
            ));
        } else if name == "a" && !empty && self.note.as_ref().is_some_and(|n| n.label.is_some()) {
            self.capture = Some((Capture::Backlink(String::new()), 1));
        } else if !empty && self.note.is_none() && is_note(e) {
            self.flush();
            let label = attr(e, "id").and_then(|id| self.notes.get(&id).cloned());
            self.note = Some(Note {
                element: name.clone(),
                depth: 1,
                label,
                written: false,
            });
            self.heading = None;
            self.list_item = false;
        } else if name == "blockquote" {
            self.flush();
            self.heading = None;
            self.list_item = false;
            if !empty {
                self.quote_depth += 1;
            }
        } else if name == "hr" {
            self.flush();
            self.heading = None;
            self.list_item = false;
            self.out.push("---".to_string());
        } else if let Some(level) = heading_level(&name) {
            self.flush();
            self.heading = Some(level);
            self.list_item = false;
        } else if name == "li" {
            self.flush();
            self.heading = None;
            self.list_item = true;
        } else if BLOCKS.contains(&name.as_str()) {
            self.flush();
            self.heading = None;
            self.list_item = false;
        } else {
            self.inline_start(&name, empty);
        }
    }

    fn inline_start(&mut self, name: &str, empty: bool) {
        let Some(mark) = Mark::of(name) else { return };
        if empty {
            return;
        }
        let free = self.inline.iter().all(|(_, m)| m.is_none());
        let kept = free.then_some(mark);
        if let Some(mark) = kept {
            self.target().push(mark.open());
        }
        self.inline.push((name.to_string(), kept));
    }

    fn end(&mut self, name: &str) {
        if let Some((depth, code)) = self.pre.as_mut() {
            if name == "pre" {
                *depth -= 1;
                if *depth == 0 {
                    if let Some(block) = fence(code) {
                        self.out.push(block);
                    }
                    self.pre = None;
                }
            }
            return;
        }
        if self.skip_depth > 0 {
            if SKIPPED.contains(&name) {
                self.skip_depth -= 1;
                if name == "svg" {
                    self.svg_depth = self.svg_depth.saturating_sub(1);
                }
            }
            return;
        }
        if name == "a" {
            if let Some((_, depth)) = self.capture.as_mut() {
                *depth -= 1;
                if *depth == 0 {
                    self.end_capture();
                }
                return;
            }
        }
        if self.capture.is_some() {
            return;
        }
        if let Some(table) = self.table.as_mut() {
            match name {
                "table" if table.nested > 0 => table.nested -= 1,
                _ if table.nested > 0 => {}
                "td" | "th" => {
                    if let Some(cell) = table.cell.take() {
                        table.row.push(styled(&cell).trim().to_string());
                    }
                }
                "tr" => {
                    if !table.row.is_empty() {
                        let row = std::mem::take(&mut table.row);
                        table.rows.push(row);
                    }
                }
                "table" => {
                    let table = self.table.take().expect("checked");
                    self.write_table(table);
                }
                _ => self.inline_end(name),
            }
            return;
        }
        let closes_note = self
            .note
            .as_mut()
            .filter(|n| n.element == name)
            .map(|n| {
                n.depth -= 1;
                n.depth == 0
            })
            .unwrap_or(false);
        if closes_note {
            self.flush();
            self.note = None;
            self.heading = None;
            self.list_item = false;
        } else if name == "blockquote" {
            self.flush();
            self.heading = None;
            self.list_item = false;
            self.quote_depth = self.quote_depth.saturating_sub(1);
        } else if heading_level(name).is_some() || name == "li" || BLOCKS.contains(&name) {
            self.flush();
            self.heading = None;
            self.list_item = false;
        } else {
            self.inline_end(name);
        }
    }

    fn inline_end(&mut self, name: &str) {
        if Mark::of(name).is_none() {
            return;
        }
        let Some(at) = self.inline.iter().rposition(|(n, _)| n == name) else {
            return;
        };
        let (_, mark) = self.inline.remove(at);
        if let Some(mark) = mark {
            self.target().push(mark.close());
        }
    }

    fn text(&mut self, text: &str) {
        if let Some((_, code)) = self.pre.as_mut() {
            code.push_str(text);
            return;
        }
        if self.skip_depth > 0 {
            return;
        }
        if let Some((Capture::NoteRef { text: held, .. } | Capture::Backlink(held), _)) =
            self.capture.as_mut()
        {
            held.push_str(text);
            return;
        }
        let held;
        let text = match self.initial.take() {
            Some(letter) if !text.trim().is_empty() => {
                held = format!("{letter}{}", text.trim_start());
                held.as_str()
            }
            Some(letter) => {
                self.initial = Some(letter);
                return;
            }
            None => text,
        };
        if let Some(table) = self.table.as_mut() {
            if table.nested == 0 && table.cell.is_none() {
                return;
            }
            if let Some(cell) = table.cell.as_mut() {
                push_text(cell, text);
            }
            return;
        }
        push_text(&mut self.line, text);
    }

    fn end_capture(&mut self) {
        let Some((capture, _)) = self.capture.take() else {
            return;
        };
        match capture {
            Capture::NoteRef { target, text } => {
                let label: String = text.chars().filter(|c| c.is_alphanumeric()).collect();
                if !label.is_empty() && label.chars().count() <= 6 && !target.is_empty() {
                    self.notes.insert(target, label.clone());
                    self.line.push_str(&format!("[^{label}]"));
                } else {
                    push_text(&mut self.line, &text);
                }
            }
            Capture::Backlink(text) => {
                let own = self.note.as_ref().and_then(|n| n.label.clone());
                let bare: String = text.chars().filter(|c| c.is_alphanumeric()).collect();
                let repeats = bare.is_empty() || own.as_deref() == Some(bare.as_str());
                if !repeats {
                    push_text(&mut self.line, &text);
                }
            }
        }
    }

    /// The buffer text is being written into: a table's cell, or the block.
    fn target(&mut self) -> &mut String {
        match self.table.as_mut().and_then(|t| t.cell.as_mut()) {
            Some(cell) => cell,
            None => &mut self.line,
        }
    }

    /// A picture that stands on its own; one in the middle of a sentence is left out, as an
    /// icon or a letter drawn as an image would break the line it sits in.
    fn figure(&mut self, src: &str, alt: &str) {
        let Some(base) = self.images else { return };
        if self.table.is_some() || self.capture.is_some() || !strip(&self.line).trim().is_empty() {
            return;
        }
        let Some(path) = image_path(base, src) else {
            return;
        };
        let alt: String = alt
            .chars()
            .map(|c| {
                if matches!(c, '[' | ']') || c.is_control() {
                    ' '
                } else {
                    c
                }
            })
            .collect();
        self.flush();
        self.out.push(format!(
            "![{}]({path})",
            alt.split_whitespace().collect::<Vec<_>>().join(" ")
        ));
    }

    fn write_table(&mut self, table: Table) {
        self.out.extend(markdown::table(table.rows));
    }

    fn flush(&mut self) {
        let open: Vec<Mark> = self.inline.iter().filter_map(|(_, m)| *m).collect();
        for mark in &open {
            self.line.push(mark.close());
        }
        let mut text = if self.heading.is_some() {
            strip(&self.line)
        } else {
            styled(&self.line)
        };
        if let Some(note) = self
            .note
            .as_mut()
            .filter(|n| !n.written && !text.trim().is_empty())
        {
            if let Some(label) = &note.label {
                text = format!("[^{label}]: {}", text.trim());
                note.written = true;
            }
            self.list_item = false;
        }
        if let Some(mut block) = markdown::block(self.heading, self.list_item, &text) {
            if self.quote_depth > 0 && self.heading.is_none() && !self.list_item {
                block = format!("> {block}");
            }
            self.out.push(block);
        }
        self.line.clear();
        // A style that runs on past the block's end carries into the next one.
        for mark in open {
            self.line.push(mark.open());
        }
    }

    fn finish(mut self) -> String {
        if let Some(letter) = self.initial.take() {
            self.line.push(letter);
        }
        if let Some(table) = self.table.take() {
            self.write_table(table);
        }
        self.flush();
        if let Some(block) = self.pre.take().and_then(|(_, code)| fence(&code)) {
            self.out.push(block);
        }
        markdown::document(&self.out)
    }
}

/// The text with every style marker dropped.
pub(crate) fn strip(text: &str) -> String {
    text.chars().filter(|c| !is_marker(*c)).collect()
}

/// Markers to markdown where a run sits on word boundaries, and to plain text where it does
/// not: `un<em>believ</em>able` written `un*believ*able` would be three words to the
/// tokeniser, and the word count is what every highlight hangs on.
pub(crate) fn styled(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let Some(mark) = Mark::opened_by(chars[i]) else {
            if !is_marker(chars[i]) {
                out.push(chars[i]);
            }
            i += 1;
            continue;
        };
        let Some(end) = (i + 1..chars.len()).find(|&j| chars[j] == mark.close()) else {
            i += 1;
            continue;
        };
        let inner: String = chars[i + 1..end]
            .iter()
            .filter(|c| !is_marker(**c))
            .collect();
        let core = inner.trim();
        let lead = inner.starts_with(char::is_whitespace);
        let trail = inner.ends_with(char::is_whitespace);
        let after = chars[end + 1..].iter().find(|c| !is_marker(**c)).copied();
        let joins =
            |c: char| c.is_alphanumeric() || matches!(c, '*' | '`' | '_' | '\'' | '\u{2019}');
        let forbidden = match mark {
            Mark::Code => core.contains('`'),
            _ => core.contains(['*', '`']) || core.starts_with('_') || core.ends_with('_'),
        };
        let fits = !core.is_empty()
            && !forbidden
            && (lead || out.chars().last().is_none_or(|c| !joins(c)))
            && (trail || after.is_none_or(|c| !joins(c)));
        if lead && !out.is_empty() && !out.ends_with(' ') {
            out.push(' ');
        }
        if fits {
            out.push_str(mark.delimiter());
            out.push_str(core);
            out.push_str(mark.delimiter());
        } else {
            out.push_str(core);
        }
        if trail {
            out.push(' ');
        }
        i = end + 1;
    }
    out
}

fn attr(e: &BytesStart, key: &str) -> Option<String> {
    e.attributes()
        .flatten()
        .find(|a| local(a.key.as_ref()) == key)
        .and_then(|a| a.unescape_value().ok().map(|v| v.into_owned()))
}

/// EPUB's `epub:type` and ARIA's `role` both say what a part of a book is.
fn has_type(e: &BytesStart, kind: &str) -> bool {
    ["type", "role"].iter().filter_map(|k| attr(e, k)).any(|v| {
        v.split_whitespace()
            .any(|t| t.trim_start_matches("doc-") == kind)
    })
}

fn is_note(e: &BytesStart) -> bool {
    ["footnote", "endnote", "rearnote"]
        .iter()
        .any(|k| has_type(e, k))
}

/// A picture's place inside the document, or `None` for one that lives elsewhere.
fn image_path(base: &str, src: &str) -> Option<String> {
    let src = src.split(['#', '?']).next()?.trim();
    let scheme = src
        .split_once(':')
        .is_some_and(|(s, _)| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric()));
    if src.is_empty() || scheme || src.starts_with("//") {
        return None;
    }
    let path = if base.is_empty() {
        src.to_string()
    } else {
        crate::epub::join(base, src)
    };
    // A markdown image's target ends at the first `)`, so the brackets and the space are
    // escaped; `resource` reads them back.
    Some(
        path.replace('%', "%25")
            .replace('(', "%28")
            .replace(')', "%29")
            .replace(' ', "%20"),
    )
}

/// A bare `&`, as in "Q&A", is not an entity; a browser shows it as written, and so does this.
fn decoded(text: &quick_xml::events::BytesText) -> String {
    match text.unescape() {
        Ok(t) => t.into_owned(),
        Err(_) => lenient(&String::from_utf8_lossy(text)),
    }
}

fn lenient(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        rest = &rest[at..];
        let entity = rest[1..]
            .find(';')
            .filter(|&end| end <= 32)
            .and_then(|end| Some((resolve(&rest[1..=end])?, end + 2)));
        match entity {
            Some((text, len)) => {
                out.push_str(&text);
                rest = &rest[len..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

fn resolve(name: &str) -> Option<String> {
    if let Some(number) = name.strip_prefix('#') {
        let code = match number.strip_prefix(['x', 'X']) {
            Some(hex) => u32::from_str_radix(hex, 16).ok()?,
            None => number.parse().ok()?,
        };
        return char::from_u32(code).map(String::from);
    }
    quick_xml::escape::resolve_predefined_entity(name).map(String::from)
}

/// `<?…>` outside XML is a comment that ends at the first `>`, and Lit writes `<?>` between
/// template parts; quick-xml wants `?>` and read an MDN page as ending at its first one.
fn bogus_comments(html: &str) -> std::borrow::Cow<'_, str> {
    if !html.contains("<?") {
        return html.into();
    }
    let mut out = String::with_capacity(html.len());
    let mut rest = html;
    while let Some(at) = rest.find("<?") {
        out.push_str(&rest[..at]);
        let tail = &rest[at..];
        let end = tail.find('>').map_or(tail.len(), |i| i + 1);
        if tail[..end].starts_with("<?xml") {
            out.push_str(&tail[..end]);
        }
        rest = &tail[end..];
    }
    out.push_str(rest);
    out.into()
}

fn fence(code: &str) -> Option<String> {
    let code = code.trim_matches('\n').trim_end();
    if code.trim().is_empty() {
        return None;
    }
    let mut marks = "```".to_string();
    while code.contains(&marks) {
        marks.push('`');
    }
    Some(format!("{marks}\n{code}\n{marks}"))
}

pub fn chapters(html: &str) -> Result<Vec<Chapter>> {
    Ok(crate::split_markdown(&to_markdown(html)?))
}

/// A page on disk, whose pictures are named relative to it.
pub fn chapters_beside(html: &str) -> Result<Vec<Chapter>> {
    Ok(crate::split_markdown(&to_markdown_with(html, Some(""))?))
}

fn heading_level(name: &str) -> Option<usize> {
    let mut chars = name.chars();
    if chars.next() != Some('h') {
        return None;
    }
    let rest: String = chars.collect();
    rest.parse::<usize>().ok().filter(|n| (1..=6).contains(n))
}

/// Append text, collapsing whitespace across the tag boundaries it crosses. A space is
/// kept only where the markup had one: "<b>thinking</b>." is "thinking.", not "thinking .".
fn push_text(line: &mut String, text: &str) {
    for (i, word) in text.split_whitespace().enumerate() {
        let spaced = i > 0 || text.starts_with(char::is_whitespace);
        if spaced && !line.is_empty() && !line.ends_with(' ') {
            line.push(' ');
        }
        line.push_str(word);
    }
    if text.ends_with(char::is_whitespace) && !line.is_empty() && !line.ends_with(' ') {
        line.push(' ');
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_entities_and_bare_ampersands_read_as_a_browser_shows_them() {
        let md = to_markdown(
            "<p>It&rsquo;s &ldquo;Q&A&rdquo; &mdash; AT&T &amp; co &#8212; &#x2014;</p>",
        )
        .unwrap();
        assert_eq!(
            md.trim(),
            "It\u{2019}s \u{201c}Q&A\u{201d} \u{2014} AT&T & co \u{2014} \u{2014}"
        );
    }

    #[test]
    fn headings_become_hashes() {
        let md = to_markdown("<h1>Title</h1><p>Prose.</p><h2>Section</h2><p>More.</p>").unwrap();
        assert_eq!(md, "# Title\n\nProse.\n\n## Section\n\nMore.\n");
    }

    #[test]
    fn script_and_style_contribute_nothing() {
        let md = to_markdown("<style>p{color:red}</style><script>x=1</script><p>Only this.</p>")
            .unwrap();
        assert_eq!(md, "Only this.\n");
    }

    #[test]
    fn a_sentence_split_across_tags_keeps_its_space() {
        let md = to_markdown("<p>one <em>two</em> three</p>").unwrap();
        assert_eq!(md, "one *two* three\n");
    }

    #[test]
    fn list_items_become_bullets() {
        let md = to_markdown("<ul><li>first</li><li>second</li></ul>").unwrap();
        assert_eq!(md, "- first\n\n- second\n");
    }

    #[test]
    fn namespaced_and_upper_case_tags_are_recognised() {
        let md = to_markdown("<h:H1 xmlns:h='x'>Title</h:H1><P>Prose.</P>").unwrap();
        assert_eq!(md, "# Title\n\nProse.\n");
    }

    #[test]
    fn malformed_markup_yields_what_it_can() {
        let md = to_markdown("<p>kept<p>also kept<span>and this").unwrap();
        assert!(md.contains("kept"), "{md:?}");
    }

    #[test]
    fn punctuation_after_a_tag_is_not_spaced_off() {
        let md =
            to_markdown("<p><strong>More thinking</strong>. And <a>a link</a>, then</p>").unwrap();
        assert_eq!(md, "**More thinking**. And a link, then\n");
    }

    #[test]
    fn a_lit_marker_does_not_end_the_page() {
        let md = to_markdown("<p>one</p><?><p>two</p><?lit$1$><p>three</p>").unwrap();
        assert_eq!(md, "one\n\ntwo\n\nthree\n");
    }

    #[test]
    fn preformatted_text_becomes_a_fence() {
        let md = to_markdown(
            "<p>Run:</p><div class=\"highlight\"><pre><span>fn</span> main() {\n    go();\n}</pre></div><p>Done.</p>",
        )
        .unwrap();
        assert_eq!(md, "Run:\n\n```\nfn main() {\n    go();\n}\n```\n\nDone.\n");
    }

    #[test]
    fn emphasis_inside_a_word_is_left_plain() {
        let md = to_markdown("<p>un<em>believ</em>able and <b>bold</b>ly</p>").unwrap();
        assert_eq!(md, "unbelievable and boldly\n");
    }

    #[test]
    fn spaces_inside_a_style_move_outside_it() {
        let md = to_markdown("<p>a<em> b </em>c <i></i>d</p>").unwrap();
        assert_eq!(md, "a *b* c d\n");
    }

    #[test]
    fn one_style_at_a_time_is_kept() {
        let md =
            to_markdown("<p><strong>all <em>of</em> it</strong> and <code>x*y</code></p>").unwrap();
        assert_eq!(md, "**all of it** and `x*y`\n");
    }

    #[test]
    fn a_heading_keeps_no_markers() {
        let md = to_markdown("<h2>The <em>Pequod</em></h2>").unwrap();
        assert_eq!(md, "## The Pequod\n");
    }

    #[test]
    fn a_style_across_paragraphs_carries_on() {
        let md = to_markdown("<p><i>one</p><p>two</i> three</p>").unwrap();
        assert_eq!(md, "*one*\n\n*two* three\n");
    }

    #[test]
    fn a_picture_on_its_own_is_a_figure_named_inside_the_book() {
        let html = "<p><img src=\"../images/whale (1).jpg\" alt=\"The [white] whale\"/></p>\
                    <p>Text <img src=\"icon.png\"/> on.</p>\
                    <p><img src=\"https://x.y/a.png\"/></p>";
        let md = to_markdown_with(html, Some("OEBPS/text")).unwrap();
        assert_eq!(
            md,
            "![The white whale](OEBPS/images/whale%20%281%29.jpg)\n\nText on.\n"
        );
        assert_eq!(to_markdown(html).unwrap(), "Text on.\n");
    }

    #[test]
    fn a_drop_cap_drawn_as_a_picture_is_its_letter() {
        let md =
            to_markdown_with("<p><img src=\"a.png\" alt=\"A\"/>LICE was</p>", Some("")).unwrap();
        assert_eq!(md, "ALICE was\n");
        let md =
            to_markdown("<div><img src=\"a.png\" alt=\"A\"/></div><div>LICE was</div>").unwrap();
        assert_eq!(md, "ALICE was\n");
    }

    #[test]
    fn a_tiny_picture_is_left_out() {
        let md = to_markdown_with(
            "<p><img src=\"dot.png\" width=\"8\" height=\"8\"/></p>",
            Some(""),
        )
        .unwrap();
        assert_eq!(md, "\n");
    }

    #[test]
    fn a_cover_wrapped_in_svg_is_its_picture() {
        let html = "<svg xmlns:xlink=\"http://www.w3.org/1999/xlink\"><image xlink:href=\"cover.jpg\"/></svg>";
        assert_eq!(
            to_markdown_with(html, Some("OEBPS")).unwrap(),
            "![](OEBPS/cover.jpg)\n"
        );
    }

    #[test]
    fn a_quotation_is_quoted_paragraph_by_paragraph() {
        let md = to_markdown(
            "<p>He said:</p><blockquote><p>One.</p><p>Two.</p></blockquote><p>After.</p>",
        )
        .unwrap();
        assert_eq!(md, "He said:\n\n> One.\n\n> Two.\n\nAfter.\n");
    }

    #[test]
    fn a_data_table_is_a_table() {
        let md = to_markdown(
            "<table><tr><th>Name</th><th>Tons</th></tr><tr><td>Pequod</td><td><b>3</b>00</td></tr>\
             <tr><td>A|B</td></tr></table>",
        )
        .unwrap();
        assert_eq!(
            md,
            "| Name | Tons |\n| --- | --- |\n| Pequod | 300 |\n| A\u{2223}B |  |\n"
        );
    }

    #[test]
    fn a_layout_table_reads_as_its_cells() {
        let md = to_markdown(
            "<table><tr><td><p>Only</p><p>column</p></td></tr><tr><td>here</td></tr></table>",
        )
        .unwrap();
        assert_eq!(md, "Only column\n\nhere\n");
    }

    #[test]
    fn a_rule_is_a_scene_break() {
        let md = to_markdown("<p>End.</p><hr/><p>Later.</p>").unwrap();
        assert_eq!(md, "End.\n\n---\n\nLater.\n");
    }

    #[test]
    fn footnotes_are_linked_to_their_references() {
        let html = "<p>A claim.<a epub:type=\"noteref\" href=\"notes.xhtml#n1\">[1]</a> More.</p>\
                    <aside epub:type=\"footnote\" id=\"n1\"><p><a href=\"#r1\">1</a> The source.</p><p>Also this.</p></aside>";
        let md = to_markdown(html).unwrap();
        assert_eq!(
            md,
            "A claim.[^1] More.\n\n[^1]: The source.\n\nAlso this.\n"
        );
    }

    /// The point of every rule above: the page counts the same words, styled or not.
    #[test]
    fn styling_never_changes_the_words() {
        let html = "<h1>Ch<em>apter</em> 1</h1><p>It was <i>the</i> best of <b>times</b>, it\
                    was the <code>worst</code> of times.</p><blockquote><p>Call me <em>Ishmael</em>.</p>\
                    </blockquote><table><tr><th>a b</th><th>c</th></tr><tr><td><i>d</i></td><td>e f</td></tr></table>\
                    <p>un<em>sty</em>led and <strong> spaced </strong> out</p><hr/><p>Fin.</p>";
        let words = |md: &str| -> Vec<String> {
            let doc = tts_doc::Document::from_markdown(md);
            let text: Vec<&str> = doc.blocks.iter().map(|b| b.text.as_str()).collect();
            let count = doc.blocks.last().map_or(0, |b| b.words.end);
            let words: Vec<String> = text
                .join(" ")
                .split(|c: char| !c.is_ascii_alphanumeric())
                .filter(|w| !w.is_empty())
                .map(str::to_string)
                .collect();
            assert_eq!(words.len(), count, "{md}");
            words
        };
        let styled = to_markdown(html).unwrap();
        let plain: String = styled
            .chars()
            .filter(|c| !matches!(c, '*' | '`' | '>' | '|'))
            .collect();
        assert_eq!(words(&styled), words(&plain));
        assert!(words(&styled).len() > 25, "{styled}");
    }

    #[test]
    fn headings_drive_the_chapter_split() {
        let chapters = chapters("<h1>One</h1><p>alpha</p><h1>Two</h1><p>beta</p>").unwrap();
        assert_eq!(chapters.len(), 2);
        assert_eq!(chapters[1].title.as_deref(), Some("Two"));
    }
}
