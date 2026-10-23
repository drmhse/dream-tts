//! DOCX, read from `word/document.xml`.
//!
//! The heading semantics are explicit here — `w:pStyle w:val="Heading1"` — which is why this
//! reads the XML directly rather than going through a text extractor. macOS `textutil` will
//! extract the words, but it flattens every heading to a styled `<p>`, and losing the
//! headings loses the chapter boundaries the whole pipeline hangs off.

use crate::html;
use crate::markdown;
use crate::xml::{local, reader};
use crate::zipped::Archive;
use crate::Chapter;
use anyhow::Result;
use quick_xml::events::Event;
use std::collections::HashMap;
use std::path::Path;

pub fn chapters(path: &Path) -> Result<Vec<Chapter>> {
    let mut archive = Archive::open(path)?;
    let xml = archive.read("word/document.xml")?;
    let lists = archive
        .read_opt("word/numbering.xml")
        .map(|n| Numbering::read(&n))
        .unwrap_or_default();
    Ok(crate::split_markdown(&to_markdown_with(&xml, &lists)?))
}

/// The page a DOCX is set on, in points, as far as it says. Word fills what is missing from its
/// own defaults — the paper from the region, the size from the template — and so may a reader.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Page {
    /// Width and height.
    pub size: Option<(f64, f64)>,
    /// Top, right, bottom, left.
    pub margins: Option<[f64; 4]>,
    /// The body text's size.
    pub body: Option<f64>,
}

/// The last `w:sectPr` in the body is the document's own; earlier ones belong to sections the
/// reader does not keep apart. The body size is `Normal`'s, falling back to the defaults.
pub fn page(path: &Path) -> Result<Page> {
    let mut archive = Archive::open(path)?;
    let mut page = Page::default();
    if let Some(xml) = archive.read_opt("word/document.xml") {
        let mut reader = reader(&xml);
        let mut size = None;
        let mut margins = None;
        loop {
            match reader.read_event() {
                Ok(Event::Eof) | Err(_) => break,
                Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                    let twips = |key: &str| {
                        e.attributes()
                            .flatten()
                            .find(|a| local(a.key.as_ref()) == key)
                            .and_then(|a| String::from_utf8(a.value.to_vec()).ok())
                            .and_then(|v| v.parse::<f64>().ok())
                            .map(|v| v / 20.0)
                    };
                    match local(e.name().as_ref()).as_str() {
                        "pgsz" => size = twips("w").zip(twips("h")),
                        "pgmar" => {
                            margins = Some([
                                twips("top").unwrap_or(72.0).abs(),
                                twips("right").unwrap_or(72.0),
                                twips("bottom").unwrap_or(72.0).abs(),
                                twips("left").unwrap_or(72.0),
                            ])
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        page.size = size.filter(|(w, h)| *w >= 72.0 && *h >= 72.0);
        page.margins = margins;
    }
    if let Some(xml) = archive.read_opt("word/styles.xml") {
        page.body = body_size(&xml);
    }
    Ok(page)
}

/// `w:sz` is in half-points: in `Normal` if it says, otherwise in the document defaults.
fn body_size(xml: &str) -> Option<f64> {
    let mut reader = reader(xml);
    let (mut defaults, mut normal) = (None, None);
    let (mut in_defaults, mut in_normal) = (false, false);
    loop {
        match reader.read_event() {
            Ok(Event::Eof) | Err(_) => break,
            Ok(Event::Start(e)) => match local(e.name().as_ref()).as_str() {
                "docdefaults" => in_defaults = true,
                "style" => in_normal = val_of(&e, "styleid").as_deref() == Some("Normal"),
                _ => {}
            },
            Ok(Event::Empty(e)) if local(e.name().as_ref()) == "sz" => {
                let pt = val(&e).and_then(|v| v.parse::<f64>().ok()).map(|v| v / 2.0);
                if in_normal {
                    normal = pt;
                } else if in_defaults {
                    defaults = pt;
                }
            }
            Ok(Event::End(e)) => match local(e.name().as_ref()).as_str() {
                "docdefaults" => in_defaults = false,
                "style" => in_normal = false,
                _ => {}
            },
            _ => {}
        }
    }
    normal.or(defaults).filter(|pt| (6.0..=36.0).contains(pt))
}

fn val_of(e: &quick_xml::events::BytesStart, key: &str) -> Option<String> {
    e.attributes()
        .flatten()
        .find(|a| local(a.key.as_ref()) == key)
        .and_then(|a| String::from_utf8(a.value.to_vec()).ok())
}

/// `w:p` is a paragraph, `w:t` its text, and `w:pStyle` its style. Everything else — runs,
/// properties, revision marks — is structure this does not need.
pub fn to_markdown(xml: &str) -> Result<String> {
    to_markdown_with(xml, &Numbering::default())
}

/// Bold, italic and code runs come through as markdown, as they do from HTML. A run's
/// properties sit in its own `w:rPr`; the paragraph's `w:pPr` has one too, which styles the
/// paragraph mark and not the text, so it is ignored.
fn to_markdown_with(xml: &str, lists: &Numbering) -> Result<String> {
    let mut reader = reader(xml);

    let mut out: Vec<String> = Vec::new();
    let mut text = String::new();
    let mut run = String::new();
    let mut marks = Marks::default();
    let mut in_run = false;
    let mut in_props = false;
    let mut in_para_props = false;
    let mut style: Option<String> = None;
    let mut in_text = false;
    // Word writes a list as a paragraph carrying `w:numPr`, with no element that says "list";
    // without this every bullet becomes its own paragraph and loses the marker. Whether it is
    // numbered, and how deep, is in `numbering.xml` under the list's id and level.
    let mut numbered = false;
    let mut num_id: Option<String> = None;
    let mut level = 0usize;
    let mut grid = markdown::Grid::default();

    loop {
        let (event, empty) = match reader.read_event() {
            Ok(Event::Eof) | Err(_) => break,
            Ok(Event::Start(e)) => (e, false),
            Ok(Event::Empty(e)) => (e, true),
            Ok(Event::End(e)) => {
                match local(e.name().as_ref()).as_str() {
                    "t" => in_text = false,
                    "ppr" => in_para_props = false,
                    "rpr" => in_props = false,
                    "r" => {
                        marks.wrap(&mut text, &run);
                        run.clear();
                        in_run = false;
                    }
                    "tbl" => out.extend(grid.close()),
                    "tr" => grid.close_row(),
                    "tc" => grid.close_cell(),
                    "p" => {
                        if !grid.paragraph(&html::styled(&text)) {
                            let item = numbered.then(|| lists.item(num_id.as_deref(), level));
                            push_paragraph(&mut out, &text, style.as_deref(), item);
                        }
                        text.clear();
                        style = None;
                        numbered = false;
                        num_id = None;
                        level = 0;
                    }
                    _ => {}
                }
                continue;
            }
            Ok(Event::Text(t)) => {
                if in_text {
                    if let Ok(decoded) = t.unescape() {
                        run.push_str(&decoded);
                    }
                }
                continue;
            }
            Ok(_) => continue,
        };
        let name = local(event.name().as_ref());
        match name.as_str() {
            "t" if !empty => in_text = true,
            "r" if !empty => {
                in_run = true;
                marks = Marks::default();
            }
            "ppr" if !empty => in_para_props = true,
            "rpr" if !empty && in_run => in_props = true,
            "b" | "i" if in_props => {
                let on = !matches!(val(&event).as_deref(), Some("0" | "false" | "off" | "none"));
                if name == "b" {
                    marks.strong = on;
                } else {
                    marks.emphasis = on;
                }
            }
            // Character styles by name: Word's own, and pandoc's for inline code.
            "rstyle" if in_props => match val(&event).as_deref() {
                Some("Strong") => marks.strong = true,
                Some("Emphasis") => marks.emphasis = true,
                Some(s) if s.contains("Verbatim") || s.contains("Code") => marks.code = true,
                _ => {}
            },
            "pstyle" if in_para_props => style = val(&event),
            "numpr" => numbered = true,
            "numid" if in_para_props => num_id = val(&event),
            "ilvl" if in_para_props => {
                level = val(&event).and_then(|v| v.parse().ok()).unwrap_or(0);
            }
            // A soft line break inside a paragraph is a space to a narrator.
            "br" | "tab" => {
                if in_run {
                    run.push(' ');
                } else {
                    text.push(' ');
                }
            }
            "tbl" if !empty => grid.open(),
            "tr" if !empty => grid.open_row(),
            "tc" if !empty => grid.open_cell(),
            _ => {}
        }
    }
    push_paragraph(&mut out, &text, style.as_deref(), None);
    Ok(markdown::document(&out))
}

fn val(e: &quick_xml::events::BytesStart) -> Option<String> {
    val_of(e, "val")
}

/// One run's inline styles. Nested markdown emphasis is not kept, so the strongest wins.
#[derive(Default, Clone, Copy, PartialEq)]
struct Marks {
    strong: bool,
    emphasis: bool,
    code: bool,
}

impl Marks {
    fn mark(self) -> Option<html::Mark> {
        if self.code {
            Some(html::Mark::Code)
        } else if self.strong {
            Some(html::Mark::Strong)
        } else if self.emphasis {
            Some(html::Mark::Emphasis)
        } else {
            None
        }
    }

    /// Append a run, joined to the one before it when both carry the same style: Word splits
    /// runs mid-word for its own bookkeeping, and two marked halves of a word would each be
    /// refused as not sitting on a word boundary.
    fn wrap(self, text: &mut String, run: &str) {
        if run.is_empty() {
            return;
        }
        let Some(mark) = self.mark() else {
            text.push_str(run);
            return;
        };
        if text.ends_with(mark.close()) {
            text.pop();
        } else {
            text.push(mark.open());
        }
        text.push_str(run);
        text.push(mark.close());
    }
}

/// What a list's id and level say about its items: numbered, or bulleted.
#[derive(Default)]
struct Numbering {
    /// `w:num` id to its abstract definition.
    nums: HashMap<String, String>,
    /// Abstract definition and level to whether that level is numbered.
    ordered: HashMap<(String, usize), bool>,
}

impl Numbering {
    fn read(xml: &str) -> Self {
        let mut reader = reader(xml);
        let mut n = Numbering::default();
        let mut abstract_id: Option<String> = None;
        let mut num_id: Option<String> = None;
        let mut level: Option<usize> = None;
        loop {
            let e = match reader.read_event() {
                Ok(Event::Eof) | Err(_) => break,
                Ok(Event::Start(e)) | Ok(Event::Empty(e)) => e,
                Ok(Event::End(e)) => {
                    match local(e.name().as_ref()).as_str() {
                        "abstractnum" => abstract_id = None,
                        "num" => num_id = None,
                        "lvl" => level = None,
                        _ => {}
                    }
                    continue;
                }
                Ok(_) => continue,
            };
            let attr = |key: &str| {
                e.attributes()
                    .flatten()
                    .find(|a| local(a.key.as_ref()) == key)
                    .and_then(|a| String::from_utf8(a.value.to_vec()).ok())
            };
            match local(e.name().as_ref()).as_str() {
                "abstractnum" => abstract_id = attr("abstractnumid"),
                "num" => num_id = attr("numid"),
                "abstractnumid" => {
                    if let (Some(num), Some(target)) = (num_id.clone(), attr("val")) {
                        n.nums.insert(num, target);
                    }
                }
                "lvl" => level = attr("ilvl").and_then(|v| v.parse().ok()),
                "numfmt" => {
                    if let (Some(id), Some(level), Some(format)) =
                        (abstract_id.clone(), level, attr("val"))
                    {
                        n.ordered
                            .insert((id, level), !matches!(format.as_str(), "bullet" | "none"));
                    }
                }
                _ => {}
            }
        }
        n
    }

    fn item(&self, num_id: Option<&str>, level: usize) -> Item {
        let ordered = num_id
            .and_then(|id| self.nums.get(id))
            .and_then(|a| self.ordered.get(&(a.clone(), level)))
            .copied()
            .unwrap_or(false);
        Item {
            ordered,
            depth: level.min(5),
        }
    }
}

struct Item {
    ordered: bool,
    depth: usize,
}

/// `Heading1`, `heading 1`, `Title` and `Subtitle` all appear in real files, and Word's own
/// exports differ from LibreOffice's. Anything unrecognised is body text.
fn heading_level(style: &str) -> Option<usize> {
    let normalised: String = style
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '-' && *c != '_')
        .collect::<String>()
        .to_ascii_lowercase();
    if normalised == "title" {
        return Some(1);
    }
    if normalised == "subtitle" {
        return Some(2);
    }
    normalised
        .strip_prefix("heading")
        .and_then(|n| n.parse::<usize>().ok())
        .filter(|n| (1..=6).contains(n))
}

fn push_paragraph(out: &mut Vec<String>, text: &str, style: Option<&str>, item: Option<Item>) {
    let level = style.and_then(heading_level);
    // A heading is its own weight; marks inside one would only print as asterisks.
    let text = if level.is_some() {
        html::strip(text)
    } else {
        html::styled(text)
    };
    match item.filter(|_| level.is_none()) {
        Some(Item { ordered, depth }) if !text.trim().is_empty() => {
            let marker = if ordered { "1." } else { "-" };
            out.push(format!("{}{marker} {}", "  ".repeat(depth), text.trim()));
        }
        _ => {
            if let Some(block) = markdown::block(level, false, &text) {
                out.push(block);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = r#"<?xml version="1.0"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>
  <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Chapter One</w:t></w:r></w:p>
  <w:p><w:r><w:t>First </w:t></w:r><w:r><w:t>paragraph.</w:t></w:r></w:p>
  <w:p><w:pPr><w:pStyle w:val="heading 2"/></w:pPr><w:r><w:t>A Section</w:t></w:r></w:p>
  <w:p><w:pPr><w:numPr><w:ilvl w:val="0"/></w:numPr></w:pPr><w:r><w:t>An item</w:t></w:r></w:p>
  <w:p/>
</w:body></w:document>"#;

    #[test]
    fn styles_become_headings_and_runs_join() {
        let md = to_markdown(DOC).unwrap();
        assert_eq!(
            md,
            "# Chapter One\n\nFirst paragraph.\n\n## A Section\n\n- An item\n"
        );
    }

    #[test]
    fn heading_styles_are_recognised_in_every_spelling_seen_in_the_wild() {
        assert_eq!(heading_level("Heading1"), Some(1));
        assert_eq!(heading_level("heading 2"), Some(2));
        assert_eq!(heading_level("Heading-3"), Some(3));
        assert_eq!(heading_level("Title"), Some(1));
        assert_eq!(heading_level("Subtitle"), Some(2));
        assert_eq!(heading_level("BodyText"), None);
        assert_eq!(heading_level("Heading9"), None);
    }

    #[test]
    fn a_table_is_a_table() {
        let cell = |t: &str| format!("<w:tc><w:p><w:r><w:t>{t}</w:t></w:r></w:p></w:tc>");
        let row = |a: &str, b: &str| format!("<w:tr>{}{}</w:tr>", cell(a), cell(b));
        let xml = format!(
            "<w:p><w:r><w:t>Before</w:t></w:r></w:p><w:tbl>{}{}</w:tbl>",
            row("Env", "Host"),
            row("Prod", "go.example.com")
        );
        assert_eq!(
            to_markdown(&xml).unwrap(),
            "Before\n\n| Env | Host |\n| --- | --- |\n| Prod | go.example.com |\n"
        );
    }

    fn para(runs: &str) -> String {
        format!("<w:p><w:pPr><w:rPr><w:b/></w:rPr></w:pPr>{runs}</w:p>")
    }

    fn run(props: &str, text: &str) -> String {
        format!("<w:r><w:rPr>{props}</w:rPr><w:t xml:space=\"preserve\">{text}</w:t></w:r>")
    }

    /// Bold and italic runs keep their weight, as an HTML document's do. The bold on the
    /// paragraph mark styles the pilcrow, not the text, and is not carried.
    #[test]
    fn bold_and_italic_runs_become_markdown() {
        let xml = para(&format!(
            "{}{}{}{}{}",
            run("", "merge into "),
            run("<w:b />", "dev"),
            run("", ", then "),
            run("<w:i/><w:b w:val=\"0\"/>", "Deployment"),
            run("<w:rStyle w:val=\"VerbatimChar\"/>", " x "),
        ));
        assert_eq!(
            to_markdown(&xml).unwrap(),
            "merge into **dev**, then *Deployment* `x`\n"
        );
    }

    /// Word splits a run mid-word for its own bookkeeping; the halves are one bold word.
    #[test]
    fn runs_split_inside_a_word_are_joined() {
        let xml = para(&format!(
            "{}{}{}",
            run("", "the "),
            run("<w:b/>", "mas"),
            run("<w:b/>", "ter"),
        ));
        assert_eq!(to_markdown(&xml).unwrap(), "the **master**\n");
    }

    /// Bold on part of a word cannot be written in markdown without splitting the word, which
    /// would change the word count every highlight hangs on.
    #[test]
    fn bold_inside_a_word_is_left_plain() {
        let xml = para(&format!("{}{}", run("<w:b/>", "un"), run("", "believable")));
        assert_eq!(to_markdown(&xml).unwrap(), "unbelievable\n");
    }

    #[test]
    fn a_bold_heading_is_not_starred() {
        let xml = format!(
            "<w:p><w:pPr><w:pStyle w:val=\"Heading1\"/></w:pPr>{}</w:p>",
            run("<w:b/>", "Plan")
        );
        assert_eq!(to_markdown(&xml).unwrap(), "# Plan\n");
    }

    /// Bullets and numbers are told apart by `numbering.xml`, and a level is an indent.
    #[test]
    fn numbering_decides_bullet_or_number_and_depth() {
        let numbering = r#"<w:numbering xmlns:w="w">
  <w:abstractNum w:abstractNumId="7">
    <w:lvl w:ilvl="0"><w:numFmt w:val="decimal"/></w:lvl>
    <w:lvl w:ilvl="1"><w:numFmt w:val="bullet"/></w:lvl>
  </w:abstractNum>
  <w:num w:numId="3"><w:abstractNumId w:val="7"/></w:num>
</w:numbering>"#;
        let item = |level: u8, text: &str| {
            format!(
                "<w:p><w:pPr><w:numPr><w:ilvl w:val=\"{level}\"/><w:numId w:val=\"3\"/></w:numPr></w:pPr><w:r><w:t>{text}</w:t></w:r></w:p>"
            )
        };
        let xml = format!("{}{}{}", item(0, "One"), item(1, "Under"), item(0, "Two"));
        let md = to_markdown_with(&xml, &Numbering::read(numbering)).unwrap();
        assert_eq!(md, "1. One\n\n  - Under\n\n1. Two\n");
    }

    #[test]
    fn the_page_and_body_size_are_read_in_points() {
        let styles = r#"<w:styles xmlns:w="w"><w:docDefaults><w:rPrDefault><w:rPr><w:sz w:val="24"/></w:rPr></w:rPrDefault></w:docDefaults>
<w:style w:styleId="Heading1"><w:rPr><w:sz w:val="32"/></w:rPr></w:style></w:styles>"#;
        assert_eq!(body_size(styles), Some(12.0));
        let normal = styles.replace("Heading1", "Normal");
        assert_eq!(body_size(&normal), Some(16.0));

        let dir = std::env::temp_dir().join("tts-import-docx-page");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("page.docx");
        let mut zip = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
        let opts = zip::write::SimpleFileOptions::default();
        use std::io::Write;
        zip.start_file("word/document.xml", opts).unwrap();
        zip.write_all(br#"<w:document xmlns:w="w"><w:body><w:p/><w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1800" w:bottom="1440" w:left="1800"/></w:sectPr></w:body></w:document>"#).unwrap();
        zip.start_file("word/styles.xml", opts).unwrap();
        zip.write_all(styles.as_bytes()).unwrap();
        zip.finish().unwrap();
        let page = page(&path).unwrap();
        assert_eq!(page.size, Some((595.3, 841.9)));
        assert_eq!(page.margins, Some([72.0, 90.0, 72.0, 90.0]));
        assert_eq!(page.body, Some(12.0));
    }

    #[test]
    fn an_empty_paragraph_contributes_nothing() {
        let md = to_markdown("<w:p/><w:p><w:r><w:t>only</w:t></w:r></w:p>").unwrap();
        assert_eq!(md, "only\n");
    }
}
