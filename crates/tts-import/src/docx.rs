//! DOCX, read from `word/document.xml`.
//!
//! The heading semantics are explicit here — `w:pStyle w:val="Heading1"` — which is why this
//! reads the XML directly rather than going through a text extractor. macOS `textutil` will
//! extract the words, but it flattens every heading to a styled `<p>`, and losing the
//! headings loses the chapter boundaries the whole pipeline hangs off.

use crate::markdown;
use crate::xml::{local, reader};
use crate::zipped::Archive;
use crate::Chapter;
use anyhow::Result;
use quick_xml::events::Event;
use std::path::Path;

pub fn chapters(path: &Path) -> Result<Vec<Chapter>> {
    let mut archive = Archive::open(path)?;
    let xml = archive.read("word/document.xml")?;
    Ok(crate::split_markdown(&to_markdown(&xml)?))
}

/// `w:p` is a paragraph, `w:t` its text, and `w:pStyle` its style. Everything else — runs,
/// properties, revision marks — is structure this does not need.
pub fn to_markdown(xml: &str) -> Result<String> {
    let mut reader = reader(xml);

    let mut out: Vec<String> = Vec::new();
    let mut text = String::new();
    let mut style: Option<String> = None;
    let mut in_text = false;
    // Word writes a numbered list as a paragraph carrying `w:numPr`, with no element that
    // says "list"; without this every bullet becomes its own paragraph and loses the marker.
    let mut numbered = false;

    loop {
        match reader.read_event() {
            Ok(Event::Eof) | Err(_) => break,
            Ok(Event::Start(e)) => match local(e.name().as_ref()).as_str() {
                "t" => in_text = true,
                "numpr" => numbered = true,
                _ => {}
            },
            Ok(Event::Empty(e)) => match local(e.name().as_ref()).as_str() {
                "pstyle" => {
                    style = e
                        .attributes()
                        .flatten()
                        .find(|a| local(a.key.as_ref()) == "val")
                        .and_then(|a| String::from_utf8(a.value.to_vec()).ok());
                }
                // A soft line break inside a paragraph is a space to a narrator.
                "br" | "tab" => text.push(' '),
                "numpr" => numbered = true,
                _ => {}
            },
            Ok(Event::End(e)) => match local(e.name().as_ref()).as_str() {
                "t" => in_text = false,
                "p" => {
                    push_paragraph(&mut out, &text, style.as_deref(), numbered);
                    text.clear();
                    style = None;
                    numbered = false;
                }
                _ => {}
            },
            Ok(Event::Text(t)) if in_text => {
                if let Ok(decoded) = t.unescape() {
                    text.push_str(&decoded);
                }
            }
            Ok(_) => {}
        }
    }
    push_paragraph(&mut out, &text, style.as_deref(), numbered);
    Ok(markdown::document(&out))
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

fn push_paragraph(out: &mut Vec<String>, text: &str, style: Option<&str>, numbered: bool) {
    if let Some(block) = markdown::block(style.and_then(heading_level), numbered, text) {
        out.push(block);
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
    fn an_empty_paragraph_contributes_nothing() {
        let md = to_markdown("<w:p/><w:p><w:r><w:t>only</w:t></w:r></w:p>").unwrap();
        assert_eq!(md, "only\n");
    }
}
