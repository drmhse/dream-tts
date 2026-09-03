//! ODT, read from `content.xml`.
//!
//! Same shape as DOCX and for the same reason — the heading is explicit, here as
//! `text:h` with a `text:outline-level` — so this is a separate small reader rather than a
//! detour through a text extractor that would drop it.

use crate::markdown;
use crate::xml::{local, reader};
use crate::zipped::Archive;
use crate::Chapter;
use anyhow::Result;
use quick_xml::events::Event;
use std::path::Path;

pub fn chapters(path: &Path) -> Result<Vec<Chapter>> {
    let mut archive = Archive::open(path)?;
    let xml = archive.read("content.xml")?;
    Ok(crate::split_markdown(&to_markdown(&xml)?))
}

pub fn to_markdown(xml: &str) -> Result<String> {
    let mut reader = reader(xml);

    let mut out: Vec<String> = Vec::new();
    let mut text = String::new();
    let mut heading: Option<usize> = None;
    let mut list_depth = 0usize;
    let mut in_block = false;

    loop {
        match reader.read_event() {
            Ok(Event::Eof) | Err(_) => break,
            Ok(Event::Start(e)) => match local(e.name().as_ref()).as_str() {
                "h" => {
                    heading = e
                        .attributes()
                        .flatten()
                        .find(|a| local(a.key.as_ref()) == "outline-level")
                        .and_then(|a| String::from_utf8(a.value.to_vec()).ok())
                        .and_then(|v| v.parse::<usize>().ok())
                        .filter(|n| (1..=6).contains(n))
                        .or(Some(1));
                    in_block = true;
                }
                "p" => in_block = true,
                "list" => list_depth += 1,
                _ => {}
            },
            Ok(Event::Empty(e)) => match local(e.name().as_ref()).as_str() {
                // `text:s` is a run of spaces, `text:tab` a tab, `text:line-break` a break.
                "s" | "tab" | "line-break" => text.push(' '),
                _ => {}
            },
            Ok(Event::End(e)) => match local(e.name().as_ref()).as_str() {
                "h" | "p" => {
                    push_paragraph(&mut out, &text, heading, list_depth > 0);
                    text.clear();
                    heading = None;
                    in_block = false;
                }
                "list" => list_depth = list_depth.saturating_sub(1),
                _ => {}
            },
            Ok(Event::Text(t)) if in_block => {
                if let Ok(decoded) = t.unescape() {
                    text.push_str(&decoded);
                }
            }
            Ok(_) => {}
        }
    }
    push_paragraph(&mut out, &text, heading, list_depth > 0);
    Ok(markdown::document(&out))
}

fn push_paragraph(out: &mut Vec<String>, text: &str, heading: Option<usize>, in_list: bool) {
    if let Some(block) = markdown::block(heading, in_list, text) {
        out.push(block);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = r#"<office:document-content
  xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
  xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
<office:body><office:text>
  <text:h text:outline-level="1">Chapter One</text:h>
  <text:p>First paragraph.</text:p>
  <text:h text:outline-level="2">A Section</text:h>
  <text:list><text:list-item><text:p>An item</text:p></text:list-item></text:list>
  <text:p/>
</office:text></office:body></office:document-content>"#;

    #[test]
    fn outline_levels_become_headings() {
        let md = to_markdown(DOC).unwrap();
        assert_eq!(
            md,
            "# Chapter One\n\nFirst paragraph.\n\n## A Section\n\n- An item\n"
        );
    }

    #[test]
    fn a_heading_with_no_level_is_h1() {
        let md = to_markdown("<text:h>Bare</text:h>").unwrap();
        assert_eq!(md, "# Bare\n");
    }
}
