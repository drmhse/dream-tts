//! FictionBook 2: one XML file, sections nested as deep as the book is.
//!
//! A section's `<title>` is a heading at the section's depth, which is the whole structural
//! signal the format has. The notes body (`<body name="notes">`) and embedded `<binary>` images
//! are not the text and are skipped.

use crate::markdown;
use crate::xml::{local, reader};
use crate::Chapter;
use anyhow::{Context, Result};
use quick_xml::events::Event;
use std::path::Path;

pub fn chapters(path: &Path) -> Result<Vec<Chapter>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(crate::split_markdown(&to_markdown(&crate::text::decode(
        &bytes,
    ))?))
}

pub fn to_markdown(xml: &str) -> Result<String> {
    let mut reader = reader(xml);
    let mut out: Vec<String> = Vec::new();
    let mut text = String::new();
    let mut depth = 0usize;
    let mut in_title = false;
    let mut in_book_title = false;
    let mut skip = 0usize;
    let mut quote = 0usize;

    loop {
        match reader.read_event() {
            Ok(Event::Eof) | Err(_) => break,
            Ok(Event::Start(e)) => {
                let name = local(e.name().as_ref());
                if skip > 0 {
                    skip += 1;
                    continue;
                }
                match name.as_str() {
                    "binary" => skip = 1,
                    "body" => {
                        let notes = e.attributes().flatten().any(|a| {
                            local(a.key.as_ref()) == "name" && a.value.as_ref() == b"notes"
                        });
                        if notes {
                            skip = 1;
                        }
                    }
                    "description" => skip = 1,
                    "book-title" => in_book_title = true,
                    "section" => depth += 1,
                    "title" => {
                        in_title = true;
                        text.clear();
                    }
                    "cite" | "epigraph" => quote += 1,
                    "p" | "v" | "subtitle" | "text-author" if !in_title => text.clear(),
                    "emphasis" => text.push('*'),
                    "strong" => text.push_str("**"),
                    _ => {}
                }
            }
            Ok(Event::End(e)) => {
                let name = local(e.name().as_ref());
                if skip > 0 {
                    skip -= 1;
                    continue;
                }
                match name.as_str() {
                    "book-title" => in_book_title = false,
                    "section" => depth = depth.saturating_sub(1),
                    "title" => {
                        in_title = false;
                        if let Some(b) = markdown::block(Some(depth.max(1)), false, &text) {
                            out.push(b);
                        }
                        text.clear();
                    }
                    "cite" | "epigraph" => quote = quote.saturating_sub(1),
                    "p" | "v" | "text-author" if !in_title => {
                        if let Some(b) = markdown::block(None, false, &text) {
                            out.push(if quote > 0 { format!("> {b}") } else { b });
                        }
                        text.clear();
                    }
                    "subtitle" => {
                        if let Some(b) = markdown::block(Some((depth + 1).min(6)), false, &text) {
                            out.push(b);
                        }
                        text.clear();
                    }
                    "emphasis" => text.push('*'),
                    "strong" => text.push_str("**"),
                    "p" if in_title => text.push(' '),
                    _ => {}
                }
            }
            Ok(Event::Text(t)) if skip == 0 && !in_book_title => {
                if let Ok(decoded) = t.unescape() {
                    text.push_str(&decoded);
                }
            }
            Ok(_) => {}
        }
    }
    Ok(markdown::document(&out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sections_nest_into_headings_and_notes_are_left_out() {
        let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
<description><title-info><book-title>The Book</book-title></title-info></description>
<body>
  <section><title><p>Part One</p></title>
    <section><title><p>Chapter</p><p>One</p></title>
      <p>It began <emphasis>quietly</emphasis>.</p>
      <cite><p>A quotation.</p></cite>
    </section>
  </section>
</body>
<body name="notes"><section><p>A footnote.</p></section></body>
<binary id="cover.jpg">AAAA</binary>
</FictionBook>"#;
        assert_eq!(
            to_markdown(xml).unwrap(),
            "# Part One\n\n## Chapter One\n\nIt began *quietly*.\n\n> A quotation.\n"
        );
    }
}
