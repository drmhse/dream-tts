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
use quick_xml::events::Event;

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
    "pre",
    "main",
    "header",
    "footer",
    "aside",
    "nav",
];

pub fn to_markdown(html: &str) -> Result<String> {
    let mut reader = reader(html);

    let mut out: Vec<String> = Vec::new();
    let mut line = String::new();
    let mut skip_depth = 0usize;
    let mut heading: Option<usize> = None;
    let mut list_item = false;

    // `read_event` on malformed markup: broken tags are common in the wild and stopping on
    // one would refuse a document a browser renders fine, so a parse error ends extraction
    // rather than failing the import.
    loop {
        match reader.read_event() {
            Ok(Event::Eof) | Err(_) => break,
            Ok(Event::Start(e)) => {
                let name = local(e.name().as_ref());
                if SKIPPED.contains(&name.as_str()) {
                    skip_depth += 1;
                } else if let Some(level) = heading_level(&name) {
                    flush(&mut out, &mut line, heading, list_item);
                    heading = Some(level);
                    list_item = false;
                } else if name == "li" {
                    flush(&mut out, &mut line, heading, list_item);
                    heading = None;
                    list_item = true;
                } else if BLOCKS.contains(&name.as_str()) {
                    flush(&mut out, &mut line, heading, list_item);
                    heading = None;
                    list_item = false;
                }
            }
            Ok(Event::End(e)) => {
                let name = local(e.name().as_ref());
                if SKIPPED.contains(&name.as_str()) {
                    skip_depth = skip_depth.saturating_sub(1);
                } else if heading_level(&name).is_some()
                    || name == "li"
                    || BLOCKS.contains(&name.as_str())
                {
                    flush(&mut out, &mut line, heading, list_item);
                    heading = None;
                    list_item = false;
                }
            }
            Ok(Event::Empty(e)) => {
                let name = local(e.name().as_ref());
                if name == "br" || name == "hr" {
                    flush(&mut out, &mut line, heading, list_item);
                    heading = None;
                    list_item = false;
                }
            }
            Ok(Event::Text(t)) if skip_depth == 0 => {
                if let Ok(text) = t.unescape() {
                    push_text(&mut line, &text);
                }
            }
            Ok(Event::CData(t)) if skip_depth == 0 => {
                push_text(&mut line, &String::from_utf8_lossy(&t));
            }
            Ok(_) => {}
        }
    }
    flush(&mut out, &mut line, heading, list_item);
    Ok(markdown::document(&out))
}

pub fn chapters(html: &str) -> Result<Vec<Chapter>> {
    Ok(crate::split_markdown(&to_markdown(html)?))
}

fn heading_level(name: &str) -> Option<usize> {
    let mut chars = name.chars();
    if chars.next() != Some('h') {
        return None;
    }
    let rest: String = chars.collect();
    rest.parse::<usize>().ok().filter(|n| (1..=6).contains(n))
}

/// Append text, collapsing whitespace across the tag boundaries it crosses. Markup routinely
/// splits a sentence across elements, and joining without this gives "onetwo".
fn push_text(line: &mut String, text: &str) {
    for word in text.split_whitespace() {
        if !line.is_empty() && !line.ends_with(' ') {
            line.push(' ');
        }
        line.push_str(word);
    }
}

fn flush(out: &mut Vec<String>, line: &mut String, heading: Option<usize>, list_item: bool) {
    if let Some(block) = markdown::block(heading, list_item, line) {
        out.push(block);
    }
    line.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(md, "one two three\n");
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
    fn headings_drive_the_chapter_split() {
        let chapters = chapters("<h1>One</h1><p>alpha</p><h1>Two</h1><p>beta</p>").unwrap();
        assert_eq!(chapters.len(), 2);
        assert_eq!(chapters[1].title.as_deref(), Some("Two"));
    }
}
