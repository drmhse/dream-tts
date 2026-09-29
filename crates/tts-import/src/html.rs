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
    "main",
    "header",
    "footer",
    "aside",
    "nav",
];

pub fn to_markdown(html: &str) -> Result<String> {
    let html = bogus_comments(html);
    let mut rest: &str = &html;
    let mut reader = reader(rest);

    let mut out: Vec<String> = Vec::new();
    let mut line = String::new();
    let mut skip_depth = 0usize;
    let mut heading: Option<usize> = None;
    let mut list_item = false;
    // Preformatted text keeps its lines and becomes a fence, which is shown and never spoken:
    // flattened into prose, a README's Rust was read out as "Fn main ( ), then Result <".
    let mut pre: Option<(usize, String)> = None;

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
            Ok(Event::Start(e)) if pre.is_some() => {
                if local(e.name().as_ref()) == "pre" {
                    pre.as_mut().unwrap().0 += 1;
                }
            }
            Ok(Event::End(e)) if pre.is_some() => {
                if local(e.name().as_ref()) == "pre" {
                    let (depth, code) = pre.as_mut().unwrap();
                    *depth -= 1;
                    if *depth == 0 {
                        if let Some(block) = fence(code) {
                            out.push(block);
                        }
                        pre = None;
                    }
                }
            }
            Ok(Event::Empty(e)) if pre.is_some() => {
                if local(e.name().as_ref()) == "br" {
                    pre.as_mut().unwrap().1.push('\n');
                }
            }
            Ok(Event::Text(t)) if pre.is_some() => {
                if let Ok(text) = t.unescape() {
                    pre.as_mut().unwrap().1.push_str(&text);
                }
            }
            Ok(Event::Start(e)) => {
                let name = local(e.name().as_ref());
                if name == "pre" && skip_depth == 0 {
                    flush(&mut out, &mut line, heading, list_item);
                    heading = None;
                    list_item = false;
                    pre = Some((1, String::new()));
                } else if SKIPPED.contains(&name.as_str()) {
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
    if let Some(block) = pre.and_then(|(_, code)| fence(&code)) {
        out.push(block);
    }
    Ok(markdown::document(&out))
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
    fn punctuation_after_a_tag_is_not_spaced_off() {
        let md =
            to_markdown("<p><strong>More thinking</strong>. And <a>a link</a>, then</p>").unwrap();
        assert_eq!(md, "More thinking. And a link, then\n");
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
    fn headings_drive_the_chapter_split() {
        let chapters = chapters("<h1>One</h1><p>alpha</p><h1>Two</h1><p>beta</p>").unwrap();
        assert_eq!(chapters.len(), 2);
        assert_eq!(chapters[1].title.as_deref(), Some("Two"));
    }
}
