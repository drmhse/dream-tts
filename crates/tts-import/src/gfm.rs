//! GitHub's markdown, as plain markdown: a README is written for a web page, and its HTML
//! wrappers and badge rows read as source text everywhere else.
//!
//! HTML tags go and their text stays; `<img>` becomes `![alt](src)`. An image that is a link
//! keeps only the image, and one that is not on this disk inside a link is a badge, which goes.
//! Code spans and fenced code are left as written.

/// Tags this removes. Anything else in angle brackets — `Vec<String>`, `<https://…>` — is text.
const TAGS: &[&str] = &[
    "a",
    "abbr",
    "b",
    "big",
    "blockquote",
    "br",
    "center",
    "code",
    "dd",
    "del",
    "details",
    "div",
    "dl",
    "dt",
    "em",
    "font",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "i",
    "img",
    "ins",
    "kbd",
    "li",
    "mark",
    "ol",
    "p",
    "picture",
    "pre",
    "s",
    "small",
    "source",
    "span",
    "strong",
    "sub",
    "summary",
    "sup",
    "table",
    "tbody",
    "td",
    "th",
    "thead",
    "tr",
    "u",
    "ul",
    "video",
];

pub fn tidy(markdown: &str) -> String {
    let mut out = String::with_capacity(markdown.len());
    let mut fence: Option<&str> = None;
    let mut comment = false;
    for line in markdown.lines() {
        let trimmed = line.trim_start();
        if comment || fence.is_none() {
            let (text, still) = inline(line, comment);
            comment = still;
            if text != line {
                // A line that was only markup is a paragraph break, not text joined across it.
                let text = if line.trim_start().starts_with('<') {
                    text.trim_start()
                } else {
                    &text
                };
                out.push_str(if text.trim().is_empty() { "" } else { text });
                out.push('\n');
                continue;
            }
        }
        match fence {
            Some(open) if trimmed.starts_with(open) => fence = None,
            None if trimmed.starts_with("```") => fence = Some("```"),
            None if trimmed.starts_with("~~~") => fence = Some("~~~"),
            _ => {}
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// One line with its markup taken out, and whether it ends inside an HTML comment.
fn inline(line: &str, mut comment: bool) -> (String, bool) {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while !rest.is_empty() {
        if comment {
            match rest.find("-->") {
                Some(end) => {
                    rest = &rest[end + 3..];
                    comment = false;
                }
                None => return (out, true),
            }
            continue;
        }
        if let Some(after) = rest.strip_prefix("<!--") {
            comment = true;
            rest = after;
            continue;
        }
        if rest.starts_with('`') {
            let run = rest.len() - rest.trim_start_matches('`').len();
            let ticks = &rest[..run];
            let end = rest[run..]
                .find(ticks)
                .map_or(rest.len(), |e| run + e + run);
            out.push_str(&rest[..end]);
            rest = &rest[end..];
            continue;
        }
        if rest.starts_with('<') {
            if let Some((tag, len)) = tag(rest) {
                if tag.name == "img" {
                    if let Some(src) = tag.attr("src").filter(|s| local(s)) {
                        out.push_str(&format!(
                            "![{}]({src})",
                            tag.attr("alt").unwrap_or_default()
                        ));
                    }
                }
                rest = &rest[len..];
                continue;
            }
        }
        if rest.starts_with("[![") {
            if let Some((alt, src, len)) = linked_image(rest) {
                if local(src) {
                    out.push_str(&format!("![{alt}]({src})"));
                }
                rest = &rest[len..];
                continue;
            }
        }
        let c = rest.chars().next().expect("not empty");
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    (out, comment)
}

struct Tag<'a> {
    name: String,
    body: &'a str,
}

impl Tag<'_> {
    fn attr(&self, name: &str) -> Option<String> {
        let mut rest = self.body;
        while let Some(at) = rest.find(name) {
            let before = rest[..at].chars().next_back();
            let after = rest[at + name.len()..].trim_start();
            rest = &rest[at + name.len()..];
            if !before.is_some_and(char::is_whitespace) {
                continue;
            }
            let Some(value) = after.strip_prefix('=').map(str::trim_start) else {
                continue;
            };
            let quote = value.chars().next()?;
            return Some(if quote == '"' || quote == '\'' {
                value[1..].split(quote).next()?.to_string()
            } else {
                value
                    .split(|c: char| c.is_whitespace() || c == '>')
                    .next()?
                    .to_string()
            });
        }
        None
    }
}

/// A known tag at the start of `text`, and how long it is.
fn tag(text: &str) -> Option<(Tag<'_>, usize)> {
    let inner = text.strip_prefix('<')?;
    let inner = inner.strip_prefix('/').unwrap_or(inner);
    let name: String = inner
        .chars()
        .take_while(char::is_ascii_alphanumeric)
        .collect::<String>()
        .to_ascii_lowercase();
    let next = inner[name.len()..].chars().next()?;
    if !TAGS.contains(&name.as_str()) || !(next.is_whitespace() || next == '>' || next == '/') {
        return None;
    }
    let end = text.find('>')?;
    Some((
        Tag {
            name,
            body: &text[..end],
        },
        end + 1,
    ))
}

/// `[![alt](src)](url)` at the start of `text`: the alt, the image, and the whole length.
fn linked_image(text: &str) -> Option<(&str, &str, usize)> {
    let inner = &text[1..];
    let alt_end = inner.find("](")?;
    let alt = &inner[2..alt_end];
    let src_from = alt_end + 2;
    let src_end = src_from + inner[src_from..].find(')')?;
    let src = inner[src_from..src_end]
        .split_whitespace()
        .next()
        .unwrap_or("");
    let link = inner[src_end + 1..].strip_prefix("](")?;
    let len = 1 + src_end + 1 + 2 + link.find(')')? + 1;
    Some((alt, src, len))
}

fn local(src: &str) -> bool {
    !src.is_empty() && !src.contains("://") && !src.starts_with("data:")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_readme_header_is_its_image_and_its_words() {
        let readme = "<p align=\"center\">\n  <a href=\"https://x.io\" target=\"_blank\"><img src=\"img/logo.png\" alt=\"Logo\" /></a>\n</p>\n\nBaaS - a service\n\n[![Bugs](https://sonarcloud.io/b.svg)](https://sonarcloud.io/s)\n[![Shot](docs/shot.png)](https://x.io)\n";
        assert_eq!(
            tidy(readme),
            "\n![Logo](img/logo.png)\n\n\nBaaS - a service\n\n\n![Shot](docs/shot.png)\n"
        );
    }

    #[test]
    fn code_and_text_in_angle_brackets_are_kept() {
        let md = "A `<div>` and Vec<String>, see <https://x.io>.\n```html\n<p>kept</p>\n```\n";
        assert_eq!(tidy(md), md);
    }

    #[test]
    fn inline_markup_keeps_its_words() {
        assert_eq!(
            tidy("Press <kbd>Ctrl</kbd> and <b>go</b><br>\n"),
            "Press Ctrl and go\n"
        );
        assert_eq!(tidy("a <!-- hidden\nstill --> b\n"), "a \n b\n");
    }
}
