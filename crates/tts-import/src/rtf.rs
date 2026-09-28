//! RTF, read directly: a control-word stream, not a zip of XML.
//!
//! Native rather than through a system converter, so an `.rtf` imports the same everywhere. The
//! heading signal is Word's `\outlinelevelN`; a list is a paragraph with `\ls` or `\ilvl`. Every
//! destination that is not body text — font and colour tables, the stylesheet, pictures,
//! headers, field instructions, anything behind `\*` — is skipped whole.

use crate::markdown;
use crate::Chapter;
use anyhow::{Context, Result};
use std::path::Path;

pub fn chapters(path: &Path) -> Result<Vec<Chapter>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(crate::split_markdown(&to_markdown(&bytes)))
}

/// Destinations whose contents are never text on the page.
const SKIPPED: &[&str] = &[
    "fonttbl",
    "colortbl",
    "stylesheet",
    "info",
    "pict",
    "header",
    "headerl",
    "headerr",
    "headerf",
    "footer",
    "footerl",
    "footerr",
    "footerf",
    "fldinst",
    "object",
    "themedata",
    "datastore",
    "listtable",
    "listoverridetable",
    "rsidtbl",
    "generator",
    "xmlnstbl",
    "mmathpr",
    "pntext",
    "pntxtb",
    "pntxta",
    "latentstyles",
    "filetbl",
    "revtbl",
    "bkmkstart",
    "bkmkend",
    "nonshppict",
];

#[derive(Clone)]
struct Group {
    skip: bool,
    /// Characters to drop after a `\uN`, per `\ucN`.
    uc: usize,
}

pub fn to_markdown(bytes: &[u8]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut text = String::new();
    let mut level: Option<usize> = None;
    let mut in_list = false;
    let mut stack = vec![Group { skip: false, uc: 1 }];
    let mut pending_skip = 0usize;
    let mut i = 0usize;

    let end_paragraph = |out: &mut Vec<String>, text: &mut String, level: Option<usize>, list| {
        if let Some(block) = markdown::block(level, list, text) {
            out.push(block);
        }
        text.clear();
    };

    while i < bytes.len() {
        let b = bytes[i];
        match b {
            b'{' => {
                let top = stack
                    .last()
                    .cloned()
                    .unwrap_or(Group { skip: false, uc: 1 });
                stack.push(top);
                i += 1;
                // `{\*\dest ...}` is a destination a reader that does not know it must skip.
                if bytes.get(i..i + 2) == Some(b"\\*") {
                    if let Some(g) = stack.last_mut() {
                        g.skip = true;
                    }
                    i += 2;
                }
            }
            b'}' => {
                if stack.len() > 1 {
                    stack.pop();
                }
                i += 1;
            }
            b'\\' => {
                i += 1;
                let Some(&c) = bytes.get(i) else { break };
                if c.is_ascii_alphabetic() {
                    let start = i;
                    while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
                        i += 1;
                    }
                    let word = std::str::from_utf8(&bytes[start..i]).unwrap_or("");
                    let num_start = i;
                    if i < bytes.len() && (bytes[i] == b'-' || bytes[i].is_ascii_digit()) {
                        i += 1;
                        while i < bytes.len() && bytes[i].is_ascii_digit() {
                            i += 1;
                        }
                    }
                    let arg: Option<i64> = std::str::from_utf8(&bytes[num_start..i])
                        .ok()
                        .and_then(|n| n.parse().ok());
                    if bytes.get(i) == Some(&b' ') {
                        i += 1;
                    }
                    let skip = stack.last().is_some_and(|g| g.skip);
                    match word {
                        w if SKIPPED.contains(&w) => {
                            if let Some(g) = stack.last_mut() {
                                g.skip = true;
                            }
                        }
                        "par" | "sect" | "page" if !skip => {
                            end_paragraph(&mut out, &mut text, level, in_list);
                        }
                        "pard" => {
                            level = None;
                            in_list = false;
                        }
                        "outlinelevel" => level = arg.map(|n| n as usize + 1),
                        "ls" | "ilvl" => in_list = true,
                        "line" | "tab" if !skip => text.push(' '),
                        "emdash" if !skip => text.push('\u{2014}'),
                        "endash" if !skip => text.push('\u{2013}'),
                        "lquote" if !skip => text.push('\u{2018}'),
                        "rquote" if !skip => text.push('\u{2019}'),
                        "ldblquote" if !skip => text.push('\u{201c}'),
                        "rdblquote" if !skip => text.push('\u{201d}'),
                        "bullet" if !skip => text.push('\u{2022}'),
                        "uc" => {
                            if let (Some(g), Some(n)) = (stack.last_mut(), arg) {
                                g.uc = n.max(0) as usize;
                            }
                        }
                        "u" => {
                            if let Some(n) = arg {
                                let code = if n < 0 { n + 65536 } else { n } as u32;
                                if !skip {
                                    if let Some(ch) = char::from_u32(code) {
                                        text.push(ch);
                                    }
                                }
                                pending_skip = stack.last().map_or(1, |g| g.uc);
                            }
                        }
                        _ => {}
                    }
                } else {
                    i += 1;
                    let skip = stack.last().is_some_and(|g| g.skip);
                    match c {
                        b'\'' => {
                            let hex = bytes
                                .get(i..i + 2)
                                .and_then(|h| std::str::from_utf8(h).ok());
                            i += 2;
                            if pending_skip > 0 {
                                pending_skip -= 1;
                            } else if let Some(byte) =
                                hex.and_then(|h| u8::from_str_radix(h, 16).ok())
                            {
                                if !skip {
                                    text.push(crate::text::cp1252(byte));
                                }
                            }
                        }
                        b'\\' | b'{' | b'}' if !skip => text.push(c as char),
                        b'~' if !skip => text.push(' '),
                        b'_' if !skip => text.push('-'),
                        b'\n' | b'\r' if !skip => {
                            end_paragraph(&mut out, &mut text, level, in_list)
                        }
                        _ => {}
                    }
                }
            }
            b'\r' | b'\n' => i += 1,
            _ => {
                let skip = stack.last().is_some_and(|g| g.skip);
                // A run of plain bytes, taken as one: the common case, and the one that
                // makes a book-length file linear rather than a byte at a time.
                let start = i;
                while i < bytes.len() && !matches!(bytes[i], b'{' | b'}' | b'\\' | b'\r' | b'\n') {
                    i += 1;
                }
                if skip {
                    continue;
                }
                for &byte in &bytes[start..i] {
                    if pending_skip > 0 {
                        pending_skip -= 1;
                        continue;
                    }
                    text.push(crate::text::cp1252(byte));
                }
            }
        }
    }
    end_paragraph(&mut out, &mut text, level, in_list);
    markdown::document(&out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_text_survives_and_tables_do_not() {
        let rtf = br"{\rtf1\ansi{\fonttbl{\f0 Times;}}{\colortbl;\red0\green0\blue0;}
{\*\generator Word;}\pard\outlinelevel0 Chapter One\par
\pard Caf\'e9 is \u8220?open\u8221? today.\par
\pard\ls1\ilvl0 An item\par}";
        assert_eq!(
            to_markdown(rtf),
            "# Chapter One\n\nCaf\u{e9} is \u{201c}open\u{201d} today.\n\n- An item\n"
        );
    }

    #[test]
    fn an_escaped_brace_is_text() {
        assert_eq!(to_markdown(br"{\rtf1 a \{b\} c\par}"), "a {b} c\n");
    }
}
