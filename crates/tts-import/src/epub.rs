//! EPUB, read through its spine.
//!
//! The best case of the lot: the OPF spine *is* the chapter list, in reading order, stated by
//! the file itself. No heuristic is involved — an EPUB that splits its chapters into separate
//! documents (which is nearly all of them) needs no heading inference at all.

use crate::xml::{local, reader};
use crate::zipped::Archive;
use crate::{html, Chapter};
use anyhow::{bail, Context, Result};
use quick_xml::events::Event;
use std::path::Path;

pub fn chapters(path: &Path) -> Result<Vec<Chapter>> {
    let mut archive = Archive::open(path)?;
    let container = archive
        .read("META-INF/container.xml")
        .context("an EPUB must carry META-INF/container.xml")?;
    let opf_path = rootfile(&container).context("META-INF/container.xml names no rootfile")?;
    let opf = archive.read(&opf_path)?;

    let base = opf_path.rsplit_once('/').map_or("", |(dir, _)| dir);
    let spine = spine_documents(&opf)?;
    if spine.is_empty() {
        bail!("the EPUB spine is empty; there is no reading order to follow");
    }

    let mut out: Vec<Chapter> = Vec::new();
    for href in spine {
        let full = join(base, &href);
        // A spine entry naming a missing file is a broken EPUB, but the rest of the book is
        // still narratable — skip it rather than refuse the whole document.
        let Some(xhtml) = archive.read_opt(&full) else {
            continue;
        };
        let markdown = html::to_markdown(&xhtml)?;
        if markdown.trim().is_empty() {
            continue;
        }
        // One spine document is one chapter, even when it carries several headings: the book
        // said where its chapters are and that is better evidence than a heading level.
        let split = crate::split_markdown(&markdown);
        match split.as_slice() {
            [only] => out.push(only.clone()),
            _ => {
                let title = split.iter().find_map(|c| c.title.clone());
                let body = split
                    .iter()
                    .map(|c| match &c.title {
                        Some(t) if Some(t) != title.as_ref() => format!("## {t}\n\n{}", c.body),
                        _ => c.body.clone(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n");
                out.push(Chapter { title, body });
            }
        }
    }
    Ok(out)
}

fn rootfile(container: &str) -> Option<String> {
    attribute_of(container, "rootfile", "full-path")
}

/// `idref` order from `<spine>`, resolved through `<manifest>` to hrefs.
fn spine_documents(opf: &str) -> Result<Vec<String>> {
    let mut reader = reader(opf);

    let mut manifest: Vec<(String, String)> = Vec::new();
    let mut order: Vec<String> = Vec::new();
    let mut in_spine = false;

    loop {
        match reader.read_event() {
            Ok(Event::Eof) | Err(_) => break,
            Ok(Event::Start(e)) if local(e.name().as_ref()) == "spine" => in_spine = true,
            Ok(Event::End(e)) if local(e.name().as_ref()) == "spine" => in_spine = false,
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                let name = local(e.name().as_ref());
                let attr = |key: &str| {
                    e.attributes()
                        .flatten()
                        .find(|a| local(a.key.as_ref()) == key)
                        .and_then(|a| String::from_utf8(a.value.to_vec()).ok())
                };
                if name == "item" {
                    if let (Some(id), Some(href)) = (attr("id"), attr("href")) {
                        manifest.push((id, href));
                    }
                } else if in_spine && name == "itemref" {
                    // `linear="no"` marks front matter a reader may skip; narration skips it
                    // too, since it is covers and copyright pages.
                    if attr("linear").as_deref() == Some("no") {
                        continue;
                    }
                    if let Some(idref) = attr("idref") {
                        order.push(idref);
                    }
                }
            }
            Ok(_) => {}
        }
    }

    Ok(order
        .iter()
        .filter_map(|id| {
            manifest
                .iter()
                .find(|(mid, _)| mid == id)
                .map(|(_, href)| decode_href(href))
        })
        .collect())
}

/// The first value of `attribute` on the first `element`. Enough for the one attribute this
/// needs out of `container.xml`, and cheaper than modelling the document.
fn attribute_of(xml: &str, element: &str, attribute: &str) -> Option<String> {
    let mut reader = reader(xml);
    loop {
        match reader.read_event() {
            Ok(Event::Eof) | Err(_) => return None,
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) if local(e.name().as_ref()) == element => {
                return e
                    .attributes()
                    .flatten()
                    .find(|a| local(a.key.as_ref()) == attribute)
                    .and_then(|a| String::from_utf8(a.value.to_vec()).ok())
                    .map(|v| decode_href(&v));
            }
            Ok(_) => {}
        }
    }
}

/// Percent-decoding, because a manifest href is a URL and a zip entry name is not: a chapter
/// filed as `ch%201.xhtml` is `ch 1.xhtml` in the archive.
fn decode_href(href: &str) -> String {
    let bytes = href.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(&href[i + 1..i + 3], 16) {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Join a manifest href to the OPF's directory, resolving the `../` that a nested OPF uses.
fn join(base: &str, href: &str) -> String {
    if base.is_empty() {
        return href.to_string();
    }
    let mut parts: Vec<&str> = base.split('/').filter(|p| !p.is_empty()).collect();
    for segment in href.split('/') {
        match segment {
            "." | "" => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    parts.join("/")
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPF: &str = r#"<package xmlns="http://www.idpf.org/2007/opf">
  <manifest>
    <item id="cover" href="cover.xhtml" media-type="application/xhtml+xml"/>
    <item id="c1" href="text/ch%201.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="text/ch2.xhtml" media-type="application/xhtml+xml"/>
    <item id="css" href="style.css" media-type="text/css"/>
  </manifest>
  <spine>
    <itemref idref="cover" linear="no"/>
    <itemref idref="c1"/>
    <itemref idref="c2"/>
  </spine>
</package>"#;

    #[test]
    fn the_spine_gives_reading_order_and_skips_non_linear_items() {
        let spine = spine_documents(OPF).unwrap();
        assert_eq!(
            spine,
            vec!["text/ch 1.xhtml".to_string(), "text/ch2.xhtml".to_string()]
        );
    }

    #[test]
    fn the_container_names_the_opf() {
        let container = r#"<container><rootfiles>
            <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
        </rootfiles></container>"#;
        assert_eq!(rootfile(container).as_deref(), Some("OEBPS/content.opf"));
    }

    #[test]
    fn hrefs_resolve_against_the_opf_directory() {
        assert_eq!(join("OEBPS", "text/ch1.xhtml"), "OEBPS/text/ch1.xhtml");
        assert_eq!(
            join("OEBPS/sub", "../text/ch1.xhtml"),
            "OEBPS/text/ch1.xhtml"
        );
        assert_eq!(join("", "ch1.xhtml"), "ch1.xhtml");
    }

    #[test]
    fn percent_escapes_are_decoded_to_zip_entry_names() {
        assert_eq!(decode_href("ch%201.xhtml"), "ch 1.xhtml");
        assert_eq!(decode_href("plain.xhtml"), "plain.xhtml");
        assert_eq!(decode_href("bad%zz.xhtml"), "bad%zz.xhtml");
    }
}
