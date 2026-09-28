//! A web page, as the article on it.
//!
//! Fetched with the system's `curl`, which carries the machine's certificates and proxy settings
//! and adds no TLS stack to this crate. What is kept is the article — `<article>`, else `<main>`,
//! else the body — with the page's furniture taken out: navigation, headers, footers, asides,
//! forms. The result is saved as HTML and read by the HTML importer like any file.

use anyhow::{bail, Context, Result};
use std::path::PathBuf;
use std::process::Command;

pub fn is_url(text: &str) -> bool {
    let t = text.trim();
    (t.starts_with("https://") || t.starts_with("http://")) && !t.contains(char::is_whitespace)
}

/// The page saved as a file the importers read, and its title.
pub fn fetch(url: &str) -> Result<(PathBuf, String)> {
    anyhow::ensure!(is_url(url), "{url} is not a web address");
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    url.trim().hash(&mut hasher);
    let dir = crate::convert::cache_root().join("web");
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let raw = dir.join(format!("{:016x}.raw.html", hasher.finish()));
    let out = dir.join(format!("{:016x}.html", hasher.finish()));

    let status = Command::new("curl")
        .args([
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--max-time",
            "60",
        ])
        .args(["--user-agent", "Mozilla/5.0 (Macintosh) DreamReader/1"])
        .arg("--output")
        .arg(&raw)
        .arg(url.trim())
        .output()
        .context("running curl")?;
    if !status.status.success() {
        bail!(
            "could not fetch {url}: {}",
            String::from_utf8_lossy(&status.stderr).trim()
        );
    }
    let bytes = std::fs::read(&raw)?;
    let html = crate::text::decode(&bytes);
    let title = title(&html).unwrap_or_else(|| url.trim().to_string());
    let body = article(&html);
    anyhow::ensure!(!body.trim().is_empty(), "{url} has no article text to read");
    let page = format!(
        "<!doctype html><html><head><title>{t}</title></head><body><h1>{t}</h1>{body}</body></html>",
        t = escape(&title)
    );
    std::fs::write(&out, page).with_context(|| format!("writing {}", out.display()))?;
    let _ = std::fs::remove_file(&raw);
    Ok((out, title))
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn title(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    // og:title is the headline; <title> often carries " | Site name".
    if let Some(at) = lower.find("property=\"og:title\"") {
        let tag_start = lower[..at].rfind('<')?;
        let tag_end = at + lower[at..].find('>')?;
        let tag = &html[tag_start..tag_end];
        if let Some(c) = tag.to_ascii_lowercase().find("content=\"") {
            let rest = &tag[c + 9..];
            if let Some(end) = rest.find('"') {
                return Some(unescape(rest[..end].trim()));
            }
        }
    }
    let start = lower.find("<title")?;
    let open_end = start + lower[start..].find('>')? + 1;
    let close = open_end + lower[open_end..].find("</title>")?;
    Some(unescape(html[open_end..close].trim())).filter(|t| !t.is_empty())
}

fn unescape(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
}

/// The article's markup: the largest `<article>`, else `<main>`, else `<body>`, with furniture
/// removed.
pub fn article(html: &str) -> String {
    let chosen = largest(html, "article")
        .or_else(|| largest(html, "main"))
        .or_else(|| largest(html, "body"))
        .unwrap_or(html);
    let mut out = chosen.to_string();
    for tag in [
        "script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg", "button",
    ] {
        out = remove_all(&out, tag);
    }
    out
}

/// The contents of the largest `<tag>…</tag>`, matched case-insensitively and counting nesting.
fn largest<'a>(html: &'a str, tag: &str) -> Option<&'a str> {
    let lower = html.to_ascii_lowercase();
    let mut best: Option<&str> = None;
    let mut from = 0;
    while let Some(start) = find_open(&lower, tag, from) {
        let content_start = start + lower[start..].find('>')? + 1;
        let end = matching_close(&lower, tag, content_start).unwrap_or(lower.len());
        let inner = &html[content_start..end];
        if best.is_none_or(|b| inner.len() > b.len()) {
            best = Some(inner);
        }
        from = end;
    }
    best
}

fn find_open(lower: &str, tag: &str, from: usize) -> Option<usize> {
    let needle = format!("<{tag}");
    let mut at = from;
    while let Some(i) = lower[at..].find(&needle) {
        let i = at + i;
        let next = lower[i + needle.len()..].chars().next();
        if matches!(next, Some('>' | ' ' | '\n' | '\t' | '\r' | '/')) {
            return Some(i);
        }
        at = i + needle.len();
    }
    None
}

fn matching_close(lower: &str, tag: &str, from: usize) -> Option<usize> {
    let close = format!("</{tag}");
    let mut depth = 1usize;
    let mut at = from;
    loop {
        let next_close = lower[at..].find(&close).map(|i| at + i)?;
        let next_open = find_open(lower, tag, at).filter(|o| *o < next_close);
        match next_open {
            Some(o) => {
                depth += 1;
                at = o + 1;
            }
            None => {
                depth -= 1;
                if depth == 0 {
                    return Some(next_close);
                }
                at = next_close + close.len();
            }
        }
    }
}

fn remove_all(html: &str, tag: &str) -> String {
    let mut out = html.to_string();
    loop {
        let lower = out.to_ascii_lowercase();
        let Some(start) = find_open(&lower, tag, 0) else {
            return out;
        };
        let Some(open_end) = lower[start..].find('>').map(|i| start + i + 1) else {
            return out;
        };
        // A self-closed element has no contents to remove.
        let end = if lower[..open_end].ends_with("/>") {
            open_end
        } else {
            matching_close(&lower, tag, open_end)
                .and_then(|c| lower[c..].find('>').map(|i| c + i + 1))
                .unwrap_or(lower.len())
        };
        out.replace_range(start..end, "");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_article_is_kept_and_the_furniture_goes() {
        let html = r#"<html><head><title>A Post | Blog</title>
<meta property="og:title" content="A Post"></head><body>
<nav><a href="/">Home</a></nav>
<article><h2>Intro</h2><p>The text worth reading.</p>
<aside>Related links</aside><div><article><p>nested</p></article></div></article>
<footer>Copyright</footer></body></html>"#;
        assert_eq!(title(html).as_deref(), Some("A Post"));
        let body = article(html);
        assert!(
            body.contains("The text worth reading.") && body.contains("nested"),
            "{body}"
        );
        assert!(
            !body.contains("Home") && !body.contains("Related") && !body.contains("Copyright"),
            "{body}"
        );
    }

    #[test]
    fn a_page_without_an_article_falls_back_to_its_body() {
        let html = "<html><body><header>Site</header><p>Only prose.</p></body></html>";
        assert_eq!(article(html).trim(), "<p>Only prose.</p>");
        assert!(is_url("https://example.com/a") && !is_url("book.pdf"));
    }
}
