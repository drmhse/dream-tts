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
    let heading = if find_open(&body.to_ascii_lowercase(), "h1", 0).is_some() {
        String::new()
    } else {
        format!("<h1>{}</h1>", escape(&title))
    };
    let page = format!(
        "<!doctype html><html><head><title>{t}</title></head><body>{heading}{body}</body></html>",
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
    let head = meta(html, "og:title").or_else(|| {
        let start = lower.find("<title")?;
        let open_end = start + lower[start..].find('>')? + 1;
        let close = open_end + lower[open_end..].find("</title>")?;
        Some(unescape(html[open_end..close].trim()))
    });
    let head = head
        .filter(|t| !t.is_empty())
        .map(|t| unsuffixed(&t, meta(html, "og:site_name")));
    // The page's own h1 carries no site name; a logo's h1 is rejected by not appearing in the head.
    let h1 = largest(html, "h1")
        .map(|inner| {
            unescape(&text_of(inner))
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        })
        .filter(|h| !h.is_empty());
    match (h1, head) {
        (Some(h1), Some(head)) if head.contains(&h1) => Some(h1),
        (_, head) => head,
    }
}

/// "Post / Blog", "Post | Blog": the site's name, which a title read aloud should not carry.
fn unsuffixed(title: &str, site: Option<String>) -> String {
    for sep in [
        " | ",
        " / ",
        " \u{2014} ",
        " \u{2013} ",
        " - ",
        " \u{00b7} ",
    ] {
        if let Some((kept, tail)) = title.rsplit_once(sep) {
            let named = site
                .as_deref()
                .is_some_and(|s| tail.trim().eq_ignore_ascii_case(s.trim()));
            if named || (site.is_none() && sep == " | ") {
                return kept.trim().to_string();
            }
        }
    }
    title.to_string()
}

fn text_of(html: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' if in_tag => {
                in_tag = false;
                out.push(' ');
            }
            c if !in_tag => out.push(c),
            _ => {}
        }
    }
    out
}

fn meta(html: &str, property: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let at = lower.find(&format!("property=\"{property}\""))?;
    let tag_start = lower[..at].rfind('<')?;
    let tag_end = at + lower[at..].find('>')?;
    let tag = &html[tag_start..tag_end];
    let c = tag.to_ascii_lowercase().find("content=\"")?;
    let rest = &tag[c + 9..];
    Some(unescape(rest[..rest.find('"')?].trim()))
}

fn unescape(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&#x27;", "'")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
}

/// The article's markup: the largest `<article>`, else `<main>`, else `<body>`, with furniture
/// removed, narrowed to the container that holds the prose.
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
    // Wikipedia's navboxes, authority control and "[edit]" links, which declare themselves.
    for marker in ["role=\"navigation\"", "class=\"mw-editsection\""] {
        out = remove_marked(&out, marker);
    }
    match densest(&out) {
        Some(range) => headline(&out[..range.start]) + &out[range],
        None => out,
    }
}

/// The last h1 before the prose, with a paragraph right after it (a standfirst): both sit in a
/// hero block beside the article rather than in it.
fn headline(before: &str) -> String {
    let lower = before.to_ascii_lowercase();
    let Some(start) = lower
        .rfind("<h1")
        .filter(|&i| find_open(&lower, "h1", i) == Some(i))
    else {
        return String::new();
    };
    let Some(end) = lower[start..].find("</h1>").map(|i| start + i + 5) else {
        return String::new();
    };
    let after = lower[end..].trim_start();
    let gap = lower.len() - end - after.len();
    let standfirst = find_open(after, "p", 0)
        .filter(|&i| i == 0)
        .and_then(|_| after.find("</p>"))
        .map_or(0, |i| i + 4);
    before[start..end + gap + standfirst].to_string()
}

/// The smallest container holding nearly all the paragraph text. A page with no `<article>`
/// puts a category chip, a byline and a related-posts list beside the prose, none of it in the
/// furniture tags; paragraphs are what the chrome lacks.
fn densest(html: &str) -> Option<std::ops::Range<usize>> {
    let lower = html.to_ascii_lowercase();
    let mut items: Vec<(usize, usize)> = Vec::new();
    for tag in ["p", "li"] {
        let mut from = 0;
        while let Some(start) = find_open(&lower, tag, from) {
            let content = start + lower[start..].find('>').map_or(1, |i| i + 1);
            // HTML lets both go unclosed; one that runs to the end of the page would weigh it all.
            let end = [
                format!("</{tag}"),
                format!("<{tag}>"),
                format!("<{tag} "),
                "</div".into(),
                "</ul".into(),
                "</ol".into(),
            ]
            .iter()
            .filter_map(|m| lower[content..].find(m.as_str()).map(|i| content + i))
            .min()
            .unwrap_or(lower.len());
            let weight = text_of(&html[content..end])
                .split_whitespace()
                .map(str::len)
                .sum();
            items.push((start, weight));
            from = content;
        }
    }
    items.sort_unstable();
    let total: usize = items.iter().map(|i| i.1).sum();
    if total < 400 {
        return None;
    }
    let mut prefix = vec![0usize];
    for (_, w) in &items {
        prefix.push(prefix.last().unwrap() + w);
    }
    let within = |r: &std::ops::Range<usize>| {
        let a = items.partition_point(|i| i.0 < r.start);
        let b = items.partition_point(|i| i.0 < r.end);
        prefix[b] - prefix[a]
    };

    const CONTAINERS: [&str; 4] = ["div", "section", "article", "main"];
    let mut stack: Vec<(&str, usize)> = Vec::new();
    let mut best: Option<std::ops::Range<usize>> = None;
    for (at, _) in lower.match_indices('<') {
        let rest = &lower[at + 1..];
        let closing = rest.starts_with('/');
        let name_start = usize::from(closing);
        let name_len = rest[name_start..]
            .find(|c: char| !c.is_ascii_alphanumeric())
            .unwrap_or(rest.len() - name_start);
        let name = &rest[name_start..name_start + name_len];
        let Some(tag) = CONTAINERS.iter().find(|t| **t == name) else {
            continue;
        };
        let Some(gt) = rest.find('>') else { break };
        if !closing {
            if !rest[..gt].ends_with('/') {
                stack.push((tag, at + 1 + gt + 1));
            }
            continue;
        }
        let Some(open) = stack.iter().rposition(|(t, _)| t == tag) else {
            continue;
        };
        let (_, content) = stack[open];
        stack.truncate(open);
        let range = content..at;
        if within(&range) * 10 >= total * 8 && best.as_ref().is_none_or(|b| range.len() < b.len()) {
            best = Some(range);
        }
    }
    best
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

fn remove_marked(html: &str, marker: &str) -> String {
    let mut out = html.to_string();
    let mut from = 0;
    loop {
        let lower = out.to_ascii_lowercase();
        let Some(at) = lower[from..].find(marker).map(|i| from + i) else {
            return out;
        };
        from = at + 1;
        let Some(start) = lower[..at].rfind('<') else {
            continue;
        };
        let name: String = lower[start + 1..]
            .chars()
            .take_while(char::is_ascii_alphanumeric)
            .collect();
        let Some(open_end) = lower[at..].find('>').map(|i| at + i + 1) else {
            return out;
        };
        if name.is_empty() || lower[start..open_end].contains("<!") {
            continue;
        }
        let end = matching_close(&lower, &name, open_end)
            .and_then(|c| lower[c..].find('>').map(|i| c + i + 1))
            .unwrap_or(open_end);
        out.replace_range(start..end, "");
        from = start;
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
    fn a_page_without_an_article_keeps_its_prose_not_its_chrome() {
        let prose =
            "<p>A paragraph of the post, long enough to count as prose on its own.</p>".repeat(8);
        let html = format!(
            "<html><head><meta property=\"og:title\" content=\"A Post / The Blog\"/>\
             <meta property=\"og:site_name\" content=\"The Blog\"/></head><body>\
             <div class=\"hero\"><span>Category</span><h1>A Post</h1><p>The standfirst.</p>\
             <div>AUTHOR</div><div>Someone</div></div><div class=\"grid\"><section>{prose}</section></div>\
             <div class=\"more\"><div>Related posts</div><a>Another post</a></div>\
             <div role=\"navigation\"><ul><li>Navbox</li></ul></div></body></html>"
        );
        assert_eq!(title(&html).as_deref(), Some("A Post"));
        let body = article(&html);
        assert!(
            body.starts_with("<h1>A Post</h1><p>The standfirst.</p>"),
            "{body}"
        );
        for chrome in ["Category", "AUTHOR", "Related", "Navbox"] {
            assert!(!body.contains(chrome), "{chrome} in {body}");
        }
    }

    #[test]
    fn a_page_without_an_article_falls_back_to_its_body() {
        let html = "<html><body><header>Site</header><p>Only prose.</p></body></html>";
        assert_eq!(article(html).trim(), "<p>Only prose.</p>");
        assert!(is_url("https://example.com/a") && !is_url("book.pdf"));
    }
}
