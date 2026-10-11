//! A web page, as the article on it.
//!
//! Fetched with the system's `curl` (Foundation's loader on iOS), which carries the machine's
//! certificates and proxy settings and adds no TLS stack to this crate. The page is parsed as a
//! browser would and the article kept: furniture, chrome that names itself, link lists and back
//! matter go, and what remains is narrowed to the prose. The result is saved as HTML and read by
//! the HTML importer like any file.

use anyhow::{Context, Result};
use dom_query::{Document, NodeId, NodeRef};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
#[cfg(not(target_os = "ios"))]
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

    download(url.trim(), &raw)?;
    let (page, title) = page(url.trim(), &std::fs::read(&raw)?)?;
    std::fs::write(&out, page).with_context(|| format!("writing {}", out.display()))?;
    let _ = std::fs::remove_file(&raw);
    Ok((out, title))
}

/// A fetched page's bytes as a document of just its article, and its title.
pub fn page(url: &str, bytes: &[u8]) -> Result<(String, String)> {
    let html = crate::text::decode(bytes);
    let title = title(&html).unwrap_or_else(|| url.to_string());
    let body = article(&html);
    let text = words(&Document::from(body.as_str()).select("body").text());
    anyhow::ensure!(!text.is_empty(), "{url} has no article text to read");
    anyhow::ensure!(
        !refused(&title, &text),
        "{url} did not give its page: the site answered “{title}”"
    );
    let heading = if body.contains("<h1") {
        String::new()
    } else {
        format!("<h1>{}</h1>", escape(&title))
    };
    let page = format!(
        "<!doctype html><html><head><title>{t}</title></head><body>{heading}{body}</body></html>",
        t = escape(&title)
    );
    Ok((page, title))
}

/// An error page sent as a success: a soft 404, or a bot check in front of the article.
fn refused(title: &str, text: &str) -> bool {
    const SAYS: &[&str] = &[
        "404", "not found", "attention required", "just a moment", "access denied", "forbidden",
        "are you a robot", "are you human", "captcha", "security check", "too many requests",
        "service unavailable", "bad gateway", "page unavailable",
    ];
    let title = title.to_lowercase();
    let short = text.split(' ').count() < 250;
    short && (SAYS.iter().any(|s| title.contains(s)) || text.contains("Cloudflare Ray ID"))
}

#[cfg(not(target_os = "ios"))]
fn download(url: &str, to: &Path) -> Result<()> {
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
        .arg(to)
        .arg(url)
        .output()
        .context("running curl")?;
    if !status.status.success() {
        anyhow::bail!(
            "could not fetch {url}: {}",
            String::from_utf8_lossy(&status.stderr).trim()
        );
    }
    Ok(())
}

/// iOS runs no other programs, so no curl: Foundation's loader, which like curl's `--fail`
/// refuses an error status rather than saving the error page as the article.
#[cfg(target_os = "ios")]
fn download(url: &str, to: &Path) -> Result<()> {
    use block2::RcBlock;
    use objc2_foundation::{
        NSData, NSError, NSHTTPURLResponse, NSMutableURLRequest, NSString, NSURLResponse,
        NSURLSession, NSURL,
    };
    let address = NSURL::URLWithString(&NSString::from_str(url))
        .with_context(|| format!("{url} is not a web address"))?;
    let request = NSMutableURLRequest::requestWithURL(&address);
    request.setTimeoutInterval(60.0);
    request.setValue_forHTTPHeaderField(
        Some(&NSString::from_str("Mozilla/5.0 (iPhone) DreamReader/1")),
        &NSString::from_str("User-Agent"),
    );
    let (sent, answer) = std::sync::mpsc::channel::<Result<Vec<u8>, String>>();
    let done = RcBlock::new(
        move |data: *mut NSData, response: *mut NSURLResponse, error: *mut NSError| {
            // SAFETY: Foundation passes each argument valid or null, for the call's duration.
            let (data, response, error) = unsafe { (data.as_ref(), response.as_ref(), error.as_ref()) };
            let status = response
                .and_then(|r| r.downcast_ref::<NSHTTPURLResponse>())
                .map_or(200, |r| r.statusCode());
            let result = match (data, error) {
                (_, Some(e)) => Err(e.localizedDescription().to_string()),
                _ if !(200..300).contains(&status) => Err(format!("the server answered {status}")),
                (Some(data), None) => Ok(data.to_vec()),
                (None, None) => Err("nothing came back".into()),
            };
            let _ = sent.send(result);
        },
    );
    // SAFETY: the block owns only a channel sender, which may be used from any thread.
    let task = unsafe {
        NSURLSession::sharedSession().dataTaskWithRequest_completionHandler(&request, &done)
    };
    task.resume();
    let bytes = answer
        .recv_timeout(std::time::Duration::from_secs(75))
        .map_err(|_| anyhow::anyhow!("could not fetch {url}: no answer"))?
        .map_err(|why| anyhow::anyhow!("could not fetch {url}: {why}"))?;
    std::fs::write(to, bytes).with_context(|| format!("writing {}", to.display()))
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// The page's title, without the site's name, and its article.
struct Reading {
    title: Option<String>,
    article: String,
}

fn title(html: &str) -> Option<String> {
    read(html).title
}

/// The article's markup, with the page's furniture, chrome and boilerplate taken out.
pub fn article(html: &str) -> String {
    read(html).article
}

fn read(html: &str) -> Reading {
    let doc = Document::from(html);
    let (head, site) = head_title(&doc);
    doc.select(FURNITURE).remove();
    let anchor = anchor(&doc, head.as_deref(), site.as_deref());
    let title = match (&anchor, head) {
        (Some(h), Some(head)) if head.to_lowercase().contains(&words(&h.text()).to_lowercase()) => {
            Some(words(&h.text()))
        }
        (_, head) => head,
    };
    let Some(body) = doc.body() else {
        return Reading {
            title,
            article: String::new(),
        };
    };
    // The headline and every element around it, which no rule may take out.
    let kept: HashSet<NodeId> = anchor
        .iter()
        .flat_map(|a| std::iter::once(*a).chain(a.ancestors_it(None)))
        .map(|n| n.id)
        .collect();
    let holds = |node: &NodeRef| kept.contains(&node.id);

    for node in doc.select("header").nodes() {
        if !holds(node) {
            node.remove_from_parent();
        }
    }
    let counts = Counts::of(&body);
    let total = counts.prose(&body).max(1);
    for node in body.descendants() {
        if !node.is_element() || holds(&node) {
            continue;
        }
        if let Some(strength) = unlikely(&node) {
            // A wrapper named for its sidebar still holds the article; a comments section may
            // outweigh the article it follows, and goes anyway.
            if strength == Strength::Strong || counts.prose(&node) * 2 < total {
                node.remove_from_parent();
            }
        }
    }

    let counts = Counts::of(&body);
    for node in body.descendants() {
        if node.is_element() && !holds(&node) && linky(&node, &counts) {
            node.remove_from_parent();
        }
    }
    cut_back_matter(&body);
    let counts = Counts::of(&body);
    let total = counts.prose(&body);
    if total == 0 {
        return Reading {
            title,
            article: body.inner_html().to_string(),
        };
    }
    let mut chosen = body;
    for node in body.descendants() {
        let block = node.node_name().is_some_and(|n| PROSE.contains(&&*n));
        if node.is_element() && !block && counts.prose(&node) * 10 >= total * 8 {
            chosen = node;
        }
    }
    // Prose split across sibling sections — an introduction apart from the body — is
    // gathered by climbing while what an ancestor adds still reads as prose. A level that adds
    // only a caption is passed through; one that adds mostly links ends the climb.
    let core = chosen;
    for parent in core.ancestors_it(None).take_while(|_| core.id != body.id) {
        let extra = counts.prose(&parent).saturating_sub(counts.prose(&core));
        let extra_text = counts.text(&parent).saturating_sub(counts.text(&core));
        let extra_links = counts.links(&parent).saturating_sub(counts.links(&core));
        if extra_links * 100 > extra_text * 35 {
            break;
        }
        if extra >= (total * 3 / 100).max(150) {
            chosen = parent;
        }
        if parent.id == body.id {
            break;
        }
    }
    // Code is never boilerplate: highlighting puts each bracket in a span of its own.
    let code: HashSet<NodeId> = chosen
        .descendants_it()
        .filter(|n| matches!(n.node_name().as_deref(), Some("pre" | "code")))
        .flat_map(|n| n.descendants())
        .map(|n| n.id)
        .collect();
    // Innermost first, so a block emptied by its children's removal is judged on what is left.
    for node in chosen.descendants().into_iter().rev() {
        let judged = node.is_element() && !holds(&node) && !code.contains(&node.id);
        if judged && (boilerplate(&node) || label(&node)) {
            node.remove_from_parent();
        }
    }
    if let Some(h1) = anchor.filter(|_| holds(&chosen)) {
        above_headline(&chosen, &h1, &kept, total);
        let headline = words(&h1.text());
        for node in chosen.descendants() {
            if node.id != h1.id && heading_level(&node).is_some() && words(&node.text()) == headline {
                node.remove_from_parent();
            }
        }
    }
    let mut article = String::new();
    if let Some(h1) = anchor.filter(|a| !holds(&chosen) && a.id != chosen.id) {
        article.push_str(&h1.html());
        if let Some(dek) = standfirst(&h1).filter(|d| d.id != chosen.id) {
            article.push_str(&dek.html());
        }
    }
    article.push_str(&chosen.inner_html());
    Reading { title, article }
}

/// What sits above the headline inside the article — a kicker, the title again, a player —
/// goes when it is short; a long run there is an introduction set above it.
fn above_headline(chosen: &NodeRef, h1: &NodeRef, kept: &HashSet<NodeId>, total: usize) {
    let mut before = Vec::new();
    for node in chosen.descendants_it() {
        if node.id == h1.id {
            break;
        }
        let top = node.parent().is_some_and(|p| kept.contains(&p.id));
        if node.is_element() && !kept.contains(&node.id) && top {
            before.push(node);
        }
    }
    let text: usize = before.iter().map(|n| words(&n.text()).len()).sum();
    if text < (total * 3 / 100).max(300) {
        for node in before {
            node.remove_from_parent();
        }
    }
}

/// Never prose, wherever it sits.
const FURNITURE: &str = "script, style, noscript, template, svg, iframe, object, embed, canvas, \
    form, button, input, select, textarea, label, dialog, nav, footer, aside, link, meta, \
    video, audio, [hidden], [aria-hidden=true], [role=navigation], [role=banner], \
    [role=contentinfo], [role=complementary], [role=dialog], [role=search], [role=menu], \
    [role=menubar], [role=toolbar], [role=tablist], a[rel=author], a[href*=\"/author/\"], \
    .mw-editsection, sup.reference, .noprint, .navbox, .infobox, .sidebar, .metadata, .ambox, \
    .hatnote, .side-box, .mw-jump-link, .catlinks, .printfooter, .mw-references-wrap, .reflist, \
    #toc, .toc";

fn words(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The head's title without the site's name, and the site's name where the page gives it.
fn head_title(doc: &Document) -> (Option<String>, Option<String>) {
    let meta = |name: &str| {
        doc.select(&format!("meta[property=\"{name}\"], meta[name=\"{name}\"]"))
            .attr("content")
            .map(|c| words(&c))
            .filter(|c| !c.is_empty())
    };
    let site = meta("og:site_name");
    let head = meta("og:title")
        .or_else(|| Some(words(&doc.select("title").text())).filter(|t| !t.is_empty()));
    (head.map(|h| unsuffixed(&h, site.clone())), site)
}

/// The article's own heading: the h1 the head's title names, preferring one inside the
/// article over a site banner's; else the first in an `<article>` or `<main>`. An h1 that is
/// the site's name is a banner — a book's title over each chapter — and the chapter's h2
/// stands in.
fn anchor<'a>(doc: &'a Document, head: Option<&str>, site: Option<&str>) -> Option<NodeRef<'a>> {
    let head = head.map(str::to_lowercase);
    let site = site.map(str::to_lowercase);
    let named = |h: &NodeRef| {
        let text = words(&h.text()).to_lowercase();
        !text.is_empty() && head.as_ref().is_some_and(|t| t.contains(&text))
    };
    let banner = |h: &NodeRef| {
        let text = words(&h.text()).to_lowercase();
        site.as_ref().is_some_and(|s| *s == text)
            || head.as_ref().is_some_and(|t| t != &text && t.ends_with(&text))
    };
    let inside = |h: &NodeRef| {
        h.ancestors_it(None)
            .any(|a| matches!(a.node_name().as_deref(), Some("article" | "main")))
    };
    let h1s: Vec<NodeRef> = doc.select("h1").nodes().iter().copied().filter(|h| !banner(h)).collect();
    let h2s: Vec<NodeRef> = doc.select("h2").nodes().to_vec();
    let pick = |headings: &[NodeRef<'a>]| {
        let named: Vec<NodeRef> = headings.iter().copied().filter(|h| named(h)).collect();
        named
            .iter()
            .find(|h| inside(h))
            .or_else(|| named.iter().max_by_key(|h| h.text().len()))
            .copied()
    };
    pick(&h1s)
        .or_else(|| pick(&h2s))
        .or_else(|| h1s.iter().find(|h| inside(h)).copied())
        .or_else(|| h1s.first().copied())
}

/// A summary set under the headline, before the article proper.
fn standfirst<'a>(h1: &NodeRef<'a>) -> Option<NodeRef<'a>> {
    let next = h1.next_element_sibling()?;
    let text = words(&next.text());
    let named = tokens(&next).iter().any(|t| DEKS.contains(&t.as_str()));
    let prose = next.node_name().as_deref() == Some("p");
    let count = text.split(' ').count();
    ((2..=80).contains(&count) && (prose || named)).then_some(next)
}

const DEKS: &[&str] = &[
    "standfirst", "dek", "deck", "subtitle", "subheadline", "subhead", "summary", "lede",
    "excerpt", "description", "strapline", "teaser",
];

/// Characters of text, of linked text, and of prose — text whose nearest block is a
/// paragraph, a list item, a quotation or code — under each node.
struct Counts {
    text: HashMap<NodeId, usize>,
    links: HashMap<NodeId, usize>,
    prose: HashMap<NodeId, usize>,
}

const PROSE: &[&str] = &["p", "li", "blockquote", "pre", "dd", "td"];
const BLOCK: &[&str] = &[
    "p", "li", "blockquote", "pre", "dd", "td", "div", "section", "article", "main", "body",
    "ul", "ol", "dl", "table", "figure", "figcaption", "h1", "h2", "h3", "h4", "h5", "h6",
];

impl Counts {
    fn of(root: &NodeRef) -> Self {
        let mut counts = Self {
            text: HashMap::new(),
            links: HashMap::new(),
            prose: HashMap::new(),
        };
        for node in root.descendants_it().filter(NodeRef::is_text) {
            let len = words(&node.text()).chars().count();
            if len == 0 {
                continue;
            }
            let mut linked = false;
            let mut block: Option<bool> = None;
            for up in node.ancestors_it(None) {
                let name = up.node_name();
                let name = name.as_deref().unwrap_or("");
                linked |= name == "a";
                if block.is_none() && BLOCK.contains(&name) {
                    block = Some(PROSE.contains(&name));
                }
                *counts.text.entry(up.id).or_default() += len;
                if linked {
                    *counts.links.entry(up.id).or_default() += len;
                }
                if block == Some(true) {
                    *counts.prose.entry(up.id).or_default() += len;
                }
                if up.id == root.id {
                    break;
                }
            }
        }
        counts
    }

    fn text(&self, node: &NodeRef) -> usize {
        self.text.get(&node.id).copied().unwrap_or(0)
    }
    fn links(&self, node: &NodeRef) -> usize {
        self.links.get(&node.id).copied().unwrap_or(0)
    }
    fn prose(&self, node: &NodeRef) -> usize {
        self.prose.get(&node.id).copied().unwrap_or(0)
    }
}

/// The words of an element's class and id, camelCase split: `TagList-styles__x` is
/// tag, list, styles, x.
fn tokens(node: &NodeRef) -> Vec<String> {
    let names = format!(
        "{} {}",
        node.attr("class").unwrap_or_default(),
        node.attr("id").unwrap_or_default()
    );
    let mut out = Vec::new();
    for part in names.split(|c: char| !c.is_ascii_alphanumeric()) {
        let mut word = String::new();
        let mut lower_before = false;
        for c in part.chars() {
            if c.is_ascii_uppercase() && lower_before && !word.is_empty() {
                out.push(std::mem::take(&mut word));
            }
            lower_before = c.is_ascii_lowercase() || c.is_ascii_digit();
            word.push(c.to_ascii_lowercase());
        }
        if !word.is_empty() {
            out.push(word);
        }
    }
    out
}

#[derive(PartialEq)]
enum Strength {
    Weak,
    Strong,
}

/// Comment threads: removed even when they outweigh the article.
const THREADS: &[&str] = &["comment", "comments", "discussion", "disqus", "replies", "responses"];

/// Chrome that names itself. Words, not substrings: "ad" is not "header" or "shadow".
const CHROME: &[&str] = &[
    "related", "recommended", "recommendations", "share", "sharing", "sharebar", "social",
    "newsletter", "subscribe", "subscription", "signup", "promo", "promotion", "advert",
    "advertisement", "ad", "ads", "adslot", "sponsor", "sponsored", "breadcrumb", "breadcrumbs",
    "tag", "tags", "taglist", "topics", "popup", "modal", "cookie", "cookies", "consent", "gdpr",
    "paywall", "sidebar", "widget", "skip", "enlarge", "credit", "credits", "copyright",
    "toolbar", "feedback", "masthead", "menu", "menubar", "dropdown", "flyout", "lightbox",
    "pagination", "pager", "outbrain", "taboola", "bio", "rail", "offscreen", "visuallyhidden",
    "embed", "audio", "player", "byline", "date", "dateline", "dateblock", "timestamp",
    "pubdate",
];

/// Two-word names whose words alone mean nothing: "sr-only", "visually-hidden".
const CHROME_PAIRS: &[(&str, &str)] = &[
    ("sr", "only"),
    ("visually", "hidden"),
    ("screen", "reader"),
    ("read", "more"),
    ("more", "stories"),
    ("author", "info"),
    ("about", "author"),
];

fn unlikely(node: &NodeRef) -> Option<Strength> {
    // A heading's id is its text, made into an anchor: "tags-git-refs", "sidebar-nav-2".
    if heading_level(node).is_some()
        || matches!(
            node.node_name().as_deref(),
            Some("html" | "body" | "main" | "article")
        )
    {
        return None;
    }
    let words = tokens(node);
    if words.iter().any(|w| THREADS.contains(&w.as_str())) {
        return Some(Strength::Strong);
    }
    let pair = words
        .windows(2)
        .any(|w| CHROME_PAIRS.contains(&(w[0].as_str(), w[1].as_str())));
    (pair || words.iter().any(|w| CHROME.contains(&w.as_str()))).then_some(Strength::Weak)
}

/// A block that is mostly links: related stories, tag lists, a row of site sections.
fn linky(node: &NodeRef, counts: &Counts) -> bool {
    let text = counts.text(node);
    if text == 0 {
        return false;
    }
    let links = counts.links(node);
    match node.node_name().as_deref() {
        Some("ul" | "ol" | "dl" | "div" | "section" | "table" | "figure") => {
            text < 2000 && links * 10 > text * 6
        }
        Some("p") => text < 200 && links * 10 > text * 8,
        _ => false,
    }
}

/// Sections after the article that are not read: its sources, links out, and the site's
/// suggestions. Notes stay; an essay's notes are part of it.
const BACK_MATTER: &[&str] = &[
    "references", "citations", "sources", "bibliography", "works cited", "external links",
    "further reading", "see also", "related", "related articles", "related posts",
    "related stories", "related reading", "comments", "responses", "you may also like",
    "you might also like", "recommended", "read next", "more stories", "most read",
    "most popular", "trending",
];

fn cut_back_matter(body: &NodeRef) {
    let headings: Vec<NodeRef> = body
        .descendants()
        .into_iter()
        .filter(|n| heading_level(n).is_some_and(|l| l >= 2))
        .collect();
    for heading in headings {
        let name = words(&heading.text()).to_lowercase();
        let name = name.trim_end_matches(':');
        if !BACK_MATTER.contains(&name) {
            continue;
        }
        let level = heading_level(&heading).unwrap_or(2);
        // Wikipedia wraps each heading in a div of its own; the section is that div's siblings.
        let block = heading
            .parent()
            .filter(|p| p.element_children().len() == 1 && p.node_name().as_deref() == Some("div"))
            .unwrap_or(heading);
        let mut next = block.next_element_sibling();
        while let Some(sibling) = next {
            let opens = |n: &NodeRef| heading_level(n).is_some_and(|l| l <= level);
            if opens(&sibling) || sibling.first_element_child().is_some_and(|c| opens(&c)) {
                break;
            }
            next = sibling.next_element_sibling();
            sibling.remove_from_parent();
        }
        block.remove_from_parent();
    }
}

fn heading_level(node: &NodeRef) -> Option<usize> {
    let name = node.node_name()?;
    let level = name.strip_prefix('h')?.parse().ok()?;
    (1..=6).contains(&level).then_some(level)
}

/// A line that is the page talking about itself: when it was posted, how to share it, whose
/// photograph this is.
fn boilerplate(node: &NodeRef) -> bool {
    if !matches!(
        node.node_name().as_deref(),
        Some("p" | "div" | "span" | "li" | "figcaption" | "time" | "small" | "a")
    ) {
        return false;
    }
    let text = words(&node.text()).to_lowercase();
    let text = text.trim_end_matches(['.', ':', '!']);
    if text.split(' ').count() > 12 {
        return false;
    }
    // What a removed byline leaves: "– |".
    if !text.is_empty() && !text.chars().any(char::is_alphanumeric) {
        return true;
    }
    if text.is_empty() {
        return false;
    }
    // A date on its own line, not one inside a sentence.
    if node.node_name().as_deref() == Some("time") {
        let around = node.parent().map_or(0, |p| words(&p.text()).split(' ').count());
        return around <= text.split(' ').count() + 2;
    }
    const SAID: &[&str] = &[
        "read more", "learn more", "enlarge this image", "advertisement", "share", "copy link",
        "subscribers only", "back to top", "back to full list", "skip to content",
        "skip to main content", "continue reading", "story continues below advertisement",
        "image", "image caption", "toggle caption", "hide caption", "listen to this article",
        "sign up", "subscribe", "print", "email", "share this article", "getty images",
        "shutterstock", "reuters", "associated press", "afp", "alamy", "epa", "read later",
        "transcript", "download", "embed", "listen", "save", "bookmark",
    ];
    const OPENS: &[&str] = &[
        "image source,", "image caption,", "media caption,", "photo:", "photo by ",
        "photo illustration by ", "illustration by ", "credit:", "image credit", "courtesy of ",
        "view full discussion", "view all comments",
    ];
    const AGENCIES: &[&str] = &["getty images", "ap", "reuters", "afp", "npr", "shutterstock"];
    SAID.contains(&text)
        || OPENS.iter().any(|o| text.starts_with(o))
        || AGENCIES.iter().any(|a| text.ends_with(&format!("/{a}")))
        || ago(text)
        || counted(text)
        || credit(&words(&node.text()))
}

/// The language named over a code sample — "http", "js" — which the sample does not need.
fn label(node: &NodeRef) -> bool {
    let text = words(&node.text());
    let precedes_code = node
        .next_element_sibling()
        .is_some_and(|n| n.node_name().as_deref() == Some("pre") || n.is("div:has(> pre:first-child)"));
    precedes_code && !text.is_empty() && !text.contains(' ') && text.len() <= 16
}

/// "Merrill Sherman/Quanta Magazine": a photographer and their publication.
fn credit(text: &str) -> bool {
    let count = text.split(' ').count();
    text.contains('/')
        && count <= 8
        && text
            .split([' ', '/'])
            .all(|w| w.chars().next().is_some_and(|c| c.is_uppercase()))
}

/// "18 hours ago", "a day ago".
fn ago(text: &str) -> bool {
    let Some(rest) = text.strip_suffix(" ago") else {
        return false;
    };
    let mut parts = rest.split(' ');
    let (Some(n), Some(unit), None) = (parts.next(), parts.next(), parts.next()) else {
        return false;
    };
    let number = n.parse::<u32>().is_ok() || matches!(n, "a" | "an" | "one");
    let unit = unit.trim_end_matches('s');
    number
        && matches!(
            unit,
            "sec" | "second" | "min" | "minute" | "hr" | "hour" | "day" | "week" | "month" | "year"
        )
}

/// "214 comments", "3 replies".
fn counted(text: &str) -> bool {
    let mut parts = text.split(' ');
    let (Some(n), Some(what), None) = (parts.next(), parts.next(), parts.next()) else {
        return false;
    };
    n.replace(',', "").parse::<u32>().is_ok()
        && matches!(what, "comment" | "comments" | "reply" | "replies" | "response" | "responses")
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
    fn comments_go_even_when_they_outweigh_the_article() {
        let comment = "<p>A reader's reply, long enough to read as prose on its own here.</p>".repeat(20);
        let html = format!(
            "<html><body><article><h1>Post</h1><p>The post itself, which is short.</p></article>\
             <section id=\"comments\">{comment}</section></body></html>"
        );
        let body = article(&html);
        assert!(body.contains("The post itself") && !body.contains("reply"), "{body}");
    }

    #[test]
    fn an_introduction_in_a_section_of_its_own_is_kept() {
        let intro = "<p>The opening, set apart in a section of its own above the rest.</p>".repeat(3);
        let rest = "<p>The body of the piece, which holds most of what there is to read.</p>".repeat(14);
        let html = format!(
            "<html><body><h1>Piece</h1><section><div>{intro}</div></section>\
             <section><div><div>{rest}</div></div></section></body></html>"
        );
        assert!(article(&html).contains("The opening"));
    }

    #[test]
    fn code_keeps_its_brackets_and_loses_its_language_label() {
        let prose = "<p>Some prose around the sample, so the page has an article.</p>".repeat(8);
        let html = format!(
            "<html><body><article>{prose}<div><div><span>js</span></div>\
             <pre><code><span>foo</span><span>(</span><span>)</span></code></pre></div></article></body></html>"
        );
        let body = article(&html);
        assert!(body.contains("<span>(</span><span>)</span>") && !body.contains(">js<"), "{body}");
    }

    #[test]
    fn a_book_title_over_a_chapter_is_not_the_title() {
        let prose = "<p>The chapter's text, long enough to count as prose on its own.</p>".repeat(8);
        let html = format!(
            "<html><head><title>Ownership - The Book</title></head><body>\
             <div class=\"menu\"><h1>The Book</h1></div><main><h2>Ownership</h2>{prose}</main></body></html>"
        );
        assert_eq!(title(&html).as_deref(), Some("Ownership"));
    }

    #[test]
    fn sources_are_cut_and_notes_kept() {
        let prose = "<p>The article, long enough to count as prose on its own here.</p>".repeat(8);
        let html = format!(
            "<html><body><article><h1>A</h1>{prose}<h2>Notes</h2><p>A note worth hearing.</p>\
             <h2>References</h2><ol><li>Smith, 2020.</li></ol></article></body></html>"
        );
        let body = article(&html);
        assert!(body.contains("worth hearing") && !body.contains("Smith"), "{body}");
    }

    #[test]
    fn an_error_page_sent_as_a_success_is_refused() {
        let page = b"<html><head><title>Page not found</title></head><body><p>Sorry.</p></body></html>";
        assert!(page_of(page).is_err());
        let bot = b"<html><head><title>Just a moment...</title></head><body><p>Checking.</p></body></html>";
        assert!(page_of(bot).is_err());
    }

    fn page_of(bytes: &[u8]) -> Result<(String, String)> {
        page("https://example.com/a", bytes)
    }

    #[test]
    fn a_page_without_an_article_falls_back_to_its_body() {
        let html = "<html><body><header>Site</header><p>Only prose.</p></body></html>";
        assert_eq!(article(html).trim(), "<p>Only prose.</p>");
        assert!(is_url("https://example.com/a") && !is_url("book.pdf"));
    }
}
