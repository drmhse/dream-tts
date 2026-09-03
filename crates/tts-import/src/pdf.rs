//! PDF, through PDFKit.
//!
//! PDF is the one format with no structure to read: it describes glyphs at positions, not
//! paragraphs. Two consequences shape this module.
//!
//! **Extraction is delegated.** `PDFPage`'s own `string` applies Quartz's layout analysis —
//! reading order, column detection, de-hyphenation — which is a large body of work no
//! pure-Rust extractor matches. PDFKit is in `/System/Library/Frameworks`, so it costs
//! nothing to install and stays inside the release audit that refuses any non-system dynamic
//! dependency.
//!
//! **Chapters come from the outline, or from nothing.** `PDFOutline` is the table of
//! contents the file itself declares, and it names both the chapter and the page it starts
//! on — real evidence, unlike inferring headings from font size. Without an outline this
//! yields a single chapter rather than guessing, because a wrong split is worse than none:
//! it desynchronises the per-chapter resume the narration pipeline is built on.

use crate::Chapter;
use anyhow::{bail, Context, Result};
use objc2::rc::Retained;
use objc2::AnyThread;
use objc2_foundation::{NSString, NSURL};
use objc2_pdf_kit::{PDFDocument, PDFOutline};
use std::path::Path;

pub fn chapters(path: &Path) -> Result<Vec<Chapter>> {
    let document = open(path)?;
    // SAFETY (every unsafe block in this module): PDFKit calls on a `Retained` handle we
    // own, on the calling thread. Nothing here escapes an autorelease pool boundary and no
    // returned object is used after its owner is dropped.
    let pages = unsafe { document.pageCount() };
    if pages == 0 {
        bail!("{} has no pages", path.display());
    }

    let page_text: Vec<String> = (0..pages).map(|i| text_of_page(&document, i)).collect();
    if page_text.iter().all(|t| t.trim().is_empty()) {
        bail!(
            "{} has {pages} page(s) and no extractable text, which means it is scanned images. \
             This extracts text and does not perform OCR — run it through an OCR tool first.",
            path.display()
        );
    }

    match outline_marks(&document) {
        marks if marks.len() > 1 => Ok(split_at(&page_text, &marks)),
        _ => Ok(vec![Chapter {
            title: None,
            body: paragraphs(&page_text.join("\n\n")),
        }]),
    }
}

fn open(path: &Path) -> Result<Retained<PDFDocument>> {
    let absolute =
        std::fs::canonicalize(path).with_context(|| format!("resolving {}", path.display()))?;
    let as_string = absolute
        .to_str()
        .with_context(|| format!("{} is not valid UTF-8", absolute.display()))?;
    unsafe {
        let url = NSURL::fileURLWithPath(&NSString::from_str(as_string));
        PDFDocument::initWithURL(PDFDocument::alloc(), &url).with_context(|| {
            format!(
                "PDFKit could not open {} — encrypted, or not a PDF",
                path.display()
            )
        })
    }
}

fn text_of_page(document: &PDFDocument, index: usize) -> String {
    unsafe {
        document
            .pageAtIndex(index)
            .and_then(|page| page.string())
            .map(|s| s.to_string())
            .unwrap_or_default()
    }
}

/// `(page index, title)` for each top-level outline entry, in page order.
///
/// Only the top level: a nested outline describes sections within a chapter, and splitting
/// on those would produce a narration file per subsection — hundreds for a textbook, each
/// too short for the engine to batch.
///
/// Entries sharing a page are all kept. An academic paper puts several sections on one page
/// and dropping the later ones lost five of this file's thirteen chapters; `split_at`
/// resolves them by finding the heading inside the page instead.
fn outline_marks(document: &PDFDocument) -> Vec<(usize, String)> {
    let mut marks: Vec<(usize, String)> = Vec::new();
    unsafe {
        let Some(root) = document.outlineRoot() else {
            return marks;
        };
        let count = root.numberOfChildren();
        for i in 0..count {
            let Some(child) = root.childAtIndex(i) else {
                continue;
            };
            let Some(page) = page_index(document, &child) else {
                continue;
            };
            let title = child.label().map(|l| l.to_string()).unwrap_or_default();
            let title = title.trim().to_string();
            if title.is_empty() {
                continue;
            }
            marks.push((page, title));
        }
    }
    // Stable, so entries on one page keep the order the outline gave them — which is the
    // order they appear on the page.
    marks.sort_by_key(|(page, _)| *page);
    marks
}

/// The page an outline entry points at. An entry whose destination PDFKit cannot resolve is
/// skipped rather than guessed at.
unsafe fn page_index(document: &PDFDocument, entry: &PDFOutline) -> Option<usize> {
    let page = entry.destination().and_then(|d| d.page())?;
    Some(document.indexForPage(&page))
}

/// Slice the document at each outline entry.
///
/// An outline destination names a *page*, which is too coarse: this paper puts "Abstract"
/// and "1 Introduction" on page 0, and slicing by page would have to drop one of them and
/// would give the whole page to whichever survived. So each mark is placed at the offset
/// where its own heading text appears within its page, and only falls back to the page
/// boundary when the heading cannot be found there — which happens when the outline label
/// does not match the printed heading, and merging is the honest outcome then.
///
/// A leading run before the first mark becomes an untitled chapter: front matter is real
/// content and is not silently dropped.
fn split_at(pages: &[String], marks: &[(usize, String)]) -> Vec<Chapter> {
    const SEPARATOR: &str = "\n\n";
    let mut joined = String::new();
    let mut page_at: Vec<usize> = Vec::with_capacity(pages.len());
    for page in pages {
        if !joined.is_empty() {
            joined.push_str(SEPARATOR);
        }
        page_at.push(joined.len());
        joined.push_str(page);
    }

    // Where each mark begins. Searching starts after the previous mark on the same page, so
    // a heading whose words recur earlier in the page cannot pull a later chapter backwards.
    let mut cuts: Vec<(usize, &str)> = Vec::with_capacity(marks.len());
    for (page, title) in marks {
        let page_start = page_at.get(*page).copied().unwrap_or(joined.len());
        let page_end = page_at.get(page + 1).map_or(joined.len(), |next| *next);
        let floor = cuts
            .last()
            .map_or(page_start, |(at, _)| (*at).max(page_start));
        let cut = find_heading(&joined[floor..page_end], title)
            .map(|offset| floor + offset)
            .unwrap_or(page_start);
        // Never move backwards: an unfound heading falls back to its page start, which may
        // be before the previous mark when both are on one page.
        let cut = cut.max(cuts.last().map_or(0, |(at, _)| *at));
        cuts.push((cut, title));
    }

    let mut out: Vec<Chapter> = Vec::new();
    if let Some((first, _)) = cuts.first() {
        let front = &joined[..*first];
        if !front.trim().is_empty() {
            out.push(Chapter {
                title: None,
                body: paragraphs(front),
            });
        }
    }
    for (i, (start, title)) in cuts.iter().enumerate() {
        let end = cuts.get(i + 1).map_or(joined.len(), |(next, _)| *next);
        if end <= *start {
            continue;
        }
        let body = strip_leading_heading(&joined[*start..end], title);
        out.push(Chapter {
            title: Some((*title).to_string()),
            body: paragraphs(&body),
        });
    }
    out
}

/// Drop the printed heading from the top of a chapter whose title already carries it.
///
/// The cut lands *on* the heading, so the text keeps it and the title repeats it — the voice
/// would announce every section twice. The printed form is also usually set in capitals, and
/// a run of capitals is spelled letter by letter by this voice rather than read.
fn strip_leading_heading<'a>(body: &'a str, title: &str) -> std::borrow::Cow<'a, str> {
    let needle = collapse(title).0;
    if needle.is_empty() {
        return body.into();
    }
    let (haystack, offsets) = collapse(body);
    let Some(rest) = haystack.strip_prefix(&needle) else {
        return body.into();
    };
    // Map back through the offset table: the boundary is where the remainder begins.
    let consumed = haystack.len() - rest.len();
    match offsets.get(consumed) {
        Some(at) => body[*at..].into(),
        // The heading is the whole chapter, which happens for a section that is a figure.
        None => "".into(),
    }
}

/// The offset of `heading` in `text`, tolerant of how PDF layout broke it.
///
/// A heading printed across two lines arrives as "3 Garnet System\nArchitecture" while the
/// outline label is one line, and the case can differ too. So both sides are compared with
/// whitespace collapsed and case folded, and the match is mapped back to an offset in the
/// original.
fn find_heading(text: &str, heading: &str) -> Option<usize> {
    let needle = collapse(heading).0;
    if needle.is_empty() {
        return None;
    }
    let (haystack, offsets) = collapse(text);
    let at = haystack.find(&needle)?;
    offsets.get(at).copied()
}

/// Whitespace-collapsed, case-folded text, plus each byte's offset in the original.
fn collapse(text: &str) -> (String, Vec<usize>) {
    let mut out = String::with_capacity(text.len());
    let mut offsets: Vec<usize> = Vec::with_capacity(text.len());
    let mut in_space = false;
    for (index, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if !in_space && !out.is_empty() {
                out.push(' ');
                offsets.push(index);
            }
            in_space = true;
            continue;
        }
        in_space = false;
        // `to_lowercase` can yield several chars; each maps back to the same source offset.
        for lowered in ch.to_lowercase() {
            let mut buf = [0u8; 4];
            for _ in lowered.encode_utf8(&mut buf).bytes() {
                offsets.push(index);
            }
            out.push(lowered);
        }
    }
    (out, offsets)
}

/// Rejoin PDFKit's hard-wrapped lines into paragraphs.
///
/// Quartz returns the page's text with a newline per *visual* line, and a newline is a
/// paragraph break to the markdown converter — so a chapter would arrive as several hundred
/// one-line paragraphs, each getting the engine's 320 ms paragraph gap. That reads as a
/// stammer. A blank line is a real break; a single newline is a wrap, unless the line before
/// it ended a sentence.
fn paragraphs(text: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    for block in text.split('\n') {
        let line = block.trim();
        if line.is_empty() {
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            continue;
        }
        if current.is_empty() {
            current.push_str(line);
            continue;
        }
        // A hyphen at a line break is ambiguous: PDF layout hyphenates words ("de-hyphen-\n
        // ation") and also wraps after a compound's own hyphen ("stronger-than-\nexpected"),
        // and nothing in the text distinguishes them without a dictionary. Keep it.
        //
        // That is the safe direction, not the neutral one. Keeping a hyphen that should have
        // gone gives narration three real words, because `clean_inline` spaces a hyphen
        // between lower-case words; dropping one that should have stayed fuses two words
        // into a non-word, which is precisely the input that sends the model into a
        // repetition loop. Words that are slightly wrong beat a token that is not a word.
        if current.ends_with('-') {
            current.push_str(line);
            continue;
        }
        if ends_paragraph(&current) {
            out.push(std::mem::take(&mut current));
            current.push_str(line);
        } else {
            current.push(' ');
            current.push_str(line);
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out.join("\n\n")
}

/// A line ends a paragraph when it ends a sentence *and* is short enough to be the last line
/// of one. A full line ending in a full stop is far more often mid-paragraph.
fn ends_paragraph(line: &str) -> bool {
    line.ends_with(['.', '!', '?', '"']) && line.len() < 60
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapped_lines_rejoin_into_one_paragraph() {
        let text = "This sentence was hard-wrapped\nacross three visual lines by the\nPDF layout.";
        assert_eq!(
            paragraphs(text),
            "This sentence was hard-wrapped across three visual lines by the PDF layout."
        );
    }

    #[test]
    fn a_blank_line_is_a_real_break() {
        assert_eq!(paragraphs("one\n\ntwo"), "one\n\ntwo");
    }

    #[test]
    fn a_hyphen_at_a_line_break_keeps_its_hyphen() {
        // Both readings stay as words. "de-hyphen-ation" narrates as three words, which is
        // mildly wrong; "thanexpected" is a fused non-word, which is damaging.
        assert_eq!(
            paragraphs("de-hyphen-\nation works"),
            "de-hyphen-ation works"
        );
        assert_eq!(
            paragraphs("stronger-than-\nexpected growth"),
            "stronger-than-expected growth"
        );
    }

    #[test]
    fn a_short_sentence_end_starts_a_new_paragraph() {
        assert_eq!(
            paragraphs("The end.\nA new one begins here."),
            "The end.\n\nA new one begins here."
        );
    }

    /// A full line ending in a full stop is mid-paragraph far more often than not, so it must
    /// not split — otherwise most body text becomes one paragraph per line.
    #[test]
    fn a_full_line_ending_in_a_stop_does_not_split() {
        let long = "This line is long enough to be a wrapped line rather than the end of one.";
        let joined = paragraphs(&format!("{long}\nand it continues"));
        assert!(!joined.contains("\n\n"), "{joined:?}");
    }

    /// The case a real paper exposed: five of thirteen chapters were lost because their
    /// outline entries shared a page with an earlier one.
    #[test]
    fn several_marks_on_one_page_are_all_kept() {
        let pages = vec![
            "Abstract\nSome abstract prose.\n1 INTRODUCTION\nIntro prose.".to_string(),
            "2 BACKGROUND\nBackground prose.".to_string(),
        ];
        let marks = vec![
            (0usize, "Abstract".to_string()),
            (0usize, "1 Introduction".to_string()),
            (1usize, "2 Background".to_string()),
        ];
        let chapters = split_at(&pages, &marks);
        let titles: Vec<&str> = chapters.iter().filter_map(|c| c.title.as_deref()).collect();
        assert_eq!(titles, ["Abstract", "1 Introduction", "2 Background"]);
        assert!(
            chapters[0].body.contains("abstract prose"),
            "{:?}",
            chapters[0].body
        );
        assert!(
            !chapters[0].body.contains("Intro prose"),
            "the cut must be at the heading"
        );
        assert!(
            chapters[1].body.contains("Intro prose"),
            "{:?}",
            chapters[1].body
        );
    }

    /// The printed heading is set in capitals and the outline label is not, and a run of
    /// capitals is spelled letter by letter rather than read.
    #[test]
    fn the_printed_heading_is_not_repeated_in_the_body() {
        let pages = vec!["3 GARNET SYSTEM\nARCHITECTURE\nFigure 1 depicts it.".to_string()];
        let marks = vec![(0usize, "3 Garnet System Architecture".to_string())];
        let chapters = split_at(&pages, &marks);
        assert_eq!(chapters.len(), 1);
        assert_eq!(chapters[0].body.trim(), "Figure 1 depicts it.");
    }

    #[test]
    fn a_heading_broken_across_lines_is_still_found() {
        assert_eq!(
            find_heading(
                "x\n3 Garnet System\nArchitecture\ny",
                "3 Garnet System Architecture"
            ),
            Some(2)
        );
        assert_eq!(find_heading("no heading here", "3 Garnet"), None);
    }

    /// An outline label that does not match the printed heading falls back to the page
    /// boundary rather than guessing, and must not move a later mark backwards.
    #[test]
    fn an_unfindable_heading_falls_back_to_its_page() {
        let pages = vec!["Page zero.".to_string(), "Page one prose.".to_string()];
        let marks = vec![
            (0usize, "Findable".to_string()),
            (1usize, "Not printed anywhere".to_string()),
        ];
        let chapters = split_at(&pages, &marks);
        let titles: Vec<&str> = chapters.iter().filter_map(|c| c.title.as_deref()).collect();
        assert_eq!(titles, ["Findable", "Not printed anywhere"]);
        assert!(chapters.last().unwrap().body.contains("Page one prose"));
    }

    #[test]
    fn outline_marks_slice_the_pages() {
        let pages: Vec<String> = (0..6).map(|i| format!("page {i}")).collect();
        let marks = vec![(1usize, "One".to_string()), (4usize, "Two".to_string())];
        let chapters = split_at(&pages, &marks);
        assert_eq!(chapters.len(), 3, "front matter, then two chapters");
        assert_eq!(chapters[0].title, None);
        assert_eq!(chapters[0].body, "page 0");
        assert_eq!(chapters[1].title.as_deref(), Some("One"));
        assert_eq!(chapters[1].body, "page 1\n\npage 2\n\npage 3");
        assert_eq!(chapters[2].body, "page 4\n\npage 5");
    }

    #[test]
    fn a_mark_on_the_first_page_creates_no_front_matter() {
        let pages: Vec<String> = (0..2).map(|i| format!("page {i}")).collect();
        let chapters = split_at(&pages, &[(0, "One".into())]);
        assert_eq!(chapters.len(), 1);
        assert_eq!(chapters[0].title.as_deref(), Some("One"));
    }
}
