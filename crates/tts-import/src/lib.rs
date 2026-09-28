//! Any document, as the markdown the narration pipeline already consumes.
//!
//! The pipeline gets its chapter structure from the *filesystem* — `chapter-NNN.md`, one per
//! file — and four working stages hang off that: verbalisation, WAV, delivery encode,
//! alignment. So this is one stage in front of them and changes none of them:
//!
//! ```text
//! document  ->  chapter-001.md, chapter-002.md, ...  ->  [the existing pipeline]
//! ```
//!
//! Which means the work is not extracting text. It is **splitting one monolithic document
//! into chapters**, and that is what decides which formats are easy: EPUB's spine *is* the
//! chapter list, DOCX and ODT mark chapters with a heading style, and PDF has no semantics
//! at all beyond its outline.
//!
//! Markdown is the intermediate rather than plain text because `tts-narrate` is built for
//! it, and because a heading is what produces the 320 ms paragraph gap a listener hears as
//! a section break.

#[cfg(target_os = "macos")]
pub mod attributed;
pub mod convert;
pub mod docx;
pub mod epub;
pub mod fb2;
pub mod html;
#[cfg(target_os = "macos")]
pub mod ocr;
pub mod odt;
#[cfg(target_os = "macos")]
pub mod pdf;
pub mod rtf;
pub mod sniff;
pub mod text;
pub mod web;

mod markdown;
mod xml;
mod zipped;

/// A path as the file URL every system framework here takes.
#[cfg(target_os = "macos")]
pub(crate) fn file_url(path: &Path) -> Result<objc2::rc::Retained<objc2_foundation::NSURL>> {
    let path = path
        .to_str()
        .with_context(|| format!("{} is not valid UTF-8", path.display()))?;
    Ok(objc2_foundation::NSURL::fileURLWithPath(
        &objc2_foundation::NSString::from_str(path),
    ))
}

use anyhow::{bail, Context, Result};
use std::path::Path;

/// One narratable unit: what becomes a single `chapter-NNN.md`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chapter {
    /// The heading the chapter was split on, if it had one. Written as an `# H1` so the
    /// converter gives it its own paragraph.
    pub title: Option<String>,
    /// Markdown body, without the title.
    pub body: String,
}

impl Chapter {
    /// The file's contents: the title as an H1, then the body.
    pub fn markdown(&self) -> String {
        match &self.title {
            Some(t) if !t.trim().is_empty() => format!("# {}\n\n{}\n", t.trim(), self.body.trim()),
            _ => format!("{}\n", self.body.trim()),
        }
    }

    fn is_empty(&self) -> bool {
        self.body.trim().is_empty() && self.title.as_deref().unwrap_or("").trim().is_empty()
    }
}

/// A whole document, split.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    pub title: Option<String>,
    pub chapters: Vec<Chapter>,
    /// Which importer produced this, for the CLI to report.
    pub format: Format,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Markdown,
    Text,
    Html,
    Docx,
    Odt,
    Epub,
    Pdf,
    Rtf,
    /// Word 97. Read by the system's text system, so macOS only.
    Doc,
    /// A directory bundle of RTF and its attachments. macOS only.
    Rtfd,
    /// Safari's saved page. macOS only.
    WebArchive,
    Fb2,
    /// A photograph or scan of a page: made a PDF, then read from its pixels. macOS only.
    Image,
    /// MOBI, AZW, AZW3: through Calibre, when it is installed.
    Ebook,
    /// PowerPoint, OpenDocument slides: through LibreOffice.
    Presentation,
    /// Keynote and Pages: their stored preview, else LibreOffice.
    IWork,
    /// reStructuredText, Org, LaTeX, notebooks and the rest of what Pandoc reads.
    Pandoc,
}

impl Format {
    pub fn name(self) -> &'static str {
        match self {
            Self::Markdown => "markdown",
            Self::Text => "plain text",
            Self::Html => "HTML",
            Self::Docx => "DOCX",
            Self::Odt => "ODT",
            Self::Epub => "EPUB",
            Self::Pdf => "PDF",
            Self::Rtf => "RTF",
            Self::Doc => "Word 97",
            Self::Rtfd => "RTFD",
            Self::WebArchive => "web archive",
            Self::Fb2 => "FictionBook",
            Self::Image => "image",
            Self::Ebook => "Kindle book",
            Self::Presentation => "slides",
            Self::IWork => "iWork",
            Self::Pandoc => "Pandoc input",
        }
    }

    /// Recognised by extension. Sniffing the bytes would be more robust in general, but every
    /// format here except plain text has a distinctive extension and a mis-sniffed zip is a
    /// worse failure than a mis-named file.
    pub fn of(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        Some(match ext.as_str() {
            "md" | "markdown" => Self::Markdown,
            "txt" | "text" => Self::Text,
            "html" | "htm" | "xhtml" => Self::Html,
            "docx" => Self::Docx,
            "odt" => Self::Odt,
            "epub" => Self::Epub,
            "pdf" => Self::Pdf,
            "rtf" => Self::Rtf,
            "doc" => Self::Doc,
            "rtfd" => Self::Rtfd,
            "webarchive" => Self::WebArchive,
            "fb2" => Self::Fb2,
            "png" | "jpg" | "jpeg" | "tif" | "tiff" | "heic" | "heif" | "gif" | "bmp" | "webp" => {
                Self::Image
            }
            "mobi" | "azw" | "azw3" | "kf8" | "prc" => Self::Ebook,
            "pptx" | "ppt" | "odp" => Self::Presentation,
            "key" | "pages" => Self::IWork,
            "rst" | "org" | "tex" | "latex" | "ipynb" | "textile" | "wiki" | "mediawiki"
            | "dbk" | "opml" | "man" => Self::Pandoc,
            _ => return None,
        })
    }

    /// Everything this build can read, for an error message that lists the options.
    pub fn supported() -> &'static [&'static str] {
        if cfg!(target_os = "macos") {
            &[
                "md",
                "txt",
                "html",
                "xhtml",
                "docx",
                "odt",
                "epub",
                "pdf",
                "rtf",
                "doc",
                "rtfd",
                "webarchive",
                "fb2",
                "png",
                "jpg",
                "heic",
                "tiff",
                "mobi*",
                "azw3*",
                "pptx*",
                "key*",
                "pages*",
                "rst*",
                "org*",
                "tex*",
                "ipynb*",
            ]
        } else {
            // PDF extraction is PDFKit and `.doc` the system text system, so they are
            // genuinely absent off macOS rather than merely slower. Saying so beats failing
            // at the call.
            &[
                "md", "txt", "html", "xhtml", "docx", "odt", "epub", "rtf", "fb2", "mobi*",
                "pptx*", "rst*",
            ]
        }
    }

    /// What a file is: its contents first, its extension when the contents do not say. A
    /// `.txt` that is really HTML, or a book with no extension at all, imports as what it is.
    pub fn detect(path: &Path) -> Option<Self> {
        sniff::format(path).or_else(|| Self::of(path))
    }
}

/// The whole document in the shared model, unsplit: what a reader pages through.
///
/// Markdown and plain text are taken as written. Anything else goes through its importer and
/// its chapters are rejoined, a chapter boundary being a heading like any other.
pub fn document(path: &Path) -> Result<tts_doc::Document> {
    let path = &convert::readable(path)?;
    let raw = matches!(
        Format::detect(path),
        Some(Format::Markdown | Format::Text) | None
    );
    let markdown = if raw {
        read_text(path)?
    } else {
        import(path)?
            .chapters
            .iter()
            .map(Chapter::markdown)
            .collect::<Vec<_>>()
            .join("\n")
    };
    Ok(tts_doc::Document::from_markdown(&markdown))
}

/// Read and split a document, choosing the importer by extension.
pub fn import(path: &Path) -> Result<Document> {
    let format = Format::detect(path).with_context(|| {
        format!(
            "cannot tell what {} is from its contents or its extension; this build reads {}",
            path.display(),
            Format::supported().join(", ")
        )
    })?;
    import_as(path, format)
}

/// Read and split a document with the importer named explicitly.
pub fn import_as(path: &Path, format: Format) -> Result<Document> {
    if matches!(
        format,
        Format::Image | Format::Ebook | Format::Presentation | Format::IWork | Format::Pandoc
    ) {
        let converted = convert::readable(path)?;
        let as_format = Format::detect(&converted).unwrap_or(Format::Markdown);
        let mut document = import_as(&converted, as_format)?;
        document.format = format;
        return Ok(document);
    }
    let chapters = match format {
        Format::Markdown => split_markdown(&read_text(path)?),
        Format::Text => vec![Chapter {
            title: None,
            body: read_text(path)?,
        }],
        Format::Html => html::chapters(&read_text(path)?)?,
        Format::Docx => docx::chapters(path)?,
        Format::Odt => odt::chapters(path)?,
        Format::Epub => epub::chapters(path)?,
        Format::Rtf => rtf::chapters(path)?,
        Format::Fb2 => fb2::chapters(path)?,
        #[cfg(target_os = "macos")]
        Format::Doc | Format::Rtfd | Format::WebArchive => attributed::chapters(path)?,
        #[cfg(not(target_os = "macos"))]
        Format::Doc | Format::Rtfd | Format::WebArchive => bail!(
            "{} is read by macOS's text system and this is not macOS",
            format.name()
        ),
        Format::Image | Format::Ebook | Format::Presentation | Format::IWork | Format::Pandoc => {
            unreachable!("converted above")
        }
        #[cfg(target_os = "macos")]
        Format::Pdf => pdf::chapters(path)?,
        #[cfg(not(target_os = "macos"))]
        Format::Pdf => bail!(
            "PDF import needs PDFKit and this is not macOS. Convert it first, or use one of: {}",
            Format::supported().join(", ")
        ),
    };
    let chapters: Vec<Chapter> = chapters
        .into_iter()
        .filter(|c: &Chapter| !c.is_empty())
        .collect();
    if chapters.is_empty() {
        bail!(
            "{} yielded no text. An image-only PDF or a document of scanned pages has none to \
             find — this extracts text and does not perform OCR.",
            path.display()
        );
    }
    let title = chapters
        .first()
        .and_then(|c| c.title.clone())
        .filter(|_| chapters.len() > 1);
    Ok(Document {
        title,
        chapters,
        format,
    })
}

fn read_text(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    // Decoded rather than refused: a stray invalid byte in a 400-page document should not
    // stop a narration run, and a Windows-1252 file keeps its accents.
    Ok(text::decode(&bytes))
}

/// Split markdown on its top-level headings.
///
/// The shallowest heading level *present* is the split, not `#` unconditionally: a document
/// whose chapters are `##` under one `#` title would otherwise come out as a single chapter.
pub fn split_markdown(text: &str) -> Vec<Chapter> {
    let level = text.lines().filter_map(heading_level).min().unwrap_or(0);
    if level == 0 {
        return vec![Chapter {
            title: None,
            body: text.to_string(),
        }];
    }
    let mut chapters: Vec<Chapter> = Vec::new();
    let mut title: Option<String> = None;
    let mut body: Vec<&str> = Vec::new();
    for line in text.lines() {
        if heading_level(line) == Some(level) {
            if title.is_some() || !body.join("\n").trim().is_empty() {
                chapters.push(Chapter {
                    title: title.take(),
                    body: body.join("\n"),
                });
                body.clear();
            }
            title = Some(line.trim_start_matches('#').trim().to_string());
        } else {
            body.push(line);
        }
    }
    if title.is_some() || !body.join("\n").trim().is_empty() {
        chapters.push(Chapter {
            title,
            body: body.join("\n"),
        });
    }
    chapters
}

fn heading_level(line: &str) -> Option<usize> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with('#') {
        return None;
    }
    let hashes = trimmed.chars().take_while(|c| *c == '#').count();
    // `#hashtag` is not a heading; markdown requires the space.
    ((1..=6).contains(&hashes) && trimmed[hashes..].starts_with(' ')).then_some(hashes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_splits_on_its_shallowest_heading() {
        let chapters = split_markdown("## One\n\nalpha\n\n## Two\n\nbeta\n");
        assert_eq!(chapters.len(), 2);
        assert_eq!(chapters[0].title.as_deref(), Some("One"));
        assert_eq!(chapters[1].body.trim(), "beta");
    }

    #[test]
    fn a_preamble_before_the_first_heading_is_its_own_chapter() {
        let chapters = split_markdown("front matter prose\n\n# One\n\nalpha\n");
        assert_eq!(chapters.len(), 2);
        assert_eq!(chapters[0].title, None);
        assert_eq!(chapters[0].body.trim(), "front matter prose");
    }

    #[test]
    fn a_document_with_no_headings_is_one_chapter() {
        let chapters = split_markdown("just prose\n");
        assert_eq!(chapters.len(), 1);
        assert_eq!(chapters[0].title, None);
    }

    #[test]
    fn a_hashtag_is_not_a_heading() {
        assert_eq!(heading_level("#hashtag"), None);
        assert_eq!(heading_level("# Heading"), Some(1));
        assert_eq!(heading_level("####### too deep"), None);
    }

    #[test]
    fn a_chapter_writes_its_title_as_a_heading() {
        let c = Chapter {
            title: Some("One".into()),
            body: "alpha".into(),
        };
        assert_eq!(c.markdown(), "# One\n\nalpha\n");
        let untitled = Chapter {
            title: None,
            body: "alpha".into(),
        };
        assert_eq!(untitled.markdown(), "alpha\n");
    }

    #[test]
    fn extensions_map_to_importers() {
        assert_eq!(Format::of(Path::new("a.EPUB")), Some(Format::Epub));
        assert_eq!(Format::of(Path::new("a.docx")), Some(Format::Docx));
        assert_eq!(Format::of(Path::new("a.rtf")), Some(Format::Rtf));
        assert_eq!(Format::of(Path::new("a.xlsx")), None);
        assert_eq!(Format::of(Path::new("noext")), None);
    }
}
