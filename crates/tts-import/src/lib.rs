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

pub mod docx;
pub mod epub;
pub mod html;
pub mod odt;
#[cfg(target_os = "macos")]
pub mod pdf;

mod markdown;
mod xml;
mod zipped;

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
            _ => return None,
        })
    }

    /// Everything this build can read, for an error message that lists the options.
    pub fn supported() -> &'static [&'static str] {
        if cfg!(target_os = "macos") {
            &["md", "txt", "html", "docx", "odt", "epub", "pdf"]
        } else {
            // PDF extraction is PDFKit, so it is genuinely absent off macOS rather than
            // merely slower. Saying so beats failing at the call.
            &["md", "txt", "html", "docx", "odt", "epub"]
        }
    }
}

/// Read and split a document, choosing the importer by extension.
pub fn import(path: &Path) -> Result<Document> {
    let format = Format::of(path).with_context(|| {
        format!(
            "cannot tell what {} is from its extension; this build reads {}",
            path.display(),
            Format::supported().join(", ")
        )
    })?;
    import_as(path, format)
}

/// Read and split a document with the importer named explicitly.
pub fn import_as(path: &Path, format: Format) -> Result<Document> {
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
    // Lossy rather than an error: a stray invalid byte in a 400-page document should not
    // stop a narration run, and the replacement character is silent to a TTS engine.
    Ok(String::from_utf8_lossy(&bytes).into_owned())
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
        assert_eq!(Format::of(Path::new("a.rtf")), None);
        assert_eq!(Format::of(Path::new("noext")), None);
    }
}
