//! End-to-end import of the three zip formats, from archives built here.
//!
//! Built rather than checked in, for two reasons: a binary fixture in git is opaque to
//! review, and generating one needs a tool CI would then depend on. The parts written here
//! are the minimum each format's spec requires, which is also the interesting case — real
//! files carry far more, and an importer that needs the extra is over-fitted to one
//! producer's output.

use std::io::Write;
use std::path::{Path, PathBuf};
use tts_import::{import, Format};
use zip::write::SimpleFileOptions;

/// Cross-format agreement: the same source content, in three containers, must import to the
/// same chapters. Any importer that disagrees with the other two is the one that is wrong.
const EXPECTED_TITLES: [&str; 2] = ["Chapter One", "Chapter Two"];

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tts-import-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir.join(name)
}

fn write_archive(path: &Path, entries: &[(&str, &str)]) {
    let file = std::fs::File::create(path).expect("create archive");
    let mut zip = zip::ZipWriter::new(file);
    for (name, body) in entries {
        zip.start_file(*name, SimpleFileOptions::default())
            .expect("start entry");
        zip.write_all(body.as_bytes()).expect("write entry");
    }
    zip.finish().expect("finish archive");
}

#[test]
fn docx_imports_its_headings_and_paragraphs() {
    let path = scratch("book.docx");
    write_archive(
        &path,
        &[(
            "word/document.xml",
            r#"<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>
  <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Chapter One</w:t></w:r></w:p>
  <w:p><w:r><w:t>First prose.</w:t></w:r></w:p>
  <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Chapter Two</w:t></w:r></w:p>
  <w:p><w:r><w:t>Second prose.</w:t></w:r></w:p>
</w:body></w:document>"#,
        )],
    );
    let doc = import(&path).expect("import docx");
    assert_eq!(doc.format, Format::Docx);
    let titles: Vec<&str> = doc
        .chapters
        .iter()
        .filter_map(|c| c.title.as_deref())
        .collect();
    assert_eq!(titles, EXPECTED_TITLES);
    assert_eq!(doc.chapters[1].body.trim(), "Second prose.");
}

#[test]
fn odt_imports_its_outline_levels() {
    let path = scratch("book.odt");
    write_archive(
        &path,
        &[(
            "content.xml",
            r#"<office:document-content
  xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
  xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
<office:body><office:text>
  <text:h text:outline-level="1">Chapter One</text:h>
  <text:p>First prose.</text:p>
  <text:h text:outline-level="1">Chapter Two</text:h>
  <text:p>Second prose.</text:p>
</office:text></office:body></office:document-content>"#,
        )],
    );
    let doc = import(&path).expect("import odt");
    assert_eq!(doc.format, Format::Odt);
    let titles: Vec<&str> = doc
        .chapters
        .iter()
        .filter_map(|c| c.title.as_deref())
        .collect();
    assert_eq!(titles, EXPECTED_TITLES);
}

#[test]
fn epub_follows_its_spine() {
    let path = scratch("book.epub");
    write_archive(
        &path,
        &[
            ("mimetype", "application/epub+zip"),
            (
                "META-INF/container.xml",
                r#"<container><rootfiles>
                    <rootfile full-path="OEBPS/content.opf"/>
                </rootfiles></container>"#,
            ),
            (
                "OEBPS/content.opf",
                r#"<package><manifest>
                    <item id="c1" href="ch1.xhtml"/>
                    <item id="c2" href="ch2.xhtml"/>
                   </manifest>
                   <spine><itemref idref="c1"/><itemref idref="c2"/></spine>
                </package>"#,
            ),
            (
                "OEBPS/ch1.xhtml",
                "<html><body><h1>Chapter One</h1><p>First prose.</p></body></html>",
            ),
            (
                "OEBPS/ch2.xhtml",
                "<html><body><h1>Chapter Two</h1><p>Second prose.</p></body></html>",
            ),
        ],
    );
    let doc = import(&path).expect("import epub");
    assert_eq!(doc.format, Format::Epub);
    let titles: Vec<&str> = doc
        .chapters
        .iter()
        .filter_map(|c| c.title.as_deref())
        .collect();
    assert_eq!(titles, EXPECTED_TITLES);
}

/// The spine, not the file order in the zip, decides reading order — and an EPUB whose
/// entries are stored out of order is common enough that this is worth pinning.
#[test]
fn epub_reading_order_comes_from_the_spine_not_the_zip() {
    let path = scratch("reordered.epub");
    write_archive(
        &path,
        &[
            (
                "META-INF/container.xml",
                r#"<container><rootfiles><rootfile full-path="content.opf"/></rootfiles></container>"#,
            ),
            ("second.xhtml", "<h1>Chapter Two</h1><p>b</p>"),
            ("first.xhtml", "<h1>Chapter One</h1><p>a</p>"),
            (
                "content.opf",
                r#"<package><manifest>
                    <item id="a" href="first.xhtml"/>
                    <item id="b" href="second.xhtml"/>
                   </manifest>
                   <spine><itemref idref="a"/><itemref idref="b"/></spine></package>"#,
            ),
        ],
    );
    let doc = import(&path).expect("import epub");
    let titles: Vec<&str> = doc
        .chapters
        .iter()
        .filter_map(|c| c.title.as_deref())
        .collect();
    assert_eq!(titles, EXPECTED_TITLES);
}

#[test]
fn a_missing_part_names_what_was_missing() {
    let path = scratch("empty.docx");
    write_archive(&path, &[("other.xml", "<x/>")]);
    let err = import(&path).expect_err("a docx without word/document.xml is not importable");
    assert!(err.to_string().contains("word/document.xml"), "{err:#}");
}

#[test]
fn a_document_with_no_text_says_so_rather_than_yielding_nothing() {
    let path = scratch("blank.docx");
    write_archive(
        &path,
        &[("word/document.xml", "<w:document><w:body/></w:document>")],
    );
    let err = import(&path).expect_err("no text is an error, not an empty success");
    assert!(err.to_string().contains("no text"), "{err:#}");
}

#[test]
fn an_unknown_extension_lists_what_is_supported() {
    let path = scratch("book.rtf");
    std::fs::write(&path, "{\\rtf1}").expect("write");
    let err = import(&path).expect_err("rtf is not supported");
    let message = err.to_string();
    assert!(message.contains("epub"), "{message}");
}

#[test]
fn markdown_and_text_need_no_container() {
    let md = scratch("plain.md");
    std::fs::write(&md, "# One\n\nalpha\n\n# Two\n\nbeta\n").expect("write");
    let doc = import(&md).expect("import markdown");
    assert_eq!(doc.format, Format::Markdown);
    assert_eq!(doc.chapters.len(), 2);

    let txt = scratch("plain.txt");
    std::fs::write(&txt, "just prose, no structure\n").expect("write");
    let doc = import(&txt).expect("import text");
    assert_eq!(doc.format, Format::Text);
    assert_eq!(
        doc.chapters.len(),
        1,
        "plain text has no chapter signal to split on"
    );
}
