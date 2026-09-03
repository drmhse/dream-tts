//! PDF import against a real PDF, generated here.
//!
//! The outline is the whole reason this goes through PDFKit rather than a pure-Rust
//! extractor: it is the table of contents the file itself declares, naming both the chapter
//! and the page it starts on. That is real evidence, unlike inferring headings from font
//! size — so it needs a test against a real file, not a mocked one.
//!
//! Written by hand because the alternative is a checked-in binary fixture nobody can review,
//! or a build dependency on a PDF generator. A PDF is a handful of objects plus a
//! byte-offset table, and computing those offsets is the only fiddly part.

#![cfg(target_os = "macos")]

use std::path::PathBuf;
use tts_import::{import, Format};

/// A minimal PDF: `pages` pages of text, and an outline entry per `(page, title)` mark.
fn write_pdf(path: &PathBuf, pages: &[&str], marks: &[(usize, &str)]) {
    // Object numbering: 1 catalog, 2 page tree, 3..3+n pages, then one content stream per
    // page, then the outline root and its children, then the font.
    let first_page = 3usize;
    let first_content = first_page + pages.len();
    let outline_root = first_content + pages.len();
    let first_mark = outline_root + 1;
    // The outline objects are only emitted when there are marks, so with none the font takes
    // the slot the root would have used. Getting this wrong pointed every page at a font
    // object that did not exist, and PDFKit then extracted no text at all — which looked
    // exactly like a scanned document.
    let font = if marks.is_empty() {
        outline_root
    } else {
        first_mark + marks.len()
    };

    let mut objects: Vec<String> = Vec::new();

    objects.push(if marks.is_empty() {
        "<< /Type /Catalog /Pages 2 0 R >>".to_string()
    } else {
        format!("<< /Type /Catalog /Pages 2 0 R /Outlines {outline_root} 0 R >>")
    });

    let kids: Vec<String> = (0..pages.len())
        .map(|i| format!("{} 0 R", first_page + i))
        .collect();
    objects.push(format!(
        "<< /Type /Pages /Kids [{}] /Count {} >>",
        kids.join(" "),
        pages.len()
    ));

    for i in 0..pages.len() {
        objects.push(format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {} 0 R \
             /Resources << /Font << /F1 {font} 0 R >> >> >>",
            first_content + i
        ));
    }

    for text in pages {
        // `Tj` with one string per line is enough: PDFKit reads the text, and layout is not
        // what this test is about.
        let stream = format!("BT /F1 12 Tf 72 720 Td ({}) Tj ET", escape(text));
        objects.push(format!(
            "<< /Length {} >>\nstream\n{stream}\nendstream",
            stream.len()
        ));
    }

    if !marks.is_empty() {
        objects.push(format!(
            "<< /Type /Outlines /First {first_mark} 0 R /Last {} 0 R /Count {} >>",
            first_mark + marks.len() - 1,
            marks.len()
        ));
        for (i, (page, title)) in marks.iter().enumerate() {
            let mut entry = format!(
                "<< /Title ({}) /Parent {outline_root} 0 R /Dest [{} 0 R /Fit]",
                escape(title),
                first_page + page
            );
            if i > 0 {
                entry.push_str(&format!(" /Prev {} 0 R", first_mark + i - 1));
            }
            if i + 1 < marks.len() {
                entry.push_str(&format!(" /Next {} 0 R", first_mark + i + 1));
            }
            entry.push_str(" >>");
            objects.push(entry);
        }
    }

    objects.push("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string());

    let mut pdf = String::from("%PDF-1.4\n");
    let mut offsets: Vec<usize> = Vec::with_capacity(objects.len());
    for (i, body) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.push_str(&format!("{} 0 obj\n{body}\nendobj\n", i + 1));
    }
    let xref_at = pdf.len();
    pdf.push_str(&format!("xref\n0 {}\n", objects.len() + 1));
    pdf.push_str("0000000000 65535 f \n");
    for offset in &offsets {
        pdf.push_str(&format!("{offset:010} 00000 n \n"));
    }
    pdf.push_str(&format!(
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n",
        objects.len() + 1
    ));

    std::fs::write(path, pdf).expect("write pdf");
}

fn escape(s: &str) -> String {
    s.replace('\\', r"\\")
        .replace('(', r"\(")
        .replace(')', r"\)")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tts-import-pdf-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir.join(name)
}

#[test]
fn pdfkit_extracts_text_from_a_real_pdf() {
    let path = scratch("plain.pdf");
    write_pdf(
        &path,
        &["The first page of prose.", "The second page."],
        &[],
    );
    let doc = import(&path).expect("import pdf");
    assert_eq!(doc.format, Format::Pdf);
    // No outline, so one chapter — a wrong split is worse than none.
    assert_eq!(doc.chapters.len(), 1);
    let body = &doc.chapters[0].body;
    assert!(body.contains("first page of prose"), "{body:?}");
    assert!(body.contains("second page"), "{body:?}");
}

#[test]
fn the_outline_splits_the_document_into_chapters() {
    let path = scratch("outlined.pdf");
    // Each page prints its own heading, as a real document does, so this also covers the
    // heading being dropped from the body that the chapter title already names.
    write_pdf(
        &path,
        &[
            "Front matter page.",
            "CHAPTER ONE Prose of the first.",
            "Still the first.",
            "CHAPTER TWO Prose of the second.",
        ],
        &[(1, "Chapter One"), (3, "Chapter Two")],
    );
    let doc = import(&path).expect("import pdf");
    assert_eq!(
        doc.chapters.len(),
        3,
        "front matter, then the two outline entries"
    );

    assert_eq!(doc.chapters[0].title, None);
    assert!(
        doc.chapters[0].body.contains("Front matter"),
        "{:?}",
        doc.chapters[0].body
    );

    assert_eq!(doc.chapters[1].title.as_deref(), Some("Chapter One"));
    let one = &doc.chapters[1].body;
    assert!(
        one.starts_with("Prose of the first"),
        "the printed heading must go: {one:?}"
    );
    assert!(
        one.contains("Still the first"),
        "pages up to the next mark belong here: {one:?}"
    );

    assert_eq!(doc.chapters[2].title.as_deref(), Some("Chapter Two"));
    let two = &doc.chapters[2].body;
    assert!(two.starts_with("Prose of the second"), "{two:?}");
}

#[test]
fn a_pdf_with_no_text_says_so_rather_than_yielding_an_empty_chapter() {
    let path = scratch("blank.pdf");
    write_pdf(&path, &[""], &[]);
    let err = import(&path).expect_err("a page of no text is an error");
    let message = err.to_string();
    assert!(
        message.contains("OCR"),
        "the message must name the fix: {message}"
    );
}

#[test]
fn a_file_that_is_not_a_pdf_is_reported_as_such() {
    let path = scratch("fake.pdf");
    std::fs::write(&path, "this is not a PDF").expect("write");
    let err = import(&path).expect_err("not a pdf");
    assert!(err.to_string().contains("PDFKit could not open"), "{err:#}");
}
