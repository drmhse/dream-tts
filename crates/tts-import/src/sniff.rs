//! What a file is, from its first bytes.
//!
//! Extensions lie — a book saved without one, an `.html` renamed `.txt`, a download named
//! `document` — and every format here but plain text has a signature. A zip is opened far enough
//! to tell EPUB from ODT from DOCX, which all start with the same four bytes.

use crate::zipped::Archive;
use crate::Format;
use std::io::Read;
use std::path::Path;

pub fn format(path: &Path) -> Option<Format> {
    if path.is_dir() {
        // `.rtfd` is a directory; nothing else here is.
        return path.join("TXT.rtf").is_file().then_some(Format::Rtfd);
    }
    let mut head = [0u8; 4096];
    let n = std::fs::File::open(path).ok()?.read(&mut head).ok()?;
    of_bytes(&head[..n], path)
}

pub fn of_bytes(head: &[u8], path: &Path) -> Option<Format> {
    if head.starts_with(b"%PDF-") {
        return Some(Format::Pdf);
    }
    if head.starts_with(b"{\\rtf") {
        return Some(Format::Rtf);
    }
    if head.starts_with(b"PK\x03\x04") {
        return zipped(path);
    }
    // An OLE compound file: Word 97 is the one this reads. Excel and PowerPoint share the
    // container, and are told apart by the extension the caller already has.
    if head.starts_with(&[0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1]) {
        let ppt = path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("ppt"));
        return Some(if ppt {
            Format::Presentation
        } else {
            Format::Doc
        });
    }
    if head.starts_with(b"bplist00") && contains(head, b"WebMainResource") {
        return Some(Format::WebArchive);
    }
    let image = head.starts_with(b"\x89PNG")
        || head.starts_with(&[0xFF, 0xD8, 0xFF])
        || head.starts_with(b"II*\0")
        || head.starts_with(b"MM\0*")
        || head.starts_with(b"GIF8")
        // "BM" alone opens plenty of prose; a bitmap also has its header's size where it says.
        || (head.starts_with(b"BM")
            && matches!(head.get(14..18), Some([12 | 40 | 52 | 56 | 108 | 124, 0, 0, 0])))
        || (head.starts_with(b"RIFF") && head.get(8..12) == Some(b"WEBP"))
        || (head.get(4..8) == Some(b"ftyp")
            && matches!(head.get(8..12), Some(b"heic" | b"heix" | b"mif1" | b"heif" | b"avif")));
    if image {
        return Some(Format::Image);
    }
    if head.get(60..68) == Some(b"BOOKMOBI") {
        return Some(Format::Ebook);
    }
    let text = crate::text::decode(&head[..head.len().min(1024)]).to_ascii_lowercase();
    let text = text.trim_start_matches('\u{feff}').trim_start();
    if text.starts_with("<?xml") && text.contains("<fictionbook") {
        return Some(Format::Fb2);
    }
    if text.starts_with("<!doctype html")
        || text.starts_with("<html")
        || (text.starts_with("<?xml") && text.contains("<html"))
    {
        return Some(Format::Html);
    }
    None
}

fn zipped(path: &Path) -> Option<Format> {
    let mut archive = Archive::open(path).ok()?;
    if let Some(mime) = archive.read_opt("mimetype") {
        match mime.trim() {
            "application/epub+zip" => return Some(Format::Epub),
            "application/vnd.oasis.opendocument.text" => return Some(Format::Odt),
            "application/vnd.oasis.opendocument.presentation" => return Some(Format::Presentation),
            _ => {}
        }
    }
    if archive.has("word/document.xml") {
        return Some(Format::Docx);
    }
    if archive.has("ppt/presentation.xml") {
        return Some(Format::Presentation);
    }
    if archive.has("Index/Document.iwa") || archive.has("index.apxl") || archive.has("index.xml") {
        return Some(Format::IWork);
    }
    if archive.has("META-INF/container.xml") {
        return Some(Format::Epub);
    }
    None
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signatures_outrank_extensions() {
        let p = Path::new("book.txt");
        assert_eq!(of_bytes(b"%PDF-1.7\n", p), Some(Format::Pdf));
        assert_eq!(of_bytes(b"{\\rtf1\\ansi", p), Some(Format::Rtf));
        assert_eq!(
            of_bytes(b"\n  <!DOCTYPE html><html>", p),
            Some(Format::Html)
        );
        assert_eq!(
            of_bytes(b"<?xml version=\"1.0\"?><FictionBook xmlns=\"x\">", p),
            Some(Format::Fb2)
        );
        assert_eq!(of_bytes(b"Just some prose.", p), None);
        assert_eq!(
            of_bytes(b"BMW sold more cars than last year, and ", p),
            None
        );
    }

    #[test]
    fn a_zip_is_opened_far_enough_to_name_it() {
        let dir = std::env::temp_dir().join("tts-import-sniff");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("no-extension");
        let file = std::fs::File::create(&path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        let stored = zip::write::SimpleFileOptions::default();
        zip.start_file("mimetype", stored).unwrap();
        std::io::Write::write_all(&mut zip, b"application/vnd.oasis.opendocument.text").unwrap();
        zip.finish().unwrap();
        assert_eq!(format(&path), Some(Format::Odt));
    }
}
