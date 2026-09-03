//! XML plumbing the three zip formats share.

use quick_xml::Reader;

/// An element or attribute name with its namespace prefix dropped, lower-cased.
///
/// Every format here is namespaced — `w:p`, `text:h`, an EPUB's `opf:item` — and which
/// prefix a producer picks is its own choice, so matching on the qualified name matches one
/// producer's output rather than the format. XHTML also arrives upper-cased.
pub fn local(raw: &[u8]) -> String {
    let name = String::from_utf8_lossy(raw);
    name.rsplit(':')
        .next()
        .unwrap_or(&name)
        .to_ascii_lowercase()
}

/// A reader that tolerates unmatched tags.
///
/// Every one of these formats is produced by tools that emit markup a strict parser rejects
/// and a browser or a word processor renders fine. Refusing such a document would be worse
/// than extracting what is there, so end-name checking is off and a parse error stops
/// extraction rather than failing the import.
pub fn reader(xml: &str) -> Reader<&[u8]> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().check_end_names = false;
    reader
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixes_are_dropped_and_case_is_folded() {
        assert_eq!(local(b"w:p"), "p");
        assert_eq!(local(b"text:outline-level"), "outline-level");
        assert_eq!(local(b"H1"), "h1");
        assert_eq!(local(b"p"), "p");
    }
}
