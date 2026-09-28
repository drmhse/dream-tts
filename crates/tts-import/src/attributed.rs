//! Word 97 (`.doc`), `.rtfd` and `.webarchive`, read by the system's own text system.
//!
//! `NSAttributedString` opens all three with nothing to install. What it returns is text with
//! fonts, not structure, so headings come from type size: a paragraph set at least a fifth larger
//! than the body is a heading, the largest size level one. A bullet or number and a tab at the
//! start of a paragraph is a list item.

use crate::markdown;
use crate::Chapter;
use anyhow::{Context, Result};
use objc2::rc::Retained;
use objc2::AnyThread;
use objc2_app_kit::{NSAttributedStringDocumentFormats, NSFont, NSFontAttributeName};
use objc2_foundation::{NSAttributedString, NSDictionary, NSRange};
use std::path::Path;

pub fn chapters(path: &Path) -> Result<Vec<Chapter>> {
    Ok(crate::split_markdown(&to_markdown(path)?))
}

pub fn to_markdown(path: &Path) -> Result<String> {
    let url = crate::file_url(path)?;
    let options: Retained<NSDictionary<_, objc2::runtime::AnyObject>> = NSDictionary::new();
    // SAFETY: a file URL and an empty options dictionary, as the method documents.
    let text: Retained<NSAttributedString> = unsafe {
        NSAttributedString::initWithURL_options_documentAttributes_error(
            NSAttributedString::alloc(),
            &url,
            &options,
            None,
        )
    }
    .map_err(|e: Retained<objc2_foundation::NSError>| {
        anyhow::anyhow!("{}", e.localizedDescription())
    })
    .with_context(|| format!("the system could not read {}", path.display()))?;

    let whole = text.string().to_string();
    // Paragraphs with the UTF-16 offset each starts at: attributes are indexed in UTF-16.
    let mut paragraphs: Vec<(String, usize, f64)> = Vec::new();
    let mut at = 0usize;
    for line in whole.split(['\n', '\u{2029}']).collect::<Vec<&str>>() {
        let units = line.encode_utf16().count();
        if !line.trim().is_empty() {
            paragraphs.push((line.to_string(), at, size_at(&text, at)));
        }
        at += units + 1;
    }

    // The body size is the one most characters are set in.
    let mut weights: Vec<(i64, usize)> = Vec::new();
    for (line, _, size) in &paragraphs {
        let key = (size * 2.0).round() as i64;
        match weights.iter_mut().find(|(k, _)| *k == key) {
            Some(w) => w.1 += line.chars().count(),
            None => weights.push((key, line.chars().count())),
        }
    }
    let body = weights
        .iter()
        .max_by_key(|w| w.1)
        .map_or(12.0, |w| w.0 as f64 / 2.0);
    let mut heading_sizes: Vec<i64> = paragraphs
        .iter()
        .filter(|(line, _, size)| *size >= body * 1.2 && line.chars().count() <= 120)
        .map(|(_, _, size)| (size * 2.0).round() as i64)
        .collect();
    heading_sizes.sort_unstable_by(|a, b| b.cmp(a));
    heading_sizes.dedup();

    let mut out = Vec::new();
    for (line, _, size) in &paragraphs {
        let key = (size * 2.0).round() as i64;
        let level = heading_sizes
            .iter()
            .position(|k| *k == key)
            .filter(|_| line.chars().count() <= 120)
            .map(|p| (p + 1).min(6));
        let (in_list, body_text) = match list_item(line) {
            Some(rest) => (true, rest),
            None => (false, line.as_str()),
        };
        if let Some(block) = markdown::block(level, in_list, body_text) {
            out.push(block);
        }
    }
    Ok(markdown::document(&out))
}

fn size_at(text: &NSAttributedString, index: usize) -> f64 {
    if index >= text.length() {
        return 0.0;
    }
    // SAFETY: an index inside the string, and a null effective-range pointer, which the
    // method accepts.
    let font = unsafe {
        text.attribute_atIndex_effectiveRange(
            NSFontAttributeName,
            index,
            std::ptr::null_mut::<NSRange>(),
        )
    };
    font.and_then(|f| f.downcast::<NSFont>().ok())
        .map_or(0.0, |f| f.pointSize())
}

/// A leading bullet or number and a tab, which is how the text system writes a list marker.
fn list_item(line: &str) -> Option<&str> {
    let (marker, rest) = line.split_once('\t')?;
    let marker = marker.trim();
    let bullet = matches!(
        marker,
        "\u{2022}" | "\u{25e6}" | "\u{25aa}" | "-" | "\u{2013}" | "*"
    );
    let numbered = marker
        .trim_end_matches(['.', ')'])
        .chars()
        .all(|c| c.is_ascii_digit() || c.is_ascii_lowercase())
        && marker.len() <= 4
        && marker.ends_with(['.', ')']);
    (bullet || numbered).then_some(rest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_list_marker_is_a_bullet_or_a_number_and_a_tab() {
        assert_eq!(list_item("\u{2022}\tAn item"), Some("An item"));
        assert_eq!(list_item("3.\tThird"), Some("Third"));
        assert_eq!(list_item("Name\tValue"), None);
    }

    #[test]
    fn a_word_document_reads_with_its_headings() {
        let dir = std::env::temp_dir().join("tts-import-attributed");
        std::fs::create_dir_all(&dir).unwrap();
        let rtf = dir.join("sample.rtf");
        std::fs::write(
            &rtf,
            r"{\rtf1\ansi{\fonttbl\f0 Helvetica;}\f0\fs48 Big Title\par\fs24 Body text that runs on.\par}",
        )
        .unwrap();
        // `textutil` writes Word 97, the format this reader exists for.
        let doc = dir.join("sample.doc");
        let made = std::process::Command::new("textutil")
            .args(["-convert", "doc", "-output"])
            .arg(&doc)
            .arg(&rtf)
            .status();
        if !made.is_ok_and(|s| s.success()) {
            eprintln!("skipped: textutil could not write a .doc");
            return;
        }
        let md = to_markdown(&doc).unwrap();
        assert_eq!(md, "# Big Title\n\nBody text that runs on.\n");
    }
}
