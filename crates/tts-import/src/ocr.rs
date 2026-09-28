//! Text from pixels, through Vision: a scanned page, or a photograph of one.
//!
//! One function answers for both callers, the importer that wants the text and the reader that
//! wants every word's rectangle, and it caches what it read. Two passes over the same page would
//! cost twice and could disagree, and a disagreement between the text and the rectangles is a
//! highlight on the wrong word.
//!
//! Pages are rendered at a fixed size, so the result is a function of the page rather than of
//! whoever asked first.

use anyhow::{Context, Result};
use objc2::rc::Retained;
use objc2::AnyThread;
use objc2_core_foundation::{CFRetained, CGPoint, CGRect, CGSize};
use objc2_core_graphics::{
    CGBitmapContextCreate, CGBitmapContextCreateImage, CGColorSpace, CGContext, CGImageAlphaInfo,
};
use objc2_foundation::{NSArray, NSDictionary, NSRange};
use objc2_pdf_kit::{PDFDisplayBox, PDFDocument};
use objc2_vision::{
    VNImageRequestHandler, VNRecognizeTextRequest, VNRecognizedTextObservation, VNRequest,
    VNRequestTextRecognitionLevel,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Pixels on the longer side a page is rendered at. Vision's accurate mode reads 8-point type
/// cleanly at this size, and past it the time grows faster than the words recognised.
const LONG_SIDE: f64 = 2400.0;

/// One recognised word, in the page's own points from the top-left.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub text: String,
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

/// One line as Vision found it, words in reading order.
#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    pub words: Vec<Word>,
}

impl Line {
    pub fn text(&self) -> String {
        self.words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

type Key = (PathBuf, u64, std::time::SystemTime, usize);
static CACHE: Mutex<Option<HashMap<Key, Vec<Line>>>> = Mutex::new(None);

fn key(path: &Path, page: usize) -> Option<Key> {
    let meta = std::fs::metadata(path).ok()?;
    Some((
        std::fs::canonicalize(path).ok()?,
        meta.len(),
        meta.modified().ok()?,
        page,
    ))
}

fn cached(key: &Option<Key>, read: impl FnOnce() -> Result<Vec<Line>>) -> Result<Vec<Line>> {
    if let Some(k) = key {
        if let Some(hit) = CACHE
            .lock()
            .expect("ocr cache")
            .get_or_insert_with(HashMap::new)
            .get(k)
        {
            return Ok(hit.clone());
        }
    }
    let lines = read()?;
    if let Some(k) = key.clone() {
        CACHE
            .lock()
            .expect("ocr cache")
            .get_or_insert_with(HashMap::new)
            .insert(k, lines.clone());
    }
    Ok(lines)
}

/// The lines on one page of a PDF.
pub fn pdf_page(path: &Path, document: &PDFDocument, index: usize) -> Result<Vec<Line>> {
    cached(&key(path, index), || {
        // SAFETY: PDFKit calls on a document the caller owns, on this thread.
        let page = unsafe { document.pageAtIndex(index) }.context("no such page")?;
        let bounds = unsafe { page.boundsForBox(PDFDisplayBox::CropBox) };
        let (width, height) = (bounds.size.width, bounds.size.height);
        anyhow::ensure!(width > 0.0 && height > 0.0, "the page has no size");
        let scale = LONG_SIDE / width.max(height);
        let context = bitmap(
            (width * scale).round() as usize,
            (height * scale).round() as usize,
        )?;
        CGContext::scale_ctm(Some(&context), scale, scale);
        unsafe { page.drawWithBox_toContext(PDFDisplayBox::CropBox, &context) };
        CGContext::flush(Some(&context));
        let image = CGBitmapContextCreateImage(Some(&context)).context("the page as an image")?;
        // SAFETY: a CGImage we own and empty options.
        let handler = unsafe {
            VNImageRequestHandler::initWithCGImage_options(
                VNImageRequestHandler::alloc(),
                &image,
                &NSDictionary::new(),
            )
        };
        recognise(&handler, width, height)
    })
}

/// The lines in an image file, in its pixels; `(width, height)` of the image comes back too.
pub fn image(path: &Path) -> Result<(Vec<Line>, (f64, f64))> {
    let (w, h) = image_size(path)?;
    let lines = cached(&key(path, 0), || {
        let url = crate::file_url(path)?;
        // SAFETY: a file URL and empty options.
        let handler = unsafe {
            VNImageRequestHandler::initWithURL_options(
                VNImageRequestHandler::alloc(),
                &url,
                &NSDictionary::new(),
            )
        };
        recognise(&handler, w, h)
    })?;
    Ok((lines, (w, h)))
}

fn image_size(path: &Path) -> Result<(f64, f64)> {
    let url = crate::file_url(path)?;
    let image = objc2_app_kit::NSImage::initByReferencingURL(objc2_app_kit::NSImage::alloc(), &url);
    let reps = image.representations();
    let rep = reps.firstObject().context("the image has no pixels")?;
    let (w, h) = (rep.pixelsWide(), rep.pixelsHigh());
    anyhow::ensure!(w > 0 && h > 0, "the image has no size");
    Ok((w as f64, h as f64))
}

fn bitmap(width: usize, height: usize) -> Result<CFRetained<CGContext>> {
    let space = CGColorSpace::new_device_rgb().context("device RGB colour space")?;
    let context = unsafe {
        CGBitmapContextCreate(
            std::ptr::null_mut(),
            width,
            height,
            8,
            width * 4,
            Some(&space),
            CGImageAlphaInfo::PremultipliedLast.0,
        )
    }
    .context("creating a bitmap context")?;
    // A PDF page has no background of its own; drawn onto transparency, dark text vanishes.
    CGContext::set_rgb_fill_color(Some(&context), 1.0, 1.0, 1.0, 1.0);
    CGContext::fill_rect(
        Some(&context),
        CGRect {
            origin: CGPoint { x: 0.0, y: 0.0 },
            size: CGSize {
                width: width as f64,
                height: height as f64,
            },
        },
    );
    Ok(context)
}

fn recognise(handler: &VNImageRequestHandler, width: f64, height: f64) -> Result<Vec<Line>> {
    let request = unsafe { VNRecognizeTextRequest::init(VNRecognizeTextRequest::alloc()) };
    request.setRecognitionLevel(VNRequestTextRecognitionLevel::Accurate);
    request.setUsesLanguageCorrection(true);
    request.setAutomaticallyDetectsLanguage(true);
    let as_request: Retained<VNRequest> =
        Retained::into_super(Retained::into_super(request.clone()));
    let requests = NSArray::from_retained_slice(&[as_request]);
    handler
        .performRequests_error(&requests)
        .map_err(|e| anyhow::anyhow!("{}", e.localizedDescription()))
        .context("text recognition")?;
    let observations = request.results().unwrap_or_default();

    let mut lines = Vec::new();
    for observation in observations.iter() {
        let observation: &VNRecognizedTextObservation = &observation;
        let Some(best) = observation.topCandidates(1).firstObject() else {
            continue;
        };
        let text = best.string().to_string();
        let whole = unsafe { observation.boundingBox() };
        let mut words = Vec::new();
        let mut offset = 0usize;
        for piece in text.split(' ') {
            let units = piece.encode_utf16().count();
            if !piece.is_empty() {
                let range = NSRange::new(offset, units);
                // Vision's rectangles are normalised, from the bottom-left.
                let rect = unsafe { best.boundingBoxForRange_error(range) }
                    .ok()
                    .map(|b| unsafe { b.boundingBox() })
                    .unwrap_or(whole);
                words.push(Word {
                    text: piece.to_string(),
                    x: rect.origin.x * width,
                    y: (1.0 - rect.origin.y - rect.size.height) * height,
                    w: rect.size.width * width,
                    h: rect.size.height * height,
                });
            }
            offset += units + 1;
        }
        if !words.is_empty() {
            lines.push(Line { words });
        }
    }
    // Top to bottom, then left to right: Vision's order is close to this, not guaranteed.
    lines.sort_by(|a, b| {
        let (ay, by) = (a.words[0].y, b.words[0].y);
        if (ay - by).abs() > a.words[0].h.min(b.words[0].h) / 2.0 {
            ay.total_cmp(&by)
        } else {
            a.words[0].x.total_cmp(&b.words[0].x)
        }
    });
    Ok(lines)
}

/// Lines to prose: one paragraph where the lines run on, a blank line where the gap between
/// them opens up or the line before stopped short on a full stop. A word hyphenated across a line
/// end is rejoined.
pub fn paragraphs(lines: &[Line]) -> String {
    let mut out = String::new();
    let widest = lines
        .iter()
        .map(|l| l.words.last().map_or(0.0, |w| w.x + w.w) - l.words[0].x)
        .fold(0.0f64, f64::max);
    for (i, line) in lines.iter().enumerate() {
        let text = line.text();
        if i > 0 {
            let before = &lines[i - 1];
            let gap = line.words[0].y - (before.words[0].y + before.words[0].h);
            let span = before.words.last().map_or(0.0, |w| w.x + w.w) - before.words[0].x;
            let closed = before.text().trim_end().ends_with(['.', '!', '?', ':']);
            let new_paragraph = gap > before.words[0].h * 0.9 || (closed && span < widest * 0.85);
            if new_paragraph {
                out.push_str("\n\n");
            } else if out.ends_with('-') && text.starts_with(|c: char| c.is_lowercase()) {
                out.pop();
            } else {
                out.push(' ');
            }
        }
        out.push_str(&text);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str, y: f64, right: f64) -> Line {
        let words: Vec<&str> = text.split(' ').collect();
        let step = right / words.len() as f64;
        Line {
            words: words
                .iter()
                .enumerate()
                .map(|(i, t)| Word {
                    text: t.to_string(),
                    x: i as f64 * step,
                    y,
                    w: step * 0.9,
                    h: 10.0,
                })
                .collect(),
        }
    }

    #[test]
    fn lines_run_on_and_a_short_closing_line_ends_a_paragraph() {
        let lines = vec![
            line("The first line of a para-", 0.0, 400.0),
            line("graph continues here and", 14.0, 400.0),
            line("ends.", 28.0, 60.0),
            line("A second paragraph.", 42.0, 400.0),
        ];
        assert_eq!(
            paragraphs(&lines),
            "The first line of a paragraph continues here and ends.\n\nA second paragraph."
        );
    }

    /// A one-page PDF with `text` set in Helvetica: enough for PDFKit to draw.
    fn one_page_pdf(text: &str) -> Vec<u8> {
        let stream = format!("BT /F1 28 Tf 60 700 Td ({text}) Tj ET");
        let objects = [
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R \
             /Resources << /Font << /F1 5 0 R >> >> >>"
                .to_string(),
            format!(
                "<< /Length {} >>\nstream\n{stream}\nendstream",
                stream.len()
            ),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
        ];
        let mut out = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for (i, body) in objects.iter().enumerate() {
            offsets.push(out.len());
            out.extend(format!("{} 0 obj\n{body}\nendobj\n", i + 1).bytes());
        }
        let xref = out.len();
        out.extend(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).bytes());
        for o in offsets {
            out.extend(format!("{o:010} 00000 n \n").bytes());
        }
        out.extend(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
                objects.len() + 1
            )
            .bytes(),
        );
        out
    }

    /// A page drawn with known text, read back: the whole round trip, on this machine.
    #[test]
    fn a_rendered_page_reads_back() {
        let dir = std::env::temp_dir().join("tts-import-ocr");
        std::fs::create_dir_all(&dir).unwrap();
        let pdf = dir.join("page.pdf");
        std::fs::write(&pdf, one_page_pdf("Reading from pixels works")).unwrap();
        let document = crate::pdf::open_document(&pdf).unwrap();
        let lines = pdf_page(&pdf, &document, 0).unwrap();
        let text: String = lines.iter().map(Line::text).collect::<Vec<_>>().join(" ");
        assert!(text.contains("Reading from pixels works"), "{text:?}");
        assert!(lines[0].words.iter().all(|w| w.w > 0.0 && w.h > 0.0));
    }
}
