//! Formats read by becoming another: an image is a one-page PDF, a Kindle book an EPUB, a slide
//! deck a PDF, a reStructuredText file markdown.
//!
//! Only the image step is built in. The rest use a tool when one is installed — Calibre,
//! LibreOffice, Pandoc — and say which to install when it is not, rather than half-reading a
//! format this crate has no reader for. Results are cached by the source's path, size and time.

use crate::Format;
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// A path the importers read directly: `path` itself, or what it was converted to.
pub fn readable(path: &Path) -> Result<PathBuf> {
    let Some(format) = Format::detect(path) else {
        return Ok(path.to_path_buf());
    };
    let target = match format {
        Format::Image => "pdf",
        Format::Ebook => "epub",
        Format::Presentation | Format::IWork => "pdf",
        Format::Pandoc => "md",
        _ => return Ok(path.to_path_buf()),
    };
    let out = cached(path, target)?;
    if out.is_file() {
        return Ok(out);
    }
    let partial = out.with_extension(format!("partial.{target}"));
    match format {
        Format::Image => image_to_pdf(path, &partial)?,
        Format::Ebook => ebook_to_epub(path, &partial)?,
        Format::IWork if iwork_preview(path, &partial)? => {}
        Format::Presentation | Format::IWork => office_to_pdf(path, &partial)?,
        Format::Pandoc => pandoc_to_markdown(path, &partial)?,
        _ => unreachable!("matched above"),
    }
    std::fs::rename(&partial, &out).with_context(|| format!("keeping {}", out.display()))?;
    Ok(out)
}

fn cached(path: &Path, extension: &str) -> Result<PathBuf> {
    use std::hash::{Hash, Hasher};
    let meta = std::fs::metadata(path).with_context(|| format!("reading {}", path.display()))?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::fs::canonicalize(path)?.hash(&mut hasher);
    meta.len().hash(&mut hasher);
    meta.modified().ok().hash(&mut hasher);
    let dir = cache_root().join("converted");
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let stem = path
        .file_stem()
        .map_or("document".into(), |s| s.to_string_lossy().into_owned());
    Ok(dir.join(format!("{stem}-{:016x}.{extension}", hasher.finish())))
}

/// Where converted documents and fetched pages are kept. `DREAM_TTS_CONVERT_CACHE` moves it.
pub fn cache_root() -> PathBuf {
    if let Some(dir) = std::env::var_os("DREAM_TTS_CONVERT_CACHE") {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME").map(PathBuf::from);
    match home {
        Some(h) if cfg!(target_os = "macos") => h.join("Library/Caches/dream-tts"),
        Some(h) => h.join(".cache/dream-tts"),
        None => std::env::temp_dir().join("dream-tts"),
    }
}

/// A tool on `PATH`, or where its installer puts it.
fn tool(name: &str, also: &[&str]) -> Option<PathBuf> {
    let on_path = std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|dir| dir.join(name))
            .find(|p| p.is_file())
    });
    on_path.or_else(|| also.iter().map(PathBuf::from).find(|p| p.is_file()))
}

fn run(mut command: Command, what: &str) -> Result<()> {
    let output = command
        .output()
        .with_context(|| format!("running {what}"))?;
    if !output.status.success() {
        let why = String::from_utf8_lossy(&output.stderr);
        let why = why
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("no reason given");
        bail!("{what} could not convert it: {why}");
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn image_to_pdf(path: &Path, out: &Path) -> Result<()> {
    use objc2::AnyThread;
    use objc2_foundation::NSString;
    use objc2_pdf_kit::{PDFDocument, PDFPage};
    let url = crate::file_url(path)?;
    let image = objc2_app_kit::NSImage::initByReferencingURL(objc2_app_kit::NSImage::alloc(), &url);
    anyhow::ensure!(
        image.isValid(),
        "{} is not an image the system can read",
        path.display()
    );
    // SAFETY: PDFKit calls on objects created here, on this thread.
    unsafe {
        let page = PDFPage::initWithImage(PDFPage::alloc(), &image)
            .with_context(|| format!("{} would not become a page", path.display()))?;
        let document = PDFDocument::init(PDFDocument::alloc());
        document.insertPage_atIndex(&page, 0);
        anyhow::ensure!(
            document.writeToFile(&NSString::from_str(out.to_str().context("utf-8 path")?)),
            "writing {}",
            out.display()
        );
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn image_to_pdf(path: &Path, _out: &Path) -> Result<()> {
    bail!(
        "{} is an image; reading one needs macOS's Vision",
        path.display()
    )
}

fn ebook_to_epub(path: &Path, out: &Path) -> Result<()> {
    let Some(convert) = tool(
        "ebook-convert",
        &["/Applications/calibre.app/Contents/MacOS/ebook-convert"],
    ) else {
        bail!(
            "{} is a Kindle book. Calibre converts it: `brew install --cask calibre`, then open \
             it again. A book with DRM cannot be opened at all.",
            path.display()
        );
    };
    let mut command = Command::new(convert);
    command.arg(path).arg(out);
    run(command, "Calibre").map_err(|e| {
        let text = format!("{e:#}");
        if text.to_ascii_lowercase().contains("drm") {
            anyhow::anyhow!("{} is DRM-protected, and cannot be opened", path.display())
        } else {
            e
        }
    })
}

/// Keynote and Pages files often carry their own rendering; it is exact where a converter is not.
fn iwork_preview(path: &Path, out: &Path) -> Result<bool> {
    let Ok(mut archive) = crate::zipped::Archive::open(path) else {
        return Ok(false);
    };
    for name in ["QuickLook/Preview.pdf", "preview.pdf"] {
        if let Some(bytes) = archive.bytes(name) {
            std::fs::write(out, bytes).with_context(|| format!("writing {}", out.display()))?;
            return Ok(true);
        }
    }
    Ok(false)
}

fn office_to_pdf(path: &Path, out: &Path) -> Result<()> {
    let Some(office) = tool(
        "soffice",
        &["/Applications/LibreOffice.app/Contents/MacOS/soffice"],
    ) else {
        bail!(
            "{} is a slide deck or an iWork document. LibreOffice converts it: `brew install \
             --cask libreoffice`, then open it again.",
            path.display()
        );
    };
    // LibreOffice names its output after the input, in a directory it is given.
    let dir = out.with_extension("lo");
    std::fs::create_dir_all(&dir)?;
    let mut command = Command::new(office);
    command
        .args(["--headless", "--convert-to", "pdf", "--outdir"])
        .arg(&dir)
        .arg(path);
    run(command, "LibreOffice")?;
    let made = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.extension().is_some_and(|x| x == "pdf"))
        .context("LibreOffice wrote no PDF")?;
    std::fs::rename(&made, out)?;
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

fn pandoc_to_markdown(path: &Path, out: &Path) -> Result<()> {
    let Some(pandoc) = tool(
        "pandoc",
        &["/opt/homebrew/bin/pandoc", "/usr/local/bin/pandoc"],
    ) else {
        bail!(
            "{} is read through Pandoc: `brew install pandoc`, then open it again.",
            path.display()
        );
    };
    let mut command = Command::new(pandoc);
    command
        .arg(path)
        .args(["-t", "commonmark", "--wrap=none", "-o"])
        .arg(out);
    run(command, "Pandoc")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_tool_names_the_install() {
        let dir = std::env::temp_dir().join("tts-import-convert");
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("DREAM_TTS_CONVERT_CACHE", dir.join("cache"));
        let book = dir.join("book.mobi");
        let mut bytes = vec![0u8; 60];
        bytes.extend(b"BOOKMOBI");
        bytes.extend([0u8; 64]);
        std::fs::write(&book, bytes).unwrap();
        assert_eq!(Format::detect(&book), Some(Format::Ebook));
        if tool(
            "ebook-convert",
            &["/Applications/calibre.app/Contents/MacOS/ebook-convert"],
        )
        .is_none()
        {
            let err = readable(&book).unwrap_err().to_string();
            assert!(err.contains("brew install --cask calibre"), "{err}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn an_image_becomes_a_page() {
        let dir = std::env::temp_dir().join("tts-import-convert-image");
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("DREAM_TTS_CONVERT_CACHE", dir.join("cache"));
        // A 2x2 24-bit BMP, which has no checksums to get wrong.
        let mut bmp = b"BM".to_vec();
        bmp.extend(70u32.to_le_bytes());
        bmp.extend([0u8; 4]);
        bmp.extend(54u32.to_le_bytes());
        bmp.extend(40u32.to_le_bytes());
        bmp.extend(2i32.to_le_bytes());
        bmp.extend(2i32.to_le_bytes());
        bmp.extend(1u16.to_le_bytes());
        bmp.extend(24u16.to_le_bytes());
        bmp.extend([0u8; 24]);
        bmp.extend([255u8; 16]);
        let png = bmp;
        let image = dir.join("scan");
        std::fs::write(&image, png).unwrap();
        assert_eq!(Format::detect(&image), Some(Format::Image));
        let pdf = readable(&image).unwrap();
        assert!(std::fs::read(&pdf).unwrap().starts_with(b"%PDF"));
    }
}
