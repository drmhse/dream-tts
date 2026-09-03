//! DOCX, ODT and EPUB are all a zip of XML, so their one shared need lives here.

use anyhow::{Context, Result};
use std::io::Read;
use std::path::Path;

pub struct Archive {
    inner: zip::ZipArchive<std::io::BufReader<std::fs::File>>,
}

impl Archive {
    pub fn open(path: &Path) -> Result<Self> {
        let file =
            std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let inner = zip::ZipArchive::new(std::io::BufReader::new(file))
            .with_context(|| format!("{} is not a readable zip archive", path.display()))?;
        Ok(Self { inner })
    }

    /// One entry, as text. Missing is an error with the path named, because every caller
    /// asks for a part the format guarantees.
    pub fn read(&mut self, name: &str) -> Result<String> {
        let mut entry = self
            .inner
            .by_name(name)
            .with_context(|| format!("the archive has no `{name}`"))?;
        let mut buf = Vec::new();
        entry
            .read_to_end(&mut buf)
            .with_context(|| format!("reading `{name}`"))?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    /// One entry, as text, or `None` when it is absent. For parts a format only sometimes
    /// carries.
    pub fn read_opt(&mut self, name: &str) -> Option<String> {
        self.read(name).ok()
    }
}
