//! Turning "narrate this" into a job, and giving it an identity.
//!
//! The identity is the whole design: a job's id is a hash of what will be narrated *and*
//! how, so resubmitting the same book with the same settings finds the run already in
//! progress. Resume therefore needs no remembered id and no `--resume` flag — the user runs
//! the same command again, which is what they were going to do anyway.
//!
//! Settings are in the hash because they change the audio. Narrating the same book in a
//! different voice is a different run and must not adopt the first one's finished chapters.

use crate::{Chapter, ChapterState, Job, Request, State};
use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// What to narrate and how.
pub struct Plan {
    /// What the user named, for display.
    pub source: String,
    /// Narration text files, in reading order.
    pub texts: Vec<PathBuf>,
    pub out_dir: PathBuf,
    pub engine: String,
    pub voice: String,
    pub quant: Option<String>,
    pub seed: Option<u64>,
}

impl Plan {
    /// Build the job, hashing the text that will actually be spoken.
    ///
    /// The *text*, not the source document: two documents that narrate to the same words are
    /// the same run, and — more usefully — editing a chapter changes the id, so a corrected
    /// book does not silently resume onto audio of the old wording.
    pub fn into_job(self) -> Result<Job> {
        if self.texts.is_empty() {
            bail!("nothing to narrate: the plan has no chapters");
        }
        let mut hasher = Sha256::new();
        hasher.update(self.engine.as_bytes());
        hasher.update([0]);
        hasher.update(self.voice.as_bytes());
        hasher.update([0]);
        hasher.update(self.quant.as_deref().unwrap_or("").as_bytes());
        hasher.update([0]);
        hasher.update(
            self.seed
                .map(|s| s.to_string())
                .unwrap_or_default()
                .as_bytes(),
        );

        let mut chapters = Vec::with_capacity(self.texts.len());
        for path in &self.texts {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?;
            let name = chapter_name(path);
            hasher.update([0]);
            hasher.update(name.as_bytes());
            hasher.update([0]);
            hasher.update(text.as_bytes());
            chapters.push(Chapter {
                wav_path: self
                    .out_dir
                    .join(format!("{name}.wav"))
                    .display()
                    .to_string(),
                name,
                text_path: path.display().to_string(),
                state: ChapterState::Pending,
                words: text.split_whitespace().count(),
                audio_seconds: None,
                wall_seconds: None,
                error: None,
            });
        }
        // Twelve hex characters: enough that a collision is not a practical concern, short
        // enough to be typed and read aloud over a desk.
        let id = format!("{:x}", hasher.finalize())[..12].to_string();

        let now = crate::now();
        Ok(Job {
            id,
            source: self.source,
            out_dir: self.out_dir.display().to_string(),
            engine: self.engine,
            voice: self.voice,
            quant: self.quant,
            seed: self.seed,
            state: State::Queued,
            request: Request::None,
            chapters,
            created: now,
            updated: now,
            pid: None,
            error: None,
        })
    }
}

/// `chapter-001.txt` -> `chapter-001`.
fn chapter_name(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "chapter".to_string())
}

/// Adopt what a previous run of the same job already finished.
///
/// The new job carries the plan; the stored one carries the progress. Chapter state is taken
/// from the store *only* where the chapter still exists in the plan and its audio is still on
/// disk — a WAV deleted to force a re-render must actually cause one, which is the documented
/// way to redo a chapter.
pub fn adopt(fresh: Job, stored: &Job) -> Job {
    let mut job = fresh;
    for chapter in &mut job.chapters {
        let Some(previous) = stored.chapters.iter().find(|c| c.name == chapter.name) else {
            continue;
        };
        if previous.state != ChapterState::Done {
            continue;
        }
        if !Path::new(&previous.wav_path).is_file() {
            continue;
        }
        chapter.state = ChapterState::Done;
        chapter.audio_seconds = previous.audio_seconds;
        chapter.wall_seconds = previous.wall_seconds;
    }
    job.created = stored.created;
    job
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("tts-plan-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn plan(dir: &Path, texts: &[(&str, &str)]) -> Plan {
        let paths = texts
            .iter()
            .map(|(name, body)| {
                let p = dir.join(format!("{name}.txt"));
                std::fs::write(&p, body).unwrap();
                p
            })
            .collect();
        Plan {
            source: "book.epub".into(),
            texts: paths,
            out_dir: dir.to_path_buf(),
            engine: "qwen3tts".into(),
            voice: "voices/a".into(),
            quant: None,
            seed: None,
        }
    }

    #[test]
    fn the_same_book_is_the_same_job() {
        let dir = scratch("same");
        let a = plan(&dir, &[("chapter-001", "one"), ("chapter-002", "two")])
            .into_job()
            .unwrap();
        let b = plan(&dir, &[("chapter-001", "one"), ("chapter-002", "two")])
            .into_job()
            .unwrap();
        assert_eq!(a.id, b.id, "resume depends on this");
    }

    #[test]
    fn edited_text_is_a_different_job() {
        let dir = scratch("edited");
        let a = plan(&dir, &[("chapter-001", "one")]).into_job().unwrap();
        let b = plan(&dir, &[("chapter-001", "one, corrected")])
            .into_job()
            .unwrap();
        assert_ne!(
            a.id, b.id,
            "resuming onto audio of the old wording would be silent damage"
        );
    }

    #[test]
    fn a_different_voice_is_a_different_job() {
        let dir = scratch("voice");
        let a = plan(&dir, &[("chapter-001", "one")]).into_job().unwrap();
        let mut second = plan(&dir, &[("chapter-001", "one")]);
        second.voice = "voices/b".into();
        assert_ne!(a.id, second.into_job().unwrap().id);
    }

    #[test]
    fn chapter_order_is_part_of_the_identity() {
        let dir = scratch("order");
        let mut a = plan(&dir, &[("chapter-001", "one"), ("chapter-002", "two")]);
        let forward = plan(&dir, &[("chapter-001", "one"), ("chapter-002", "two")])
            .into_job()
            .unwrap();
        a.texts.reverse();
        assert_ne!(forward.id, a.into_job().unwrap().id);
    }

    #[test]
    fn a_plan_with_no_chapters_is_an_error_not_an_empty_job() {
        let dir = scratch("empty");
        let mut p = plan(&dir, &[]);
        p.texts.clear();
        assert!(p.into_job().is_err());
    }

    #[test]
    fn words_are_counted_up_front_so_an_estimate_exists_before_anything_runs() {
        let dir = scratch("words");
        let job = plan(&dir, &[("chapter-001", "one two three")])
            .into_job()
            .unwrap();
        assert_eq!(job.chapters[0].words, 3);
        assert_eq!(job.words_total(), 3);
    }

    #[test]
    fn adopting_resumes_only_chapters_whose_audio_is_still_there() {
        let dir = scratch("adopt");
        let fresh = plan(&dir, &[("chapter-001", "one"), ("chapter-002", "two")])
            .into_job()
            .unwrap();

        let mut stored = fresh.clone();
        stored.created = 42;
        for chapter in &mut stored.chapters {
            chapter.state = ChapterState::Done;
            chapter.wall_seconds = Some(5.0);
        }
        // Only the first chapter's audio survives; the second was deleted to force a redo.
        std::fs::write(&stored.chapters[0].wav_path, b"RIFF").unwrap();
        std::fs::remove_file(&stored.chapters[1].wav_path).ok();

        let resumed = adopt(fresh, &stored);
        assert_eq!(resumed.chapters[0].state, ChapterState::Done);
        assert_eq!(resumed.chapters[0].wall_seconds, Some(5.0));
        assert_eq!(
            resumed.chapters[1].state,
            ChapterState::Pending,
            "a deleted WAV must cause a re-render"
        );
        assert_eq!(resumed.next_chapter(), Some(1));
        assert_eq!(resumed.created, 42, "the run keeps its original start time");
    }
}
