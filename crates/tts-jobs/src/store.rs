//! Jobs on disk, so any process can see them.
//!
//! One JSON file per job under `<data_dir>/jobs/`. Files rather than a database for a
//! reason that outlives the choice: the question "is something already narrating?" has to be
//! answerable when the service is not running, by a shell script, a status bar, or a user
//! with `cat`. A database would make the service the only thing that can answer.
//!
//! Every write is a write-and-rename. A reader that catches a job mid-write would see a
//! truncated file, and the reader is often a progress display refreshing several times a
//! second — so the failure would be constant rather than rare.

use crate::{Job, State};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub struct Store {
    dir: PathBuf,
}

impl Store {
    /// `<data_dir>/jobs`.
    pub fn open(data_dir: &Path) -> Result<Self> {
        let dir = data_dir.join("jobs");
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating the job directory {}", dir.display()))?;
        Ok(Self { dir })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.json"))
    }

    pub fn get(&self, id: &str) -> Result<Option<Job>> {
        let path = self.path(id);
        if !path.is_file() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        // A job whose file is corrupt is reported, not skipped: silently losing a run's
        // record is how eight hours of work becomes unresumable without anyone noticing.
        let job = serde_json::from_str(&text)
            .with_context(|| format!("{} is not a readable job record", path.display()))?;
        Ok(Some(job))
    }

    pub fn put(&self, job: &Job) -> Result<()> {
        let path = self.path(&job.id);
        // Same directory, so the rename is atomic; a temp elsewhere could be another
        // filesystem and would degrade to a copy.
        let temp = self
            .dir
            .join(format!(".{}.{}.tmp", job.id, std::process::id()));
        let json = serde_json::to_string_pretty(job)?;
        std::fs::write(&temp, json).with_context(|| format!("writing {}", temp.display()))?;
        std::fs::rename(&temp, &path).with_context(|| format!("replacing {}", path.display()))?;
        Ok(())
    }

    pub fn remove(&self, id: &str) -> Result<bool> {
        let path = self.path(id);
        if !path.is_file() {
            return Ok(false);
        }
        std::fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
        Ok(true)
    }

    /// Every job, newest first.
    ///
    /// An unreadable file is skipped here rather than failing the listing: one corrupt
    /// record must not hide the others, and `get` reports it when asked for by name.
    pub fn list(&self) -> Result<Vec<Job>> {
        let entries = std::fs::read_dir(&self.dir)
            .with_context(|| format!("reading {}", self.dir.display()))?;
        let mut jobs: Vec<Job> = entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .filter_map(|p| std::fs::read_to_string(p).ok())
            .filter_map(|text| serde_json::from_str::<Job>(&text).ok())
            .collect();
        jobs.sort_by(|a, b| b.created.cmp(&a.created));
        Ok(jobs)
    }

    /// Jobs that believe they are running, with a live process behind them.
    ///
    /// This is the discovery question: what a caller checks before starting work, and the
    /// reason the store is readable without the service. A job left behind by a killed
    /// server is excluded — it is not running, whatever its file says.
    pub fn active(&self) -> Result<Vec<Job>> {
        Ok(self
            .list()?
            .into_iter()
            .filter(|j| j.state.is_active() && !j.is_stale())
            .collect())
    }

    /// Mark jobs whose process is gone as paused, so they can be resumed rather than
    /// appearing to run for ever. Returns what it changed.
    ///
    /// Paused rather than failed: nothing went wrong with the *work*, the machine or the
    /// service stopped, and every finished chapter is still on disk.
    pub fn reap_stale(&self) -> Result<Vec<String>> {
        let mut reaped = Vec::new();
        for mut job in self.list()? {
            if !job.is_stale() {
                continue;
            }
            job.state = State::Paused;
            job.pid = None;
            job.updated = crate::now();
            job.error = Some(
                "the process running this job exited; resume to continue from the last \
                 finished chapter"
                    .into(),
            );
            self.put(&job)?;
            reaped.push(job.id);
        }
        Ok(reaped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Chapter, ChapterState, Request};

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("tts-jobs-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn job(id: &str, state: State, pid: Option<u32>) -> Job {
        Job {
            id: id.into(),
            source: "book.epub".into(),
            out_dir: "out".into(),
            engine: "qwen3tts".into(),
            voice: "voices/x".into(),
            quant: None,
            seed: None,
            state,
            request: Request::None,
            chapters: vec![Chapter {
                name: "chapter-001".into(),
                text_path: "a.txt".into(),
                wav_path: "a.wav".into(),
                state: ChapterState::Pending,
                words: 10,
                audio_seconds: None,
                wall_seconds: None,
                error: None,
            }],
            created: 1,
            updated: 1,
            pid,
            error: None,
        }
    }

    #[test]
    fn a_job_round_trips() {
        let store = Store::open(&scratch("round")).unwrap();
        let j = job("aaa", State::Queued, None);
        store.put(&j).unwrap();
        assert_eq!(store.get("aaa").unwrap().as_ref(), Some(&j));
        assert_eq!(store.get("missing").unwrap(), None);
        assert!(store.remove("aaa").unwrap());
        assert!(!store.remove("aaa").unwrap());
    }

    #[test]
    fn writes_leave_no_partial_file_behind() {
        let dir = scratch("atomic");
        let store = Store::open(&dir).unwrap();
        store
            .put(&job("bbb", State::Running, Some(std::process::id())))
            .unwrap();
        let leftovers: Vec<_> = std::fs::read_dir(store.dir())
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "temp files left: {leftovers:?}");
    }

    #[test]
    fn listing_is_newest_first_and_survives_a_corrupt_record() {
        let dir = scratch("list");
        let store = Store::open(&dir).unwrap();
        let mut older = job("old", State::Done, None);
        older.created = 10;
        let mut newer = job("new", State::Done, None);
        newer.created = 20;
        store.put(&older).unwrap();
        store.put(&newer).unwrap();
        std::fs::write(store.dir().join("broken.json"), "{not json").unwrap();

        let ids: Vec<String> = store.list().unwrap().into_iter().map(|j| j.id).collect();
        assert_eq!(
            ids,
            ["new", "old"],
            "one bad file must not hide the good ones"
        );
        // ...but asked for by name, the corruption is reported rather than looking absent.
        assert!(store.get("broken").is_err());
    }

    #[test]
    fn active_excludes_a_job_left_by_a_dead_process() {
        let dir = scratch("active");
        let store = Store::open(&dir).unwrap();
        store
            .put(&job("live", State::Running, Some(std::process::id())))
            .unwrap();
        store
            .put(&job("dead", State::Running, Some(4_000_000_000)))
            .unwrap();
        store.put(&job("finished", State::Done, None)).unwrap();

        let ids: Vec<String> = store.active().unwrap().into_iter().map(|j| j.id).collect();
        assert_eq!(ids, ["live"]);
    }

    #[test]
    fn reaping_makes_a_dead_run_resumable_rather_than_eternally_running() {
        let dir = scratch("reap");
        let store = Store::open(&dir).unwrap();
        store
            .put(&job("dead", State::Running, Some(4_000_000_000)))
            .unwrap();
        assert_eq!(store.reap_stale().unwrap(), ["dead"]);

        let reaped = store.get("dead").unwrap().unwrap();
        assert_eq!(reaped.state, State::Paused, "resumable, not failed");
        assert_eq!(reaped.pid, None);
        assert!(reaped.error.as_deref().unwrap().contains("resume"));
        // Idempotent: a second pass finds nothing to do.
        assert!(store.reap_stale().unwrap().is_empty());
    }
}
