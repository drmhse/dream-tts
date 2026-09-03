//! Where a running chapter is, right now.
//!
//! Kept out of [`crate::Job`] because it is not durable: it changes several times a second
//! and is worth nothing after a restart, whereas the job is worth everything. Writing it to
//! disk at that rate would also make the store's atomic-rename cost the dominant expense of
//! narrating a book.

use serde::{Deserialize, Serialize};

/// One engine stage, and how far through its segments it is.
///
/// Engines report *segments* — the unit the caller's text was split into — per named stage,
/// and each stage counts from zero. A three-stage engine therefore reports three passes,
/// which is honest about where the time goes rather than merging them into one fiction.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageProgress {
    pub stage: String,
    pub done: usize,
    pub total: usize,
}

impl StageProgress {
    pub fn fraction(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.done as f64 / self.total as f64).clamp(0.0, 1.0)
        }
    }
}

/// A snapshot a watcher can render without knowing anything else.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Progress {
    pub job: String,
    /// Zero-based index into the job's chapters.
    pub chapter_index: usize,
    pub chapter: String,
    pub chapters_total: usize,
    /// Absent between chapters, and before the engine reports anything.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<StageProgress>,
    /// Seconds since this chapter started.
    pub elapsed: f64,
    /// Whole-job estimate, when the job has finished a chapter to measure with.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eta_seconds: Option<f64>,
    pub words_done: usize,
    pub words_total: usize,
}

impl Progress {
    /// Share of the *book*, by words, counting the running chapter's stage progress.
    ///
    /// Words rather than chapters, because a reference book's chapters differ by an order of
    /// magnitude in length and a chapter-counting bar sticks and then leaps.
    pub fn fraction(&self) -> f64 {
        if self.words_total == 0 {
            return 0.0;
        }
        let chapter_words = self
            .stage
            .as_ref()
            .map_or(0.0, |s| s.fraction() * self.current_chapter_words() as f64);
        ((self.words_done as f64 + chapter_words) / self.words_total as f64).clamp(0.0, 1.0)
    }

    /// Only the runner knows this; a watcher reconstructs it from the job. Kept simple
    /// rather than carried, since it is only used to weight the running chapter.
    fn current_chapter_words(&self) -> usize {
        if self.chapters_total == 0 {
            return 0;
        }
        self.words_total.saturating_sub(self.words_done) / self.remaining_chapters().max(1)
    }

    fn remaining_chapters(&self) -> usize {
        self.chapters_total.saturating_sub(self.chapter_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stage_with_no_segments_is_not_a_division_by_zero() {
        assert_eq!(StageProgress::default().fraction(), 0.0);
    }

    #[test]
    fn stage_fraction_is_clamped() {
        let s = StageProgress {
            stage: "talker".into(),
            done: 12,
            total: 10,
        };
        assert_eq!(s.fraction(), 1.0);
    }

    #[test]
    fn book_progress_is_measured_in_words_not_chapters() {
        let p = Progress {
            words_done: 2_500,
            words_total: 10_000,
            chapters_total: 4,
            chapter_index: 1,
            ..Default::default()
        };
        assert_eq!(
            p.fraction(),
            0.25,
            "no stage yet: only finished words count"
        );

        let half = Progress {
            stage: Some(StageProgress {
                stage: "talker".into(),
                done: 1,
                total: 2,
            }),
            ..p
        };
        // 2500 done, plus half of the running chapter's share of the remaining 7500.
        assert!(
            half.fraction() > 0.25 && half.fraction() < 0.5,
            "{}",
            half.fraction()
        );
    }

    #[test]
    fn an_empty_job_has_no_progress_rather_than_a_panic() {
        assert_eq!(Progress::default().fraction(), 0.0);
    }
}
