//! Narration jobs: what is being narrated, how far it got, and whether to keep going.
//!
//! A thousand-page book is hours of synthesis. That length changes what the software has to
//! be: a run has to survive being interrupted, report progress fine enough to be worth
//! watching, and be discoverable by a process that did not start it — otherwise the second
//! thing a user does is start a second run on top of the first.
//!
//! # Why the store is a crate and not part of the service
//!
//! The service owns the engine and runs the work. But "is something already narrating?" has
//! to be answerable when the service is *not* running — that is exactly the moment a user is
//! about to start one — so the state lives in files under the data directory and any process
//! can read it. That also makes the observatory story real: a status bar, a menu-bar app or
//! a shell script can watch a job without going through the HTTP API or being trusted with
//! it.
//!
//! # Identity is the document
//!
//! A job's id is a hash of what is being narrated, so running the same book again *is* the
//! same job. Resume needs no flag and no remembered id: submit the document, get back the
//! run already in progress.

pub mod plan;
pub mod progress;
pub mod store;

pub use plan::{adopt, Plan};
pub use progress::{Progress, StageProgress};
pub use store::Store;

use serde::{Deserialize, Serialize};

/// Where a job is. The states a *caller* can distinguish, not the runner's internals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    /// Accepted, nothing started.
    Queued,
    Running,
    /// A pause was asked for and has taken effect. Resuming continues from here.
    Paused,
    /// Every chapter finished.
    Done,
    /// Stopped by a caller. Distinct from `Done` because the output is incomplete, and
    /// distinct from `Failed` because nothing went wrong.
    Cancelled,
    /// A chapter failed and the run stopped. The error is on the chapter.
    Failed,
}

impl State {
    /// Whether the runner should be doing work for this job.
    pub fn is_active(self) -> bool {
        matches!(self, Self::Queued | Self::Running)
    }

    /// Whether it will never change again without a caller asking.
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Done | Self::Cancelled | Self::Failed)
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Paused => "paused",
            Self::Done => "done",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
        }
    }
}

/// What a caller asked for that the runner has not acted on yet.
///
/// Separate from [`State`] because a pause is not instant: the runner finishes what it is
/// doing first, and a caller needs to see "pausing" rather than a request that appears to
/// have done nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Request {
    #[default]
    None,
    /// Stop after the chapter in flight. Its work is kept.
    Pause,
    /// Stop after the chapter in flight and mark the job cancelled.
    Cancel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChapterState {
    Pending,
    Running,
    Done,
    Failed,
}

/// One unit of work: a chapter of narration text, and the audio it produces.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Chapter {
    /// Stable name, used for the output filenames. `chapter-001`.
    pub name: String,
    /// The narration text to speak, on disk.
    pub text_path: String,
    /// Where the WAV master goes.
    pub wav_path: String,
    pub state: ChapterState,
    /// For the estimate. Counted at submit, so an estimate exists before anything runs.
    pub words: usize,
    /// Audio produced, once it has been.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_seconds: Option<f64>,
    /// Synthesis wall time, for the estimate of what remains.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall_seconds: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl Chapter {
    pub fn is_done(&self) -> bool {
        self.state == ChapterState::Done
    }
}

/// A narration run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Job {
    /// Hash of the document's content and the chapter list. The same book resubmitted is
    /// the same job, which is what makes resume need no remembered id.
    pub id: String,
    /// What was submitted, for a human to recognise it by.
    pub source: String,
    /// Where the output goes.
    pub out_dir: String,
    pub engine: String,
    pub voice: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    pub state: State,
    #[serde(default)]
    pub request: Request,
    pub chapters: Vec<Chapter>,
    /// Unix seconds. Not a timestamp type: this is read by shell scripts and browsers as
    /// often as by Rust, and an integer needs no library to agree about.
    pub created: u64,
    pub updated: u64,
    /// The process that owns the run, so a stale job from a killed server is recognisable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl Job {
    pub fn chapters_done(&self) -> usize {
        self.chapters.iter().filter(|c| c.is_done()).count()
    }

    pub fn words_done(&self) -> usize {
        self.chapters
            .iter()
            .filter(|c| c.is_done())
            .map(|c| c.words)
            .sum()
    }

    pub fn words_total(&self) -> usize {
        self.chapters.iter().map(|c| c.words).sum()
    }

    pub fn audio_seconds(&self) -> f64 {
        self.chapters.iter().filter_map(|c| c.audio_seconds).sum()
    }

    /// The next chapter to work on, skipping what is already done. This is resume: nothing
    /// records "where we were", because finished work is the record.
    pub fn next_chapter(&self) -> Option<usize> {
        self.chapters
            .iter()
            .position(|c| c.state != ChapterState::Done)
    }

    /// Seconds of synthesis left, fitted to this job's own observed cost per chapter.
    ///
    /// **A flat words-per-second rate is wrong here, and wrong by a factor of two.** These
    /// engines have strong economies of scale: `qwen3tts` batches across segments, which
    /// only engages once a chapter has enough of them, so the same voice runs at RTF 0.66 on
    /// a 132-word passage and 0.26 on a 1612-word chapter. Extrapolating a short chapter's
    /// rate over a long one therefore predicts roughly twice the time it will take, and the
    /// error is in the direction that makes a user abandon a run that would have finished.
    ///
    /// So the model is `wall ≈ fixed + marginal × words`, fitted by least squares over the
    /// chapters that have finished, and applied to each remaining chapter *individually*.
    /// A long chapter then amortises the fixed cost the way it really does, and a short one
    /// still pays it.
    ///
    /// `None` until there is enough to fit: a confident wrong estimate on an eight-hour run
    /// is worse than none at all.
    pub fn eta_seconds(&self) -> Option<f64> {
        let samples: Vec<(f64, f64)> = self
            .chapters
            .iter()
            .filter(|c| c.is_done())
            .filter_map(|c| c.wall_seconds.map(|w| (c.words as f64, w)))
            .collect();
        if samples.is_empty() {
            return None;
        }
        // One chapter is not a sample. A book's front matter is 27 words of title page whose
        // seconds-per-word is nothing like its prose; extrapolating from it put a real
        // estimate at 2h 09m against an actual hour.
        let done_words: f64 = samples.iter().map(|(w, _)| w).sum();
        let total_words = self.words_total().max(1) as f64;
        if samples.len() < 2 && done_words / total_words < 0.05 {
            return None;
        }

        let cost = ChapterCost::fit(&samples)?;
        Some(
            self.chapters
                .iter()
                .filter(|c| !c.is_done())
                .map(|c| cost.predict(c.words as f64))
                .sum(),
        )
    }

    /// Whether this job's process is still alive. A server killed mid-run leaves a job that
    /// says `Running` for ever, and a caller has to be able to tell that apart from work
    /// actually happening.
    pub fn is_stale(&self) -> bool {
        match (self.state, self.pid) {
            (State::Running, Some(pid)) => !process_alive(pid),
            (State::Running, None) => true,
            _ => false,
        }
    }
}

/// What a chapter costs: a fixed part and a part that scales with its length.
///
/// The fixed part is real and large — prompt assembly, the first batch's ramp, the codec
/// pass — and it is what makes a short chapter expensive per word. The marginal part is
/// what a long chapter mostly pays, and it is roughly 2.5x cheaper per word once batching
/// engages.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ChapterCost {
    fixed: f64,
    marginal: f64,
}

impl ChapterCost {
    /// Least squares over `(words, wall)`.
    ///
    /// Degenerate inputs — one sample, or every chapter the same length — have no slope to
    /// find, so they fall back to a flat rate through the origin. That is the old behaviour,
    /// which is right when there is nothing better to say.
    fn fit(samples: &[(f64, f64)]) -> Option<Self> {
        let n = samples.len() as f64;
        let mean_w = samples.iter().map(|(w, _)| w).sum::<f64>() / n;
        let mean_t = samples.iter().map(|(_, t)| t).sum::<f64>() / n;
        let variance: f64 = samples.iter().map(|(w, _)| (w - mean_w).powi(2)).sum();
        let covariance: f64 = samples
            .iter()
            .map(|(w, t)| (w - mean_w) * (t - mean_t))
            .sum();

        let flat = || {
            let words: f64 = samples.iter().map(|(w, _)| w).sum();
            let time: f64 = samples.iter().map(|(_, t)| t).sum();
            (words > 0.0 && time > 0.0).then_some(Self {
                fixed: 0.0,
                marginal: time / words,
            })
        };
        if variance <= f64::EPSILON {
            return flat();
        }
        let marginal = covariance / variance;
        let fixed = mean_t - marginal * mean_w;
        // A negative marginal rate means longer chapters cost *less in total*, which is
        // noise rather than a discount; a negative fixed cost means the fit ran off the end
        // of the data. Either way the flat rate is the honest answer.
        if marginal <= 0.0 || fixed < 0.0 {
            return flat();
        }
        Some(Self { fixed, marginal })
    }

    fn predict(&self, words: f64) -> f64 {
        self.fixed + self.marginal * words
    }
}

/// `kill(pid, 0)`: does the process exist and can we signal it.
fn process_alive(pid: u32) -> bool {
    // SAFETY: signal 0 performs the permission and existence checks without delivering
    // anything, which is the documented way to ask this question.
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

/// Unix seconds.
pub fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chapter(name: &str, words: usize, state: ChapterState) -> Chapter {
        Chapter {
            name: name.into(),
            text_path: format!("{name}.txt"),
            wav_path: format!("{name}.wav"),
            state,
            words,
            audio_seconds: None,
            wall_seconds: None,
            error: None,
        }
    }

    fn job(chapters: Vec<Chapter>) -> Job {
        Job {
            id: "abc".into(),
            source: "book.epub".into(),
            out_dir: "out".into(),
            engine: "qwen3tts".into(),
            voice: "voices/x".into(),
            quant: None,
            seed: None,
            state: State::Running,
            request: Request::None,
            chapters,
            created: 0,
            updated: 0,
            pid: None,
            error: None,
        }
    }

    #[test]
    fn resume_is_the_first_chapter_not_done() {
        let j = job(vec![
            chapter("a", 10, ChapterState::Done),
            chapter("b", 10, ChapterState::Done),
            chapter("c", 10, ChapterState::Pending),
        ]);
        assert_eq!(j.next_chapter(), Some(2));
        assert_eq!(j.chapters_done(), 2);
    }

    /// A chapter that failed is retried on resume rather than skipped: the failure may have
    /// been the machine, and skipping it would silently ship a book with a hole in it.
    #[test]
    fn a_failed_chapter_is_next_again() {
        let j = job(vec![
            chapter("a", 10, ChapterState::Done),
            chapter("b", 10, ChapterState::Failed),
            chapter("c", 10, ChapterState::Pending),
        ]);
        assert_eq!(j.next_chapter(), Some(1));
    }

    #[test]
    fn a_finished_job_has_no_next_chapter() {
        let j = job(vec![chapter("a", 10, ChapterState::Done)]);
        assert_eq!(j.next_chapter(), None);
    }

    #[test]
    fn there_is_no_estimate_before_the_first_chapter_finishes() {
        assert_eq!(
            job(vec![chapter("a", 100, ChapterState::Running)]).eta_seconds(),
            None
        );
    }

    /// A book's front matter is 27 words of title page whose seconds-per-word is nothing
    /// like its prose. Extrapolating from it put a real estimate at 2h 09m against an
    /// actual hour.
    #[test]
    fn one_tiny_chapter_is_not_a_sample_worth_extrapolating() {
        let mut chapters = vec![chapter("front", 27, ChapterState::Done)];
        chapters[0].wall_seconds = Some(15.0);
        chapters.push(chapter("body", 13_880, ChapterState::Pending));
        assert_eq!(job(chapters).eta_seconds(), None);
    }

    /// The whole point of fitting rather than averaging.
    ///
    /// The documented figures: `qwen3tts` runs `examples/senior.txt` (132 words) at RTF
    /// 0.665 and `examples/chapter.txt` (1612 words) at 0.260, because batching across
    /// segments only engages once a chapter has enough of them. At this material's ~139
    /// words per minute of speech that is 37.9 s of synthesis for the short one and 180.9 s
    /// for the long one — **0.287 s/word against 0.112 s/word, a 2.6x economy of scale.**
    ///
    /// A rate learned from the short chapter and applied to a long one therefore predicts
    /// 463 s where the truth is 181 s. Fitting a fixed cost plus a marginal rate gets it
    /// right, because the long chapter amortises the fixed part the way it really does.
    #[test]
    fn the_estimate_accounts_for_batching_rather_than_averaging_it_away() {
        let mut chapters = vec![
            chapter("short", 132, ChapterState::Done),
            chapter("long", 1612, ChapterState::Done),
            chapter("next", 1612, ChapterState::Pending),
        ];
        chapters[0].wall_seconds = Some(37.9);
        chapters[1].wall_seconds = Some(180.9);

        let fitted = job(chapters)
            .eta_seconds()
            .expect("two differing samples is a fit");
        assert!(
            (fitted - 180.9).abs() < 5.0,
            "a chapter the same length as the long one should cost about the same: {fitted}"
        );

        // What the old flat rate would have said, learned from the short chapter alone.
        let short_only = 37.9 / 132.0 * 1612.0;
        assert!(
            short_only > 400.0,
            "the naive estimate really is 2.6x out: {short_only}"
        );
    }

    /// ...and the fixed cost is still charged to a short chapter, which is the other half
    /// of the same fact: economies of scale mean the *small* chapters are the expensive
    /// ones per word, not that they are free.
    #[test]
    fn a_short_remaining_chapter_still_pays_the_per_chapter_overhead() {
        let mut chapters = vec![
            chapter("short", 132, ChapterState::Done),
            chapter("long", 1612, ChapterState::Done),
            chapter("tiny", 30, ChapterState::Pending),
        ];
        chapters[0].wall_seconds = Some(37.9);
        chapters[1].wall_seconds = Some(180.9);
        let eta = job(chapters).eta_seconds().unwrap();
        // Below the 132-word chapter's cost, but well above 30 x the marginal rate: the
        // fixed part dominates a tiny chapter.
        assert!(eta > 20.0 && eta < 37.9, "{eta}");
    }

    /// Chapters that are all the same length give no slope, so the flat rate is the only
    /// honest answer and must not become a division by zero.
    #[test]
    fn equal_length_chapters_fall_back_to_a_flat_rate() {
        let mut chapters = vec![
            chapter("a", 100, ChapterState::Done),
            chapter("b", 100, ChapterState::Done),
            chapter("c", 100, ChapterState::Pending),
        ];
        chapters[0].wall_seconds = Some(50.0);
        chapters[1].wall_seconds = Some(50.0);
        assert_eq!(job(chapters).eta_seconds(), Some(50.0));
    }

    /// Noise can make longer chapters look cheaper in total. That is not a discount to
    /// extrapolate from.
    #[test]
    fn a_nonsense_fit_falls_back_rather_than_predicting_negative_time() {
        let mut chapters = vec![
            chapter("a", 100, ChapterState::Done),
            chapter("b", 1000, ChapterState::Done),
            chapter("c", 500, ChapterState::Pending),
        ];
        chapters[0].wall_seconds = Some(90.0);
        chapters[1].wall_seconds = Some(10.0);
        let eta = job(chapters).eta_seconds().expect("still an estimate");
        assert!(eta > 0.0, "never negative: {eta}");
    }

    #[test]
    fn a_running_job_with_no_live_process_is_stale() {
        let mut j = job(vec![chapter("a", 10, ChapterState::Pending)]);
        j.state = State::Running;
        j.pid = None;
        assert!(
            j.is_stale(),
            "a running job that names no process cannot be running"
        );

        j.pid = Some(std::process::id());
        assert!(!j.is_stale(), "this process is alive");

        // A pid that cannot exist: the kernel's maximum is well below this.
        j.pid = Some(4_000_000_000);
        assert!(j.is_stale());

        j.state = State::Done;
        assert!(!j.is_stale(), "only a running job can be stale");
    }

    #[test]
    fn states_partition_into_active_and_terminal() {
        assert!(State::Queued.is_active() && State::Running.is_active());
        assert!(
            !State::Paused.is_active(),
            "a paused job is not work in flight"
        );
        for s in [State::Done, State::Cancelled, State::Failed] {
            assert!(s.is_terminal() && !s.is_active());
        }
        assert!(
            !State::Paused.is_terminal(),
            "a paused job can still be resumed"
        );
    }
}
