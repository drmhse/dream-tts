//! The engine contract.

use crate::voice::Voice;
use anyhow::Result;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Mono PCM in `[-1, 1]`.
pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl Audio {
    pub fn seconds(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }
}

/// How an engine supports voice cloning.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cloning {
    /// No reference voice; the model has one built-in speaker.
    None,
    /// Clones from a precomputed [`Voice`] asset. Producing that asset needs Python;
    /// consuming it does not.
    PrecomputedAsset,
}

/// What an engine can and cannot do. Returned even by engines that are not yet
/// runnable, so a caller can enumerate and choose before committing to a request.
#[derive(Clone, Debug)]
pub struct Capabilities {
    /// Stable identifier a client passes to select this engine.
    pub id: &'static str,
    pub description: &'static str,
    pub sample_rate: u32,
    /// Acoustic frames per second — the unit RTF projections are quoted in.
    pub frame_rate: f64,
    pub cloning: Cloning,
    pub streaming: bool,
    /// Weight formats `EngineConfig::quant` accepts, most faithful first.
    pub quantization: &'static [&'static str],
    /// `Some` when the engine only speaks a closed set, lowercase English names.
    /// `None` means unrestricted, not unknown. A closed list is why an engine must not
    /// become the default silently — see `tts_engines::default_caveat`.
    pub languages: Option<&'static [&'static str]>,
    /// False when the engine is registered but cannot synthesize yet. `reason` says
    /// why. Registering an unfinished engine is deliberate: a client can discover it
    /// exists and code against the identifier before it lands.
    pub available: bool,
    pub reason: Option<&'static str>,
}

/// Sampling controls. Engines map these onto their own samplers and document any
/// knob they ignore rather than silently accepting it.
#[derive(Clone, Debug)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
    pub greedy: bool,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            seed: 1234,
            greedy: false,
        }
    }
}

/// Silence inserted when stitching segments back together.
#[derive(Clone, Copy, Debug)]
pub struct Gaps {
    pub segment_ms: usize,
    pub paragraph_ms: usize,
}

impl Default for Gaps {
    fn default() -> Self {
        Self {
            segment_ms: 90,
            paragraph_ms: 320,
        }
    }
}

/// What an engine tells a caller while it works.
///
/// Engines report *segments*, not tokens or frames: it is the one unit every engine has,
/// it is what the caller's text was split into, and it does not require a client to know
/// anything about the architecture. Each stage counts from zero, so a three-stage engine
/// reports three passes rather than one merged fiction — which is honest about where the
/// time goes and matches the stage breakdown printed at the end.
#[derive(Clone, Copy, Debug)]
pub enum ProgressEvent {
    /// Emitted once, after segmentation and before any compute.
    Planned { segments: usize },
    Advanced {
        /// The stage name, the same string that appears in [`Stats::breakdown`].
        stage: &'static str,
        done: usize,
        total: usize,
    },
}

/// A progress sink. `Arc` rather than a borrow so [`SynthesisRequest`] needs no lifetime;
/// `Send + Sync` because the service hands requests to a worker.
pub type Progress = std::sync::Arc<dyn Fn(ProgressEvent) + Send + Sync>;

/// Asked between segments; `true` stops the render.
///
/// Synthesis is minutes long for a chapter and hours for a book, so "stop" has to mean
/// something sooner than "when it finishes". Checked at segment boundaries because that is
/// where an engine's state is consistent — mid-segment there is a partly generated waveform
/// that is not audio yet.
pub type Interrupt = std::sync::Arc<dyn Fn() -> bool + Send + Sync>;

/// The error an interrupted render returns.
///
/// A distinct marker rather than a message, so a caller can tell "the user stopped this"
/// from "this failed" — one is a job to resume and the other is a job to report.
#[derive(Debug)]
pub struct Interrupted {
    /// Segments completed before stopping, for a caller that wants to say how far it got.
    pub segments: usize,
}

impl std::fmt::Display for Interrupted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "synthesis interrupted after {} segment(s)",
            self.segments
        )
    }
}

impl std::error::Error for Interrupted {}

pub struct SynthesisRequest {
    pub text: String,
    pub voice: Option<Voice>,
    pub sampling: Sampling,
    /// Segment length budget in characters. Segmentation is what keeps prompts inside
    /// the model's context and bounds how much audio one AR run must stay coherent
    /// over, so it is a core concern rather than an engine detail.
    pub max_chars: usize,
    pub max_new_tokens: usize,
    pub gaps: Gaps,
    /// Called as the engine works. `None` is the quiet path and costs nothing.
    pub progress: Option<Progress>,
    /// Asked between segments. `None` means never interrupted.
    pub interrupt: Option<Interrupt>,
}

impl SynthesisRequest {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            voice: None,
            sampling: Sampling::default(),
            max_chars: 220,
            max_new_tokens: 512,
            gaps: Gaps::default(),
            progress: None,
            interrupt: None,
        }
    }

    pub fn with_voice(mut self, voice: Voice) -> Self {
        self.voice = Some(voice);
        self
    }

    pub fn with_progress(mut self, progress: Progress) -> Self {
        self.progress = Some(progress);
        self
    }

    pub fn with_interrupt(mut self, interrupt: Interrupt) -> Self {
        self.interrupt = Some(interrupt);
        self
    }

    /// Whether a caller has asked to stop. Engines call this at segment boundaries.
    pub fn interrupted(&self) -> bool {
        self.interrupt.as_ref().is_some_and(|f| f())
    }

    /// `Err(Interrupted)` when a caller has asked to stop, for `?` in a segment loop.
    pub fn check_interrupt(&self, segments: usize) -> Result<()> {
        if self.interrupted() {
            return Err(Interrupted { segments }.into());
        }
        Ok(())
    }

    /// Report progress if anyone is listening. Engines call this; the `Option` check keeps
    /// an uninstrumented render free of any cost beyond a null test.
    pub fn notify(&self, event: ProgressEvent) {
        if let Some(f) = &self.progress {
            f(event);
        }
    }

    /// Convenience for the common `Advanced` shape.
    pub fn advanced(&self, stage: &'static str, done: usize, total: usize) {
        self.notify(ProgressEvent::Advanced { stage, done, total });
    }
}

/// Where the time went. Reported per request because the split between stages is the
/// number that decides what to optimize next, and this project has twice been wrong
/// about it.
///
/// The stages are a named list rather than fixed fields because engines genuinely differ:
/// Audio8 is autoregressive-then-vocoder, CosyVoice is LLM-then-flow-then-vocoder, and
/// flattening the second into the first would hide the stage that dominates it. Names are
/// the engine's own and are only meaningful next to that engine.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub segments: usize,
    pub frames: usize,
    /// Per-stage wall time, in the order the engine ran them.
    pub stages: Vec<(&'static str, f64)>,
    pub total_s: f64,
}

impl Stats {
    pub fn rtf(&self, audio_seconds: f64) -> f64 {
        if audio_seconds <= 0.0 {
            f64::NAN
        } else {
            self.total_s / audio_seconds
        }
    }

    /// Add to a stage's accumulated time, creating it on first use.
    pub fn add(&mut self, stage: &'static str, seconds: f64) {
        match self.stages.iter_mut().find(|(n, _)| *n == stage) {
            Some((_, t)) => *t += seconds,
            None => self.stages.push((stage, seconds)),
        }
    }

    /// `stage  0.512 s  61%` lines, for a CLI to print without knowing the engine.
    pub fn breakdown(&self, audio_seconds: f64) -> Vec<(String, f64, f64, f64)> {
        self.stages
            .iter()
            .map(|(name, t)| {
                let share = if self.total_s > 0.0 {
                    t / self.total_s * 100.0
                } else {
                    0.0
                };
                let rtf = if audio_seconds > 0.0 {
                    t / audio_seconds
                } else {
                    f64::NAN
                };
                (name.to_string(), *t, rtf, share)
            })
            .collect()
    }
}

pub struct Synthesis {
    pub audio: Audio,
    pub stats: Stats,
}

/// How to load an engine. `model_root` plus conventional filenames covers the normal
/// case; `overrides` exists because this repo keeps converted weights and upstream
/// checkpoints in different trees.
pub struct EngineConfig {
    pub model_root: PathBuf,
    pub quant: Option<String>,
    pub cpu: bool,
    pub overrides: BTreeMap<String, PathBuf>,
}

impl EngineConfig {
    pub fn new(model_root: impl Into<PathBuf>) -> Self {
        Self {
            model_root: model_root.into(),
            quant: None,
            cpu: false,
            overrides: BTreeMap::new(),
        }
    }

    /// Resolve a conventional filename, honouring an override if one was given.
    pub fn path(&self, key: &str, default_relative: &str) -> PathBuf {
        match self.overrides.get(key) {
            Some(p) => p.clone(),
            None => self.model_root.join(default_relative),
        }
    }
}

pub trait Engine: Send + Sync {
    fn capabilities(&self) -> Capabilities;

    /// Synthesize the whole request, segmenting and stitching internally.
    fn synthesize(&self, request: &SynthesisRequest) -> Result<Synthesis>;

    /// Reject a request this engine cannot honour, with a reason naming the engine.
    ///
    /// The default checks the capability flags. Engines that need more should call
    /// [`validate_against`] and then add their own, rather than reimplementing these.
    fn validate(&self, request: &SynthesisRequest) -> Result<()> {
        validate_against(&self.capabilities(), request)
    }
}

/// The capability checks every engine shares: availability, cloning support, and that a
/// voice asset was built for the engine being asked to use it.
pub fn validate_against(caps: &Capabilities, request: &SynthesisRequest) -> Result<()> {
    if !caps.available {
        anyhow::bail!(
            "engine `{}` is not available: {}",
            caps.id,
            caps.reason.unwrap_or("no reason given")
        );
    }
    if let Some(voice) = &request.voice {
        if caps.cloning == Cloning::None {
            anyhow::bail!(
                "engine `{}` does not support voice cloning, but a voice was supplied",
                caps.id
            );
        }
        if voice.engine != caps.id {
            anyhow::bail!(
                "voice `{}` was built for engine `{}`, not `{}` — voice assets are not \
                 interchangeable between engines",
                voice.name,
                voice.engine,
                caps.id
            );
        }
    }
    Ok(())
}
