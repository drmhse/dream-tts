//! `nano` behind the engine-neutral [`Engine`] trait: Kokoro's voices and front end through
//! [`NanoTts`], 3.27M parameters and no Kokoro network.

use crate::decoder::SeededDraws;
use crate::engine::word_times;
use crate::model::Voices;
use crate::standalone::NanoTts;
use anyhow::{Context, Result};
use candle_core::Device;
use std::time::Instant;
use tts_core::{
    text, Audio, Capabilities, Cloning, Engine, EngineConfig, Stats, Synthesis, SynthesisRequest,
    WordTime,
};

pub const ID: &str = "nano";

const DEFAULT_VOICE: &str = "af_heart";
const SAMPLE_RATE: u32 = 24_000;
/// Samples per text-side frame (40 frames/s).
const HOP: usize = 600;
/// The voice tables are indexed by phoneme count up to 510, so segments stay about Kokoro's size.
const MAX_CHARS: usize = 400;

pub fn capabilities() -> Capabilities {
    Capabilities {
        id: ID,
        description:
            "nano-edge-tts-standalone — 3.27M parameters distilled from Kokoro-82M, 24 kHz",
        sample_rate: SAMPLE_RATE,
        frame_rate: SAMPLE_RATE as f64 / HOP as f64,
        cloning: Cloning::None,
        streaming: false,
        word_timings: true,
        quantization: &["f32"],
        languages: Some(&["english"]),
        available: true,
        reason: None,
    }
}

pub struct NanoEngine {
    model: NanoTts,
    voices: Voices,
    g2p: tts_phoneme::g2p::G2P,
    british: bool,
    other: std::sync::OnceLock<Option<tts_phoneme::g2p::G2P>>,
    frontend: std::path::PathBuf,
    fallback: bool,
    voice: String,
}

impl NanoEngine {
    pub fn load(config: &EngineConfig) -> Result<Self> {
        // The CPU on every machine: the model is small enough that Metal's per-dispatch cost loses.
        // `examples/chapter.txt` on an M4: 3.6 s on the CPU against 6.0 s on Metal.
        let device = Device::Cpu;
        let setting = |key: &str| config.overrides.get(key).and_then(|p| p.to_str());
        let voice = setting("voice").unwrap_or(DEFAULT_VOICE).to_string();
        let british = match setting("british") {
            Some("true") => true,
            Some("false") => false,
            Some(other) => anyhow::bail!("--set british= takes true or false, got {other:?}"),
            None => voice.starts_with('b'),
        };
        let fallback = match setting("fallback") {
            Some("off") => false,
            Some("on") | None => true,
            Some(other) => anyhow::bail!("--set fallback= takes on or off, got {other:?}"),
        };
        let model = NanoTts::load(&config.model_root, &device)?;
        let voices = Voices::load(&config.path("voices", "voices.safetensors"), &device)?;
        anyhow::ensure!(
            voices.names().contains(&voice.as_str()),
            "unknown voice `{voice}`; `--set voice=<name>` takes one of: {}",
            voices.names().join(", ")
        );
        let frontend = config.path("frontend", "frontend");
        let g2p = tts_phoneme::g2p::G2P::load(&frontend, british)?.with_fallback(fallback);
        Ok(Self {
            model,
            voices,
            g2p,
            british,
            other: std::sync::OnceLock::new(),
            frontend,
            fallback,
            voice,
        })
    }

    /// A `bf_`/`bm_` voice reads with the British lexicon, whichever was loaded first.
    fn voice_for<'a>(
        &'a self,
        request: &'a SynthesisRequest,
    ) -> (&'a str, &'a tts_phoneme::g2p::G2P) {
        let Some(voice) = request.speaker.as_deref() else {
            return (&self.voice, &self.g2p);
        };
        if voice.starts_with('b') == self.british {
            return (voice, &self.g2p);
        }
        let other = self.other.get_or_init(|| {
            tts_phoneme::g2p::G2P::load(&self.frontend, !self.british)
                .ok()
                .map(|g| g.with_fallback(self.fallback))
        });
        (voice, other.as_ref().unwrap_or(&self.g2p))
    }
}

impl Engine for NanoEngine {
    fn capabilities(&self) -> Capabilities {
        capabilities()
    }

    fn validate(&self, request: &SynthesisRequest) -> Result<()> {
        if request.voice.is_some() {
            anyhow::bail!(
                "engine `{ID}` cannot clone: its voices are Kokoro's fixed style tables. Pick one \
                 with `--set voice=<name>` instead of `--voice`"
            );
        }
        if let Some(speaker) = &request.speaker {
            anyhow::ensure!(
                self.voices.names().contains(&speaker.as_str()),
                "unknown voice `{speaker}`; engine `{ID}` carries: {}",
                self.voices.names().join(", ")
            );
        }
        tts_core::engine::validate_against(&self.capabilities(), request)
    }

    fn synthesize(&self, request: &SynthesisRequest) -> Result<Synthesis> {
        self.validate(request)?;
        let paragraphs = text::segment(&request.text, request.max_chars.min(MAX_CHARS));
        let flat: Vec<(usize, &String)> = paragraphs
            .iter()
            .enumerate()
            .flat_map(|(pi, para)| para.iter().map(move |s| (pi, s)))
            .collect();
        anyhow::ensure!(!flat.is_empty(), "no text to speak");
        request.notify(tts_core::ProgressEvent::Planned {
            segments: flat.len(),
        });

        let mut draws = SeededDraws::new(request.sampling.seed);
        let mut stats = Stats::default();
        let mut unknown: Vec<String> = Vec::new();
        let mut pieces: Vec<tts_core::Piece> = Vec::new();
        let mut piece_words: Vec<Vec<WordTime>> = Vec::new();
        let t0 = Instant::now();

        // The front end runs a segment ahead on another thread: it is now a large share of the work.
        let (voice, g2p) = self.voice_for(request);
        let segments = &flat;
        let (tx, rx) = std::sync::mpsc::sync_channel(2);
        std::thread::scope(|sc| -> Result<()> {
            sc.spawn(move || {
                for (_, segment) in segments {
                    if tx.send(g2p.phonemize_tagged(segment)).is_err() {
                        break;
                    }
                }
            });
            for (k, (pi, segment)) in flat.iter().enumerate() {
                let mut tagged = rx.recv().context("the frontend thread stopped")?;
                unknown.append(&mut tagged.unknown);
                let (ids, tags, offsets) = self.model.encode(&tagged.phonemes, &tagged.tags);
                if ids.len() <= 2 {
                    continue;
                }
                let style = self.voices.style(voice, ids.len() - 2)?;
                let started = Instant::now();
                let (samples, durations) = self
                    .model
                    .synthesize(&ids, &tags, &style, 1.0, &mut draws)?;
                stats.add("render", started.elapsed().as_secs_f64());
                stats.frames += samples.len() / HOP;
                stats.segments += 1;
                let frames: usize = durations.iter().sum();
                let per_frame = samples.len() as f64 / frames.max(1) as f64 / SAMPLE_RATE as f64;
                piece_words.push(word_times(&tagged.spans, &offsets, &durations, per_frame));
                pieces.push(tts_core::Piece {
                    paragraph: *pi,
                    text: (*segment).clone(),
                    samples,
                });
                request.advanced("render", k + 1, flat.len());
                request.check_interrupt(k + 1)?;
            }
            Ok(())
        })?;
        stats.total_s = t0.elapsed().as_secs_f64();

        if !unknown.is_empty() {
            unknown.sort_unstable();
            unknown.dedup();
            eprintln!(
                "engine {ID}: {} word(s) not in the lexicon: {}",
                unknown.len(),
                unknown.join(", ")
            );
        }

        let (samples, segments) =
            tts_core::join_segments(pieces, request.gaps, SAMPLE_RATE as usize);
        let mut words: Vec<WordTime> = Vec::new();
        for (piece, said) in segments.iter().zip(&piece_words) {
            words.extend(said.iter().map(|w| WordTime {
                text: w.text.clone(),
                start: w.start + piece.start,
                end: w.end + piece.start,
            }));
        }
        anyhow::ensure!(!samples.is_empty(), "engine {ID} produced no audio");
        Ok(Synthesis {
            audio: Audio {
                samples,
                sample_rate: SAMPLE_RATE,
            },
            stats,
            segments: Some(segments),
            words: Some(words),
        })
    }
}
