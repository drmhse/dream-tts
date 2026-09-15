//! Kokoro behind the engine-neutral [`Engine`] trait.
//!
//! The adapter is thinner than the others because there is no autoregressive loop to
//! batch: a segment is one forward pass. What it does own is the frontend — text becomes
//! IPA here, not in the model — and the 510-token ceiling that the ALBERT position table
//! imposes on a segment.

use crate::cfg::Config;
use crate::decoder::SeededDraws;
use crate::model::{Model, Voices};
use anyhow::{Context, Result};
use candle_core::Device;
use std::time::Instant;
use tts_core::{
    text, wav, Audio, Capabilities, Cloning, Engine, EngineConfig, Stats, Synthesis,
    SynthesisRequest,
};

pub const ID: &str = crate::ID;

/// Not a clone of anything, so a caller that says nothing gets this rather than an error.
const DEFAULT_VOICE: &str = "af_heart";

/// `plbert.max_position_embeddings`, minus nothing — the two pad ids are inside it.
const MAX_TOKENS: usize = 510;

/// Phonemes are never more numerous than the characters they came from in English, so a
/// character budget bounds the token count. 400 leaves room for the ones that are: a
/// digit is a word, and `ˈ` and `ˌ` are tokens with no character of their own. A caller
/// asking for more is clamped, not refused — the streaming route passes `usize::MAX` on
/// the assumption that an engine will not re-split, and this one has to.
const MAX_CHARS: usize = 400;

/// Samples per predictor frame: the two upsamples and the iSTFT hop.
const HOP: usize = 300;

pub struct KokoroEngine {
    model: Model,
    voices: Voices,
    g2p: tts_phoneme::g2p::G2P,
    voice: String,
}

pub fn capabilities() -> Capabilities {
    Capabilities {
        id: ID,
        description: "Kokoro-82M — non-autoregressive StyleTTS2 + iSTFTNet, 24 kHz",
        sample_rate: Config::SAMPLE_RATE,
        frame_rate: Config::SAMPLE_RATE as f64 / HOP as f64,
        cloning: Cloning::None,
        streaming: false,
        quantization: &["f32"],
        languages: Some(&["english"]),
        available: true,
        reason: None,
    }
}

impl KokoroEngine {
    pub fn load(config: &EngineConfig) -> Result<Self> {
        let device = if config.cpu {
            Device::Cpu
        } else {
            Device::new_metal(0).context("opening the Metal device")?
        };
        let voice = match config.overrides.get("voice") {
            Some(p) => p.to_str().context("--set voice=<name> is not utf-8")?.to_string(),
            None => DEFAULT_VOICE.to_string(),
        };
        // `bf_`/`bm_` are the British voicepacks, and the lexicon is the half of the
        // frontend that differs; an American lexicon under a British voice is a wrong
        // pronunciation rather than a wrong accent.
        let british = match config.overrides.get("british").and_then(|p| p.to_str()) {
            Some("true") => true,
            Some("false") => false,
            Some(other) => anyhow::bail!("--set british= takes true or false, got {other:?}"),
            None => voice.starts_with('b'),
        };
        let model = Model::load(&config.model_root, &device)?;
        let voices = Voices::load(&config.path("voices", "voices.safetensors"), &device)?;
        anyhow::ensure!(
            voices.names().contains(&voice.as_str()),
            "unknown voice `{voice}`; `--set voice=<name>` takes one of: {}",
            voices.names().join(", ")
        );
        let g2p = tts_phoneme::g2p::G2P::load(&config.path("frontend", "frontend"), british)?;
        Ok(Self { model, voices, g2p, voice })
    }
}

impl Engine for KokoroEngine {
    fn capabilities(&self) -> Capabilities {
        capabilities()
    }

    fn validate(&self, request: &SynthesisRequest) -> Result<()> {
        if request.voice.is_some() {
            anyhow::bail!(
                "engine `{ID}` cannot clone: its voices are fixed style tables, and there is \
                 no path from a reference clip to one of them. Pick one with \
                 `--set voice=<name>` instead of `--voice`"
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
        request.notify(tts_core::ProgressEvent::Planned { segments: flat.len() });

        let mut draws = SeededDraws::new(request.sampling.seed);
        let mut stats = Stats::default();
        let mut unknown: Vec<String> = Vec::new();
        let mut pieces: Vec<(usize, Vec<f32>)> = Vec::new();
        let t0 = Instant::now();

        for (k, (pi, segment)) in flat.iter().enumerate() {
            let (phonemes, mut oov) = self.g2p.phonemize_report(segment);
            unknown.append(&mut oov);
            let ids = self.model.cfg.encode(&phonemes);
            anyhow::ensure!(
                ids.len() <= MAX_TOKENS,
                "segment {k} is {} tokens, over the {MAX_TOKENS} the position table holds; \
                 lower --max-chars",
                ids.len()
            );
            if ids.len() <= 2 {
                continue;
            }
            let style = self.voices.style(&self.voice, ids.len() - 2)?;
            let (samples, timings) =
                self.model.synthesize_timed(&ids, &style, 1.0, &mut draws)?;
            for (stage, secs) in timings {
                stats.add(stage, secs);
            }
            stats.frames += samples.len() / HOP;
            stats.segments += 1;
            pieces.push((*pi, samples));
            request.advanced("decoder", k + 1, flat.len());
            request.check_interrupt(k + 1)?;
        }
        stats.total_s = t0.elapsed().as_secs_f64();

        if !unknown.is_empty() {
            unknown.sort_unstable();
            unknown.dedup();
            // The lexicon is the whole frontend — there is no espeak fallback to guess
            // with — so an unpronounceable word is dropped rather than approximated, and
            // naming it is the only way a caller can find that out.
            eprintln!(
                "engine {ID}: {} word(s) not in the lexicon: {}",
                unknown.len(),
                unknown.join(", ")
            );
        }

        let rate = Config::SAMPLE_RATE as usize;
        let gap = wav::silence(rate, request.gaps.segment_ms);
        let para_gap = wav::silence(rate, request.gaps.paragraph_ms);
        let mut samples: Vec<f32> = Vec::new();
        let mut prev: Option<usize> = None;
        for (pi, piece) in &pieces {
            if let Some(p) = prev {
                samples.extend_from_slice(if *pi != p { &para_gap } else { &gap });
            }
            samples.extend_from_slice(piece);
            prev = Some(*pi);
        }
        anyhow::ensure!(!samples.is_empty(), "engine {ID} produced no audio");

        Ok(Synthesis {
            audio: Audio { samples, sample_rate: Config::SAMPLE_RATE },
            stats,
        })
    }
}
