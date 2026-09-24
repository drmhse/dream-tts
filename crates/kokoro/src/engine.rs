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
    SynthesisRequest, WordTime,
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

/// Footprint and Metal allocation, under `KOKORO_TIMING`.
fn mem_line(when: &str, device: &Device) {
    if std::env::var("KOKORO_TIMING").is_err() {
        return;
    }
    let gb = |b: Option<u64>| b.map_or("?".into(), |b| format!("{:.2} GB", b as f64 / 1e9));
    eprintln!(
        "engine {ID}: {when}: footprint {}, Metal allocated {}",
        gb(tts_core::system::footprint()),
        gb(tts_nn::allocated_bytes(device)),
    );
}

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
        // The duration predictor already computes a length for every phoneme on the
        // way to the audio; returning it is the whole cost.
        word_timings: true,
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
            Some(p) => p
                .to_str()
                .context("--set voice=<name> is not utf-8")?
                .to_string(),
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
        mem_line("loaded", &device);
        Ok(Self {
            model,
            voices,
            g2p,
            voice,
        })
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
        request.notify(tts_core::ProgressEvent::Planned {
            segments: flat.len(),
        });

        let mut draws = SeededDraws::new(request.sampling.seed);
        let mut stats = Stats::default();
        let mut unknown: Vec<String> = Vec::new();
        let mut pieces: Vec<tts_core::Piece> = Vec::new();
        // Word times relative to each piece, offset into the whole when the pieces are joined.
        let mut piece_words: Vec<Vec<WordTime>> = Vec::new();
        let t0 = Instant::now();

        // The frontend runs a segment ahead on another thread, so the GPU never waits on it.
        let (g2p, segments) = (&self.g2p, &flat);
        let (tx, rx) = std::sync::mpsc::sync_channel(2);
        std::thread::scope(|sc| -> Result<()> {
            sc.spawn(move || {
                for (_, segment) in segments {
                    if tx.send(g2p.phonemize_all(segment)).is_err() {
                        break;
                    }
                }
            });
            for (k, (pi, segment)) in flat.iter().enumerate() {
                let (phonemes, mut oov, spans) =
                    rx.recv().context("the frontend thread stopped")?;
                unknown.append(&mut oov);
                let (ids, offsets) = self.model.cfg.encode_spans(&phonemes);
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
                let (samples, timings, durations) = self
                    .model
                    .synthesize_aligned(&ids, &style, 1.0, &mut draws)?;
                for (stage, secs) in timings {
                    stats.add(stage, secs);
                }
                stats.frames += samples.len() / HOP;
                stats.segments += 1;
                // Seconds per predictor frame, taken from the audio this render actually produced
                // rather than from a constant. The decoder's upsampling is a property of the
                // checkpoint, and a constant that is right for one and wrong for another would
                // compress the whole clock silently — which is exactly what a wrong `HOP` did.
                let frames: usize = durations.iter().sum();
                let per_frame = if frames == 0 {
                    0.0
                } else {
                    samples.len() as f64 / frames as f64 / Config::SAMPLE_RATE as f64
                };
                piece_words.push(word_times(&spans, &offsets, &durations, per_frame));
                pieces.push(tts_core::Piece {
                    paragraph: *pi,
                    text: (*segment).clone(),
                    samples,
                });
                request.advanced("decoder", k + 1, flat.len());
                request.check_interrupt(k + 1)?;
            }
            Ok(())
        })?;
        stats.total_s = t0.elapsed().as_secs_f64();
        mem_line("after synthesis", self.model.device());

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
        // Where each piece lands is decided once, in `tts-core`, so the word clock's offsets
        // and the segment times cannot disagree about where a gap went.
        let (samples, segments) = tts_core::join_segments(pieces, request.gaps, rate);
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
                sample_rate: Config::SAMPLE_RATE,
            },
            stats,
            segments: Some(segments),
            words: Some(words),
        })
    }
}

/// Frames per phoneme, into seconds per word.
///
/// `offsets[i]` is where in the phoneme string id `i` came from, and a span says which bytes of
/// that string one source word produced — so the id belongs to the word whose span contains its
/// offset. Both are in reading order, so one walk places every phoneme.
///
/// The pad ids at each end belong to no word and are given an offset past every span. Their
/// frames still advance the clock, because they are real audio: a word's start is where it
/// starts in the file, not where it starts among the words.
fn word_times(
    spans: &[tts_phoneme::g2p::WordSpan],
    offsets: &[Option<usize>],
    durations: &[usize],
    per_frame: f64,
) -> Vec<WordTime> {
    let seconds = |frames: usize| frames as f64 * per_frame;
    let mut bounds: Vec<Option<(usize, usize)>> = vec![None; spans.len()];
    let mut frame = 0usize;
    let mut at = 0usize;

    for (id, length) in durations.iter().enumerate() {
        // A pad belongs to no phoneme, so it moves the clock and not the cursor.
        if let Some(offset) = offsets.get(id).copied().flatten() {
            while at < spans.len() && spans[at].phonemes.end <= offset {
                at += 1;
            }
            if at < spans.len() && spans[at].phonemes.contains(&offset) {
                bounds[at] = Some(match bounds[at] {
                    Some((start, _)) => (start, frame + length),
                    None => (frame, frame + length),
                });
            }
        }
        frame += length;
    }

    spans
        .iter()
        .zip(bounds)
        .filter_map(|(span, bound)| {
            let (start, end) = bound?;
            Some(WordTime {
                text: span.text.clone(),
                start: seconds(start),
                end: seconds(end),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tts_phoneme::g2p::WordSpan;

    fn span(text: &str, phonemes: std::ops::Range<usize>) -> WordSpan {
        WordSpan {
            text: text.to_string(),
            phonemes,
        }
    }

    /// The pads carry real audio and no word, so the first word does not start at zero and the
    /// last does not end at the file's end.
    #[test]
    fn a_word_is_timed_by_the_phonemes_it_owns() {
        // "hi there": phonemes "hI DEr", two words either side of a space at byte 2.
        let spans = vec![span("hi", 0..2), span("there", 3..6)];
        //          pad  h  I  (space dropped)  D  E  r  pad
        let offsets = vec![None, Some(0), Some(1), Some(3), Some(4), Some(5), None];
        let durations = vec![4, 2, 6, 3, 5, 2, 4];
        let frame = 0.0125;
        let out = word_times(&spans, &offsets, &durations, frame);

        assert_eq!(out.len(), 2);
        assert_eq!(out[0].text, "hi");
        // Starts after the leading pad's 4 frames, runs for 2 + 6.
        assert!((out[0].start - 4.0 * frame).abs() < 1e-9, "{:?}", out[0]);
        assert!((out[0].end - 12.0 * frame).abs() < 1e-9, "{:?}", out[0]);
        assert_eq!(out[1].text, "there");
        assert!((out[1].start - 12.0 * frame).abs() < 1e-9, "{:?}", out[1]);
        assert!((out[1].end - 22.0 * frame).abs() < 1e-9, "{:?}", out[1]);
    }

    /// A word whose phonemes were all dropped by the vocabulary has no time to report, and
    /// inventing one would be the interpolation this whole path exists to avoid.
    #[test]
    fn a_word_with_no_phonemes_is_left_out_rather_than_guessed() {
        let spans = vec![
            span("hi", 0..2),
            span("\u{1f600}", 2..2),
            span("there", 3..6),
        ];
        let offsets = vec![None, Some(0), Some(1), Some(3), Some(4), Some(5), None];
        let durations = vec![4, 2, 6, 3, 5, 2, 4];
        let out = word_times(&spans, &offsets, &durations, 0.0125);
        assert_eq!(
            out.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(),
            ["hi", "there"]
        );
    }

    #[test]
    fn a_word_clock_never_runs_backwards() {
        let spans = vec![span("a", 0..1), span("b", 2..3), span("c", 4..5)];
        let offsets = vec![None, Some(0), Some(2), Some(4), None];
        let durations = vec![1, 7, 3, 9, 1];
        let out = word_times(&spans, &offsets, &durations, 0.0125);
        for pair in out.windows(2) {
            assert!(pair[0].end <= pair[1].start, "{pair:?}");
            assert!(pair[0].start < pair[0].end, "{pair:?}");
        }
    }
}
