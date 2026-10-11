//! What `/tts` and `/v1/batch` do, without the server: for a host that keeps the engine in its
//! own process. A socket to oneself is one more thing to fail, and iOS takes a suspended app's
//! listening sockets away.

use crate::batch::{pieces, Piece};
use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;
use tts_core::{Engine, EngineConfig, SynthesisRequest, Voice};

pub struct Local {
    engine: Box<dyn Engine>,
    id: String,
    sample_rate: u32,
    word_timings: bool,
    speakers: Vec<String>,
    segment_chars: usize,
    /// One synthesis at a time, as the server's GPU permit keeps it.
    busy: Mutex<()>,
}

/// One text's audio and clock, or a batch's whole run.
pub struct Said {
    pub pieces: Vec<Piece>,
    pub wall: f64,
}

impl Local {
    pub fn load(engine: &str, model_root: &Path, segment_chars: usize, cpu: bool) -> Result<Self> {
        let config = EngineConfig {
            model_root: model_root.to_path_buf(),
            quant: None,
            cpu,
            overrides: Default::default(),
        };
        let started = Instant::now();
        let engine = tts_engines::load(engine, &config)
            .with_context(|| format!("loading engine {engine}"))?;
        let caps = engine.capabilities();
        eprintln!(
            "loaded in {:.2}s — {}",
            started.elapsed().as_secs_f64(),
            caps.id
        );
        Ok(Self {
            id: caps.id.to_string(),
            sample_rate: caps.sample_rate,
            word_timings: caps.word_timings,
            speakers: tts_engines::builtin_voices(caps.id, model_root).unwrap_or_default(),
            engine,
            segment_chars,
            busy: Mutex::new(()),
        })
    }

    pub fn engine(&self) -> &str {
        &self.id
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn word_timings(&self) -> bool {
        self.word_timings
    }

    pub fn speakers(&self) -> &[String] {
        &self.speakers
    }

    /// Several paragraphs in one run, cut back into one piece each, as `/v1/batch` answers.
    pub fn speak(
        &self,
        texts: &[&str],
        speaker: Option<&str>,
        asset: Option<&Path>,
    ) -> Result<Said> {
        anyhow::ensure!(!texts.is_empty(), "nothing to say");
        // One paragraph per text: the segmenter splits paragraphs on lines.
        let texts: Vec<String> = texts
            .iter()
            .map(|t| t.split_whitespace().collect::<Vec<_>>().join(" "))
            .collect();
        anyhow::ensure!(texts.iter().all(|t| !t.is_empty()), "text is empty");
        let mut request = SynthesisRequest::new(texts.join("\n"));
        if let Some(asset) = asset {
            request = request.with_voice(
                Voice::load(asset)
                    .with_context(|| format!("loading voice asset {}", asset.display()))?,
            );
        }
        request.speaker = speaker.filter(|s| !s.trim().is_empty()).map(str::to_string);
        request.max_chars = self.segment_chars;
        self.engine.validate(&request)?;

        let _busy = self.busy.lock().unwrap_or_else(|e| e.into_inner());
        let started = Instant::now();
        let out = self.engine.synthesize(&request)?;
        let wall = started.elapsed().as_secs_f64();
        let triples = |t: &str, s: f64, e: f64| (t.to_string(), s, e);
        let pieces = if texts.len() == 1 {
            // Whole, as `/tts` answers: nothing to cut, and an engine need not report segments.
            vec![Piece {
                words: out
                    .words
                    .map(|w| w.iter().map(|w| triples(&w.text, w.start, w.end)).collect()),
                segments: out
                    .segments
                    .unwrap_or_default()
                    .iter()
                    .map(|s| triples(&s.text, s.start, s.end))
                    .collect(),
                audio: out.audio,
            }]
        } else {
            let segments = out.segments.unwrap_or_default();
            pieces(&out.audio, &segments, out.words.as_deref(), texts.len()).context(
                "the texts came back as a different number of paragraphs; send them singly",
            )?
        };
        Ok(Said { pieces, wall })
    }
}
