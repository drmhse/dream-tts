//! The five stages wired together: text ids and a voice in, samples out.

use crate::albert::Albert;
use crate::blocks::Linear;
use crate::cfg::Config;
use crate::decoder::Decoder;
use crate::predictor::Predictor;
use crate::source::Draws;
use crate::text_encoder::TextEncoder;
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::Path;
use tts_nn::Weights;

pub struct Model {
    pub cfg: Config,
    bert: Albert,
    bert_encoder: Linear,
    predictor: Predictor,
    text_encoder: TextEncoder,
    decoder: Decoder,
    device: Device,
}

/// The 54 shipped voices, each a `[510, 256]` table indexed by phoneme count.
///
/// A voice is 522 KB of style vectors, not a model — which is why they all fit in one file
/// and why this engine cannot clone: there is no path from reference audio to one of these.
pub struct Voices(HashMap<String, Tensor>);

impl Voices {
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let w = Weights::load(path.to_str().context("voice path")?, device)?;
        let mut map = HashMap::new();
        for name in w.names() {
            map.insert(name.clone(), w.get(&name)?);
        }
        Ok(Self(map))
    }

    pub fn names(&self) -> Vec<&str> {
        let mut n: Vec<&str> = self.0.keys().map(String::as_str).collect();
        n.sort_unstable();
        n
    }

    /// The style row for a phoneme count. Upstream indexes by `len(phonemes) - 1`, which
    /// is how the same voice speaks a short phrase and a long one differently.
    pub fn style(&self, name: &str, phonemes: usize) -> Result<Tensor> {
        let table = self
            .0
            .get(name)
            .with_context(|| format!("unknown voice `{name}`; have {}", self.names().join(", ")))?;
        let row = phonemes.saturating_sub(1).min(table.dim(0)? - 1);
        Ok(table.narrow(0, row, 1)?.contiguous()?)
    }
}

impl Model {
    pub fn load(root: &Path, device: &Device) -> Result<Self> {
        let cfg = Config::load(&root.join("config.json"))?;
        let path = root.join("kokoro.safetensors");
        let w = Weights::load(path.to_str().context("weight path")?, device)?;
        Ok(Self {
            bert: Albert::load(&w, &cfg)?,
            bert_encoder: Linear::load(&w, "bert_encoder")?,
            predictor: Predictor::load(&w, &cfg)?,
            text_encoder: TextEncoder::load(&w, &cfg)?,
            decoder: Decoder::load(&w, &cfg)?,
            cfg,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// `style` is one `[1, 256]` row: the first half conditions the decoder, the second
    /// half the prosody predictor.
    pub fn synthesize(
        &self,
        ids: &[u32],
        style: &Tensor,
        speed: f32,
        draws: &mut (dyn Draws + Send),
    ) -> Result<Vec<f32>> {
        Ok(self.synthesize_timed(ids, style, speed, draws)?.0)
    }

    pub fn synthesize_timed(
        &self,
        ids: &[u32],
        style: &Tensor,
        speed: f32,
        draws: &mut (dyn Draws + Send),
    ) -> Result<(Vec<f32>, Vec<(&'static str, f64)>)> {
        let mut timings: Vec<(&'static str, f64)> = Vec::new();
        // Metal dispatch is asynchronous: a timer that stops when a stage returns measures
        // enqueue time and bills the work to whatever is timed next.
        let mut mark = std::time::Instant::now();
        let mut lap = |name: &'static str,
                       timings: &mut Vec<(&'static str, f64)>,
                       mark: &mut std::time::Instant,
                       device: &Device|
         -> Result<()> {
            device.synchronize()?;
            timings.push((name, mark.elapsed().as_secs_f64()));
            *mark = std::time::Instant::now();
            Ok(())
        };
        let _ = &mut lap;
        let s_pred = style.narrow(1, 128, 128)?.contiguous()?;
        let s_dec = style.narrow(1, 0, 128)?.contiguous()?;

        let hidden = self.bert.forward(ids, &self.device)?;
        let d_en = self.bert_encoder.apply(&hidden)?.transpose(1, 2)?.contiguous()?;
        lap("bert", &mut timings, &mut mark, &self.device)?;

        let d = self.predictor.text_encoder.forward(&d_en, &s_pred)?;
        let durations = self.predictor.durations(&d, speed)?;
        lap("duration", &mut timings, &mut mark, &self.device)?;

        let aln = Predictor::alignment(&durations, &self.device)?;
        let en = d.transpose(1, 2)?.contiguous()?.matmul(&aln)?;
        let (f0, energy) = self.predictor.f0_and_energy(&en, &s_pred)?;
        lap("prosody", &mut timings, &mut mark, &self.device)?;

        let asr = self.text_encoder.forward(ids, &self.device)?.matmul(&aln)?;
        lap("encoder", &mut timings, &mut mark, &self.device)?;

        let audio = self.decoder.forward(&asr, &f0, &energy, &s_dec, draws)?;
        lap("decoder", &mut timings, &mut mark, &self.device)?;
        Ok((audio, timings))
    }
}
