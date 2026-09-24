//! Duration and prosody: how long each phoneme lasts, and the pitch and energy over it.
//!
//! This is where a non-autoregressive model earns its speed. Duration is predicted for
//! every phoneme in one pass, and the encoding is then stretched by an alignment matrix
//! rather than generated step by step.

use crate::blocks::{AdaLayerNorm, AdainResBlk, Conv1d, Linear};
use crate::cfg::Config;
use crate::lstm::BiLstm;
use anyhow::Result;
use candle_core::{Device, Tensor};
use tts_nn::Weights;

pub struct DurationEncoder {
    lstms: Vec<BiLstm>,
    norms: Vec<AdaLayerNorm>,
}

impl DurationEncoder {
    pub fn load(w: &Weights, cfg: &Config) -> Result<Self> {
        let mut lstms = Vec::new();
        let mut norms = Vec::new();
        for i in 0..cfg.n_layer {
            lstms.push(BiLstm::load(
                w,
                &format!("predictor.text_encoder.lstms.{}", i * 2),
            )?);
            norms.push(AdaLayerNorm::load(
                w,
                &format!("predictor.text_encoder.lstms.{}", i * 2 + 1),
            )?);
        }
        Ok(Self { lstms, norms })
    }

    /// `d_en` is `[1, 512, T]`, `s` is `[1, 128]`; returns `[1, T, 640]`.
    ///
    /// The style is concatenated onto the channels before every LSTM and again after every
    /// norm, so each layer sees the voice rather than only the first.
    pub fn forward(&self, d_en: &Tensor, s: &Tensor) -> Result<Tensor> {
        let t = d_en.dim(2)?;
        let style = s
            .reshape((1, s.dim(1)?, 1))?
            .broadcast_as((1, s.dim(1)?, t))?
            .contiguous()?;
        let mut x = Tensor::cat(&[d_en, &style], 1)?;
        for (lstm, norm) in self.lstms.iter().zip(&self.norms) {
            x = lstm
                .forward(&x.transpose(1, 2)?.contiguous()?)?
                .transpose(1, 2)?
                .contiguous()?;
            x = norm.apply(&x, s)?;
            x = Tensor::cat(&[&x, &style], 1)?;
        }
        Ok(x.transpose(1, 2)?.contiguous()?)
    }
}

pub struct Predictor {
    pub text_encoder: DurationEncoder,
    lstm: BiLstm,
    duration_proj: Linear,
    shared: BiLstm,
    f0: Vec<AdainResBlk>,
    n: Vec<AdainResBlk>,
    f0_proj: Conv1d,
    n_proj: Conv1d,
}

impl Predictor {
    pub fn load(w: &Weights, cfg: &Config) -> Result<Self> {
        let hid = cfg.hidden_dim;
        let stack = |name: &str| -> Result<Vec<AdainResBlk>> {
            Ok(vec![
                AdainResBlk::load(w, &format!("predictor.{name}.0"), hid, false)?,
                AdainResBlk::load(w, &format!("predictor.{name}.1"), hid, true)?,
                AdainResBlk::load(w, &format!("predictor.{name}.2"), hid / 2, false)?,
            ])
        };
        Ok(Self {
            text_encoder: DurationEncoder::load(w, cfg)?,
            lstm: BiLstm::load(w, "predictor.lstm")?,
            duration_proj: Linear::load(w, "predictor.duration_proj.linear_layer")?,
            shared: BiLstm::load(w, "predictor.shared")?,
            f0: stack("F0")?,
            n: stack("N")?,
            f0_proj: Conv1d::load(w, "predictor.F0_proj", 1, 1)?,
            n_proj: Conv1d::load(w, "predictor.N_proj", 1, 1)?,
        })
    }

    /// Frames per phoneme. `d` is the duration encoder's output.
    pub fn durations(&self, d: &Tensor, speed: f32) -> Result<Vec<usize>> {
        let x = self.lstm.forward(d)?;
        let proj = self.duration_proj.apply(&x)?;
        let total = candle_nn::ops::sigmoid(&proj)?.sum(2)?.squeeze(0)?;
        let raw: Vec<f32> = (total / speed as f64)?.to_vec1()?;
        // torch.round is half-to-even. A phoneme landing on exactly x.5 is rare and the
        // difference is one frame, but a frame here shifts every later frame.
        Ok(raw
            .iter()
            .map(|v| (round_half_even(*v) as usize).max(1))
            .collect())
    }

    /// The `[T, frames]` matrix that stretches one column per phoneme into its duration.
    pub fn alignment(durations: &[usize], device: &Device) -> Result<Tensor> {
        let frames: usize = durations.iter().sum();
        let mut data = vec![0f32; durations.len() * frames];
        let mut at = 0;
        for (i, d) in durations.iter().enumerate() {
            for f in 0..*d {
                data[i * frames + at + f] = 1.0;
            }
            at += d;
        }
        Ok(Tensor::from_vec(
            data,
            (1, durations.len(), frames),
            device,
        )?)
    }

    /// Pitch and energy curves, at twice the frame rate — the middle block upsamples.
    pub fn f0_and_energy(&self, en: &Tensor, s: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = self.shared.forward(&en.transpose(1, 2)?.contiguous()?)?;
        let x = x.transpose(1, 2)?.contiguous()?;
        let mut f0 = x.clone();
        for block in &self.f0 {
            f0 = block.apply(&f0, s)?;
        }
        let mut n = x;
        for block in &self.n {
            n = block.apply(&n, s)?;
        }
        Ok((
            self.f0_proj.apply(&f0)?.squeeze(1)?,
            self.n_proj.apply(&n)?.squeeze(1)?,
        ))
    }
}

fn round_half_even(v: f32) -> f32 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - v.signum()
    } else {
        r
    }
}
