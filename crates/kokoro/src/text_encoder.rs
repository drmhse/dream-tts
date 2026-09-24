//! Phoneme identities to a 512-wide encoding: three convolutions and one BiLSTM.
//!
//! Separate from the prosody path on purpose — the predictor decides *how long* each
//! phoneme is, this decides *what* it sounds like, and both are stretched by the same
//! alignment afterwards.

use crate::blocks::{ChannelNorm, Conv1d};
use crate::cfg::Config;
use crate::lstm::BiLstm;
use anyhow::Result;
use candle_core::{Device, Tensor};
use tts_nn::Weights;

pub struct TextEncoder {
    embedding: Tensor,
    convs: Vec<(Conv1d, ChannelNorm)>,
    lstm: BiLstm,
}

impl TextEncoder {
    pub fn load(w: &Weights, cfg: &Config) -> Result<Self> {
        let mut convs = Vec::new();
        for i in 0..cfg.n_layer {
            convs.push((
                Conv1d::load(w, &format!("text_encoder.cnn.{i}.0"), 1, 1)?,
                ChannelNorm::load(w, &format!("text_encoder.cnn.{i}.1"))?,
            ));
        }
        Ok(Self {
            embedding: w.get("text_encoder.embedding.weight")?,
            convs,
            lstm: BiLstm::load(w, "text_encoder.lstm")?,
        })
    }

    /// Returns `[1, 512, T]`.
    pub fn forward(&self, ids: &[u32], device: &Device) -> Result<Tensor> {
        let t = ids.len();
        let idx = Tensor::from_slice(ids, (t,), device)?;
        let mut x = self
            .embedding
            .index_select(&idx, 0)?
            .unsqueeze(0)?
            .transpose(1, 2)?
            .contiguous()?;
        for (conv, norm) in &self.convs {
            x = tts_nn::leaky_relu(&norm.apply(&conv.apply(&x)?)?, 0.2)?;
        }
        let x = self.lstm.forward(&x.transpose(1, 2)?.contiguous()?)?;
        Ok(x.transpose(1, 2)?.contiguous()?)
    }
}
