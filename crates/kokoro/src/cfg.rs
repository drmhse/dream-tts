//! `config.json`, as shipped with the checkpoint.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Deserialize, Debug, Clone)]
pub struct Plbert {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
}

#[derive(Deserialize, Debug, Clone)]
pub struct IstftNet {
    pub upsample_kernel_sizes: Vec<usize>,
    pub upsample_rates: Vec<usize>,
    pub gen_istft_hop_size: usize,
    pub gen_istft_n_fft: usize,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub upsample_initial_channel: usize,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    pub istftnet: IstftNet,
    pub plbert: Plbert,
    pub dim_in: usize,
    pub hidden_dim: usize,
    pub max_dur: usize,
    pub n_layer: usize,
    pub n_mels: usize,
    pub n_token: usize,
    pub style_dim: usize,
    pub text_encoder_kernel_size: usize,
    pub vocab: HashMap<String, u32>,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Ok(serde_json::from_slice(&raw)?)
    }

    /// ALBERT's embedding size is not in the file — it is the HF default, and the only
    /// place it shows up is the shape of `word_embeddings`. Audited against the
    /// checkpoint rather than assumed.
    pub const EMBEDDING_SIZE: usize = 128;
    pub const SAMPLE_RATE: u32 = 24_000;

    /// Phonemes the model has no symbol for are dropped, exactly as upstream does — the
    /// alternative is an id the embedding table does not have.
    pub fn sample_rate(&self) -> u32 {
        Self::SAMPLE_RATE
    }

    pub fn encode(&self, phonemes: &str) -> Vec<u32> {
        let mut ids = vec![0u32];
        ids.extend(phonemes.chars().filter_map(|c| self.vocab.get(&c.to_string()).copied()));
        ids.push(0);
        ids
    }
}
