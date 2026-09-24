//! The PL-BERT front half: an ALBERT whose twelve layers are one layer applied twelve
//! times. Shared parameters are the whole reason an 82M model can afford a 12-layer
//! encoder — worth knowing before looking for the other eleven in the checkpoint.

use crate::cfg::Config;
use anyhow::Result;
use candle_core::{Device, Tensor};
use tts_nn::Weights;

const LN_EPS: f64 = 1e-12;

struct Linear {
    w: Tensor,
    b: Tensor,
}

impl Linear {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            // Transposed once at load: every call is x @ wᵀ.
            w: w.get_t(&format!("{prefix}.weight"))?,
            b: w.get(&format!("{prefix}.bias"))?,
        })
    }

    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.broadcast_matmul(&self.w)?.broadcast_add(&self.b)?)
    }
}

struct Norm {
    w: Tensor,
    b: Tensor,
}

impl Norm {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w: w.get(&format!("{prefix}.weight"))?,
            b: w.get(&format!("{prefix}.bias"))?,
        })
    }

    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        tts_nn::layer_norm(x, &self.w, &self.b, LN_EPS)
    }
}

pub struct Albert {
    word: Tensor,
    position: Tensor,
    token_type: Tensor,
    embed_norm: Norm,
    map_in: Linear,
    query: Linear,
    key: Linear,
    value: Linear,
    dense: Linear,
    attn_norm: Norm,
    ffn: Linear,
    ffn_out: Linear,
    full_norm: Norm,
    heads: usize,
    head_dim: usize,
    layers: usize,
}

impl Albert {
    pub fn load(w: &Weights, cfg: &Config) -> Result<Self> {
        let p = "bert.encoder.albert_layer_groups.0.albert_layers.0";
        let hidden = cfg.plbert.hidden_size;
        Ok(Self {
            word: w.get("bert.embeddings.word_embeddings.weight")?,
            position: w.get("bert.embeddings.position_embeddings.weight")?,
            token_type: w.get("bert.embeddings.token_type_embeddings.weight")?,
            embed_norm: Norm::load(w, "bert.embeddings.LayerNorm")?,
            map_in: Linear::load(w, "bert.encoder.embedding_hidden_mapping_in")?,
            query: Linear::load(w, &format!("{p}.attention.query"))?,
            key: Linear::load(w, &format!("{p}.attention.key"))?,
            value: Linear::load(w, &format!("{p}.attention.value"))?,
            dense: Linear::load(w, &format!("{p}.attention.dense"))?,
            attn_norm: Norm::load(w, &format!("{p}.attention.LayerNorm"))?,
            ffn: Linear::load(w, &format!("{p}.ffn"))?,
            ffn_out: Linear::load(w, &format!("{p}.ffn_output"))?,
            full_norm: Norm::load(w, &format!("{p}.full_layer_layer_norm"))?,
            heads: cfg.plbert.num_attention_heads,
            head_dim: hidden / cfg.plbert.num_attention_heads,
            layers: cfg.plbert.num_hidden_layers,
        })
    }

    /// `input_ids` is `[1, T]`. There is no padding to mask: the engine runs one sequence
    /// at a time, so the attention mask upstream builds is all ones.
    pub fn forward(&self, ids: &[u32], device: &Device) -> Result<Tensor> {
        let t = ids.len();
        let ids = Tensor::from_slice(ids, (t,), device)?;
        let mut x = self.word.index_select(&ids, 0)?;
        let pos = Tensor::arange(0u32, t as u32, device)?;
        x = (x + self.position.index_select(&pos, 0)?)?;
        // Every token is type 0; the second row of the table is never read.
        x = x.broadcast_add(&self.token_type.narrow(0, 0, 1)?)?;
        x = self.embed_norm.apply(&x.unsqueeze(0)?)?;
        x = self.map_in.apply(&x)?;

        for _ in 0..self.layers {
            let q = self.split_heads(&self.query.apply(&x)?, t)?;
            let k = self.split_heads(&self.key.apply(&x)?, t)?;
            let v = self.split_heads(&self.value.apply(&x)?, t)?;
            let scale = (self.head_dim as f64).sqrt();
            let scores = ((q.matmul(&k.transpose(2, 3)?)?) / scale)?;
            let ctx = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
            let ctx = ctx
                .transpose(1, 2)?
                .reshape((1, t, self.heads * self.head_dim))?
                .contiguous()?;
            // ALBERT normalises after the residual, inside the attention block.
            let attended = self.attn_norm.apply(&(self.dense.apply(&ctx)? + &x)?)?;
            let ffn = tts_nn::gelu_tanh(&self.ffn.apply(&attended)?)?;
            x = self
                .full_norm
                .apply(&(self.ffn_out.apply(&ffn)? + &attended)?)?;
        }
        Ok(x)
    }

    fn split_heads(&self, x: &Tensor, t: usize) -> Result<Tensor> {
        Ok(x.reshape((1, t, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?)
    }
}
