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

    /// Several projections of one input as one, joined on the host; `scale[i]` multiplies the
    /// `i`th, weights and bias.
    fn stacked(w: &Weights, prefixes: &[&str], scale: &[f64]) -> Result<Self> {
        let host = |name: String| -> Result<Tensor> {
            Ok(w.cpu(&name)?.to_dtype(candle_core::DType::F32)?)
        };
        let ws = prefixes
            .iter()
            .zip(scale)
            .map(|(p, s)| Ok((host(format!("{p}.weight"))? * *s)?))
            .collect::<Result<Vec<_>>>()?;
        let bs = prefixes
            .iter()
            .zip(scale)
            .map(|(p, s)| Ok((host(format!("{p}.bias"))? * *s)?))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            w: Tensor::cat(&ws, 0)?
                .t()?
                .contiguous()?
                .to_device(w.device())?,
            b: Tensor::cat(&bs, 0)?.to_device(w.device())?,
        })
    }

    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.broadcast_matmul(&self.w)?.broadcast_add(&self.b)?)
    }

    /// The bias broadcast to `[1, t, out]` once, for a layer applied twelve times: a broadcast
    /// add is 33 us at a sentence's length and a plain one 4.
    fn bias_for(&self, t: usize) -> Result<Tensor> {
        Ok(self.b.broadcast_as((1, t, self.b.dim(0)?))?.contiguous()?)
    }

    fn apply_with(&self, x: &Tensor, bias: &Tensor) -> Result<Tensor> {
        Ok((x.broadcast_matmul(&self.w)? + bias)?)
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
    /// Query, key and value as one projection, so one bias add and one head split serve all
    /// three: at a sentence's length those small passes cost more than the GEMMs.
    qkv: Linear,
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
            qkv: Linear::stacked(
                w,
                &[
                    &format!("{p}.attention.query"),
                    &format!("{p}.attention.key"),
                    &format!("{p}.attention.value"),
                ],
                // 1/sqrt(head_dim) folded into the queries: a power of two, so exact.
                &[
                    1.0 / ((hidden / cfg.plbert.num_attention_heads) as f64).sqrt(),
                    1.0,
                    1.0,
                ],
            )?,
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

        let (b_qkv, b_dense, b_ffn, b_out) = (
            self.qkv.bias_for(t)?,
            self.dense.bias_for(t)?,
            self.ffn.bias_for(t)?,
            self.ffn_out.bias_for(t)?,
        );
        for _ in 0..self.layers {
            let qkv = self
                .qkv
                .apply_with(&x, &b_qkv)?
                .reshape((t, 3, self.heads, self.head_dim))?
                .permute((1, 2, 0, 3))?
                .contiguous()?;
            let (q, k, v) = (qkv.get(0)?, qkv.get(1)?, qkv.get(2)?);
            let scores = q.matmul(&k.t()?)?;
            let ctx = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
            let ctx = ctx
                .transpose(0, 1)?
                .reshape((1, t, self.heads * self.head_dim))?
                .contiguous()?;
            // ALBERT normalises after the residual, inside the attention block.
            let attended = self
                .attn_norm
                .apply(&(self.dense.apply_with(&ctx, &b_dense)? + &x)?)?;
            let ffn = tts_nn::gelu_tanh(&self.ffn.apply_with(&attended, &b_ffn)?)?;
            x = self
                .full_norm
                .apply(&(self.ffn_out.apply_with(&ffn, &b_out)? + &attended)?)?;
        }
        Ok(x)
    }
}
