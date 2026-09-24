//! A Qwen3 decoder stack, shared by all three transformers in this engine.
//!
//! The talker, the depth predictor and the codec's pre-transformer have the same layer
//! shape and differ only in flags: QK-norm (talker, predictor), per-residual `LayerScale`
//! (codec), and a sliding window (codec only). One implementation with those as [`Geometry`]
//! fields beats three near-copies.
//!
//! RoPE is plain half-split at the configured theta — see trap 1, the config's M-RoPE is
//! degenerate. Attention has no biases anywhere.

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use tts_nn::{rms_norm, rope_table_f32, Proj, Weight, Weights};

#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub eps: f32,
    pub rope_base: f64,
    pub qk_norm: bool,
    pub layer_scale: bool,
    pub window: Option<usize>,
}

impl Geometry {
    pub fn gqa(&self) -> usize {
        self.heads / self.n_kv
    }
    pub fn q_width(&self) -> usize {
        self.heads * self.head_dim
    }
    pub fn kv_width(&self) -> usize {
        self.n_kv * self.head_dim
    }
}

struct Layer {
    attn_norm: Tensor,
    ffn_norm: Tensor,
    /// Fused at load: one `[q_width + 2 * kv_width, dim]` matmul per step instead of three.
    wqkv: Proj,
    wo: Proj,
    gate: Proj,
    up: Proj,
    down: Proj,
    /// `[head_dim]` each, applied per-head before RoPE.
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    /// `[q_norm; k_norm]`, for `tts_nn::qkrope`.
    qk_norms: Option<Tensor>,
    attn_scale: Option<Tensor>,
    mlp_scale: Option<Tensor>,
}

impl Layer {
    fn load(
        w: &Weights,
        prefix: &str,
        geo: &Geometry,
        how: Weight,
        device: &Device,
    ) -> Result<Self> {
        let a = format!("{prefix}.self_attn");
        // On the host: a device-side cat would leave its three inputs pooled for good.
        let wqkv = Tensor::cat(
            &[
                w.cpu(&format!("{a}.q_proj.weight"))?,
                w.cpu(&format!("{a}.k_proj.weight"))?,
                w.cpu(&format!("{a}.v_proj.weight"))?,
            ],
            0,
        )?;
        // `1/sqrt(head_dim)` folded in. QK-norm rescales q to unit RMS and then multiplies by
        // this weight, and RoPE is a rotation, so scaling the weight is exactly scaling the
        // scores — one fewer dispatch per layer per step. Layers without QK-norm scale
        // explicitly.
        let q_norm = if geo.qk_norm {
            let s = (geo.head_dim as f64).sqrt().recip();
            Some((w.get(&format!("{a}.q_norm.weight"))? * s)?)
        } else {
            None
        };
        let k_norm = if geo.qk_norm {
            Some(w.get(&format!("{a}.k_norm.weight"))?)
        } else {
            None
        };
        let qk_norms = match (&q_norm, &k_norm) {
            (Some(q), Some(k)) => Some(Tensor::stack(&[q, k], 0)?.contiguous()?),
            _ => None,
        };
        Ok(Self {
            attn_norm: w.get(&format!("{prefix}.input_layernorm.weight"))?,
            ffn_norm: w.get(&format!("{prefix}.post_attention_layernorm.weight"))?,
            wqkv: Proj::from_tensor_as(&wqkv, how, device)?,
            wo: Proj::load_as(w, &format!("{a}.o_proj.weight"), how, device)?,
            gate: Proj::load_as(w, &format!("{prefix}.mlp.gate_proj.weight"), how, device)?,
            up: Proj::load_as(w, &format!("{prefix}.mlp.up_proj.weight"), how, device)?,
            down: Proj::load_as(w, &format!("{prefix}.mlp.down_proj.weight"), how, device)?,
            q_norm,
            k_norm,
            qk_norms,
            attn_scale: if geo.layer_scale {
                Some(w.get(&format!("{prefix}.self_attn_layer_scale.scale"))?)
            } else {
                None
            },
            mlp_scale: if geo.layer_scale {
                Some(w.get(&format!("{prefix}.mlp_layer_scale.scale"))?)
            } else {
                None
            },
        })
    }
}

struct Cache {
    k: Tensor,
    v: Tensor,
}

/// Decode state: preallocated K/V plus how many positions are written.
///
/// Capacity lives here rather than on the [`Stack`] because it is the one dimension that has
/// to shrink as the batch grows: a position costs 229 KB across 28 layers of k and v, so the
/// talker's 1536 is 352 MB for one lane and 2.8 GB for eight.
pub struct State {
    caches: Vec<Cache>,
    pub width: usize,
    pub batch: usize,
    capacity: usize,
}

impl State {
    /// Keep only the first `live` lanes.
    ///
    /// A prefix narrow is a view over the same storage, so shedding a finished *tail* costs a
    /// view where shedding an interior lane would mean rebuilding all 28 caches. It is also
    /// why a batch has to be ordered longest-first: that is what puts the early finishers at
    /// the end.
    /// A view of lanes `[off, off + width)`, sharing the parent's cache storage.
    ///
    /// Writes land in the parent: `slice_set` adds the view's own start offset, and a row range
    /// of the outermost dimension keeps natural strides, so the view is contiguous and the copy
    /// strides are the parent's. The parent's `width` is deliberately *not* advanced — every
    /// window writes the same positions, so the caller advances it once, after the last one.
    pub fn lane_window(&self, off: usize, width: usize) -> Result<State> {
        if width == 0 || off + width > self.batch {
            bail!(
                "lane window {off}..{} outside a batch of {}",
                off + width,
                self.batch
            );
        }
        let caches = self
            .caches
            .iter()
            .map(|c| {
                Ok(Cache {
                    k: c.k.narrow(0, off, width)?,
                    v: c.v.narrow(0, off, width)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(State {
            caches,
            width: self.width,
            batch: width,
            capacity: self.capacity,
        })
    }

    /// Copy a one-lane state's positions into every lane, and advance to its width.
    pub fn fill_prefix(&mut self, from: &State) -> Result<()> {
        if from.batch != 1 || from.width > self.capacity || from.caches.len() != self.caches.len() {
            bail!("cannot fill a {}-lane state from this prefix", self.batch);
        }
        let n = from.width;
        for (dst, src) in self.caches.iter_mut().zip(&from.caches) {
            for (d, s) in [(&mut dst.k, &src.k), (&mut dst.v, &src.v)] {
                let (_, h, _, hd) = s.dims4()?;
                let s = s
                    .narrow(2, 0, n)?
                    .broadcast_as((self.batch, h, n, hd))?
                    .contiguous()?;
                d.slice_set(&s, 2, 0)?;
            }
        }
        self.width = n;
        Ok(())
    }

    pub fn narrow_to(&mut self, live: usize) -> Result<()> {
        if live == self.batch {
            return Ok(());
        }
        if live == 0 || live > self.batch {
            bail!("cannot narrow a batch of {} to {live}", self.batch);
        }
        for c in self.caches.iter_mut() {
            c.k = c.k.narrow(0, 0, live)?;
            c.v = c.v.narrow(0, 0, live)?;
        }
        self.batch = live;
        Ok(())
    }
}

pub struct Stack {
    layers: Vec<Layer>,
    /// KV cache dtype. f16 with f16 weights: 114 KB per position per lane instead of 229, which
    /// is what caps how many lanes a batch can hold. Decode reads it through `tts_nn::attn`,
    /// which takes either; prefill casts, being once per segment.
    kv: DType,
    norm: Tensor,
    cos: Tensor,
    sin: Tensor,
    pub geo: Geometry,
    capacity: usize,
    device: Device,
}

impl Stack {
    pub fn load(
        w: &Weights,
        prefix: &str,
        geo: Geometry,
        how: Weight,
        capacity: usize,
        device: &Device,
    ) -> Result<Self> {
        let layers = (0..geo.layers)
            .map(|i| Layer::load(w, &format!("{prefix}layers.{i}"), &geo, how, device))
            .collect::<Result<Vec<_>>>()?;
        let (cos, sin) = rope_table_f32(capacity, geo.head_dim, geo.rope_base, device)?;
        Ok(Self {
            layers,
            kv: if how == Weight::F16 {
                DType::F16
            } else {
                DType::F32
            },
            norm: w.get(&format!("{prefix}norm.weight"))?,
            cos,
            sin,
            geo,
            capacity,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn new_state(&self, batch: usize) -> Result<State> {
        self.new_state_with(batch, self.capacity)
    }

    /// A state holding only `capacity` positions. Clamped to the RoPE tables, which are built
    /// for `self.capacity` and are the real ceiling.
    pub fn new_state_with(&self, batch: usize, capacity: usize) -> Result<State> {
        let capacity = capacity.clamp(1, self.capacity);
        let shape = (batch, self.geo.n_kv, capacity, self.geo.head_dim);
        let caches = (0..self.geo.layers)
            .map(|_| {
                Ok(Cache {
                    k: Tensor::zeros(shape, self.kv, &self.device)?,
                    v: Tensor::zeros(shape, self.kv, &self.device)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(State {
            caches,
            width: 0,
            batch,
            capacity,
        })
    }

    /// The largest state [`Self::new_state_with`] will build — the RoPE table length.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Run `t` positions, appending to the cache. Returns the normed hidden states,
    /// `[b, t, dim]` — all of them, since the codec needs the whole sequence and the
    /// decode loops take the last.
    pub fn forward(&self, x: &Tensor, state: &mut State) -> Result<Tensor> {
        let (b, t, dim) = x.dims3()?;
        if b != state.batch {
            bail!("batch {b} != state batch {}", state.batch);
        }
        if dim != self.geo.dim {
            bail!("input width {dim} != {}", self.geo.dim);
        }
        let start = state.width;
        if start + t > state.capacity {
            bail!(
                "{} positions exceeds cache capacity {}",
                start + t,
                state.capacity
            );
        }
        let g = &self.geo;
        // Folded into `q_norm` wherever QK-norm exists; see `Layer::load`.
        let scale = if g.qk_norm {
            None
        } else {
            Some(1.0 / (g.head_dim as f64).sqrt())
        };
        // `[b, t, n, head_dim] -> [b, n, t, head_dim]`. At `t == 1` those are the same bytes
        // in the same order, so the transpose is a free reshape rather than a copy — three
        // fewer whole-tensor copies per layer on every decode step.
        let heads_first = |x: &Tensor, n: usize| -> Result<Tensor> {
            if t == 1 {
                Ok(x.reshape((b, n, 1, g.head_dim))?)
            } else {
                Ok(x.transpose(1, 2)?.contiguous()?)
            }
        };
        let mut h = x.clone();
        // Prefill's mask, tiled over the query heads that share a kv head (see the prefill
        // branch). Built once: per layer it was 28 host builds and uploads.
        let mask = if t > 1 {
            let block = tts_nn::causal_window_mask(start + t, g.window, &self.device)?
                .narrow(0, start, t)?;
            Some(Tensor::cat(&vec![&block; g.gqa()], 0)?)
        } else {
            None
        };

        for (li, layer) in self.layers.iter().enumerate() {
            let normed = rms_norm(&h, &layer.attn_norm, g.eps)?;
            let qkv = layer.wqkv.forward(&normed)?;
            // `fresh` is the new positions' k and v at f32, which only prefill reads back.
            let cache = &mut state.caches[li];
            let (q, fresh) = match &layer.qk_norms {
                Some(norms) if tts_nn::qkrope::eligible(&qkv, &cache.k, g.head_dim) => {
                    tts_nn::qkrope::apply(
                        &qkv,
                        norms,
                        &self.cos,
                        &self.sin,
                        &cache.k,
                        &cache.v,
                        g.heads,
                        start,
                        g.eps,
                        t > 1,
                    )?
                }
                _ => {
                    let q = qkv.narrow(candle_core::D::Minus1, 0, g.q_width())?;
                    let kk = qkv.narrow(candle_core::D::Minus1, g.q_width(), g.kv_width())?;
                    let v = qkv.narrow(
                        candle_core::D::Minus1,
                        g.q_width() + g.kv_width(),
                        g.kv_width(),
                    )?;

                    // QK-norm is over the head dim, so it must happen on the [.., heads,
                    // head_dim] view and before the transpose — normalising the flat
                    // projection is a different, still-running model.
                    let q = q.reshape((b, t, g.heads, g.head_dim))?;
                    let kk = kk.reshape((b, t, g.n_kv, g.head_dim))?;
                    let q = match &layer.q_norm {
                        Some(n) => rms_norm(&q, n, g.eps)?,
                        None => q,
                    };
                    let kk = match &layer.k_norm {
                        Some(n) => rms_norm(&kk, n, g.eps)?,
                        None => kk,
                    };
                    let q = heads_first(&q, g.heads)?;
                    let kk = heads_first(&kk, g.n_kv)?;
                    let v = heads_first(&v.reshape((b, t, g.n_kv, g.head_dim))?, g.n_kv)?;

                    let cos = self.cos.narrow(0, start, t)?;
                    let sin = self.sin.narrow(0, start, t)?;
                    let q = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
                    let kk = candle_nn::rotary_emb::rope(&kk, &cos, &sin)?;

                    // In place. `slice_assign` reallocates the whole cache per token and cost
                    // 2.0x on CosyVoice's LLM stage.
                    cache.k.slice_set(&kk.to_dtype(self.kv)?, 2, start)?;
                    cache.v.slice_set(&v.to_dtype(self.kv)?, 2, start)?;
                    (q, Some((kk, v)))
                }
            };

            let span = start + t;

            let attn =
                if t == 1 {
                    // Reads the cache in place; candle's route copies the span twice per layer.
                    // Folded into `q_norm` for the talker and predictor; applied here for any
                    // layer without QK-norm, since the kernel takes no scale.
                    let qs = match scale {
                        Some(s) => (q.clone() * s)?,
                        None => q.clone(),
                    };
                    let qg = qs.reshape((b, g.n_kv, g.gqa(), g.head_dim))?;
                    let wstart = match g.window {
                        Some(w) if span > w => span - w,
                        _ => 0,
                    };
                    tts_nn::attn::decode_attention(&qg, &cache.k, &cache.v, span, wstart)?
                        .reshape((b, 1, g.q_width()))?
                } else {
                    // The new positions attend over their own f32 k/v, not the cache's copy, and
                    // only an earlier prefix is read back. Query heads sharing a kv head are
                    // stacked along rows, `[b, n_kv, gqa * t, hd]`, a free reshape where
                    // `repeat_kv` copied k and v per layer; these two cost 535 ms of an 8-lane
                    // window's 1.87 s.
                    let (kk, v) = fresh.expect("prefill k and v");
                    let (k_all, v_all) = if start == 0 {
                        (kk, v)
                    } else {
                        let prior = |c: &Tensor| c.narrow(2, 0, start)?.to_dtype(DType::F32);
                        (
                            Tensor::cat(&[&prior(&cache.k)?, &kk], 2)?,
                            Tensor::cat(&[&prior(&cache.v)?, &v], 2)?,
                        )
                    };
                    let qg = q.reshape((b, g.n_kv, g.gqa() * t, g.head_dim))?;
                    let scores = qg.matmul(&k_all.contiguous()?.t()?)?;
                    let scores = match scale {
                        Some(s) => (scores * s)?,
                        None => scores,
                    };
                    let scores = scores.broadcast_add(mask.as_ref().expect("prefill mask"))?;
                    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
                    probs
                        .matmul(&v_all.contiguous()?)?
                        .reshape((b, g.heads, t, g.head_dim))?
                        .transpose(1, 2)?
                        .reshape((b, t, g.q_width()))?
                };

            let attn = layer.wo.forward(&attn)?;
            let attn = match &layer.attn_scale {
                Some(s) => attn.broadcast_mul(s)?,
                None => attn,
            };
            h = (h + attn)?;

            let normed = rms_norm(&h, &layer.ffn_norm, g.eps)?;
            let tail = tts_nn::fused::swiglu_mul(
                &layer.gate.forward(&normed)?,
                &layer.up.forward(&normed)?,
            )?;
            let mlp = layer.down.forward(&tail)?;
            let mlp = match &layer.mlp_scale {
                Some(s) => mlp.broadcast_mul(s)?,
                None => mlp,
            };
            h = (h + mlp)?;
        }
        state.width = start + t;
        rms_norm(&h, &self.norm, g.eps)
    }
}
