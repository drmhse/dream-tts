//! The 12 Hz RVQ decoder: 16 codes per frame to 24 kHz waveform.
//!
//! ```text
//! codes [T,16] -> split RVQ dequant -> pre_conv -> 8L transformer (window 72)
//!              -> 2 x [transconv, ConvNeXt] -> conv + 4 decoder blocks -> SnakeBeta -> conv
//! ```
//!
//! 1920 samples per frame. Decoded in chunks with left context (see [`Codec::decode`]).

use crate::cfg::codec as k;
use crate::qwen3::{Geometry, Stack};
use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use tts_nn::fused::snake_beta_nlc;
use tts_nn::{layer_norm, matmul_2d, nlc, Linear, Weights};

/// Frames-major `[T][16]` to codebook-major `[16][T]`: the quantizers index one at a time.
fn by_codebook(frames: &[Vec<u32>]) -> Result<Vec<Vec<u32>>> {
    let mut out: Vec<Vec<u32>> = (0..k::QUANTIZERS)
        .map(|_| Vec::with_capacity(frames.len()))
        .collect();
    for f in frames {
        if f.len() != k::QUANTIZERS {
            bail!("frame has {} codes, expected {}", f.len(), k::QUANTIZERS);
        }
        for (q, &c) in f.iter().enumerate() {
            out[q].push(c);
        }
    }
    Ok(out)
}

/// Dtype of the waveform stack: `head_conv`, the four decoder blocks, and the output conv.
///
/// **f32, and f16 is not available here** — it was 1.17x faster and missed the fixtures by 6x
/// (`codec.wav` rel 3.1e-2 against a 5.0e-3 tolerance). Not range: the waveform sits well inside
/// f16. `SnakeBeta` is what does it. It computes `sin(exp(alpha) * x)`, so any error in `x` is
/// amplified by `exp(alpha)` before the sine, and there are 24 of them in series down the
/// upsampling stack. Storage rounding that would random-walk to ~5e-3 gets multiplied instead.
///
/// Left as a constant because that is the whole switch, and because the plumbing costs nothing:
/// see `docs/reference.md#what-did-not-work`.
const WAV_DTYPE: DType = DType::F32;

/// Residual units use dilations 1, 3, 9 — hardcoded in the reference's `DecoderBlock`, not
/// taken from any config field.
const DILATIONS: [usize; 3] = [1, 3, 9];

/// `SnakeBeta`: `x + 1/(exp(beta) + 1e-9) * sin^2(x * exp(alpha))`.
///
/// Both parameters are stored as logs and exponentiated at use, so the folded tensors are
/// `exp(alpha)` and `1/(exp(beta) + 1e-9)`. Folding at load rather than per call.
///
/// Kept flat `[C]` because [`snake_beta`]'s kernel indexes the channel directly.
struct SnakeBeta {
    alpha: Tensor,
    beta_recip: Tensor,
}

impl SnakeBeta {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        let alpha = w.get(&format!("{prefix}.alpha"))?.flatten_all()?;
        let beta = w.get(&format!("{prefix}.beta"))?.exp()?;
        Ok(Self {
            alpha: alpha.exp()?,
            beta_recip: (beta + 1e-9)?.recip()?.flatten_all()?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(snake_beta_nlc(x, &self.alpha, &self.beta_recip)?)
    }
}

/// MPSGraph where it measures faster, `nlc`'s fused kernel elsewhere; bit-identical in f32.
fn causal_conv(x: &Tensor, w: &Tensor, b: &Tensor, dilation: usize) -> Result<Tensor> {
    if tts_nn::mpsnlc::eligible(x, w) {
        tts_nn::mpsnlc::causal_conv1d(x, w, b, dilation)
    } else {
        nlc::causal_conv1d(x, w, Some(b), dilation)
    }
}

/// A causal conv with its own weight and bias. Stride is always 1 here; the reference's
/// extra-padding arithmetic collapses to a left pad of `(k-1)*dilation` at stride 1.
///
/// Channels-last, so each tap is a contiguous slice and the conv is `k` accumulating GEMMs with
/// no im2col matrix at all — see [`tts_nn::nlc`].
struct Conv {
    w: Tensor,
    b: Tensor,
    dilation: usize,
}

impl Conv {
    fn load(w: &Weights, prefix: &str, dilation: usize) -> Result<Self> {
        Self::load_as(w, prefix, dilation, DType::F32)
    }

    /// `dt` is the dtype of the *activations* this conv will see, so the weights match and the
    /// GEMM needs no cast. See [`Codec::forward`] for why the waveform stack is f16.
    fn load_as(w: &Weights, prefix: &str, dilation: usize, dt: DType) -> Result<Self> {
        Ok(Self {
            // Rearranged on the host so only the result is uploaded; see `Weights::get`.
            w: nlc::tap_weight(
                &w.cpu(&format!("{prefix}.conv.weight"))?
                    .to_dtype(DType::F32)?,
            )?
            .to_dtype(dt)?
            .to_device(w.device())?,
            b: w.get(&format!("{prefix}.conv.bias"))?.to_dtype(dt)?,
            dilation,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        causal_conv(x, &self.w, &self.b, self.dilation)
    }
}

/// Transposed conv as one GEMM per tap group over the `stride` output phases.
///
/// `kernel / stride` taps reach any output — 1 for the ratio-2 upsamplers, 2 for the decoder
/// blocks — so candle's `conv_transpose1d` is not needed. See [`tts_nn::nlc::transpose_weights`].
struct TransConv {
    /// Tap-major and reversed, so this runs through [`nlc::causal_conv1d`] — see
    /// [`nlc::transpose_tap_weight`].
    w: Tensor,
    b: Tensor,
    stride: usize,
}

impl TransConv {
    fn load(w: &Weights, prefix: &str, stride: usize) -> Result<Self> {
        Self::load_as(w, prefix, stride, DType::F32)
    }

    fn load_as(w: &Weights, prefix: &str, stride: usize, dt: DType) -> Result<Self> {
        let raw = w
            .cpu(&format!("{prefix}.conv.weight"))?
            .to_dtype(DType::F32)?;
        let tapped = nlc::transpose_tap_weight(&raw, stride)?
            .to_dtype(dt)?
            .to_device(w.device())?;
        Ok(Self {
            w: tapped,
            b: nlc::transpose_bias(&w.get(&format!("{prefix}.conv.bias"))?, stride)?
                .to_dtype(dt)?,
            stride,
        })
    }
    /// `[b, L, C_in] -> [b, L * stride, C_out]`. The conv emits the `stride` output phases
    /// interleaved on the channel axis, which is the layout the reshape wants — no copy.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, len, _) = x.dims3()?;
        let y = causal_conv(x, &self.w, &self.b, 1)?;
        let wide = y.dim(2)?;
        Ok(y.reshape((b, len * self.stride, wide / self.stride))?)
    }
}

/// Depthwise k=7 causal conv, LayerNorm, 4x pointwise MLP with GELU, `gamma` scale, residual.
struct ConvNext {
    dw_w: Tensor,
    dw_b: Tensor,
    norm_w: Tensor,
    norm_b: Tensor,
    pw1: Linear,
    pw2: Linear,
    gamma: Tensor,
}

impl ConvNext {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            dw_w: w.get(&format!("{prefix}.dwconv.conv.weight"))?,
            dw_b: w.get(&format!("{prefix}.dwconv.conv.bias"))?,
            norm_w: w.get(&format!("{prefix}.norm.weight"))?,
            norm_b: w.get(&format!("{prefix}.norm.bias"))?,
            pw1: Linear::load(w, &format!("{prefix}.pwconv1"), true)?,
            pw2: Linear::load(w, &format!("{prefix}.pwconv2"), true)?,
            gamma: w.get(&format!("{prefix}.gamma"))?,
        })
    }

    /// `[b, t, c] -> [b, t, c]`. Channels-last removes all four transposes this used to need:
    /// LayerNorm and both pointwise convs already want the channel on the last axis.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = nlc::depthwise(x, &self.dw_w, Some(&self.dw_b))?;
        let h = layer_norm(&h, &self.norm_w, &self.norm_b, 1e-6)?;
        let h = self
            .pw2
            .forward(&tts_nn::gelu_exact(&self.pw1.forward(&h)?)?)?;
        let h = h.broadcast_mul(&self.gamma)?;
        Ok((x + h)?)
    }
}

/// SnakeBeta, k=7 dilated conv, SnakeBeta, k=1 conv, residual.
struct ResidualUnit {
    act1: SnakeBeta,
    conv1: Conv,
    act2: SnakeBeta,
    conv2: Conv,
}

impl ResidualUnit {
    fn load(w: &Weights, prefix: &str, dilation: usize, dt: DType) -> Result<Self> {
        Ok(Self {
            act1: SnakeBeta::load(w, &format!("{prefix}.act1"))?,
            conv1: Conv::load_as(w, &format!("{prefix}.conv1"), dilation, dt)?,
            act2: SnakeBeta::load(w, &format!("{prefix}.act2"))?,
            conv2: Conv::load_as(w, &format!("{prefix}.conv2"), 1, dt)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(&self.act1.forward(x)?)?;
        let (c, a) = (&self.conv2, &self.act2);
        // The kernel reloads its input once per 32 output channels, so the fused sin runs
        // `cout / 32` times per element: a 9.5% win on the residual units up to 384 channels,
        // nothing at 768.
        if h.dim(2)? <= 384 {
            let snake = (&a.alpha, &a.beta_recip);
            if let Some(y) = nlc::snake_conv_residual(&h, &c.w, &c.b, c.dilation, snake, x)? {
                return Ok(y);
            }
        }
        let h = c.forward(&a.forward(&h)?)?;
        Ok((x + h)?)
    }
}

/// SnakeBeta, upsampling transposed conv, then three residual units.
struct DecoderBlock {
    act: SnakeBeta,
    up: TransConv,
    units: Vec<ResidualUnit>,
}

impl DecoderBlock {
    fn load(w: &Weights, prefix: &str, rate: usize, dt: DType) -> Result<Self> {
        // `block` is an nn.ModuleList: index 0 the activation, 1 the transposed conv, 2..4
        // the residual units.
        let act = SnakeBeta::load(w, &format!("{prefix}.block.0"))?;
        let up = TransConv::load_as(w, &format!("{prefix}.block.1"), rate, dt)?;
        let units = DILATIONS
            .iter()
            .enumerate()
            .map(|(i, &d)| ResidualUnit::load(w, &format!("{prefix}.block.{}", i + 2), d, dt))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { act, up, units })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.up.forward(&self.act.forward(x)?)?;
        for u in &self.units {
            h = u.forward(&h)?;
        }
        Ok(h)
    }
}

/// One `ResidualVectorQuantizer` stack: 1x1 conv in, per-layer codebook lookups summed,
/// 1x1 conv out.
struct Rvq {
    input_proj: Tensor,
    output_proj: Tensor,
    /// Folded `embedding_sum / cluster_usage.clamp(1e-5)` — trap 4.
    codebooks: Vec<Tensor>,
}

impl Rvq {
    fn load(w: &Weights, prefix: &str, layers: usize) -> Result<Self> {
        let mut codebooks = Vec::with_capacity(layers);
        for i in 0..layers {
            let p = format!("{prefix}.vq.layers.{i}._codebook");
            let sum = w.get(&format!("{p}.embedding_sum"))?;
            let usage = w.get(&format!("{p}.cluster_usage"))?;
            let usage = usage.clamp(k::CLUSTER_USAGE_EPSILON as f32, f32::INFINITY)?;
            codebooks.push(sum.broadcast_div(&usage.unsqueeze(1)?)?);
        }
        Ok(Self {
            input_proj: w.get(&format!("{prefix}.input_proj.weight"))?,
            // `[out, in, 1]` -> `[in, out]`, the orientation a channels-last GEMM wants.
            output_proj: w
                .get(&format!("{prefix}.output_proj.weight"))?
                .squeeze(2)?
                .t()?
                .contiguous()?,
            codebooks,
        })
    }

    /// `codes[layer][frame]` -> `[1, T, CODEBOOK_DIM]`, channels-last.
    ///
    /// The lookups are already `[T, inner]`, so staying channels-last removes a transpose per
    /// quantizer and turns the 1x1 output conv into a plain GEMM.
    fn decode(&self, codes: &[Vec<u32>], device: &Device) -> Result<Tensor> {
        let t = codes[0].len();
        let mut sum: Option<Tensor> = None;
        for (i, layer) in codes.iter().enumerate() {
            let idx = Tensor::from_vec(layer.clone(), t, device)?;
            let q = self.codebooks[i].index_select(&idx, 0)?;
            sum = Some(match sum {
                None => q,
                Some(s) => (s + q)?,
            });
        }
        let q = sum.expect("at least one quantizer layer");
        Ok(matmul_2d(&q, &self.output_proj)?.reshape((1, t, ()))?)
    }

    /// The input projection is only needed on the encode path; kept so the tensor is
    /// accounted for rather than silently unused.
    #[allow(dead_code)]
    fn input_width(&self) -> Result<usize> {
        Ok(self.input_proj.dim(1)?)
    }
}

pub struct Codec {
    semantic: Rvq,
    acoustic: Rvq,
    pre_conv: Conv,
    pre_in: Linear,
    pre_out: Linear,
    pre_tf: Stack,
    upsample: Vec<(TransConv, ConvNext)>,
    head_conv: Conv,
    blocks: Vec<DecoderBlock>,
    out_act: SnakeBeta,
    out_conv: Conv,
    device: Device,
}

impl Codec {
    pub fn load(path: &str, device: &Device) -> Result<Self> {
        let w = Weights::load(path, device)?;
        let geo = Geometry {
            dim: k::TF_DIM,
            layers: k::TF_LAYERS,
            heads: k::TF_HEADS,
            n_kv: k::TF_N_KV,
            head_dim: k::TF_HEAD_DIM,
            ffn: k::TF_FFN,
            eps: k::TF_NORM_EPS,
            rope_base: k::TF_ROPE_BASE,
            qk_norm: false,
            layer_scale: true,
            window: Some(k::TF_SLIDING_WINDOW),
        };
        // Chunks are bounded, so the cache only needs one chunk plus its left context.
        let capacity = (k::CHUNK_FRAMES + k::CHUNK_LEFT_CONTEXT) * 4 + 8;

        let upsample = k::UPSAMPLING_RATIOS
            .iter()
            .enumerate()
            .map(|(i, &r)| {
                Ok((
                    TransConv::load(&w, &format!("decoder.upsample.{i}.0"), r)?,
                    ConvNext::load(&w, &format!("decoder.upsample.{i}.1"))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        // `decoder` is an nn.ModuleList: 0 the head conv, 1..4 the blocks, 5 SnakeBeta,
        // 6 the output conv.
        let blocks = k::UPSAMPLE_RATES
            .iter()
            .enumerate()
            .map(|(i, &r)| {
                DecoderBlock::load(&w, &format!("decoder.decoder.{}", i + 1), r, WAV_DTYPE)
            })
            .collect::<Result<Vec<_>>>()?;
        let last = k::UPSAMPLE_RATES.len() + 1;

        Ok(Self {
            semantic: Rvq::load(&w, "decoder.quantizer.rvq_first", k::SEMANTIC_QUANTIZERS)?,
            acoustic: Rvq::load(&w, "decoder.quantizer.rvq_rest", k::ACOUSTIC_QUANTIZERS)?,
            pre_conv: Conv::load(&w, "decoder.pre_conv", 1)?,
            pre_in: Linear::load(&w, "decoder.pre_transformer.input_proj", true)?,
            pre_out: Linear::load(&w, "decoder.pre_transformer.output_proj", true)?,
            pre_tf: Stack::load(
                &w,
                "decoder.pre_transformer.",
                geo,
                tts_nn::Weight::F32,
                capacity,
                device,
            )?,
            upsample,
            head_conv: Conv::load_as(&w, "decoder.decoder.0", 1, WAV_DTYPE)?,
            blocks,
            out_act: SnakeBeta::load(&w, &format!("decoder.decoder.{last}"))?,
            out_conv: Conv::load_as(&w, &format!("decoder.decoder.{}", last + 1), 1, WAV_DTYPE)?,
            device: device.clone(),
        })
    }

    /// One forward over all given frames. `frames[i]` is a 16-code frame.
    ///
    /// Channels-last end to end: the RVQ lookups produce `[b, T, C]`, the pre-transformer already
    /// wants that, and the whole conv stack now works in it. Only the final waveform is
    /// transposed, and it has one channel so that is a reshape.
    pub fn forward(&self, frames: &[Vec<u32>]) -> Result<Tensor> {
        if frames.is_empty() {
            bail!("no frames to decode");
        }
        let by_book = by_codebook(frames)?;

        // Split RVQ: semantic then acoustic, summed.
        let q = self
            .semantic
            .decode(&by_book[..k::SEMANTIC_QUANTIZERS], &self.device)?;
        let q = (q + self
            .acoustic
            .decode(&by_book[k::SEMANTIC_QUANTIZERS..], &self.device)?)?;

        let h = self.pre_conv.forward(&q)?;
        let mut state = self.pre_tf.new_state(1)?;
        let h = self.pre_tf.forward(&self.pre_in.forward(&h)?, &mut state)?;
        let mut h = self.pre_out.forward(&h)?;

        for (up, next) in &self.upsample {
            h = next.forward(&up.forward(&h)?)?;
        }

        let mut wav = self.head_conv.forward(&h.to_dtype(WAV_DTYPE)?)?;
        for b in &self.blocks {
            wav = b.forward(&wav)?;
        }
        let wav = self.out_conv.forward(&self.out_act.forward(&wav)?)?;
        // `[1, L, 1]` -> `[1, 1, L]`: one channel, so this is a reshape not a copy.
        let len = wav.dim(1)?;
        Ok(wav
            .reshape((1, 1, len))?
            .to_dtype(DType::F32)?
            .clamp(-k::CLAMP, k::CLAMP)?)
    }

    /// The two stage boundaries the gate checks: the split-RVQ output and the pre-transformer's
    /// output after `output_proj`.
    ///
    /// `quantized` is transposed back to `[1, C, T]` because that is the layout the reference
    /// dumps; everything internal is channels-last now.
    pub fn trace(&self, frames: &[Vec<u32>]) -> Result<(Tensor, Tensor)> {
        let by_book = by_codebook(frames)?;
        let q = self
            .semantic
            .decode(&by_book[..k::SEMANTIC_QUANTIZERS], &self.device)?;
        let quantized = (q + self
            .acoustic
            .decode(&by_book[k::SEMANTIC_QUANTIZERS..], &self.device)?)?;
        let h = self.pre_conv.forward(&quantized)?;
        let mut state = self.pre_tf.new_state(1)?;
        let h = self.pre_tf.forward(&self.pre_in.forward(&h)?, &mut state)?;
        Ok((
            quantized.transpose(1, 2)?.contiguous()?,
            self.pre_out.forward(&h)?,
        ))
    }

    /// Per-stage cost of one chunk, each timer synchronised at both ends.
    ///
    /// Same order as [`Self::forward`]; the caller divides by `reps`. Only for the
    /// `qwen3tts-codecsplit` bin — `forward` stays free of instrumentation.
    pub fn profile(&self, frames: &[Vec<u32>], reps: usize) -> Result<Vec<(&'static str, f64)>> {
        use std::time::Instant;
        let mut acc = vec![0f64; 9];
        for _ in 0..reps {
            let mut at = |i: usize, t: Instant| -> Result<()> {
                self.device.synchronize()?;
                acc[i] += t.elapsed().as_secs_f64();
                Ok(())
            };
            self.device.synchronize()?;

            let t = Instant::now();
            let by_book = by_codebook(frames)?;
            let q = self
                .semantic
                .decode(&by_book[..k::SEMANTIC_QUANTIZERS], &self.device)?;
            let q = (q + self
                .acoustic
                .decode(&by_book[k::SEMANTIC_QUANTIZERS..], &self.device)?)?;
            at(0, t)?;

            let t = Instant::now();
            let h = self.pre_conv.forward(&q)?;
            at(1, t)?;

            let t = Instant::now();
            let mut state = self.pre_tf.new_state(1)?;
            let h = self.pre_tf.forward(&self.pre_in.forward(&h)?, &mut state)?;
            let mut h = self.pre_out.forward(&h)?;
            at(2, t)?;

            let t = Instant::now();
            for (up, next) in &self.upsample {
                h = next.forward(&up.forward(&h)?)?;
            }
            at(3, t)?;

            let t = Instant::now();
            let mut wav = self.head_conv.forward(&h.to_dtype(WAV_DTYPE)?)?;
            at(4, t)?;

            for b in &self.blocks {
                let t = Instant::now();
                let a = b.act.forward(&wav)?;
                at(5, t)?;
                let t = Instant::now();
                let mut h = b.up.forward(&a)?;
                at(7, t)?;
                let t = Instant::now();
                for u in &b.units {
                    h = u.forward(&h)?;
                }
                at(8, t)?;
                wav = h;
            }

            let t = Instant::now();
            let wav = self.out_conv.forward(&self.out_act.forward(&wav)?)?;
            let _ = wav.to_dtype(DType::F32)?.clamp(-k::CLAMP, k::CLAMP)?;
            at(6, t)?;
        }
        let names = [
            "rvq dequant",
            "pre_conv",
            "pre_transformer",
            "upsample+convnext",
            "head_conv",
            "block snake",
            "out snake+conv",
            "block transconv",
            "block residual units",
        ];
        Ok(names
            .iter()
            .zip(acc)
            .map(|(n, s)| (*n, s / reps as f64))
            .collect())
    }

    /// Frames per decode chunk, overridable for the memory/throughput sweep.
    ///
    /// The peak is a *chunk*, not a document: one 300-frame chunk costs 6.50 GB of activations
    /// where the whole codec's weights are a fraction of that, which is why footprint is flat
    /// across document length once the first full chunk is decoded. Free to vary — the gate
    /// checks `decode` against the *unchunked* reference (`codec.wav_chunked` reports the same
    /// 4.12e-5 as `codec.wav`), so chunking is validated rather than pinned.
    fn chunk_frames() -> usize {
        static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *N.get_or_init(|| {
            std::env::var("QWEN3TTS_CHUNK_FRAMES")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|n| *n > 0)
                .unwrap_or(k::CHUNK_FRAMES)
        })
    }

    /// Decode in chunks with left context, discarding the context's audio.
    ///
    /// Two reasons this is the default path rather than one big `forward`: memory stays
    /// bounded, and candle's Metal device pools buffers by size, so a decoder called many
    /// times recycles where one called once pays every allocation cold.
    ///
    /// **Every forward is the same length, and the padding goes on the right.** The natural
    /// loop runs two lengths — the first chunk has no history to spend, so 300 frames, and
    /// every later one 325 — and candle's pool keeps a full set of activation buffers per
    /// length. One 300-frame chunk is 6.50 GB of those, so the second length cost about that
    /// again. Extending each window *rightwards* to a fixed span fixes it without touching the
    /// audio: this decoder is causal — left-padded convs, a causal transposed conv, a masked
    /// sliding-window transformer — so frames after the ones being kept cannot affect them.
    /// Extending *leftwards* would not be free, and moving the seams is not either: the
    /// reference's own chunked output differs from its unchunked output (see the two
    /// `codec.long.wav*` gate rows), so seam positions are part of matching upstream.
    ///
    /// A document shorter than one span stays a single forward at its own length; padding it
    /// out would multiply the codec's work on a one-sentence render for nothing.
    pub fn decode(&self, frames: &[Vec<u32>]) -> Result<Vec<f32>> {
        let n = frames.len();
        let chunk = Self::chunk_frames();
        let span = chunk + k::CHUNK_LEFT_CONTEXT;
        let mut out: Vec<f32> = Vec::with_capacity(n * crate::cfg::SAMPLES_PER_FRAME);
        let mut start = 0usize;
        while start < n {
            let end = (start + chunk).min(n);
            let from = start.saturating_sub(k::CHUNK_LEFT_CONTEXT);
            let to = (from + span).min(n);
            let mut window: Vec<Vec<u32>> = frames[from..to].to_vec();
            if n >= span {
                // Repeat the last frame rather than invent codes; it is discarded either way.
                let filler = window.last().cloned().expect("non-empty window");
                window.resize(span, filler);
            }
            let wav = self.forward(&window)?;
            let drop = (start - from) * crate::cfg::SAMPLES_PER_FRAME;
            let keep = ((end - start) * crate::cfg::SAMPLES_PER_FRAME).min(wav.dim(2)? - drop);
            out.extend(wav.narrow(2, drop, keep)?.flatten_all()?.to_vec1::<f32>()?);
            start = end;
        }
        Ok(out)
    }
}
