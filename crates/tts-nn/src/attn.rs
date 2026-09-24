//! Decode attention that reads the KV cache in place.
//!
//! `narrow(2, 0, span)` of a `[b, n_kv, capacity, head_dim]` cache is not contiguous, so
//! candle copies the span twice per layer per step — once transposed for the scores, once for
//! the value product. ~112 MB/step on the talker at span 250, pure overhead.
//!
//! Two kernels rather than one fused flash-style pass: candle's `softmax_last_dim` is already
//! a single kernel, and splitting keeps both of these free of threadgroup memory and cross-lane
//! reductions. See `docs/reference.md#what-did-not-work` for the cheaper fix that failed.

#[cfg(feature = "metal")]
use crate::mtl;
use anyhow::Result as AnyResult;
#[cfg(feature = "metal")]
use candle_core::DType;
use candle_core::{CpuStorage, Layout, Shape, Tensor};

/// `q` is `[b, n_kv, gqa, head_dim]`, the cache `[b, n_kv, capacity, head_dim]`.
///
/// Positions `>= span`, and `< window_start` when a sliding window applies, are masked. Any
/// scale must already be folded into `q`.
pub fn decode_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    span: usize,
    window_start: usize,
) -> AnyResult<Tensor> {
    let (b, n_kv, gqa, head_dim) = q.dims4()?;
    let capacity = k.dim(2)?;
    anyhow::ensure!(span <= capacity, "span {span} exceeds capacity {capacity}");

    #[cfg(feature = "metal")]
    if fused_eligible(q, k, head_dim, gqa) {
        return Ok(q.contiguous()?.apply_op3_no_bwd(
            &k.contiguous()?,
            &v.contiguous()?,
            &Fused {
                b,
                n_kv,
                gqa,
                capacity,
                span,
                window_start,
            },
        )?);
    }

    let scores = q.contiguous()?.apply_op2_no_bwd(
        &k.contiguous()?,
        &Scores {
            b,
            n_kv,
            gqa,
            head_dim,
            capacity,
            span,
            window_start,
        },
    )?;
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    Ok(probs.contiguous()?.apply_op2_no_bwd(
        &v.contiguous()?,
        &Weighted {
            b,
            n_kv,
            gqa,
            head_dim,
            capacity,
            span,
        },
    )?)
}

struct Scores {
    b: usize,
    n_kv: usize,
    gqa: usize,
    head_dim: usize,
    capacity: usize,
    span: usize,
    window_start: usize,
}

impl candle_core::CustomOp2 for Scores {
    fn name(&self) -> &'static str {
        "decode_scores"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let kf: Vec<f32>;
        let (q, k) = match (s1, s2) {
            (CpuStorage::F32(q), CpuStorage::F32(k)) => (q, &k[..]),
            (CpuStorage::F32(q), CpuStorage::F16(k)) => {
                kf = k.iter().map(|x| x.to_f32()).collect();
                (q, &kf[..])
            }
            _ => candle_core::bail!("decode_scores: cache must be f32 or f16"),
        };
        let (o1, o2) = (l1.start_offset(), l2.start_offset());
        let hd = self.head_dim;
        let mut dst = vec![f32::NEG_INFINITY; self.b * self.n_kv * self.gqa * self.span];
        for bh in 0..self.b * self.n_kv {
            for g in 0..self.gqa {
                let qb = o1 + (bh * self.gqa + g) * hd;
                for p in self.window_start..self.span {
                    let kb = o2 + (bh * self.capacity + p) * hd;
                    let mut acc = 0f32;
                    for d in 0..hd {
                        acc += q[qb + d] * k[kb + d];
                    }
                    dst[(bh * self.gqa + g) * self.span + p] = acc;
                }
            }
        }
        Ok((
            CpuStorage::F32(dst),
            (self.b, self.n_kv, self.gqa, self.span).into(),
        ))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
    ) -> candle_core::Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::MetalStorage;
        use objc2_metal::{MTLResourceUsage, MTLSize};

        if s1.dtype() != DType::F32 {
            candle_core::bail!("decode_scores: query must be f32");
        }
        let kernel = match s2.dtype() {
            DType::F32 => "decode_scores_f32",
            DType::F16 => "decode_scores_f16",
            d => candle_core::bail!("decode_scores: cache must be f32 or f16, got {d:?}"),
        };
        let kv_size = s2.dtype().size_in_bytes();
        let count = self.b * self.n_kv * self.gqa * self.span;
        let device = s1.device();
        let p = mtl::pipeline(device, kernel)?;
        let dst = device.new_buffer(count, DType::F32, "decode_scores")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::decode_scores");
        encoder.set_compute_pipeline_state(&p);
        encoder.set_buffer(0, Some(s1.buffer()), l1.start_offset() * 4);
        encoder.set_buffer(1, Some(s2.buffer()), l2.start_offset() * kv_size);
        encoder.set_buffer(2, Some(dst.as_ref()), 0);
        encoder.set_bytes(3, &(self.span as u32));
        encoder.set_bytes(4, &(self.capacity as u32));
        encoder.set_bytes(5, &(self.head_dim as u32));
        encoder.set_bytes(6, &(self.gqa as u32));
        encoder.set_bytes(7, &(self.window_start as u32));
        encoder.use_resource(s1.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(s2.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        encoder.dispatch_threads(
            MTLSize {
                width: self.span,
                height: self.gqa,
                depth: self.b * self.n_kv,
            },
            MTLSize {
                width: mtl::group_width(&p, self.span),
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), count, DType::F32),
            (self.b, self.n_kv, self.gqa, self.span).into(),
        ))
    }
}

struct Weighted {
    b: usize,
    n_kv: usize,
    gqa: usize,
    head_dim: usize,
    capacity: usize,
    span: usize,
}

impl candle_core::CustomOp2 for Weighted {
    fn name(&self) -> &'static str {
        "decode_weighted"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let vf: Vec<f32>;
        let (probs, v) = match (s1, s2) {
            (CpuStorage::F32(p), CpuStorage::F32(v)) => (p, &v[..]),
            (CpuStorage::F32(p), CpuStorage::F16(v)) => {
                vf = v.iter().map(|x| x.to_f32()).collect();
                (p, &vf[..])
            }
            _ => candle_core::bail!("decode_weighted: cache must be f32 or f16"),
        };
        let (o1, o2) = (l1.start_offset(), l2.start_offset());
        let hd = self.head_dim;
        let mut dst = vec![0f32; self.b * self.n_kv * self.gqa * hd];
        for bh in 0..self.b * self.n_kv {
            for g in 0..self.gqa {
                let pb = o1 + (bh * self.gqa + g) * self.span;
                for p in 0..self.span {
                    let w = probs[pb + p];
                    let vb = o2 + (bh * self.capacity + p) * hd;
                    for d in 0..hd {
                        dst[(bh * self.gqa + g) * hd + d] += w * v[vb + d];
                    }
                }
            }
        }
        Ok((
            CpuStorage::F32(dst),
            (self.b, self.n_kv, self.gqa, hd).into(),
        ))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
    ) -> candle_core::Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::MetalStorage;
        use objc2_metal::{MTLResourceUsage, MTLSize};

        if s1.dtype() != DType::F32 {
            candle_core::bail!("decode_weighted: probs must be f32");
        }
        let kernel = match s2.dtype() {
            DType::F32 => "decode_weighted_f32",
            DType::F16 => "decode_weighted_f16",
            d => candle_core::bail!("decode_weighted: cache must be f32 or f16, got {d:?}"),
        };
        let kv_size = s2.dtype().size_in_bytes();
        let count = self.b * self.n_kv * self.gqa * self.head_dim;
        let device = s1.device();
        let p = mtl::pipeline(device, kernel)?;
        let dst = device.new_buffer(count, DType::F32, "decode_weighted")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::decode_weighted");
        encoder.set_compute_pipeline_state(&p);
        encoder.set_buffer(0, Some(s1.buffer()), l1.start_offset() * 4);
        encoder.set_buffer(1, Some(s2.buffer()), l2.start_offset() * kv_size);
        encoder.set_buffer(2, Some(dst.as_ref()), 0);
        encoder.set_bytes(3, &(self.span as u32));
        encoder.set_bytes(4, &(self.capacity as u32));
        encoder.set_bytes(5, &(self.head_dim as u32));
        encoder.set_bytes(6, &(self.gqa as u32));
        encoder.use_resource(s1.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(s2.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        encoder.dispatch_threads(
            MTLSize {
                width: self.head_dim,
                height: self.gqa,
                depth: self.b * self.n_kv,
            },
            MTLSize {
                width: mtl::group_width(&p, self.head_dim),
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), count, DType::F32),
            (self.b, self.n_kv, self.gqa, self.head_dim).into(),
        ))
    }
}

/// The fused kernel's shapes: an f16 cache on Metal, `head_dim` 128, at most two query heads per
/// kv head. `TTS_NN_FUSED_ATTN=0` keeps the two-kernel path, for A/B.
#[cfg(feature = "metal")]
fn fused_eligible(q: &Tensor, k: &Tensor, head_dim: usize, gqa: usize) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TTS_NN_FUSED_ATTN").as_deref() != Ok("0"))
        && q.device().is_metal()
        && q.dtype() == DType::F32
        && k.dtype() == DType::F16
        && head_dim == 128
        && (1..=2).contains(&gqa)
}

#[cfg(feature = "metal")]
struct Fused {
    b: usize,
    n_kv: usize,
    gqa: usize,
    capacity: usize,
    span: usize,
    window_start: usize,
}

#[cfg(feature = "metal")]
impl candle_core::CustomOp3 for Fused {
    fn name(&self) -> &'static str {
        "decode_attn"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("decode_attn: Metal only")
    }

    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
        s3: &candle_core::MetalStorage,
        l3: &Layout,
    ) -> candle_core::Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::MetalStorage;
        use objc2_metal::{MTLResourceUsage, MTLSize};

        let count = self.b * self.n_kv * self.gqa * 128;
        let device = s1.device();
        let p = mtl::pipeline(device, "decode_attn_f16")?;
        let dst = device.new_buffer(count, DType::F32, "decode_attn")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::decode_attn");
        encoder.set_compute_pipeline_state(&p);
        encoder.set_buffer(0, Some(s1.buffer()), l1.start_offset() * 4);
        encoder.set_buffer(1, Some(s2.buffer()), l2.start_offset() * 2);
        encoder.set_buffer(2, Some(s3.buffer()), l3.start_offset() * 2);
        encoder.set_buffer(3, Some(dst.as_ref()), 0);
        encoder.set_bytes(4, &(self.span as u32));
        encoder.set_bytes(5, &(self.capacity as u32));
        encoder.set_bytes(6, &(self.gqa as u32));
        encoder.set_bytes(7, &(self.window_start as u32));
        encoder.use_resource(s1.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(s2.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(s3.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: self.b * self.n_kv,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), count, DType::F32),
            (self.b, self.n_kv, self.gqa, 128).into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// The fused kernel against the CPU path, which is neither Metal kernel.
    #[test]
    fn fused_matches_cpu() -> anyhow::Result<()> {
        #[cfg(feature = "metal")]
        let _gpu = crate::gpu_guard();
        let Some(d) = crate::usable_metal() else {
            return Ok(());
        };
        for (b, n_kv, gqa, cap, span, wstart) in [
            (48usize, 8usize, 2usize, 400usize, 250usize, 0usize),
            (40, 8, 2, 17, 3, 0),
            (8, 8, 2, 300, 211, 90),
            (4, 8, 1, 64, 33, 0),
        ] {
            let q = (Tensor::randn(0f32, 1., (b, n_kv, gqa, 128), &d)? * 0.1)?;
            let k = Tensor::randn(0f32, 1., (b, n_kv, cap, 128), &d)?.to_dtype(DType::F16)?;
            let v = Tensor::randn(0f32, 1., (b, n_kv, cap, 128), &d)?.to_dtype(DType::F16)?;
            let got = decode_attention(&q, &k, &v, span, wstart)?;
            let cpu = Device::Cpu;
            let want = decode_attention(
                &q.to_device(&cpu)?,
                &k.to_device(&cpu)?,
                &v.to_device(&cpu)?,
                span,
                wstart,
            )?;
            let (abs, rel) = crate::abs_and_rel(&got.to_device(&cpu)?, &want)?;
            assert!(
                rel < 1e-5,
                "b={b} gqa={gqa} span={span} wstart={wstart}: abs {abs:.2e} rel {rel:.2e}"
            );
        }
        Ok(())
    }
}
