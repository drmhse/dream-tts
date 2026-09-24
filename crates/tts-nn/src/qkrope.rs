//! QK-norm, rope and the KV cache write for one attention layer, in one dispatch.
//!
//! The composed form was two norms, two ropes, two casts and two `slice_set`s per layer per
//! step. The caches are written in place, as `slice_set` writes them. See `qk_rope_f32` in
//! [`crate::mtl`].

use anyhow::Result;
use candle_core::{CpuStorage, CustomOp1, DType, Layout, Shape, Tensor};

/// The shapes the kernel takes: Metal, f32 activations, an f16 cache, head_dim 128.
pub fn eligible(qkv: &Tensor, k_cache: &Tensor, head_dim: usize) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TTS_NN_QK_ROPE").as_deref() != Ok("0"))
        && qkv.device().is_metal()
        && qkv.dtype() == DType::F32
        && k_cache.dtype() == DType::F16
        && head_dim == 128
}

/// `qkv` `[b, t, (heads + 2 * n_kv) * 128]` in; q `[b, heads, t, 128]` out, and k and v written
/// to the caches `[b, n_kv, capacity, 128]` at positions `start..start + t`. With `f32_kv` the
/// new k and v also come back as f32 `[b, n_kv, t, 128]`, for prefill's matmul.
///
/// `norms` is `[2, 128]`, q's weight then k's; `cos` and `sin` are `[positions, 64]`.
#[allow(clippy::too_many_arguments)]
pub fn apply(
    qkv: &Tensor,
    norms: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    heads: usize,
    start: usize,
    eps: f32,
    f32_kv: bool,
) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
    let (b, t, _) = qkv.dims3()?;
    let (_, n_kv, capacity, _) = k_cache.dims4()?;
    anyhow::ensure!(
        start + t <= capacity,
        "{} positions exceed {capacity}",
        start + t
    );
    for c in [k_cache, v_cache] {
        anyhow::ensure!(c.is_contiguous(), "qk_rope: cache must be contiguous");
    }
    let side = if f32_kv {
        let z = || Tensor::zeros((b, n_kv, t, 128), DType::F32, qkv.device());
        Some((z()?, z()?))
    } else {
        None
    };
    let op = QkRope {
        norms: norms.contiguous()?,
        cos: cos.contiguous()?,
        sin: sin.contiguous()?,
        k_cache: k_cache.clone(),
        v_cache: v_cache.clone(),
        side: side.clone(),
        heads,
        n_kv,
        start,
        capacity,
        eps,
    };
    let q = qkv.contiguous()?.apply_op1_no_bwd(&op)?;
    Ok((q, side))
}

struct QkRope {
    norms: Tensor,
    cos: Tensor,
    sin: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    side: Option<(Tensor, Tensor)>,
    heads: usize,
    n_kv: usize,
    start: usize,
    capacity: usize,
    eps: f32,
}

impl CustomOp1 for QkRope {
    fn name(&self) -> &'static str {
        "qk_rope"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("qk_rope: Metal only")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
    ) -> candle_core::Result<(candle_core::MetalStorage, Shape)> {
        use crate::mtl;
        use candle_core::backend::BackendStorage;
        use candle_core::{MetalStorage, Storage};
        use objc2_metal::{MTLResourceUsage, MTLSize};

        // Buffer and byte offset of a tensor the op reads or writes beside its input.
        let buf = |t: &Tensor| -> candle_core::Result<_> {
            let (s, l) = t.storage_and_layout();
            match &*s {
                Storage::Metal(m) => Ok((
                    m.buffer().clone(),
                    l.start_offset() * t.dtype().size_in_bytes(),
                )),
                _ => candle_core::bail!("qk_rope: operand must be on the device"),
            }
        };

        let (b, t, _) = l1.shape().dims3()?;
        let device = s1.device();
        let p = mtl::pipeline(device, "qk_rope_f32")?;
        let count = b * self.heads * t * 128;
        let q = device.new_buffer(count, DType::F32, "qk_rope")?;

        let norms = buf(&self.norms)?;
        let cos = buf(&self.cos)?;
        let sin = buf(&self.sin)?;
        let kc = buf(&self.k_cache)?;
        let vc = buf(&self.v_cache)?;
        let (kf, vf) = match &self.side {
            Some((k, v)) => (Some(buf(k)?), Some(buf(v)?)),
            None => (None, None),
        };

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::qk_rope");
        encoder.set_compute_pipeline_state(&p);
        encoder.set_buffer(0, Some(s1.buffer()), l1.start_offset() * 4);
        encoder.set_buffer(1, Some(&norms.0), norms.1);
        encoder.set_buffer(2, Some(&cos.0), cos.1);
        encoder.set_buffer(3, Some(&sin.0), sin.1);
        encoder.set_buffer(4, Some(q.as_ref()), 0);
        encoder.set_buffer(5, Some(&kc.0), kc.1);
        encoder.set_buffer(6, Some(&vc.0), vc.1);
        // Unused without f32 k/v; any valid buffer satisfies the binding.
        let (kfb, kfo) = kf.as_ref().map(|(b, o)| (b, *o)).unwrap_or((&kc.0, kc.1));
        let (vfb, vfo) = vf.as_ref().map(|(b, o)| (b, *o)).unwrap_or((&vc.0, vc.1));
        encoder.set_buffer(7, Some(kfb), kfo);
        encoder.set_buffer(8, Some(vfb), vfo);
        encoder.set_bytes(9, &(t as u32));
        encoder.set_bytes(10, &(self.heads as u32));
        encoder.set_bytes(11, &(self.n_kv as u32));
        encoder.set_bytes(12, &(self.start as u32));
        encoder.set_bytes(13, &(self.capacity as u32));
        encoder.set_bytes(14, &(kf.is_some() as u32));
        encoder.set_bytes(15, &self.eps);
        encoder.use_resource(s1.buffer(), MTLResourceUsage::Read);
        for r in [&norms.0, &cos.0, &sin.0] {
            encoder.use_resource(r, MTLResourceUsage::Read);
        }
        encoder.use_resource(q.as_ref(), MTLResourceUsage::Write);
        for r in [&kc.0, &vc.0, kfb, vfb] {
            encoder.use_resource(r, MTLResourceUsage::Write);
        }
        encoder.dispatch_thread_groups(
            MTLSize {
                width: b * t,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: (self.heads + self.n_kv) * 32,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(q, device.clone(), count, DType::F32),
            (b, self.heads, t, 128).into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Against candle's composed norm, rope and cast, on a lane window of a larger cache so
    /// the offsets are exercised too.
    #[test]
    fn matches_composed_ops() -> anyhow::Result<()> {
        #[cfg(feature = "metal")]
        let _gpu = crate::gpu_guard();
        let Some(d) = crate::usable_metal() else {
            return Ok(());
        };
        let (heads, n_kv, cap, eps) = (16usize, 8usize, 64usize, 1e-6f32);
        let (cos, sin) = crate::rope_table_f32(cap, 128, 1e6, &d)?;
        let norms = Tensor::rand(0.5f32, 1.5, (2, 128), &d)?;
        for (b, t, start, off) in [
            (8usize, 1usize, 37usize, 2usize),
            (4, 5, 0, 1),
            (4, 7, 10, 0),
        ] {
            let qkv = Tensor::randn(0f32, 1., (b, t, (heads + 2 * n_kv) * 128), &d)?;
            let full = || Tensor::zeros((b + 3, n_kv, cap, 128), DType::F16, &d);
            let (kc, vc) = (full()?, full()?);
            let (kw, vw) = (kc.narrow(0, off, b)?, vc.narrow(0, off, b)?);
            let (q, side) = apply(&qkv, &norms, &cos, &sin, &kw, &vw, heads, start, eps, true)?;

            let split = |from: usize, n: usize| -> anyhow::Result<Tensor> {
                Ok(qkv
                    .narrow(2, from * 128, n * 128)?
                    .reshape((b, t, n, 128))?)
            };
            let c = cos.narrow(0, start, t)?;
            let s = sin.narrow(0, start, t)?;
            let rope = |x: &Tensor, w: &Tensor| -> anyhow::Result<Tensor> {
                let x = candle_nn::ops::rms_norm(&x.contiguous()?, w, eps)?;
                Ok(candle_nn::rotary_emb::rope(
                    &x.transpose(1, 2)?.contiguous()?,
                    &c,
                    &s,
                )?)
            };
            let want_q = rope(&split(0, heads)?, &norms.get(0)?)?;
            let want_k = rope(&split(heads, n_kv)?, &norms.get(1)?)?;
            let want_v = split(heads + n_kv, n_kv)?.transpose(1, 2)?.contiguous()?;

            let close = |got: &Tensor, want: &Tensor, tol: f32, what: &str| -> anyhow::Result<()> {
                let (abs, rel) =
                    crate::abs_and_rel(&got.to_dtype(DType::F32)?, &want.to_dtype(DType::F32)?)?;
                assert!(
                    rel < tol,
                    "b={b} t={t} start={start} {what}: abs {abs:.2e} rel {rel:.2e}"
                );
                Ok(())
            };
            close(&q, &want_q, 1e-5, "q")?;
            let (fk, fv) = side.expect("f32 k and v");
            close(&fk, &want_k, 1e-5, "f32 k")?;
            close(&fv, &want_v, 1e-6, "f32 v")?;
            close(&kw.narrow(2, start, t)?, &want_k, 1e-3, "cached k")?;
            close(&vw.narrow(2, start, t)?, &want_v, 1e-3, "cached v")?;
            // Nothing outside the window and the positions was touched.
            let outside = kc
                .narrow(0, 0, off)?
                .abs()?
                .sum_all()?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()?
                + kc.narrow(0, off + b, 3 - off)?
                    .abs()?
                    .sum_all()?
                    .to_dtype(DType::F32)?
                    .to_scalar::<f32>()?;
            assert_eq!(outside, 0.0, "writes escaped the lane window");
        }
        Ok(())
    }
}
