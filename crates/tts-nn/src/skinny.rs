//! A GEMM for the shapes an autoregressive decode step has.
//!
//! `m` is the lane count. candle reaches 3.64 TFLOP/s on a 2048-cube and 1.1-2.35 at m = 48,
//! and the talker is 75% of a qwen3tts render, so that gap is the largest single thing left in
//! the engine. See `crates/tts-probe/src/bin/gemm.rs` for the measurement.

use candle_core::{CpuStorage, CustomOp2, DType, Layout, Result, Shape, Tensor};

struct Skinny {
    m: usize,
    k: usize,
    n: usize,
}

impl CustomOp2 for Skinny {
    fn name(&self) -> &'static str {
        "gemm_skinny"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("gemm_skinny: Metal only")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use crate::mtl;
        use candle_core::backend::BackendStorage;
        use candle_core::MetalStorage;
        use objc2_metal::{MTLResourceUsage, MTLSize};

        if !l1.is_contiguous() || !l2.is_contiguous() {
            candle_core::bail!("gemm_skinny: inputs must be contiguous");
        }
        if s1.dtype() != DType::F16 || s2.dtype() != DType::F16 {
            candle_core::bail!("gemm_skinny: only f16");
        }
        let device = s1.device();
        let p = mtl::pipeline(device, "gemm_skinny_f16")?;
        let count = self.m * self.n;
        let dst = device.new_buffer(count, DType::F32, "gemm_skinny")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::gemm_skinny");
        encoder.set_compute_pipeline_state(&p);
        encoder.set_buffer(0, Some(s1.buffer()), l1.start_offset() * 2);
        encoder.set_buffer(1, Some(s2.buffer()), l2.start_offset() * 2);
        encoder.set_buffer(2, Some(dst.as_ref()), 0);
        encoder.set_bytes(3, &(self.m as u32));
        encoder.set_bytes(4, &(self.k as u32));
        encoder.set_bytes(5, &(self.n as u32));
        encoder.use_resource(s1.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(s2.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        encoder.dispatch_threads(
            MTLSize {
                width: (self.n / 64) * 256,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), count, DType::F32),
            (self.m, self.n).into(),
        ))
    }
}

/// Whether these shapes take the kernel: `[m, k] x [k, n]`, f16, on Metal.
pub fn eligible(m: usize, k: usize, n: usize) -> bool {
    m <= MAX_M && m.is_multiple_of(8) && k.is_multiple_of(32) && n.is_multiple_of(64)
}

/// Past this the tile stops being the right shape and candle's own GEMM is closing on its
/// ceiling anyway — prefill runs here, not just decode.
const MAX_M: usize = 48;

/// `[m, k] f16 x [k, n] f16 -> [m, n] f32`.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (m, k) = a.dims2()?;
    let (k2, n) = b.dims2()?;
    if k != k2 {
        candle_core::bail!("gemm_skinny: {k} != {k2}");
    }
    a.contiguous()?
        .apply_op2_no_bwd(&b.contiguous()?, &Skinny { m, k, n })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Against candle's own GEMM at the talker's projection shapes, and at the widths a shed
    /// batch narrows to. The fixture gate cannot see this kernel — it decodes one lane, and
    /// one lane is not eligible — so this is the only check on it.
    #[test]
    fn matches_candle_at_decode_shapes() -> anyhow::Result<()> {
        #[cfg(feature = "metal")]
        let _gpu = crate::gpu_guard();
        let Some(dev) = crate::usable_metal() else {
            return Ok(());
        };
        for &(m, k, n) in &[
            (48, 2048, 4096),
            (48, 2048, 6144),
            (48, 6144, 2048),
            (48, 1024, 3072),
            (40, 3072, 1024),
            (32, 1024, 2048),
            (8, 1024, 4096),
        ] {
            assert!(eligible(m, k, n), "{m}x{k}x{n} should be eligible");
            let a = Tensor::randn(0f32, 1.0, (m, k), &dev)?.to_dtype(DType::F16)?;
            let b = Tensor::randn(0f32, 0.02, (k, n), &dev)?.to_dtype(DType::F16)?;
            let want = a.matmul(&b)?.to_dtype(DType::F32)?;
            let got = matmul(&a, &b)?;
            assert_eq!(got.dims(), want.dims());
            // Both sides are f16 inputs over a k-long reduction, and this one accumulates in
            // f32 where candle does not, so they disagree at f16's own resolution.
            let (abs, rel) = crate::abs_and_rel(&got, &want)?;
            assert!(rel < 2e-3, "{m}x{k}x{n}: abs {abs:.2e} rel {rel:.2e}");
        }
        Ok(())
    }

    #[test]
    fn declines_shapes_it_cannot_tile() {
        assert!(!eligible(1, 1024, 4096));
        assert!(!eligible(56, 1024, 4096));
        assert!(!eligible(48, 1000, 4096));
        assert!(!eligible(48, 1024, 96));
    }
}
