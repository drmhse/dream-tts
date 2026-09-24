//! A centred 1-D convolution as one GEMM that never materialises its im2col matrix.
//!
//! **This loses, and is kept only so the measurement can be rerun.** Nothing calls it;
//! `tts-probe`'s `kokorogen` times it against the gather route it was meant to replace.
//! Its failure is what pointed at [`crate::mpsconv`], which does the same thing by asking
//! MPSGraph instead, and wins by 1.7x.
//!
//! The premise was sound. `centered_conv1d_gemm` gathers `[k * cin, len]` and hands it to
//! candle's matmul: at `128ch @ 48240, k=11` that matrix is 271 MB written and read back,
//! and the gather is 4.9 ms of a 10.4 ms convolution. A kernel that stages the tap window
//! in threadgroup memory writes none of it, and the tile here takes the whole of `M` so
//! the input is read once — possible only because these convolutions have 128 to 512
//! output channels, not thousands.
//!
//! What it ran into is that the GEMM it has to replace is already near the hardware:
//! MPS reaches ~3.0 TFLOP/s on these shapes against an M4's ~4.3 peak. Two attempts:
//!
//! - **simdgroup matrices**, 128x64 tile, 4x4 accumulator blocks per simdgroup: 24.1 ms
//!   against 10.4, or 0.73 TFLOP/s. Widening the K block to 64 to cut barriers made it
//!   *worse* (28.0 ms) — the 16 KB staging buffer costs more occupancy than the barriers
//!   cost time. Dropping the epilogue's scratch tile, by seeding the accumulator with the
//!   bias through a stride-0 transposed load, bought nothing either.
//! - **A classical register tile**, 128x128 with 8x8 outputs per thread: 43.8 ms, 0.40
//!   TFLOP/s. 64 accumulators per thread spills.
//!
//! So the gather is real overhead — a third of the generator's convolution time — but it
//! is cheaper than any GEMM this codebase can write by hand. M3/M4 have no matrix unit in
//! the GPU; `simdgroup_multiply_accumulate` is a scheduled ALU sequence, and MPS schedules
//! it better. The idea is not wrong, it is just waiting on a GEMM worth building on.
//!
//! The narrow contract, for whoever picks it up: batch 1, stride 1, `cin` a multiple of 8,
//! `cout` a multiple of 128, f32, Metal.

use candle_core::{CpuStorage, CustomOp2, Layout, Result, Shape, Tensor};

pub(crate) struct TapConv {
    pub(crate) kernel: &'static str,
    pub(crate) k: usize,
    pub(crate) dilation: usize,
    pub(crate) pad_left: usize,
    pub(crate) bias: Option<Tensor>,
}

impl CustomOp2 for TapConv {
    fn name(&self) -> &'static str {
        "conv1d_tap_gemm"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("conv1d_tap_gemm: Metal only")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::{MTLResourceUsage, MTLSize};

        for l in [l1, l2] {
            if !l.is_contiguous() {
                candle_core::bail!("conv1d_tap_gemm: inputs must be contiguous");
            }
        }
        let (_, cin, len) = l1.shape().dims3()?;
        let cout = l2.shape().dims2()?.0;
        let device = s1.device();
        let p = crate::mtl::pipeline(device, self.kernel)?;
        let dst = device.new_buffer(cout * len, DType::F32, "conv1d_tap_gemm")?;

        // The kernel always reads a bias buffer; without one it reads and discards the
        // weights, which is cheaper than a second pipeline.
        let (bias_buf, bias_off, usebias) = match &self.bias {
            Some(b) => {
                let (s, l) = b.storage_and_layout();
                match &*s {
                    candle_core::Storage::Metal(m) => {
                        (m.buffer().clone(), l.start_offset() * 4, 1u32)
                    }
                    _ => candle_core::bail!("conv1d_tap_gemm: bias must be on the same device"),
                }
            }
            None => (s2.buffer().clone(), 0, 0u32),
        };

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::conv1d_tap_gemm");
        encoder.set_compute_pipeline_state(&p);
        encoder.set_buffer(0, Some(s1.buffer()), l1.start_offset() * 4);
        encoder.set_buffer(1, Some(s2.buffer()), l2.start_offset() * 4);
        encoder.set_buffer(2, Some(&bias_buf), bias_off);
        encoder.set_buffer(3, Some(dst.as_ref()), 0);
        for (i, v) in [len, cin, cout, self.k, self.dilation, self.pad_left]
            .iter()
            .enumerate()
        {
            encoder.set_bytes(4 + i, &(*v as u32));
        }
        encoder.set_bytes(10, &usebias);
        encoder.use_resource(s1.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(s2.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(&bias_buf, MTLResourceUsage::Read);
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        let nt = if self.kernel == "conv1d_tap_reg_f32" {
            128
        } else {
            64
        };
        encoder.dispatch_thread_groups(
            MTLSize {
                width: len.div_ceil(nt),
                height: cout / 128,
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
            MetalStorage::new(dst, device.clone(), cout * len, DType::F32),
            (1, cout, len).into(),
        ))
    }
}

/// The centred convolution `centered_conv1d_gemm` computes, without the gather.
pub fn centered_conv1d_fused(
    x: &Tensor,
    w_tap: &Tensor,
    b: Option<&Tensor>,
    k: usize,
    dilation: usize,
    pad_left: usize,
) -> Result<Tensor> {
    centered_conv1d_fused_with(x, w_tap, b, k, dilation, pad_left, "conv1d_tap_reg_f32")
}

/// The same, naming which of the two kernels to run. `tts-probe`'s `kokorogen` compares
/// them; nothing else should need this.
pub fn centered_conv1d_fused_with(
    x: &Tensor,
    w_tap: &Tensor,
    b: Option<&Tensor>,
    k: usize,
    dilation: usize,
    pad_left: usize,
    kernel: &'static str,
) -> Result<Tensor> {
    let (_, cin, len) = x.dims3()?;
    let cout = w_tap.dim(0)?;
    let op = TapConv {
        kernel,
        k,
        dilation,
        pad_left,
        bias: b.map(|t| t.flatten_all()).transpose()?,
    };
    let y = x
        .contiguous()?
        .apply_op2_no_bwd(&w_tap.contiguous()?, &op)?;
    crate::stats::record(
        cout,
        k * cin,
        len,
        (cout * k * cin + cin * len + cout * len) as u64 * 4,
    );
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Against the gather route it replaces, at the generator's own shapes plus the
    /// awkward ones: a length that is not a multiple of the tile, and every dilation.
    #[test]
    fn matches_gather_route() -> anyhow::Result<()> {
        #[cfg(feature = "metal")]
        let _gpu = crate::gpu_guard();
        let Some(d) = crate::usable_metal() else {
            return Ok(());
        };
        for (cin, cout, len) in [
            (128usize, 128usize, 4441usize),
            (128, 128, 48240),
            (256, 256, 8040),
            (128, 256, 511),
        ] {
            for (k, dil) in [(1usize, 1usize), (3, 1), (3, 5), (7, 3), (11, 1), (11, 5)] {
                let x = Tensor::randn(0f32, 1., (1, cin, len), &d)?;
                let w = Tensor::randn(0f32, 0.02, (cout, cin, k), &d)?;
                let b = Tensor::randn(0f32, 1., cout, &d)?;
                let w_tap = crate::tap_major_weight(&w)?;
                let pad = (k - 1) * dil / 2;
                let want = crate::centered_conv1d_gemm(&x, &w_tap, Some(&b), k, dil, pad)?;
                for kern in ["conv1d_tap_gemm_f32", "conv1d_tap_reg_f32"] {
                    let got = centered_conv1d_fused_with(&x, &w_tap, Some(&b), k, dil, pad, kern)?;
                    assert_eq!(want.dims(), got.dims());
                    let (abs, rel) = crate::abs_and_rel(&want, &got)?;
                    // Looser than the biasless bound below on purpose: the bias seeds the
                    // accumulator here rather than being added to a finished sum.
                    assert!(
                        rel < 1e-5,
                        "{kern} {cin}->{cout} @ {len} k={k} d={dil}: abs {abs:.3e} rel {rel:.3e}"
                    );
                }

                let want = crate::centered_conv1d_gemm(&x, &w_tap, None, k, dil, pad)?;
                let got = centered_conv1d_fused(&x, &w_tap, None, k, dil, pad)?;
                let (abs, rel) = crate::abs_and_rel(&want, &got)?;
                assert!(
                    rel < 1e-5,
                    "biasless {k}/{dil}: abs {abs:.3e} rel {rel:.3e}"
                );
            }
        }
        Ok(())
    }
}
