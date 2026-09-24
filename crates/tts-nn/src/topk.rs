//! Top-k sampling on the device, so an autoregressive loop need not read logits back per step.
//!
//! The draws come from the caller, in the order its host sampler would have consumed them, so
//! a seeded render keeps its random stream. See `topk_sample_f32` in [`crate::mtl`].

use candle_core::{CpuStorage, CustomOp2, DType, Layout, Result, Shape, Tensor};

/// Largest row and `k` the kernel's threadgroup arrays hold.
pub const MAX_N: usize = 4096;
pub const MAX_K: usize = 64;

struct TopkSample {
    k: usize,
    temperature: f32,
}

impl CustomOp2 for TopkSample {
    fn name(&self) -> &'static str {
        "topk_sample"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("topk_sample: Metal only")
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
            candle_core::bail!("topk_sample: inputs must be contiguous");
        }
        if s1.dtype() != DType::F32 || s2.dtype() != DType::F32 {
            candle_core::bail!("topk_sample: only f32");
        }
        let (rows, n) = l1.shape().dims2()?;
        if l2.shape().elem_count() != rows {
            candle_core::bail!(
                "topk_sample: {} draws for {rows} rows",
                l2.shape().elem_count()
            );
        }
        let device = s1.device();
        let p = mtl::pipeline(device, "topk_sample_f32")?;
        let dst = device.new_buffer(rows, DType::U32, "topk_sample")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::topk_sample");
        encoder.set_compute_pipeline_state(&p);
        encoder.set_buffer(0, Some(s1.buffer()), l1.start_offset() * 4);
        encoder.set_buffer(1, Some(s2.buffer()), l2.start_offset() * 4);
        encoder.set_buffer(2, Some(dst.as_ref()), 0);
        encoder.set_bytes(3, &(n as u32));
        encoder.set_bytes(4, &(self.k as u32));
        encoder.set_bytes(5, &self.temperature);
        encoder.use_resource(s1.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(s2.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: rows,
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
            MetalStorage::new(dst, device.clone(), rows, DType::U32),
            (rows,).into(),
        ))
    }
}

/// One index per row of `logits` `[rows, n]`, drawn from its top `k` at `temperature` with
/// `draws[row]` in `[0, 1)`. No nucleus cut and no penalty: callers needing either stay on
/// the host.
pub fn sample(logits: &Tensor, draws: &Tensor, k: usize, temperature: f32) -> Result<Tensor> {
    let (_, n) = logits.dims2()?;
    if k == 0 || k > MAX_K || k > n || n > MAX_N {
        candle_core::bail!("topk_sample: k={k} n={n} outside the kernel's limits");
    }
    let temperature = if temperature > 0.0 { temperature } else { 1.0 };
    logits
        .contiguous()?
        .apply_op2_no_bwd(&draws.contiguous()?, &TopkSample { k, temperature })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// qwen3tts's host sampler at top_p = 1 and no penalty, as a reference.
    fn host(l: &[f32], k: usize, t: f32, u: f32) -> u32 {
        let mut order: Vec<usize> = (0..l.len()).collect();
        order.sort_by(|&a, &b| l[b].total_cmp(&l[a]).then(a.cmp(&b)));
        order.truncate(k);
        let mx = l[order[0]];
        let mut p: Vec<f32> = order.iter().map(|&i| ((l[i] - mx) / t).exp()).collect();
        let total: f32 = p.iter().sum();
        for q in &mut p {
            *q /= total;
        }
        let mut u = u * p.iter().sum::<f32>();
        for (j, q) in p.iter().enumerate() {
            u -= q;
            if u <= 0.0 {
                return order[j] as u32;
            }
        }
        order[k - 1] as u32
    }

    #[test]
    fn picks_what_the_host_picks() -> anyhow::Result<()> {
        #[cfg(feature = "metal")]
        let _gpu = crate::gpu_guard();
        let Some(d) = crate::usable_metal() else {
            return Ok(());
        };
        let (rows, n, k, t) = (48usize, 2048usize, 50usize, 0.9f32);
        let mut total = 0;
        let mut differ = 0;
        for seed in 0..40u64 {
            let logits = (Tensor::randn(0f32, 3., (rows, n), &d)? + seed as f64 * 1e-3)?;
            let draws = Tensor::rand(0f32, 1., rows, &d)?;
            let got = sample(&logits, &draws, k, t)?.to_vec1::<u32>()?;
            let l = logits.to_vec2::<f32>()?;
            let u = draws.to_vec1::<f32>()?;
            for r in 0..rows {
                total += 1;
                if got[r] != host(&l[r], k, t, u[r]) {
                    differ += 1;
                }
            }
        }
        // Only a draw within an ulp of a boundary may differ, through `exp`'s last bit.
        assert!(differ * 1000 <= total, "{differ} of {total} picks differ");

        let tied = Tensor::from_vec(vec![1f32, 5., 5., 2., 5.], (1, 5), &d)?;
        let first = Tensor::from_vec(vec![0f32], 1, &d)?;
        assert_eq!(sample(&tied, &first, 3, 1.0)?.to_vec1::<u32>()?, vec![1]);
        Ok(())
    }
}
