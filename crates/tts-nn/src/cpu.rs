//! Single-pass CPU kernels for ops candle composes from several allocating passes. Off the CPU
//! each falls back to the composed form, which the tests check the kernels against.

use candle_core::{CpuStorage, CustomOp1, CustomOp3, Layout, Result, Shape, Tensor};

/// erf as Eigen's rational fit, `x P(x^2) / Q(x^2)` on [-4, 4]: within 4.5e-7 of the exact erf,
/// branch-free, so the loop vectorises. candle's `gelu_erf` calls libm's scalar `erff` per element,
/// which was half the nano decoder's CPU time.
#[inline(always)]
fn erf(x: f32) -> f32 {
    let x = x.clamp(-4.0, 4.0);
    let x2 = x * x;
    let mut p = -2.726_142_3e-10f32;
    p = p * x2 + 2.770_681_4e-8;
    p = p * x2 - 2.101_024e-6;
    p = p * x2 - 5.692_506_4e-5;
    p = p * x2 - 7.349_906_3e-4;
    p = p * x2 - 2.954_6e-3;
    p = p * x2 - 1.609_603_3e-2;
    let mut q = -1.456_607_2e-5f32;
    q = q * x2 - 2.133_740_6e-4;
    q = q * x2 - 1.682_827e-3;
    q = q * x2 - 7.373_329e-3;
    q = q * x2 - 1.426_474e-2;
    x * p / q
}

/// Cephes `expf` (Cody-Waite reduction, degree-5 minimax): 0.67 ulp against libm's 0.50, and
/// branch-free, so loops over it vectorise where libm's scalar `expf` does not. Saturates at
/// 2^±126. Not used for `tanh`: a Cephes tanh on it (1.1 ulp) inside ALBERT's GELU moved
/// kokoro-validate's F0_proj from 1.85e-3 to 2.14e-3, over its 2e-3 tolerance, and candle's
/// Accelerate build already runs that GELU on vForce's `vvtanhf`.
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E + 0.5).floor();
    let r = x - n * 0.693_359_4 + n * 2.121_944_4e-4;
    let mut p = 1.987_569_1e-4f32;
    p = p * r + 1.398_2e-3;
    p = p * r + 8.333_452e-3;
    p = p * r + 4.166_579_6e-2;
    p = p * r + 1.666_666_5e-1;
    p = p * r + 5.000_000_1e-1;
    let y = p * r * r + r + 1.0;
    y * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

struct SoftmaxLast;

impl CustomOp1 for SoftmaxLast {
    fn name(&self) -> &'static str {
        "softmax_last_cpu"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let CpuStorage::F32(x) = s else {
            candle_core::bail!("softmax_last_cpu: only f32")
        };
        let n = l.shape().dims().last().copied().unwrap_or(1).max(1);
        let x = &x[l.start_offset()..l.start_offset() + l.shape().elem_count()];
        let mut dst = vec![0f32; x.len()];
        for (row, out) in x.chunks_exact(n).zip(dst.chunks_exact_mut(n)) {
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for (o, &v) in out.iter_mut().zip(row) {
                *o = exp(v - m);
            }
            let inv = 1.0 / sum(out);
            out.iter_mut().for_each(|o| *o *= inv);
        }
        Ok((CpuStorage::F32(dst), l.shape().clone()))
    }
}

/// Eight partial sums, so the loop vectorises: one accumulator is a serial chain, since the
/// compiler may not reorder float additions.
fn sum(v: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let c = v.chunks_exact(8);
    let tail: f32 = c.remainder().iter().sum();
    for x in c {
        for k in 0..8 {
            acc[k] += x[k];
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// Softmax over the last dimension. candle's CPU form is a scalar `expf` per element: attention
/// scores were 8% of a one-thread Kokoro render.
pub fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    if x.device().is_cpu() && x.dtype() == candle_core::DType::F32 {
        return x.contiguous()?.apply_op1_no_bwd(&SoftmaxLast);
    }
    candle_nn::ops::softmax_last_dim(x)
}

struct GeluErf;

impl CustomOp1 for GeluErf {
    fn name(&self) -> &'static str {
        "gelu_erf_cpu"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let CpuStorage::F32(x) = s else {
            candle_core::bail!("gelu_erf_cpu: only f32")
        };
        let x = &x[l.start_offset()..l.start_offset() + l.shape().elem_count()];
        let dst = x
            .iter()
            .map(|&v| 0.5 * v * (1.0 + erf(v * std::f32::consts::FRAC_1_SQRT_2)))
            .collect();
        Ok((CpuStorage::F32(dst), l.shape().clone()))
    }
}

/// Exact (erf) GELU.
pub fn gelu_erf(x: &Tensor) -> Result<Tensor> {
    if x.device().is_cpu() && x.dtype() == candle_core::DType::F32 {
        return x.contiguous()?.apply_op1_no_bwd(&GeluErf);
    }
    x.gelu_erf()
}

struct DepthwiseTc {
    t: usize,
    c: usize,
    k: usize,
}

impl CustomOp3 for DepthwiseTc {
    fn name(&self) -> &'static str {
        "depthwise_tc_cpu"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (CpuStorage::F32(x), CpuStorage::F32(w), CpuStorage::F32(b)) = (s1, s2, s3) else {
            candle_core::bail!("depthwise_tc_cpu: only f32")
        };
        let (t, c, k) = (self.t, self.c, self.k);
        let x = &x[l1.start_offset()..l1.start_offset() + t * c];
        let w = &w[l2.start_offset()..l2.start_offset() + k * c];
        let b = &b[l3.start_offset()..l3.start_offset() + c];
        let p = k / 2;
        let mut dst = Vec::with_capacity(t * c);
        for i in 0..t {
            let row_start = dst.len();
            dst.extend_from_slice(b);
            let out = &mut dst[row_start..];
            // Taps that fall in the zero padding are skipped rather than read.
            for j in i.saturating_sub(p)..(i + k - p).min(t) {
                let tap = &w[(j + p - i) * c..(j + p - i + 1) * c];
                let src = &x[j * c..(j + 1) * c];
                for ((o, &v), &g) in out.iter_mut().zip(src).zip(tap) {
                    *o += v * g;
                }
            }
        }
        Ok((CpuStorage::F32(dst), l1.shape().clone()))
    }
}

/// Depthwise conv over time, zero-padded `k / 2` each side, on time-major `x` `[1, T, C]`, with
/// `taps` `[k, C]` and `bias` `[C]`: one pass where the composed form is `2k` broadcast ops.
pub fn depthwise_tc(x: &Tensor, taps: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let (_, t, c) = x.dims3()?;
    let k = taps.dim(0)?;
    if x.device().is_cpu() && x.dtype() == candle_core::DType::F32 && x.dim(0)? == 1 {
        let op = DepthwiseTc { t, c, k };
        return x
            .contiguous()?
            .apply_op3_no_bwd(&taps.contiguous()?, &bias.contiguous()?, &op);
    }
    depthwise_tc_composed(x, taps, bias)
}

fn depthwise_tc_composed(x: &Tensor, taps: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let (_, t, c) = x.dims3()?;
    let k = taps.dim(0)?;
    let p = k / 2;
    let padded = x.pad_with_zeros(1, p, k - 1 - p)?;
    let mut h = bias
        .reshape((1, 1, c))?
        .broadcast_as(x.shape())?
        .contiguous()?;
    for j in 0..k {
        let tap = taps.narrow(0, j, 1)?.unsqueeze(0)?;
        h = (h + padded.narrow(1, j, t)?.broadcast_mul(&tap)?)?;
    }
    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// Worst relative error against f64 over a dense sweep, in f32 ulps (2^-23 relative).
    fn ulps(f: impl Fn(f32) -> f32, g: impl Fn(f64) -> f64, lo: f32, hi: f32) -> f64 {
        (0..=200_000)
            .map(|i| lo + (hi - lo) * i as f32 / 200_000.0)
            .map(|x| {
                let want = g(x as f64);
                ((f(x) as f64 - want).abs() / want.abs().max(1e-30)) * (1u64 << 23) as f64
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn exp_is_within_an_ulp() {
        let e = ulps(exp, f64::exp, -87.0, 88.0);
        assert!(e < 1.0, "exp: {e:.2} ulp");
    }

    #[test]
    fn softmax_matches_candle() {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 4.0, (3, 37, 151), &dev).unwrap();
        let d = max_diff(
            &softmax_last_dim(&x).unwrap(),
            &candle_nn::ops::softmax_last_dim(&x).unwrap(),
        );
        assert!(d < 1e-6, "softmax {d}");
    }

    #[test]
    fn gelu_matches_candle() {
        let x = Tensor::arange(-12_000f32, 12_000f32, &Device::Cpu)
            .unwrap()
            .affine(1e-3, 0.0)
            .unwrap();
        let d = max_diff(&gelu_erf(&x).unwrap(), &x.gelu_erf().unwrap());
        assert!(d < 2e-6, "{d}");
    }

    #[test]
    fn depthwise_matches_composed() {
        let dev = Device::Cpu;
        for (t, c, k) in [(1, 8, 7), (3, 8, 7), (50, 16, 7), (40, 12, 5), (9, 4, 1)] {
            let x = Tensor::randn(0f32, 1.0, (1, t, c), &dev).unwrap();
            let w = Tensor::randn(0f32, 1.0, (k, c), &dev).unwrap();
            let b = Tensor::randn(0f32, 1.0, c, &dev).unwrap();
            let d = max_diff(
                &depthwise_tc(&x, &w, &b).unwrap(),
                &depthwise_tc_composed(&x, &w, &b).unwrap(),
            );
            assert!(d < 1e-5, "{t}x{c} k{k}: {d}");
        }
    }
}
