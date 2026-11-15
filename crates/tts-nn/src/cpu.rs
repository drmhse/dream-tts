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

/// Cephes `tanhf` (1.1 ulp against libm's 0.6): an odd polynomial below 0.625, where
/// `1 - 2 / (e^2x + 1)` would cancel, and that form above, selected without a branch.
#[inline(always)]
pub fn tanh(x: f32) -> f32 {
    let z = x * x;
    let mut p = -5.704_988_7e-3f32;
    p = p * z + 2.063_908_9e-2;
    p = p * z - 5.373_971_6e-2;
    p = p * z + 1.333_144_2e-1;
    p = p * z - 3.333_328_2e-1;
    let small = p * z * x + x;
    let big = (1.0 - 2.0 / (exp(2.0 * x.abs()) + 1.0)).copysign(x);
    if x.abs() < 0.625 {
        small
    } else {
        big
    }
}

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

/// exp in double precision (Cody-Waite reduction, Taylor to r^12: under 1e-14 relative), branch-free so
/// two lanes vectorise. For f32 work that must be at least as accurate as libm: computed here and
/// rounded once, the result is within half an f32 ulp.
#[inline(always)]
pub fn exp64(x: f64) -> f64 {
    let x = x.clamp(-700.0, 700.0);
    let n = (x * std::f64::consts::LOG2_E).round();
    let r = x - n * 6.931_471_803_691_238_2e-1 - n * 1.908_214_929_270_587_7e-10;
    let mut p = 1.0 / 479_001_600.0;
    for d in [
        39_916_800.0,
        3_628_800.0,
        362_880.0,
        40_320.0,
        5040.0,
        720.0,
        120.0,
        24.0,
        6.0,
        2.0,
        1.0,
    ] {
        p = p * r + 1.0 / d;
    }
    (p * r + 1.0) * f64::from_bits(((n as i64 + 1023) as u64) << 52)
}

/// tanh in double precision from [`exp64`]; below 1e-4 the series, where the exp form cancels.
#[inline(always)]
pub fn tanh64(x: f64) -> f64 {
    let big = (1.0 - 2.0 / (exp64(2.0 * x.abs()) + 1.0)).copysign(x);
    if x.abs() < 1e-4 {
        x - x * x * x / 3.0
    } else {
        big
    }
}

#[inline(always)]
pub fn sigmoid64(x: f64) -> f64 {
    1.0 / (1.0 + exp64(-x))
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

struct NormFilm {
    t: usize,
    c: usize,
    over_time: bool,
    eps: f32,
}

impl CustomOp3 for NormFilm {
    fn name(&self) -> &'static str {
        "norm_film_cpu"
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
        let (CpuStorage::F32(x), CpuStorage::F32(g), CpuStorage::F32(b)) = (s1, s2, s3) else {
            candle_core::bail!("norm_film_cpu: only f32")
        };
        let (t, c) = (self.t, self.c);
        let x = &x[l1.start_offset()..l1.start_offset() + t * c];
        let g = &g[l2.start_offset()..l2.start_offset() + c];
        let b = &b[l3.start_offset()..l3.start_offset() + c];
        let mut dst = vec![0f32; t * c];
        if self.over_time {
            // Per channel, over time: column statistics, accumulated row by row so the loops run
            // along the contiguous channels.
            let mut mean = vec![0f32; c];
            for row in x.chunks_exact(c) {
                mean.iter_mut().zip(row).for_each(|(m, &v)| *m += v);
            }
            mean.iter_mut().for_each(|m| *m /= t as f32);
            let mut var = vec![0f32; c];
            for row in x.chunks_exact(c) {
                for ((s, &v), &m) in var.iter_mut().zip(row).zip(&mean) {
                    *s += (v - m) * (v - m);
                }
            }
            let scale: Vec<f32> = var
                .iter()
                .zip(g)
                .map(|(&s, &g)| g / (s / t as f32 + self.eps).sqrt())
                .collect();
            for (row, out) in x.chunks_exact(c).zip(dst.chunks_exact_mut(c)) {
                for ((((o, &v), &m), &k), &bb) in
                    out.iter_mut().zip(row).zip(&mean).zip(&scale).zip(b)
                {
                    *o = (v - m) * k + bb;
                }
            }
        } else {
            for (row, out) in x.chunks_exact(c).zip(dst.chunks_exact_mut(c)) {
                let m = sum(row) / c as f32;
                let var = row.iter().map(|&v| (v - m) * (v - m)).sum::<f32>() / c as f32;
                let inv = 1.0 / (var + self.eps).sqrt();
                for (((o, &v), &gg), &bb) in out.iter_mut().zip(row).zip(g).zip(b) {
                    *o = (v - m) * inv * gg + bb;
                }
            }
        }
        Ok((CpuStorage::F32(dst), l1.shape().clone()))
    }
}

/// Affine-free norm then a per-channel gain and bias, on time-major `x` `[1, T, C]` with `g`, `b`
/// `[1, 1, C]`: over time per channel (instance norm) or over channels per frame (layer norm). One pass
/// where the composed form is eight broadcast ops.
pub fn norm_film(x: &Tensor, g: &Tensor, b: &Tensor, over_time: bool, eps: f64) -> Result<Tensor> {
    let (n, t, c) = x.dims3()?;
    if x.device().is_cpu() && x.dtype() == candle_core::DType::F32 && n == 1 {
        let op = NormFilm {
            t,
            c,
            over_time,
            eps: eps as f32,
        };
        return x
            .contiguous()?
            .apply_op3_no_bwd(&g.contiguous()?, &b.contiguous()?, &op);
    }
    norm_film_composed(x, g, b, over_time, eps)
}

fn norm_film_composed(
    x: &Tensor,
    g: &Tensor,
    b: &Tensor,
    over_time: bool,
    eps: f64,
) -> Result<Tensor> {
    let dim = if over_time { 1 } else { 2 };
    let mean = x.mean_keepdim(dim)?;
    let c = x.broadcast_sub(&mean)?;
    let h = c.broadcast_div(&(c.sqr()?.mean_keepdim(dim)? + eps)?.sqrt()?)?;
    h.broadcast_mul(g)?.broadcast_add(b)
}

struct ScaledResidual;

impl CustomOp3 for ScaledResidual {
    fn name(&self) -> &'static str {
        "scaled_residual_cpu"
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
        let (CpuStorage::F32(x), CpuStorage::F32(h), CpuStorage::F32(s)) = (s1, s2, s3) else {
            candle_core::bail!("scaled_residual_cpu: only f32")
        };
        let n = l1.shape().elem_count();
        let c = l3.shape().elem_count();
        let x = &x[l1.start_offset()..l1.start_offset() + n];
        let h = &h[l2.start_offset()..l2.start_offset() + n];
        let s = &s[l3.start_offset()..l3.start_offset() + c];
        let mut dst = Vec::with_capacity(n);
        for (xr, hr) in x.chunks_exact(c).zip(h.chunks_exact(c)) {
            dst.extend(xr.iter().zip(hr).zip(s).map(|((&a, &b), &k)| a + b * k));
        }
        Ok((CpuStorage::F32(dst), l1.shape().clone()))
    }
}

/// `x + h * scale` with `scale` per channel (the last dimension), in one pass.
pub fn scaled_residual(x: &Tensor, h: &Tensor, scale: &Tensor) -> Result<Tensor> {
    if x.device().is_cpu() && x.dtype() == candle_core::DType::F32 && x.shape() == h.shape() {
        return x.contiguous()?.apply_op3_no_bwd(
            &h.contiguous()?,
            &scale.contiguous()?,
            &ScaledResidual,
        );
    }
    let c = scale.elem_count();
    x + h.broadcast_mul(&scale.reshape((1, 1, c))?)?
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
        let t = ulps(tanh, f64::tanh, -12.0, 12.0);
        assert!(t < 1.5, "tanh: {t:.2} ulp");
        let s = ulps(sigmoid, |x| 1.0 / (1.0 + (-x).exp()), -80.0, 80.0);
        assert!(s < 2.5, "sigmoid: {s:.2} ulp");
        // Rounded once from double precision: within half an ulp, like a correctly rounded libm.
        let e = ulps(|x| exp64(x as f64) as f32, f64::exp, -87.0, 88.0);
        assert!(e <= 0.5001, "exp64: {e:.4} ulp");
        let t = ulps(|x| tanh64(x as f64) as f32, f64::tanh, -12.0, 12.0);
        assert!(t <= 0.5001, "tanh64: {t:.4} ulp");
        let s = ulps(
            |x| sigmoid64(x as f64) as f32,
            |x| 1.0 / (1.0 + (-x).exp()),
            -80.0,
            80.0,
        );
        assert!(s <= 0.5001, "sigmoid64: {s:.4} ulp");
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
    fn norm_film_and_residual_match_composed() {
        let dev = Device::Cpu;
        for (t, c) in [(1, 8), (37, 64), (200, 512)] {
            let x = Tensor::randn(0.3f32, 2.0, (1, t, c), &dev).unwrap();
            let g = Tensor::randn(1f32, 0.3, (1, 1, c), &dev).unwrap();
            let b = Tensor::randn(0f32, 0.3, (1, 1, c), &dev).unwrap();
            for over_time in [true, false] {
                if over_time && t == 1 {
                    continue;
                }
                let d = max_diff(
                    &norm_film(&x, &g, &b, over_time, 1e-5).unwrap(),
                    &norm_film_composed(&x, &g, &b, over_time, 1e-5).unwrap(),
                );
                assert!(d < 2e-5, "{t}x{c} over_time {over_time}: {d}");
            }
            let h = Tensor::randn(0f32, 1.0, (1, t, c), &dev).unwrap();
            let s = Tensor::randn(0f32, 0.1, c, &dev).unwrap();
            let want = (&x + h.broadcast_mul(&s.reshape((1, 1, c)).unwrap()).unwrap()).unwrap();
            assert!(max_diff(&scaled_residual(&x, &h, &s).unwrap(), &want) < 1e-6);
        }
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
