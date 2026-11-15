//! Single-pass CPU kernels for ops candle composes from several allocating passes. Off the CPU
//! each falls back to the composed form, which the tests check the kernels against.

use candle_core::{CpuStorage, CustomOp1, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor};

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
    p = p * r + 5e-1;
    let y = p * r * r + r + 1.0;
    y * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// Cephes `tanhf` (1.1 ulp against libm's 0.6): an odd polynomial below 0.625, where
/// `1 - 2 / (e^2x + 1)` would cancel, and that form above, selected without a branch.
#[inline(always)]
pub fn tanh(x: f32) -> f32 {
    let z = x * x;
    let mut p = -5.704_988_7e-3f32;
    p = p * z + 2.063_909e-2;
    p = p * z - 5.373_971_6e-2;
    p = p * z + 1.333_144_2e-1;
    p = p * z - 3.333_328e-1;
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
    let r = x - n * 6.931_471_803_691_238e-1 - n * 1.908_214_929_270_587_7e-10;
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

/// Threads for `units` independent pieces of work: candle's count (RAYON_NUM_THREADS, else the
/// cores), at most four; past that the decoder's ops are too short to split further.
pub fn threads_for(units: usize) -> usize {
    candle_core::utils::get_num_threads()
        .min(4)
        .min(units)
        .max(1)
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

/// Accelerate's vForce and BLAS, declared here rather than through a crate: candle's `accelerate`
/// feature already links the framework on every Apple target.
#[cfg(target_vendor = "apple")]
mod accelerate {
    extern "C" {
        pub fn vvsinf(y: *mut f32, x: *const f32, n: *const i32);
        #[allow(clippy::too_many_arguments)]
        pub fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }
    pub const ROW_MAJOR: i32 = 101;
    pub const NO_TRANS: i32 = 111;
    pub const TRANS: i32 = 112;
}

/// `sin` over a slice in place: vForce on Apple hardware, libm elsewhere. An inline Cephes `sin^2`
/// fused into the Snake loop instead was slower (0.96-1.00 s against 0.92 for the Kokoro decoder).
fn sin_in_place(v: &mut [f32]) {
    #[cfg(target_vendor = "apple")]
    {
        let n = v.len() as i32;
        let p = v.as_mut_ptr();
        // vvsinf allows the output to alias the input.
        unsafe { accelerate::vvsinf(p, p as *const f32, &n) };
    }
    #[cfg(not(target_vendor = "apple"))]
    v.iter_mut().for_each(|x| *x = x.sin());
}

struct AdainSnakeCpu {
    c: usize,
    len: usize,
    eps: f32,
}

impl CustomOp2 for AdainSnakeCpu {
    fn name(&self) -> &'static str {
        "adain_snake_cpu"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (CpuStorage::F32(x), CpuStorage::F32(p)) = (s1, s2) else {
            candle_core::bail!("adain_snake_cpu: only f32")
        };
        let (c, len) = (self.c, self.len);
        let x = &x[l1.start_offset()..l1.start_offset() + c * len];
        // Packed rows: mean, var, gamma, beta, alpha, beta_recip.
        let p = &p[l2.start_offset()..l2.start_offset() + 6 * c];
        let col = |r: usize, j: usize| p[r * c + j];
        let mut dst = vec![0f32; c * len];
        // Channel rows `j0..`, one per `len`-chunk of `out`.
        let rows = |j0: usize, out: &mut [f32]| {
            const CHUNK: usize = 2048;
            let mut u = vec![0f32; CHUNK];
            for (i, os) in out.chunks_exact_mut(len).enumerate() {
                let (j, row) = (j0 + i, &x[(j0 + i) * len..(j0 + i + 1) * len]);
                let scale = (1.0 + col(2, j)) / (col(1, j) + self.eps).sqrt();
                let shift = col(3, j) - col(0, j) * scale;
                let (alpha, br) = (col(4, j), col(5, j));
                for (xs, os) in row.chunks(CHUNK).zip(os.chunks_mut(CHUNK)) {
                    let u = &mut u[..xs.len()];
                    for ((o, uu), &v) in os.iter_mut().zip(u.iter_mut()).zip(xs) {
                        *o = v * scale + shift;
                        *uu = *o * alpha;
                    }
                    sin_in_place(u);
                    for (o, &sv) in os.iter_mut().zip(u.iter()) {
                        *o += br * sv * sv;
                    }
                }
            }
        };
        let threads = threads_for(c);
        if threads <= 1 {
            rows(0, &mut dst);
        } else {
            let per = c.div_ceil(threads);
            let rows = &rows;
            std::thread::scope(|sc| {
                for (i, part) in dst.chunks_mut(per * len).enumerate() {
                    sc.spawn(move || rows(i * per, part));
                }
            });
        }
        Ok((CpuStorage::F32(dst), l1.shape().clone()))
    }
}

/// AdaIN then SnakeBeta on `[1, C, L]`, one pass per channel row. `packed` is `[6, C]`: mean, var,
/// gamma, beta, alpha, 1/beta. Composed on the CPU it was six full passes and the generator's
/// largest cost.
pub fn adain_snake(x: &Tensor, packed: &Tensor, eps: f64) -> Result<Tensor> {
    let (_, c, len) = x.dims3()?;
    x.contiguous()?.apply_op2_no_bwd(
        &packed.contiguous()?,
        &AdainSnakeCpu {
            c,
            len,
            eps: eps as f32,
        },
    )
}

#[cfg(target_vendor = "apple")]
struct ConvTiled {
    cin: usize,
    len: usize,
    out: usize,
    k: usize,
    dilation: usize,
    pad_left: usize,
    tile: usize,
    /// The weight arrives as `[k * cin, out]` (row-major) to be read transposed.
    trans: bool,
}

#[cfg(target_vendor = "apple")]
impl CustomOp3 for ConvTiled {
    fn name(&self) -> &'static str {
        "conv_tiled_cpu"
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
            candle_core::bail!("conv_tiled_cpu: only f32")
        };
        let (cin, len, out, k, d) = (self.cin, self.len, self.out, self.k, self.dilation);
        let x = &x[l1.start_offset()..l1.start_offset() + cin * len];
        let w = &w[l2.start_offset()..l2.start_offset() + out * k * cin];
        let b = &b[l3.start_offset()..l3.start_offset() + out];
        let mut y = Vec::with_capacity(out * len);
        for &bias in b {
            y.extend(std::iter::repeat_n(bias, len));
        }
        let tile = self.tile.min(len);
        let (ta, lda) = if self.trans {
            (accelerate::TRANS, out)
        } else {
            (accelerate::NO_TRANS, k * cin)
        };
        let gemm = |a: *const f32, kk: usize, bm: *const f32, ldb: usize, n: usize, c: *mut f32| unsafe {
            accelerate::cblas_sgemm(
                accelerate::ROW_MAJOR,
                ta,
                accelerate::NO_TRANS,
                out as i32,
                n as i32,
                kk as i32,
                1.0,
                a,
                lda as i32,
                bm,
                ldb as i32,
                1.0,
                c,
                len as i32,
            )
        };
        // Columns [lo, hi) of every output row, tile by tile. Threads take disjoint, tile-aligned
        // column ranges, so the tiles and the result are the same however many there are.
        let y_addr = y.as_mut_ptr() as usize;
        let run = |lo: usize, hi: usize| {
            let mut cols = vec![0f32; k * cin * tile];
            let mut c0 = lo;
            while c0 < hi {
                let n = tile.min(hi - c0);
                // cols[(t, ci), j] = x[ci, c0 + j + t*d - pad_left], zero outside.
                for t in 0..k {
                    let shift = (t * d) as isize - self.pad_left as isize;
                    for ci in 0..cin {
                        let dst = &mut cols[(t * cin + ci) * n..(t * cin + ci + 1) * n];
                        let src = &x[ci * len..(ci + 1) * len];
                        let start = c0 as isize + shift;
                        let lo = (-start).clamp(0, n as isize) as usize;
                        let hi = (len as isize - start).clamp(0, n as isize) as usize;
                        dst[..lo].fill(0.0);
                        if hi > lo {
                            let s = (start + lo as isize) as usize;
                            dst[lo..hi].copy_from_slice(&src[s..s + hi - lo]);
                        }
                        dst[hi.max(lo)..].fill(0.0);
                    }
                }
                gemm(
                    w.as_ptr(),
                    k * cin,
                    cols.as_ptr(),
                    n,
                    n,
                    (y_addr as *mut f32).wrapping_add(c0),
                );
                c0 += n;
            }
        };
        let threads = threads_for(len.div_ceil(tile));
        if threads <= 1 {
            run(0, len);
        } else {
            let per = len.div_ceil(threads).div_ceil(tile) * tile;
            let run = &run;
            std::thread::scope(|sc| {
                for i in 0..threads {
                    let (lo, hi) = (i * per, ((i + 1) * per).min(len));
                    if lo < hi {
                        sc.spawn(move || run(lo, hi));
                    }
                }
            });
        }
        Ok((CpuStorage::F32(y), (1, out, len).into()))
    }
}

/// A 1-D conv on `[1, cin, len]` with `w_tap` `[out, k * cin]` (tap-major): im2col one tile of
/// columns at a time, sized to stay in cache, each multiplied by Accelerate straight into its slice
/// of the output. The whole-sequence im2col is hundreds of MB at Kokoro's widest layers, and the
/// GEMM streamed it from memory. Apple CPU only; `None` elsewhere.
pub fn conv_tiled(
    x: &Tensor,
    w_tap: &Tensor,
    b: Option<&Tensor>,
    k: usize,
    dilation: usize,
    pad_left: usize,
) -> Result<Option<Tensor>> {
    #[cfg(target_vendor = "apple")]
    {
        let (batch, cin, len) = x.dims3()?;
        let out = w_tap.dim(0)?;
        if batch != 1 || !x.device().is_cpu() || x.dtype() != candle_core::DType::F32 {
            return Ok(None);
        }
        // 512 columns: the tile, its GEMM operand and the output slice stay in L2. Measured on a
        // quiet M4, Kokoro decoder, one thread: whole-sequence im2col 1.09 s, 256 0.93, 512 0.91,
        // 1024 0.97, 4096 1.29. One GEMM per tap from a single padded copy instead: 0.95.
        let tile = 512;
        let bias = match b {
            Some(b) => b.flatten_all()?.contiguous()?,
            None => Tensor::zeros(out, candle_core::DType::F32, x.device())?,
        };
        // A transposed view of a contiguous `[k * cin, out]` weight (how Kokoro's decoder keeps
        // them) is read in place with a transposed GEMM; copying it was a seventh of the decoder.
        let (w, trans) = if w_tap.is_contiguous() {
            (w_tap.clone(), false)
        } else if w_tap.t()?.is_contiguous() {
            (w_tap.t()?, true)
        } else {
            (w_tap.contiguous()?, false)
        };
        let op = ConvTiled {
            cin,
            len,
            out,
            k,
            dilation,
            pad_left,
            tile,
            trans,
        };
        Ok(Some(x.contiguous()?.apply_op3_no_bwd(&w, &bias, &op)?))
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (x, w_tap, b, k, dilation, pad_left);
        Ok(None)
    }
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

    /// Against double precision, on fixed scores: candle's own f32 softmax rounds differently
    /// (its sums run in another order), and an unseeded comparison with it failed one draw in a few.
    #[test]
    fn softmax_is_accurate() {
        let (rows, n) = (111, 151);
        let mut state = 3u64;
        let x: Vec<f32> = (0..rows * n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * 12.0
            })
            .collect();
        let got = softmax_last_dim(&Tensor::from_slice(&x, (rows, n), &Device::Cpu).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mut worst = 0f64;
        for (row, out) in x.chunks(n).zip(got.chunks(n)) {
            let m = row.iter().fold(f64::MIN, |a, &v| a.max(v as f64));
            let e: Vec<f64> = row.iter().map(|&v| (v as f64 - m).exp()).collect();
            let s: f64 = e.iter().sum();
            for (&o, &ev) in out.iter().zip(&e) {
                worst = worst.max((o as f64 - ev / s).abs());
            }
        }
        assert!(worst < 5e-7, "softmax {worst:.3e}");
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
    fn conv_tiled_matches_im2col() {
        let dev = Device::Cpu;
        for (cin, out, len, k, d) in [
            (8, 6, 50, 3, 1),
            (16, 16, 3000, 7, 3),
            (32, 24, 2500, 11, 5),
            (5, 3, 9, 1, 1),
        ] {
            let x = Tensor::randn(0f32, 1.0, (1, cin, len), &dev).unwrap();
            let w = Tensor::randn(0f32, 0.2, (out, cin, k), &dev).unwrap();
            let b = Tensor::randn(0f32, 0.5, out, &dev).unwrap();
            let wt = crate::tap_major_weight(&w).unwrap();
            for pad_left in [(k - 1) * d / 2, 0, (k - 1) * d] {
                let Some(got) = conv_tiled(&x, &wt, Some(&b), k, d, pad_left).unwrap() else {
                    return;
                };
                let wtt = wt.t().unwrap().contiguous().unwrap().t().unwrap();
                let got_t = conv_tiled(&x, &wtt, Some(&b), k, d, pad_left)
                    .unwrap()
                    .unwrap();
                assert!(max_diff(&got, &got_t) < 1e-4);
                let want =
                    crate::conv1d_gemm_whole(&x, &wt, Some(&b), k, d, Some(pad_left)).unwrap();
                let diff = max_diff(&got, &want);
                assert!(
                    diff < 1e-4,
                    "{cin}->{out} @{len} k{k} d{d} pad {pad_left}: {diff}"
                );
            }
        }
    }

    #[test]
    fn adain_snake_matches_composed() {
        let dev = Device::Cpu;
        let (c, len) = (12, 5000);
        let x = Tensor::randn(0.2f32, 1.5, (1, c, len), &dev).unwrap();
        let r = |m: f32, s: f32| Tensor::randn(m, s, c, &dev).unwrap();
        let (mean, var, gamma, beta) = (
            r(0.2, 0.1),
            r(2.0, 0.3).abs().unwrap(),
            r(0.0, 0.3),
            r(0.0, 0.3),
        );
        let (alpha, br) = (r(1.0, 0.3), r(1.0, 0.3));
        let packed = Tensor::stack(&[&mean, &var, &gamma, &beta, &alpha, &br], 0).unwrap();
        let got = adain_snake(&x, &packed, 1e-5).unwrap();
        let y = crate::fused::adain_apply(&x, &mean, &var, &gamma, &beta, 1e-5).unwrap();
        let u = y.broadcast_mul(&alpha.reshape((1, c, 1)).unwrap()).unwrap();
        let want = y
            .broadcast_add(
                &u.sin()
                    .unwrap()
                    .sqr()
                    .unwrap()
                    .broadcast_mul(&br.reshape((1, c, 1)).unwrap())
                    .unwrap(),
            )
            .unwrap();
        let diff = max_diff(&got, &want);
        assert!(diff < 2e-5, "{diff}");
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
