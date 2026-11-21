//! The decoder half of nano-edge-tts-standalone (huggingface.co/drmhse/nano-edge-tts-standalone): the
//! text side's features, F0 and energy in, audio out. A mapper (80 frames/s) and a renderer (160 frames/s) set a gain and
//! rotation per bin of a pulse train's 25 ms STFT and a gain per bin of white noise's 5 ms STFT; two
//! inverse STFTs make the audio. Nothing learned runs at audio rate, so the signal work stays on the host
//! and the device only sees frame-rate matmuls. Everything is time-major `[1, T, C]`, the layout the
//! norms and linears want, so the network never transposes.

use crate::blocks::Linear;
use crate::source::{interpolate_linear, Draws};
use anyhow::Result;
use candle_core::{Device, Tensor};
use serde::Deserialize;
use tts_nn::Weights;

const SR: f32 = 24_000.0;
/// Samples per Kokoro F0 frame (80 frames/s).
const UP: usize = 300;
const VOICED_HZ: f32 = 10.0;
const EPS: f64 = 1e-5;

#[derive(Deserialize)]
pub struct MapperCfg {
    dim: usize,
    depth: usize,
    kernel: usize,
}

#[derive(Deserialize)]
pub struct RendererCfg {
    n_fft: usize,
    hop: usize,
    depth: usize,
    noise_nfft: usize,
    creak: f32,
    /// The creak's harmonics stop here (Hz).
    creak_hi: f32,
    /// The harmonic log gain above 2 kHz is averaged over this many bins.
    harm_fsmooth: usize,
    /// The post-filter: (alpha, from Hz, bins, frames). Above `from` the noise's log gain is pushed
    /// away from its mean over `bins` bins by alpha, the push averaged over `frames` noise frames.
    noise_sharpen: Option<[f32; 4]>,
}

/// A 1x1 conv as the linear it is, on time-major input.
fn pointwise(w: &Weights, prefix: &str) -> Result<Linear> {
    let k = w
        .cpu(&format!("{prefix}.weight"))?
        .to_dtype(candle_core::DType::F32)?;
    let k = k.squeeze(2)?.t()?.contiguous()?.to_device(w.device())?;
    Ok(Linear::from_parts(
        k,
        Some(w.get(&format!("{prefix}.bias"))?),
    ))
}

/// Gain and bias from the voice: `[1, 1, C]` each.
fn film(fc: &Linear, s: &Tensor, dim: usize) -> Result<(Tensor, Tensor)> {
    let gb = fc.apply(s)?.unsqueeze(1)?;
    Ok(((gb.narrow(2, 0, dim)? + 1.0)?, gb.narrow(2, dim, dim)?))
}

/// ConvNeXt block. The depthwise conv is k shifted multiply-adds (candle runs a grouped conv as one
/// conv per group): one pass on the CPU, 14 dispatches on Metal.
struct Block {
    taps: Tensor,
    dw_bias: Tensor,
    fc: Linear,
    up: Linear,
    down: Linear,
    scale: Tensor,
    k: usize,
    dim: usize,
    /// Statistics over time (the mapper, as Kokoro's decoder) rather than over channels.
    instance: bool,
}

impl Block {
    fn load(w: &Weights, prefix: &str, instance: bool) -> Result<Self> {
        let dw = w.get(&format!("{prefix}.dw.weight"))?; // [C, 1, k]
        let (dim, _, k) = dw.dims3()?;
        Ok(Self {
            taps: dw.squeeze(1)?.t()?.contiguous()?, // [k, C]
            dw_bias: w.get(&format!("{prefix}.dw.bias"))?,
            fc: Linear::load(w, &format!("{prefix}.norm.fc"))?,
            up: Linear::load(w, &format!("{prefix}.up"))?,
            down: Linear::load(w, &format!("{prefix}.down"))?,
            scale: w.get(&format!("{prefix}.scale"))?,
            k,
            dim,
            instance,
        })
    }

    fn apply(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let h = tts_nn::cpu::depthwise_tc(x, &self.taps, &self.dw_bias)?;
        let (g, b) = film(&self.fc, s, self.dim)?;
        let h = tts_nn::cpu::norm_film(&h, &g, &b, self.instance, EPS)?;
        let h = self.down.apply(&tts_nn::gelu_exact(&self.up.apply(&h)?)?)?;
        Ok(tts_nn::cpu::scaled_residual(x, &h, &self.scale)?)
    }
}

struct Trunk {
    inp: Linear,
    blocks: Vec<Block>,
    fc: Linear,
    head: Linear,
    dim: usize,
}

impl Trunk {
    fn load(w: &Weights, prefix: &str, depth: usize, instance: bool) -> Result<Self> {
        let blocks = (0..depth)
            .map(|i| Block::load(w, &format!("{prefix}.blocks.{i}"), instance))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            inp: pointwise(w, &format!("{prefix}.inp"))?,
            dim: blocks[0].dim,
            blocks,
            fc: Linear::load(w, &format!("{prefix}.norm.fc"))?,
            head: Linear::load(w, &format!("{prefix}.head"))?,
        })
    }

    fn apply(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let mut x = self.inp.apply(x)?;
        for b in &self.blocks {
            x = b.apply(&x, s)?;
        }
        let (g, b) = film(&self.fc, s, self.dim)?;
        self.head
            .apply(&tts_nn::cpu::norm_film(&x, &g, &b, false, EPS)?)
    }
}

/// STFT and its inverse as matmuls against one table each, the framing and overlap-add on the host.
struct Stft {
    n_fft: usize,
    hop: usize,
    bins: usize,
    window: Vec<f32>,
    /// `[n_fft, 2 * bins]`: windowed cos | -sin.
    fwd: Tensor,
    /// `[2 * bins, n_fft]`: the one-sided inverse DFT, windowed.
    inv: Tensor,
}

impl Stft {
    fn new(n_fft: usize, hop: usize, device: &Device) -> Result<Self> {
        let bins = n_fft / 2 + 1;
        let window: Vec<f32> = (0..n_fft)
            .map(|i| {
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos()) as f32
            })
            .collect();
        let mut fwd = vec![0f32; n_fft * 2 * bins];
        let mut inv = vec![0f32; 2 * bins * n_fft];
        for k in 0..bins {
            let pair = if k == 0 || k == bins - 1 { 1.0 } else { 2.0 };
            for n in 0..n_fft {
                // The angle rounded through f32 as the training code's table was: at k * n near 2e5 one
                // f32 ulp of it is 0.06 rad, and the network learned against that table, not the exact one.
                let a = ((2 * k * n) as f32 * std::f32::consts::PI / n_fft as f32) as f64;
                let (c, s, w) = (a.cos(), a.sin(), window[n] as f64);
                fwd[n * 2 * bins + k] = (c * w) as f32;
                fwd[n * 2 * bins + bins + k] = (-s * w) as f32;
                inv[k * n_fft + n] = (c * pair / n_fft as f64 * w) as f32;
                inv[(bins + k) * n_fft + n] = (-s * pair / n_fft as f64 * w) as f32;
            }
        }
        Ok(Self {
            fwd: Tensor::from_vec(fwd, (n_fft, 2 * bins), device)?,
            inv: Tensor::from_vec(inv, (2 * bins, n_fft), device)?,
            n_fft,
            hop,
            bins,
            window,
        })
    }

    /// `x` -> (re, im), each `[frames, bins]`; centred, reflect padding.
    fn forward(&self, x: &[f32], device: &Device) -> Result<(Tensor, Tensor)> {
        let p = self.n_fft / 2;
        let len = x.len();
        let frames = len / self.hop + 1;
        let at = |i: isize| -> f32 {
            let i = if i < 0 {
                -i
            } else if i >= len as isize {
                2 * (len as isize - 1) - i
            } else {
                i
            };
            x[i as usize]
        };
        let mut f = vec![0f32; frames * self.n_fft];
        for (t, row) in f.chunks_mut(self.n_fft).enumerate() {
            let start = (t * self.hop) as isize - p as isize;
            for (n, v) in row.iter_mut().enumerate() {
                *v = at(start + n as isize);
            }
        }
        let y = Tensor::from_vec(f, (frames, self.n_fft), device)?.matmul(&self.fwd)?;
        Ok((
            y.narrow(1, 0, self.bins)?,
            y.narrow(1, self.bins, self.bins)?,
        ))
    }

    /// (re, im) `[frames, bins]` -> `length` samples, normalised by the summed squared window.
    #[cfg(test)]
    fn inverse(&self, re: &Tensor, im: &Tensor, length: usize) -> Result<Vec<f32>> {
        self.inverse_rows(&Tensor::cat(&[re, im], 1)?, length)
    }

    /// The same from `[frames, 2 * bins]` rows already laid out as re | im.
    fn inverse_rows(&self, spec: &Tensor, length: usize) -> Result<Vec<f32>> {
        let frames = spec.dim(0)?;
        let rows: Vec<f32> = spec.matmul(&self.inv)?.flatten_all()?.to_vec1()?;
        let total = (frames - 1) * self.hop + self.n_fft;
        let (mut y, mut env) = (vec![0f32; total], vec![0f32; total]);
        for (t, row) in rows.chunks(self.n_fft).enumerate() {
            let o = t * self.hop;
            for n in 0..self.n_fft {
                y[o + n] += row[n];
                env[o + n] += self.window[n] * self.window[n];
            }
        }
        let p = self.n_fft / 2;
        Ok((p..p + length).map(|i| y[i] / env[i].max(1e-8)).collect())
    }
}

/// `sum cos(k theta)` for `k = 1..K` in closed form: a band-limited pulse at a cost independent of K.
fn dirichlet(theta: f32, k: f32) -> f32 {
    let half = (theta / 2.0).sin();
    if half.abs() < 1e-3 {
        k
    } else {
        ((k + 0.5) * theta).sin() / (2.0 * half) - 0.5
    }
}

/// The F0 track (80 frames/s) to one value per sample, as `F.interpolate(scale_factor=300, mode="linear")`.
/// With a scale factor torch's index is f32 and fused, `fma(1/300, i + 0.5, -0.5)`, unlike the size-based
/// path `source::interpolate_linear` matches. The phase integrates this, so the f64 index drifted the
/// pulses by 0.016 cycles over five seconds.
pub fn upsample(f0: &[f32], len: usize) -> Vec<f32> {
    let scale = (1.0f32 / UP as f32) as f64;
    let last = f0.len() - 1;
    (0..len)
        .map(|i| {
            let src = ((scale * (i as f64 + 0.5) - 0.5) as f32).max(0.0);
            let lo = (src.floor() as usize).min(last);
            let l = src - lo as f32;
            (1.0 - l) * f0[lo] + l * f0[(lo + 1).min(last)]
        })
        .collect()
}

/// torch's `F.interpolate(mode="nearest")` indices, computed in f32 as its kernel does.
fn nearest(from: usize, to: usize) -> Vec<u32> {
    let scale = from as f32 / to as f32;
    (0..to)
        .map(|i| ((i as f32 * scale).floor() as usize).min(from - 1) as u32)
        .collect()
}

/// One pulse per cycle of `f` (Hz per sample) where `on`, zero elsewhere. The phase accumulates in f64
/// and is stored as f32 per sample, as torch's CPU cumsum does.
fn pulses(f: &[f32], hi: f32, on: impl Fn(usize) -> bool) -> Vec<f32> {
    let mut acc = 0f64;
    f.iter()
        .enumerate()
        .map(|(i, &fi)| {
            let step = if on(i) { fi / SR } else { 0.0 };
            acc += step as f64;
            let phase = acc as f32;
            let theta = 2.0 * std::f32::consts::PI * (phase - phase.floor());
            if on(i) {
                dirichlet(theta, (hi / fi).floor().clamp(1.0, 240.0)) * 0.1
            } else {
                0.0
            }
        })
        .collect()
}

/// Moving average over `k` neighbours (edges average what exists): `avg_pool1d(count_include_pad=False)`.
fn pool(x: &[f32], k: usize) -> Vec<f32> {
    let (n, r) = (x.len(), k / 2);
    let mut acc = vec![0f64; n + 1];
    for (i, v) in x.iter().enumerate() {
        acc[i + 1] = acc[i] + *v as f64;
    }
    (0..n)
        .map(|i| {
            let (lo, hi) = (i.saturating_sub(r), (i + r + 1).min(n));
            ((acc[hi] - acc[lo]) / (hi - lo) as f64) as f32
        })
        .collect()
}

/// Every harmonic of the per-sample pitch below Nyquist. Unvoiced samples still advance the phase at
/// 100 Hz, as the reference does, so a pulse lands where it would have.
pub fn pulse_train(f: &[f32]) -> Vec<f32> {
    let g: Vec<f32> = f
        .iter()
        .map(|&x| if x > VOICED_HZ { x } else { 100.0 })
        .collect();
    let mut acc = 0f64;
    g.iter()
        .zip(f)
        .map(|(&gi, &fi)| {
            acc += (gi / SR) as f64;
            let phase = acc as f32;
            let theta = 2.0 * std::f32::consts::PI * (phase - phase.floor());
            if fi > VOICED_HZ {
                dirichlet(theta, (SR / 2.0 / gi).floor().clamp(1.0, 240.0)) * 0.1
            } else {
                0.0
            }
        })
        .collect()
}

/// Slow irregular pulses where the pitch track is unvoiced, harmonics up to `hi` Hz: creaky voice. The
/// irregularity is white noise brick-walled at 25 Hz and scaled to 300 cents rms. The reference draws
/// that noise from a fixed torch seed; torch's generator is not reproduced, so a render draws its own.
pub fn creak_train(f: &[f32], rate: f32, hi: f32, noise: Vec<f32>) -> Vec<f32> {
    let len = f.len();
    let keep = (25.0 * len as f64 / SR as f64) as usize + 1;
    let x = if len % 2 == 0 {
        brickwall_real(&noise, keep)
    } else {
        brickwall_complex(noise, keep)
    };
    let rms = (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / len as f64).sqrt() as f32;
    let fc: Vec<f32> = x
        .iter()
        .map(|v| rate * (v * 300.0 / (rms + 1e-9) / 1200.0).exp2())
        .collect();
    pulses(&fc, hi, |i| f[i] <= VOICED_HZ)
}

/// `exp(min(z, 8) - 3) / scale`, the renderer's gain.
fn gain(z: f32, scale: f32) -> f32 {
    tts_nn::cpu::exp(z.min(8.0) - 3.0) / scale
}

/// Real noise brick-walled to its first `keep` bins: the full complex FFT and back.
fn brickwall_complex(noise: Vec<f32>, keep: usize) -> Vec<f32> {
    use rustfft::{num_complex::Complex, FftPlanner};
    let len = noise.len();
    let mut buf: Vec<Complex<f32>> = noise.into_iter().map(|v| Complex::new(v, 0.0)).collect();
    let mut planner = FftPlanner::<f32>::new();
    planner.plan_fft_forward(len).process(&mut buf);
    // One-sided cut, mirrored, so the inverse stays real as irfft's does.
    for (k, v) in buf.iter_mut().enumerate() {
        if k != 0 && k.min(len - k) >= keep {
            *v = Complex::new(0.0, 0.0);
        }
    }
    planner.plan_fft_inverse(len).process(&mut buf);
    buf.iter().map(|c| c.re / len as f32).collect()
}

/// The same for even lengths through a half-length complex FFT each way (even samples real, odd
/// imaginary): half the work of [`brickwall_complex`], which was a fifth of the decoder's CPU time.
fn brickwall_real(noise: &[f32], keep: usize) -> Vec<f32> {
    use rustfft::{num_complex::Complex, FftPlanner};
    let len = noise.len();
    let n = len / 2;
    let mut planner = FftPlanner::<f32>::new();
    let mut z: Vec<Complex<f32>> = noise
        .chunks_exact(2)
        .map(|p| Complex::new(p[0], p[1]))
        .collect();
    planner.plan_fft_forward(n).process(&mut z);
    let tw = |k: usize, sign: f64| {
        let a = sign * 2.0 * std::f64::consts::PI * k as f64 / len as f64;
        Complex::new(a.cos() as f32, a.sin() as f32)
    };
    // X[k] = E + W^k O for k < keep, zero above (X[n] at Nyquist is never kept).
    let kept = keep.min(n);
    let mut spec = vec![Complex::new(0.0f32, 0.0); n + 1];
    for k in 0..kept {
        let (zk, zc) = (z[k % n], z[(n - k) % n].conj());
        let e = (zk + zc) * 0.5;
        let o = (zk - zc) * Complex::new(0.0, -0.5);
        spec[k] = e + tw(k, -1.0) * o;
    }
    // Back: Z[k] = (X[k] + X*[n-k]) + i (X[k] - X*[n-k]) W^-k, and z's real and imaginary parts are the
    // even and odd samples.
    for (k, out) in z.iter_mut().enumerate() {
        let (xk, xc) = (spec[k], spec[n - k].conj());
        *out = (xk + xc) + Complex::new(0.0, 1.0) * (xk - xc) * tw(k, 1.0);
    }
    planner.plan_fft_inverse(n).process(&mut z);
    let scale = 1.0 / len as f32;
    z.iter()
        .flat_map(|c| [c.re * scale, c.im * scale])
        .collect()
}

/// The audio and the two network outputs the gate compares: the mapper's features `[1, 2T, 512]` and
/// the renderer's head `[1, frames, 907]`.
pub struct Rendered {
    pub audio: Vec<f32>,
    pub mapper: Tensor,
    pub head: Tensor,
}

pub struct Nano {
    mapper: Trunk,
    mean: Tensor,
    std: Tensor,
    renderer: Trunk,
    stft: Stft,
    nstft: Stft,
    /// Noise gains per network frame.
    sub: usize,
    creak: f32,
    creak_hi: f32,
    harm_fsmooth: usize,
    sharpen: Option<[f32; 4]>,
    h_scale: f32,
    n_scale: f32,
}

impl Nano {
    /// The `mapper.*` and `renderer.*` tensors of `w`, f16 or f32.
    pub fn load(w: &Weights, m: &MapperCfg, r: &RendererCfg, device: &Device) -> Result<Self> {
        let mapper = Trunk::load(w, "mapper", m.depth, true)?;
        anyhow::ensure!(
            mapper.dim == m.dim && mapper.blocks[0].k == m.kernel,
            "nano mapper shape"
        );
        let stft = Stft::new(r.n_fft, r.hop, device)?;
        let wsum: f32 = stft.window.iter().sum();
        let wsq: f32 = stft.window.iter().map(|v| v * v).sum();
        Ok(Self {
            mapper,
            mean: w.get("mapper.mean")?,
            std: w.get("mapper.std")?,
            renderer: Trunk::load(w, "renderer", r.depth, false)?,
            nstft: Stft::new(r.noise_nfft, r.noise_nfft / 4, device)?,
            sub: r.hop / (r.noise_nfft / 4),
            creak: r.creak,
            creak_hi: r.creak_hi,
            harm_fsmooth: r.harm_fsmooth,
            sharpen: r.noise_sharpen,
            h_scale: 0.05 * wsum,
            n_scale: 0.1 * wsq.sqrt(),
            stft,
        })
    }

    /// `feat` `[1, T, 512]` (time-major, 40 frames/s), `f0_curve` and `energy` `[1, 2T]`, `s` `[1, 128]`.
    pub fn forward(
        &self,
        feat: &Tensor,
        f0_curve: &Tensor,
        energy: &Tensor,
        s: &Tensor,
        draws: &mut (dyn Draws + Send),
    ) -> Result<Vec<f32>> {
        Ok(self.render(feat, f0_curve, energy, s, draws)?.audio)
    }

    /// The creak's rate and its harmonics' ceiling, in Hz.
    pub fn creak(&self) -> (f32, f32) {
        (self.creak, self.creak_hi)
    }

    /// The head applied on the host: gains and rotations straight into the `[re | im]` rows the inverse
    /// STFTs multiply, where the tensor form is a dozen whole-spectrum passes and a concatenation.
    fn tail_host(&self, h: &Tensor, spec: [&Tensor; 4], len: usize) -> Result<Vec<f32>> {
        let host = |t: &Tensor| -> Result<Vec<f32>> { Ok(t.flatten_all()?.to_vec1()?) };
        let (hv, hr, hi, nr, ni) = (
            host(h)?,
            host(spec[0])?,
            host(spec[1])?,
            host(spec[2])?,
            host(spec[3])?,
        );
        let (b, nb, sub) = (self.stft.bins, self.nstft.bins, self.sub);
        let (t, width) = (h.dim(0)?, h.dim(1)?);
        let dev = h.device();

        // The bin 2 kHz falls in, as the reference's `int(2000 * n_fft / SR)`.
        let i0 = (2000 * self.stft.n_fft) / SR as usize;
        let mut rows = vec![0f32; t * 2 * b];
        for i in 0..t {
            let hrow = &hv[i * width..];
            let smooth = (self.harm_fsmooth > 1).then(|| pool(&hrow[i0..b], self.harm_fsmooth));
            let out = &mut rows[i * 2 * b..(i + 1) * 2 * b];
            for k in 0..b {
                let z = match &smooth {
                    Some(s) if k >= i0 => s[k - i0],
                    _ => hrow[k],
                };
                let g = gain(z, self.h_scale);
                let (sn, c) = hrow[b + k].sin_cos();
                let (r, m) = (hr[i * b + k], hi[i * b + k]);
                out[k] = (r * c - m * sn) * g;
                out[b + k] = (r * sn + m * c) * g;
            }
        }
        let harmonic = self
            .stft
            .inverse_rows(&Tensor::from_vec(rows, (t, 2 * b), dev)?, len)?;

        // Noise gains are `sub` frames per network frame, sub-frame major; frame f reads sub-frame f + sub / 2.
        let frames = nr.len() / nb;
        let mut g = vec![0f32; frames * nb];
        for f in 0..frames {
            let r = f + sub / 2;
            let hrow = &hv[(r / sub) * width + 2 * b + (r % sub) * nb..];
            for j in 0..nb {
                g[f * nb + j] = gain(hrow[j], self.n_scale);
            }
        }
        if let Some(sharpen) = self.sharpen {
            self.sharpen_noise(&mut g, frames, sharpen);
        }
        let mut rows = vec![0f32; frames * 2 * nb];
        for f in 0..frames {
            let out = &mut rows[f * 2 * nb..(f + 1) * 2 * nb];
            for j in 0..nb {
                out[j] = nr[f * nb + j] * g[f * nb + j];
                out[nb + j] = ni[f * nb + j] * g[f * nb + j];
            }
        }
        let noise = self
            .nstft
            .inverse_rows(&Tensor::from_vec(rows, (frames, 2 * nb), dev)?, len)?;
        Ok(harmonic.iter().zip(&noise).map(|(a, b)| a + b).collect())
    }

    /// The post-filter on the noise gains `[frames, nb]`: deeper high-band formants.
    fn sharpen_noise(&self, g: &mut [f32], frames: usize, [alpha, from, bins, span]: [f32; 4]) {
        let nb = self.nstft.bins;
        let lg: Vec<f32> = g.iter().map(|v| (v + 1e-12).ln()).collect();
        let mut d = vec![0f32; frames * nb];
        for f in 0..frames {
            let row = &lg[f * nb..(f + 1) * nb];
            for (j, m) in pool(row, bins as usize).into_iter().enumerate() {
                d[f * nb + j] = row[j] - m;
            }
        }
        let first = (0..nb)
            .find(|&j| j as f32 * SR / self.nstft.n_fft as f32 >= from)
            .unwrap_or(nb);
        let mut col = vec![0f32; frames];
        for j in 0..nb {
            if j >= first && span as usize > 1 {
                for f in 0..frames {
                    col[f] = d[f * nb + j];
                }
                col = pool(&col, span as usize);
            } else if j >= first {
                for f in 0..frames {
                    col[f] = d[f * nb + j];
                }
            }
            for f in 0..frames {
                let push = if j >= first {
                    (alpha - 1.0) * col[f]
                } else {
                    0.0
                };
                g[f * nb + j] = (lg[f * nb + j] + push).exp();
            }
        }
    }

    pub fn render(
        &self,
        feat: &Tensor,
        f0_curve: &Tensor,
        energy: &Tensor,
        s: &Tensor,
        draws: &mut (dyn Draws + Send),
    ) -> Result<Rendered> {
        let dev = feat.device();
        let f0: Vec<f32> = f0_curve.flatten_all()?.to_vec1()?;
        let en: Vec<f32> = energy.flatten_all()?.to_vec1()?;
        let m = f0.len();
        let len = m * UP;
        let pros: Vec<f32> = (0..m)
            .flat_map(|i| {
                [
                    f0[i].max(0.0).ln_1p() / 6.0,
                    (f0[i] > VOICED_HZ) as u8 as f32,
                    en[i] / 4.0,
                ]
            })
            .collect();
        let pros = Tensor::from_vec(pros, (1, m, 3), dev)?;

        // Mapper at 80 frames/s: the features repeated to the F0 rate (nearest), then prosody.
        let a = feat.contiguous()?;
        let idx = Tensor::from_vec(nearest(a.dim(1)?, m), m, dev)?;
        let x = Tensor::cat(&[&a.index_select(&idx, 1)?, &pros], 2)?;
        let mapper = || -> Result<Tensor> {
            Ok(self
                .mapper
                .apply(&x, s)?
                .broadcast_mul(&self.std)?
                .broadcast_add(&self.mean)?)
        };
        // Everything else the renderer reads, none of it from the mapper.
        let signals = |draws: &mut (dyn Draws + Send)| -> Result<[Tensor; 5]> {
            let f = upsample(&f0, len);
            let pulse = pulse_train(&f);
            let noise: Vec<f32> = draws.randn(len).into_iter().map(|v| v * 0.1).collect();
            let creak = (self.creak > 0.0)
                .then(|| creak_train(&f, self.creak, self.creak_hi, draws.randn(len)));
            let (hr, hi) = self.stft.forward(&pulse, dev)?;
            let (nr, ni) = self.nstft.forward(&noise, dev)?;
            let logmag =
                ((((hr.sqr()? + hi.sqr()?)? + 1e-4)?.log()? * 0.5)? - (self.h_scale as f64).ln())?;
            let (hr, hi) = match &creak {
                Some(c) => {
                    let (cr, ci) = self.stft.forward(c, dev)?;
                    ((hr + cr)?, (hi + ci)?)
                }
                None => (hr, hi),
            };
            Ok([hr, hi, nr, ni, logmag])
        };
        // With CPU threads to spare the two run side by side. Metal runs ahead asynchronously, so there the
        // signals already overlap the mapper; one thread keeps them in order.
        let (mapped, [hr, hi, nr, ni, logmag]) =
            if dev.is_cpu() && candle_core::utils::get_num_threads() > 1 {
                std::thread::scope(|sc| -> Result<_> {
                    let side = sc.spawn(|| signals(draws));
                    let mapped = mapper()?;
                    let sig = side
                        .join()
                        .map_err(|_| anyhow::anyhow!("nano signal thread panicked"))??;
                    Ok((mapped, sig))
                })?
            } else {
                let mapped = mapper()?;
                (mapped, signals(draws)?)
            };
        let t = hr.dim(0)?;

        // Renderer at 160 frames/s: features nearest, prosody linear (torch's index conventions).
        let feat = mapped.index_select(&Tensor::from_vec(nearest(m, t), t, dev)?, 1)?;
        let pros_t: Vec<f32> = pros
            .squeeze(0)?
            .t()?
            .contiguous()?
            .flatten_all()?
            .to_vec1()?;
        let lin = interpolate_linear(&pros_t, 3, t);
        let lin = Tensor::from_vec(lin, (3, t), dev)?.t()?.unsqueeze(0)?;
        let x = Tensor::cat(&[&feat, &lin, &(logmag.unsqueeze(0)? / 4.0)?], 2)?;
        let head = self.renderer.apply(&x, s)?;
        let h = head.squeeze(0)?; // [t, 2 * bins + sub * nbins]
                                  // The smoothing and the post-filter want whole rows and columns of the head, so the tail runs
                                  // on the host on every device.
        let audio = self.tail_host(&h, [&hr, &hi, &nr, &ni], len)?;
        Ok(Rendered {
            audio,
            mapper: mapped,
            head,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stft_round_trip() {
        let dev = Device::Cpu;
        for (n, h) in [(600, 150), (120, 30)] {
            let s = Stft::new(n, h, &dev).unwrap();
            let x: Vec<f32> = (0..9000)
                .map(|i| ((i as f32) * 0.013).sin() + ((i * 7 % 13) as f32 - 6.0) * 0.01)
                .collect();
            let (re, im) = s.forward(&x, &dev).unwrap();
            let y = s.inverse(&re, &im, x.len()).unwrap();
            let err = x
                .iter()
                .zip(&y)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(err < 1e-4, "{n}/{h}: {err}");
        }
    }

    #[test]
    fn real_brickwall_matches_complex() {
        let mut state = 9u64;
        for len in [300 * 37, 300 * 401, 600] {
            let noise: Vec<f32> = (0..len)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
                })
                .collect();
            let keep = (25.0 * len as f64 / SR as f64) as usize + 1;
            let a = brickwall_real(&noise, keep);
            let b = brickwall_complex(noise, keep);
            let peak = b.iter().fold(0f32, |m, v| m.max(v.abs()));
            let err = a
                .iter()
                .zip(&b)
                .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
            assert!(
                err < 1e-5 * peak.max(1e-3),
                "{len}: {err} against peak {peak}"
            );
        }
    }

    #[test]
    fn pulse_is_silent_where_unvoiced() {
        let f: Vec<f32> = (0..3000)
            .map(|i| if i < 1500 { 200.0 } else { 0.0 })
            .collect();
        let p = pulse_train(&f);
        assert!(p[..1500].iter().any(|v| v.abs() > 1.0));
        assert!(p[1500..].iter().all(|v| *v == 0.0));
    }
}
