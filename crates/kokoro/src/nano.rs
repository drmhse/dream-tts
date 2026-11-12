//! nano-edge-tts: a 1.75M-parameter stand-in for Kokoro's 53M decoder (huggingface.co/drmhse/nano-edge-tts).
//!
//! Same inputs, same output length. A mapper (80 frames/s) and a renderer (160 frames/s) set a gain and
//! rotation per bin of a pulse train's 25 ms STFT and a gain per bin of white noise's 5 ms STFT; two
//! inverse STFTs make the audio. Nothing learned runs at audio rate, so the signal work stays on the host
//! and the device only sees frame-rate matmuls. Everything is time-major `[1, T, C]`, the layout the
//! norms and linears want, so the network never transposes.

use crate::blocks::Linear;
use crate::source::{interpolate_linear, Draws};
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use serde::Deserialize;
use std::path::Path;
use tts_nn::Weights;

const SR: f32 = 24_000.0;
/// Samples per Kokoro F0 frame (80 frames/s).
const UP: usize = 300;
const VOICED_HZ: f32 = 10.0;
const EPS: f64 = 1e-5;
pub const DIR: &str = "nano";
const WEIGHTS: &str = "nano-edge-tts.safetensors";

#[derive(Deserialize)]
struct MapperCfg {
    dim: usize,
    depth: usize,
    kernel: usize,
}

#[derive(Deserialize)]
struct RendererCfg {
    n_fft: usize,
    hop: usize,
    depth: usize,
    noise_nfft: usize,
    creak: f32,
}

#[derive(Deserialize)]
struct Cfg {
    mapper: MapperCfg,
    renderer: RendererCfg,
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

/// ConvNeXt block. The depthwise conv is k shifted multiply-adds: candle runs a grouped conv as one
/// conv per group, 128 dispatches where this is 14.
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
        let t = x.dim(1)?;
        let p = self.k / 2;
        let padded = x.pad_with_zeros(1, p, p)?;
        let mut h = self
            .dw_bias
            .reshape((1, 1, self.dim))?
            .broadcast_as(x.shape())?
            .contiguous()?;
        for j in 0..self.k {
            let tap = self.taps.narrow(0, j, 1)?.unsqueeze(0)?;
            h = (h + padded.narrow(1, j, t)?.broadcast_mul(&tap)?)?;
        }
        let h = if self.instance {
            let mean = h.mean_keepdim(1)?;
            let c = h.broadcast_sub(&mean)?;
            c.broadcast_div(&(c.sqr()?.mean_keepdim(1)? + EPS)?.sqrt()?)?
        } else {
            tts_nn::layer_norm_plain(&h, EPS)?
        };
        let (g, b) = film(&self.fc, s, self.dim)?;
        let h = h.broadcast_mul(&g)?.broadcast_add(&b)?;
        let h = self.down.apply(&self.up.apply(&h)?.gelu_erf()?)?;
        Ok((x + h.broadcast_mul(&self.scale.reshape((1, 1, self.dim))?)?)?)
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
        let x = tts_nn::layer_norm_plain(&x, EPS)?
            .broadcast_mul(&g)?
            .broadcast_add(&b)?;
        self.head.apply(&x)
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
    fn inverse(&self, re: &Tensor, im: &Tensor, length: usize) -> Result<Vec<f32>> {
        let frames = re.dim(0)?;
        let rows: Vec<f32> = Tensor::cat(&[re, im], 1)?
            .matmul(&self.inv)?
            .flatten_all()?
            .to_vec1()?;
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
fn pulses(f: &[f32], on: impl Fn(usize) -> bool) -> Vec<f32> {
    let mut acc = 0f64;
    f.iter()
        .enumerate()
        .map(|(i, &fi)| {
            let step = if on(i) { fi / SR } else { 0.0 };
            acc += step as f64;
            let phase = acc as f32;
            let theta = 2.0 * std::f32::consts::PI * (phase - phase.floor());
            if on(i) {
                dirichlet(theta, (SR / 2.0 / fi).floor().clamp(1.0, 240.0)) * 0.1
            } else {
                0.0
            }
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

/// Slow irregular pulses where the pitch track is unvoiced: creaky voice. The irregularity is white
/// noise brick-walled at 25 Hz and scaled to 300 cents rms.
pub fn creak_train(f: &[f32], rate: f32, noise: Vec<f32>) -> Vec<f32> {
    use rustfft::{num_complex::Complex, FftPlanner};
    let len = f.len();
    let mut buf: Vec<Complex<f32>> = noise.into_iter().map(|v| Complex::new(v, 0.0)).collect();
    let mut planner = FftPlanner::<f32>::new();
    planner.plan_fft_forward(len).process(&mut buf);
    let keep = (25.0 * len as f64 / SR as f64) as usize + 1;
    // One-sided cut, mirrored, so the inverse stays real as irfft's does.
    for (k, v) in buf.iter_mut().enumerate() {
        if k != 0 && k.min(len - k) >= keep {
            *v = Complex::new(0.0, 0.0);
        }
    }
    planner.plan_fft_inverse(len).process(&mut buf);
    let x: Vec<f32> = buf.iter().map(|c| c.re / len as f32).collect();
    let rms = (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / len as f64).sqrt() as f32;
    let fc: Vec<f32> = x
        .iter()
        .map(|v| rate * (v * 300.0 / (rms + 1e-9) / 1200.0).exp2())
        .collect();
    pulses(&fc, |i| f[i] <= VOICED_HZ)
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
    h_scale: f32,
    n_scale: f32,
}

impl Nano {
    /// From `root/nano/`: `config.json` and the weights, f16 or f32.
    pub fn load(root: &Path, device: &Device) -> Result<Self> {
        let dir = root.join(DIR);
        let cfg: Cfg = serde_json::from_slice(
            &std::fs::read(dir.join("config.json")).context("nano config")?,
        )?;
        let w = Weights::load(dir.join(WEIGHTS).to_str().context("nano path")?, device)?;
        let r = &cfg.renderer;
        let mapper = Trunk::load(&w, "mapper", cfg.mapper.depth, true)?;
        anyhow::ensure!(
            mapper.dim == cfg.mapper.dim && mapper.blocks[0].k == cfg.mapper.kernel,
            "nano mapper shape"
        );
        let stft = Stft::new(r.n_fft, r.hop, device)?;
        let wsum: f32 = stft.window.iter().sum();
        let wsq: f32 = stft.window.iter().map(|v| v * v).sum();
        Ok(Self {
            mapper,
            mean: w.get("mapper.mean")?,
            std: w.get("mapper.std")?,
            renderer: Trunk::load(&w, "renderer", r.depth, false)?,
            nstft: Stft::new(r.noise_nfft, r.noise_nfft / 4, device)?,
            sub: r.hop / (r.noise_nfft / 4),
            creak: r.creak,
            h_scale: 0.05 * wsum,
            n_scale: 0.1 * wsq.sqrt(),
            stft,
        })
    }

    /// Kokoro's decoder call: `asr` `[1, 512, T]`, `f0_curve` and `energy` `[1, 2T]`, `s` `[1, 128]`.
    pub fn forward(
        &self,
        asr: &Tensor,
        f0_curve: &Tensor,
        energy: &Tensor,
        s: &Tensor,
        draws: &mut (dyn Draws + Send),
    ) -> Result<Vec<f32>> {
        Ok(self.render(asr, f0_curve, energy, s, draws)?.audio)
    }

    pub fn creak_rate(&self) -> f32 {
        self.creak
    }

    pub fn render(
        &self,
        asr: &Tensor,
        f0_curve: &Tensor,
        energy: &Tensor,
        s: &Tensor,
        draws: &mut (dyn Draws + Send),
    ) -> Result<Rendered> {
        let dev = asr.device();
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

        // Mapper at 80 frames/s: asr repeated to the F0 rate (nearest), then prosody.
        let a = asr.transpose(1, 2)?.contiguous()?;
        let idx = Tensor::from_vec(nearest(a.dim(1)?, m), m, dev)?;
        let x = Tensor::cat(&[&a.index_select(&idx, 1)?, &pros], 2)?;
        let mapped = self
            .mapper
            .apply(&x, s)?
            .broadcast_mul(&self.std)?
            .broadcast_add(&self.mean)?;

        // Metal runs ahead asynchronously, so the signals below overlap the mapper.
        let f = upsample(&f0, len);
        let pulse = pulse_train(&f);
        let noise: Vec<f32> = draws.randn(len).into_iter().map(|v| v * 0.1).collect();
        let creak = (self.creak > 0.0).then(|| creak_train(&f, self.creak, draws.randn(len)));

        let (hr, hi) = self.stft.forward(&pulse, dev)?;
        let (nr, ni) = self.nstft.forward(&noise, dev)?;
        let t = hr.dim(0)?;
        let logmag =
            ((((hr.sqr()? + hi.sqr()?)? + 1e-4)?.log()? * 0.5)? - (self.h_scale as f64).ln())?;
        let (hr, hi) = match &creak {
            Some(c) => {
                let (cr, ci) = self.stft.forward(c, dev)?;
                ((hr + cr)?, (hi + ci)?)
            }
            None => (hr, hi),
        };

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

        let b = self.stft.bins;
        let gain = |z: Tensor, scale: f32| -> Result<Tensor> {
            Ok(((z.clamp(f32::MIN, 8f32)? - 3.0)?.exp()? / scale as f64)?)
        };
        let gh = gain(h.narrow(1, 0, b)?, self.h_scale)?;
        let rot = h.narrow(1, b, b)?;
        let (c, sn) = (rot.cos()?, rot.sin()?);
        let re = ((&hr * &c)? - (&hi * &sn)?)?.mul(&gh)?;
        let im = ((&hr * &sn)? + (&hi * &c)?)?.mul(&gh)?;
        let harmonic = self.stft.inverse(&re, &im, len)?;

        // Noise gains are `sub` frames per network frame, sub-frame major: row t, column s * nb + j.
        let nb = self.nstft.bins;
        let frames = nr.dim(0)?;
        let g = gain(h.narrow(1, 2 * b, self.sub * nb)?, self.n_scale)?
            .reshape((t * self.sub, nb))?
            .narrow(0, self.sub / 2, frames)?;
        let noise = self.nstft.inverse(&nr.mul(&g)?, &ni.mul(&g)?, len)?;
        Ok(Rendered {
            audio: harmonic.iter().zip(&noise).map(|(a, b)| a + b).collect(),
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
    fn pulse_is_silent_where_unvoiced() {
        let f: Vec<f32> = (0..3000)
            .map(|i| if i < 1500 { 200.0 } else { 0.0 })
            .collect();
        let p = pulse_train(&f);
        assert!(p[..1500].iter().any(|v| v.abs() > 1.0));
        assert!(p[1500..].iter().all(|v| *v == 0.0));
    }
}
