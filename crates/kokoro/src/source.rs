//! The harmonic-plus-noise excitation and the 20-point transforms around it.
//!
//! The excitation stays host-side f32: sequential f64 phase accumulation with
//! torch-matched blending, plus replayable draws the gate compares exactly.
//! The transforms moved to `tts_nn::stft` — one dispatch each with tabled
//! twiddles — after measurement showed the host pair costing tens of
//! milliseconds per utterance in trig calls alone. That refutes the old claim
//! here that a direct DFT costs less than its dispatch.

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::f32::consts::PI;

pub const HARMONICS: usize = 8;
const SINE_AMP: f32 = 0.1;
const NOISE_STD: f32 = 0.003;
const VOICED_THRESHOLD: f32 = 10.0;

/// torch's `F.interpolate(mode="linear", align_corners=False)`.
///
/// The source module downsamples the per-sample phase increments by the upsample factor
/// and immediately puts them back, which looks like a no-op and is not: it is what keeps
/// the phase continuous across a frame boundary. The half-pixel convention has to match,
/// or every harmonic drifts.
fn interpolate_linear(x: &[f32], channels: usize, out_len: usize) -> Vec<f32> {
    let in_len = x.len() / channels;
    // f64 throughout, and `(1-λ)a + λb` rather than `a + (b-a)λ`. Both match torch on
    // purpose: its CPU kernel accumulates float input in double, and the phase reaching
    // ~62,000 radians makes one ulp 0.004 — enough to move a harmonic audibly.
    let scale = in_len as f64 / out_len as f64;
    let mut out = vec![0f32; channels * out_len];
    for i in 0..out_len {
        let src = (scale * (i as f64 + 0.5) - 0.5).max(0.0);
        let lo = src.floor() as usize;
        let hi = (lo + 1).min(in_len - 1);
        let frac = src - lo as f64;
        // torch computes the source index and the two weights in double, then blends in
        // the tensor's own dtype. Blending in double instead is *more* accurate and does
        // not match: at a phase of 142,000 radians one ulp is 0.016, and three of them
        // move a harmonic.
        let (w0, w1) = ((1.0 - frac) as f32, frac as f32);
        for c in 0..channels {
            out[c * out_len + i] = w0 * x[c * in_len + lo] + w1 * x[c * in_len + hi];
        }
    }
    out
}

/// Random draws, in the order the reference makes them.
///
/// Replayable so the gate can compare the whole path: a stochastic excitation that is
/// merely "statistically similar" makes every downstream stage uncheckable.
pub trait Draws {
    /// `[1, channels]` initial phase offsets.
    fn rand(&mut self, n: usize) -> Vec<f32>;
    /// Standard normal, `n` values.
    fn randn(&mut self, n: usize) -> Vec<f32>;
}

/// `f0` is one value per output sample. Returns the merged excitation, `[samples]`.
pub fn excitation(
    f0: &[f32],
    linear_w: &[f32],
    linear_b: f32,
    upsample: usize,
    sample_rate: f32,
    draws: &mut (dyn Draws + Send),
) -> Vec<f32> {
    let n = f0.len();
    let dim = HARMONICS + 1;

    // Phase increment per sample for the fundamental and each overtone, channel-major.
    let mut rad = vec![0f32; dim * n];
    for (c, slot) in rad.chunks_mut(n).enumerate() {
        let mult = (c + 1) as f32;
        for (i, v) in slot.iter_mut().enumerate() {
            *v = (f0[i] * mult / sample_rate).rem_euclid(1.0);
        }
    }
    let rand_ini = draws.rand(dim);
    for c in 1..dim {
        // Channel 0 keeps zero phase: the fundamental is the one the listener tracks.
        rad[c * n] += rand_ini[c];
    }

    let coarse_len = n / upsample;
    let coarse = interpolate_linear(&rad, dim, coarse_len);

    // The reference accumulates phase unwrapped, which reaches 165,000 radians — where one
    // f32 ulp is 0.0156 rad, so `sin` of it is uncertain at 1.6% from the representation
    // alone and no arithmetic ordering reproduces torch's result. Only `sin(phase)` is ever
    // used, so the phase is kept wrapped here and the within-frame interpolation is done on
    // the *increment*, which is small. That is deliberately not bit-identical to upstream:
    // it is the same computation done accurately, and upstream cannot reproduce itself
    // across backends here.
    let tau = 2.0 * PI as f64;
    let mut wrapped = vec![0f64; dim * coarse_len];
    let mut increment = vec![0f64; dim * coarse_len];
    for c in 0..dim {
        let mut acc = 0f64;
        for k in 0..coarse_len {
            let step = coarse[c * coarse_len + k] as f64 * upsample as f64;
            acc = (acc + step).fract();
            wrapped[c * coarse_len + k] = acc * tau;
            increment[c * coarse_len + k] = step * tau;
        }
    }

    let scale = coarse_len as f64 / n as f64;
    let mut phase = vec![0f32; dim * n];
    for i in 0..n {
        let src = (scale * (i as f64 + 0.5) - 0.5).max(0.0);
        let lo = src.floor() as usize;
        let hi = (lo + 1).min(coarse_len - 1);
        let frac = src - lo as f64;
        for c in 0..dim {
            let base = wrapped[c * coarse_len + lo];
            let delta = if hi > lo { increment[c * coarse_len + hi] } else { 0.0 };
            phase[c * n + i] = (base + frac * delta) as f32;
        }
    }

    let uv: Vec<f32> = f0.iter().map(|v| (*v > VOICED_THRESHOLD) as u8 as f32).collect();
    // The draw is torch-ordered `[time, channel]` while the phase is channel-major, so the
    // two are indexed differently on purpose. Reading the noise channel-major gives a
    // waveform that is statistically identical and sample-for-sample wrong.
    let noise = draws.randn(dim * n);
    let mut sines = vec![0f32; dim * n];
    for c in 0..dim {
        for i in 0..n {
            let amp = uv[i] * NOISE_STD + (1.0 - uv[i]) * SINE_AMP / 3.0;
            sines[c * n + i] =
                phase[c * n + i].sin() * SINE_AMP * uv[i] + amp * noise[i * dim + c];
        }
    }

    // The unused noise branch still draws, so the sequence stays aligned.
    let _ = draws.randn(n);

    (0..n)
        .map(|i| {
            let mut acc = linear_b;
            for c in 0..dim {
                acc += linear_w[c] * sines[c * n + i];
            }
            acc.tanh()
        })
        .collect()
}

pub struct Stft {
    tables: tts_nn::stft::Tables,
}

impl Stft {
    pub fn new(n_fft: usize, hop: usize) -> Self {
        Self { tables: tts_nn::stft::Tables::new(n_fft, hop) }
    }

    pub fn bins(&self) -> usize {
        self.tables.bins
    }

    /// Magnitude rows then phase rows, `[1, 2 * bins, frames]`, straight onto
    /// the device: one upload in, one kernel, no host trig.
    pub fn transform_stacked(&self, x: &[f32], device: &Device) -> Result<Tensor> {
        let xt = Tensor::from_vec(x.to_vec(), x.len(), device)?;
        let (stacked, frames) = tts_nn::stft::forward(&xt, &self.tables)?;
        Ok(stacked.reshape((1, 2 * self.bins(), frames))?)
    }

    /// Overlap-add to samples. One download out; the trig stays on the device.
    pub fn inverse_stacked(&self, stacked: &Tensor, frames: usize) -> Result<Tensor> {
        Ok(tts_nn::stft::inverse(&stacked.squeeze(0)?.contiguous()?, frames, &self.tables)?)
    }
}
