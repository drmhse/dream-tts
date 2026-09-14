//! A short-time Fourier pair as single dispatches.
//!
//! `torch.stft` / `torch.istft` with a periodic Hann window, reflect padding and a
//! small hop — the 20-point pair around Kokoro's vocoder. On the host the pair
//! costs tens of milliseconds per utterance in trig calls alone (tens of millions
//! of per-tap sin/cos/atan2); here the twiddles come from tables and each output
//! is one thread. The overlap-add runs gathered per output sample, so there are
//! no races and no envelope array — the envelope is signal-independent, and each
//! thread rebuilds its own few terms.
//!
//! The tables hold the pure DFT exponentials; windowing, the conjugate-pair
//! weight and the 1/n stay runtime multiplies in the reference's order, so the
//! only numeric distance from the composed loops is libm-vs-Metal trig (~1 ulp
//! each). Off Metal the same tables drive plain CPU loops, which is what the
//! engine gates check.

use candle_core::{CpuStorage, CustomOp2, DType, Layout, Result, Shape, Tensor};
use std::f32::consts::PI;

/// Everything a transform needs beyond the signal, precomputed once per `(n_fft, hop)`.
pub struct Tables {
    pub n_fft: usize,
    pub hop: usize,
    pub bins: usize,
    pub pad: usize,
    pub window: Vec<f32>,
    fwd_re: Vec<f32>,
    fwd_im: Vec<f32>,
    inv_c: Vec<f32>,
    inv_s: Vec<f32>,
}

impl Tables {
    pub fn new(n_fft: usize, hop: usize) -> Self {
        let bins = n_fft / 2 + 1;
        // Periodic Hann, the convention torch.hann_window uses by default.
        let window: Vec<f32> = (0..n_fft)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n_fft as f32).cos())
            .collect();
        let mut fwd_re = vec![0f32; bins * n_fft];
        let mut fwd_im = vec![0f32; bins * n_fft];
        let mut inv_c = vec![0f32; bins * n_fft];
        let mut inv_s = vec![0f32; bins * n_fft];
        for k in 0..bins {
            // Bins 1..n/2 stand for a conjugate pair; the DC and Nyquist bins do not.
            let pair = if k == 0 || k == bins - 1 { 1.0 } else { 2.0 };
            for nn in 0..n_fft {
                let a = 2.0 * PI * (k * nn) as f32 / n_fft as f32;
                fwd_re[k * n_fft + nn] = (-a).cos();
                fwd_im[k * n_fft + nn] = (-a).sin();
                inv_c[k * n_fft + nn] = a.cos() * pair / n_fft as f32;
                inv_s[k * n_fft + nn] = a.sin() * pair / n_fft as f32;
            }
        }
        Self {
            n_fft,
            hop,
            bins,
            pad: n_fft / 2,
            window,
            fwd_re,
            fwd_im,
            inv_c,
            inv_s,
        }
    }

    /// `[tre | tim | window]`, the forward kernel's table buffer.
    fn fwd_packed(&self) -> Vec<f32> {
        [
            self.fwd_re.as_slice(),
            self.fwd_im.as_slice(),
            self.window.as_slice(),
        ]
        .concat()
    }

    /// `[tc | ts | window]`, the inverse kernel's table buffer.
    fn inv_packed(&self) -> Vec<f32> {
        [
            self.inv_c.as_slice(),
            self.inv_s.as_slice(),
            self.window.as_slice(),
        ]
        .concat()
    }
}

/// Magnitude rows then phase rows, `[2 * bins, frames]`, from `x` (`[S]`).
pub fn forward(x: &Tensor, t: &Tables) -> Result<(Tensor, usize)> {
    let len = x.dim(0)?;
    let frames = len / t.hop + 1;
    if x.device().is_metal() {
        let dev = x.device().clone();
        let tables = Tensor::from_vec(t.fwd_packed(), (2 * t.bins * t.n_fft + t.n_fft,), &dev)?;
        let op = Forward {
            bins: t.bins,
            frames,
            n_fft: t.n_fft,
            hop: t.hop,
            pad: t.pad,
            len,
        };
        let y = x.contiguous()?.apply_op2_no_bwd(&tables, &op)?;
        return Ok((y, frames));
    }
    let xv: Vec<f32> = x.flatten_all()?.to_vec1()?;
    let mut dst = vec![0f32; 2 * t.bins * frames];
    cpu_forward(&xv, t, frames, &mut dst);
    Ok((
        Tensor::from_vec(dst, (2 * t.bins, frames), x.device())?,
        frames,
    ))
}

/// Overlap-add to `[out_len]`, from stacked mag/phase `[2 * bins, frames]`.
pub fn inverse(stacked: &Tensor, frames: usize, t: &Tables) -> Result<Tensor> {
    let out_len = (frames - 1) * t.hop;
    if stacked.device().is_metal() {
        let dev = stacked.device().clone();
        let tables = Tensor::from_vec(t.inv_packed(), (2 * t.bins * t.n_fft + t.n_fft,), &dev)?;
        let op = Inverse {
            bins: t.bins,
            frames,
            n_fft: t.n_fft,
            hop: t.hop,
            pad: t.pad,
            out_len,
        };
        return stacked.contiguous()?.apply_op2_no_bwd(&tables, &op);
    }
    let sv: Vec<f32> = stacked.flatten_all()?.to_vec1()?;
    let mut dst = vec![0f32; out_len];
    cpu_inverse(&sv, frames, t, &mut dst);
    Ok(Tensor::from_vec(dst, out_len, stacked.device())?)
}

fn cpu_forward(x: &[f32], t: &Tables, frames: usize, dst: &mut [f32]) {
    let len = x.len();
    for f in 0..frames {
        for k in 0..t.bins {
            let (mut re, mut im) = (0f32, 0f32);
            for nn in 0..t.n_fft {
                let j = f * t.hop + nn;
                let s = if j < t.pad {
                    x[t.pad - j]
                } else if j < t.pad + len {
                    x[j - t.pad]
                } else {
                    x[2 * len + t.pad - 2 - j]
                };
                let v = s * t.window[nn];
                re += v * t.fwd_re[k * t.n_fft + nn];
                im += v * t.fwd_im[k * t.n_fft + nn];
            }
            dst[k * frames + f] = (re * re + im * im).sqrt();
            dst[(t.bins + k) * frames + f] = im.atan2(re);
        }
    }
}

fn cpu_inverse(stacked: &[f32], frames: usize, t: &Tables, dst: &mut [f32]) {
    let out_len = (frames - 1) * t.hop;
    let at = |row: usize, f: usize| stacked[row * frames + f];
    for tt in 0..out_len {
        let g = t.pad + tt;
        let f_lo = if g < t.n_fft {
            0
        } else {
            (g - t.n_fft + t.hop) / t.hop
        };
        let f_hi = (g / t.hop).min(frames - 1);
        let (mut acc, mut env) = (0f32, 0f32);
        for f in f_lo..=f_hi {
            let nn = g - f * t.hop;
            let mut v = 0f32;
            for k in 0..t.bins {
                let m = at(k, f);
                let p = at(t.bins + k, f);
                v +=
                    m * (p.cos() * t.inv_c[k * t.n_fft + nn] - p.sin() * t.inv_s[k * t.n_fft + nn]);
            }
            let w = t.window[nn];
            acc += v * w;
            env += w * w;
        }
        dst[tt] = if env > 1e-11 { acc / env } else { 0.0 };
    }
}

struct Forward {
    bins: usize,
    frames: usize,
    n_fft: usize,
    hop: usize,
    pad: usize,
    len: usize,
}

impl CustomOp2 for Forward {
    fn name(&self) -> &'static str {
        "stft_forward"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("stft_forward: dispatched on Metal only; CPU uses the loops")
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
            candle_core::bail!("stft_forward: inputs must be contiguous");
        }
        if s1.dtype() != DType::F32 || s2.dtype() != DType::F32 {
            candle_core::bail!("stft_forward: only f32");
        }
        let device = s1.device();
        let p = mtl::pipeline(device, "stft_forward_f32")?;
        let out_len = 2 * self.bins * self.frames;
        let dst = device.new_buffer(out_len, DType::F32, "stft_forward")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::stft_forward");
        encoder.set_compute_pipeline_state(&p);
        encoder.set_buffer(0, Some(s1.buffer()), l1.start_offset() * 4);
        // Tables pack as [tre | tim | win]; the kernel splits them by size.
        let bn = (self.bins * self.n_fft) as u32;
        encoder.set_buffer(1, Some(s2.buffer()), (l2.start_offset() + 0) * 4);
        encoder.set_buffer(2, Some(s2.buffer()), (l2.start_offset() + bn as usize) * 4);
        encoder.set_buffer(
            3,
            Some(s2.buffer()),
            (l2.start_offset() + 2 * bn as usize) * 4,
        );
        encoder.set_buffer(4, Some(dst.as_ref()), 0);
        for (i, v) in [
            self.len,
            self.frames,
            self.n_fft,
            self.hop,
            self.pad,
            self.bins,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_bytes(5 + i, &(*v as u32));
        }
        for s in [s1.buffer(), s2.buffer()] {
            encoder.use_resource(s, MTLResourceUsage::Read);
        }
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        let w = mtl::group_width(&p, self.frames);
        encoder.dispatch_threads(
            MTLSize {
                width: self.frames,
                height: self.bins,
                depth: 1,
            },
            MTLSize {
                width: w,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), out_len, DType::F32),
            (2 * self.bins, self.frames).into(),
        ))
    }
}

struct Inverse {
    bins: usize,
    frames: usize,
    n_fft: usize,
    hop: usize,
    pad: usize,
    out_len: usize,
}

impl CustomOp2 for Inverse {
    fn name(&self) -> &'static str {
        "stft_inverse"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("stft_inverse: dispatched on Metal only; CPU uses the loops")
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
            candle_core::bail!("stft_inverse: inputs must be contiguous");
        }
        if s1.dtype() != DType::F32 || s2.dtype() != DType::F32 {
            candle_core::bail!("stft_inverse: only f32");
        }
        let device = s1.device();
        let p = mtl::pipeline(device, "stft_inverse_f32")?;
        let dst = device.new_buffer(self.out_len, DType::F32, "stft_inverse")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("tts_nn::stft_inverse");
        encoder.set_compute_pipeline_state(&p);
        // Stacked input is mag rows then phase rows; the kernel takes them apart.
        let bf = self.bins * self.frames;
        encoder.set_buffer(0, Some(s1.buffer()), l1.start_offset() * 4);
        encoder.set_buffer(1, Some(s1.buffer()), (l1.start_offset() + bf) * 4);
        // Tables pack as [tc | ts | win].
        let bn = (self.bins * self.n_fft) as usize;
        encoder.set_buffer(2, Some(s2.buffer()), (l2.start_offset() + 0) * 4);
        encoder.set_buffer(3, Some(s2.buffer()), (l2.start_offset() + bn) * 4);
        encoder.set_buffer(4, Some(s2.buffer()), (l2.start_offset() + 2 * bn) * 4);
        encoder.set_buffer(5, Some(dst.as_ref()), 0);
        for (i, v) in [
            self.out_len,
            self.frames,
            self.bins,
            self.n_fft,
            self.hop,
            self.pad,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_bytes(6 + i, &(*v as u32));
        }
        for s in [s1.buffer(), s2.buffer()] {
            encoder.use_resource(s, MTLResourceUsage::Read);
        }
        encoder.use_resource(dst.as_ref(), MTLResourceUsage::Write);
        let w = mtl::group_width(&p, self.out_len);
        encoder.dispatch_threads(
            MTLSize {
                width: self.out_len,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: w,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);

        Ok((
            MetalStorage::new(dst, device.clone(), self.out_len, DType::F32),
            self.out_len.into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn tables() -> Tables {
        Tables::new(20, 5)
    }

    #[test]
    fn roundtrip_is_near_identity() -> anyhow::Result<()> {
        // A Hann window with hop 5 is invertible up to the reflect-padded edges,
        // so interior samples must come back. This checks the pair against
        // itself, independent of any reference implementation.
        let dev = Device::Cpu;
        let x: Vec<f32> = (0..2400)
            .map(|i| (i as f32 * 0.11).sin() + 0.5 * (i as f32 * 0.031).cos())
            .collect();
        let t = tables();
        let xt = Tensor::from_vec(x.clone(), 2400, &dev)?;
        let (stacked, frames) = forward(&xt, &t)?;
        let back = inverse(&stacked, frames, &t)?.to_vec1::<f32>()?;
        let edge = t.pad + 1;
        let mut worst = 0f32;
        for i in edge..2400 - edge {
            worst = worst.max((back[i] - x[i]).abs());
        }
        assert!(
            worst < 1e-3,
            "roundtrip drifted {worst:.3e} in the interior"
        );
        Ok(())
    }

    #[test]
    fn metal_matches_cpu() -> anyhow::Result<()> {
        #[cfg(feature = "metal")]
        let _gpu = crate::gpu_guard();
        let Some(dev) = crate::usable_metal() else {
            return Ok(());
        };
        let cpu = Device::Cpu;
        let t = tables();
        let x: Vec<f32> = (0..7331).map(|i| (i as f32 * 0.061).sin()).collect();
        let (want_s, frames) = forward(&Tensor::from_vec(x.clone(), 7331, &cpu)?, &t)?;
        let (got_s, frames_g) = forward(&Tensor::from_vec(x.clone(), 7331, &dev)?, &t)?;
        assert_eq!(frames, frames_g);
        let got_a = inverse(&got_s, frames, &t)?
            .to_device(&cpu)?
            .to_vec1::<f32>()?;
        let want_a = inverse(&want_s, frames, &t)?.to_vec1::<f32>()?;
        // Phase compares through the complex spectrum: values near an atan2
        // branch cut differ by whole turns, which is not an error.
        let complex = |s: &Tensor| -> anyhow::Result<Tensor> {
            let b = s.dim(0)? / 2;
            let f = s.dim(1)?;
            let mag = s.narrow(0, 0, b)?;
            let ph = s.narrow(0, b, b)?;
            Ok(Tensor::cat(&[(&mag * ph.cos()?)?, (&mag * ph.sin()?)?], 0)?
                .reshape((2 * b * f,))?)
        };
        let (abs, rel) =
            crate::abs_and_rel(&complex(&got_s.to_device(&cpu)?)?, &complex(&want_s)?)?;
        assert!(rel < 1e-5, "forward: abs {abs:.3e} rel {rel:.3e}");
        let want_a = inverse(&want_s, frames, &t)?.to_vec1::<f32>()?;
        let (abs, rel) = crate::abs_and_rel(
            &Tensor::from_vec(got_a, (frames - 1) * 5, &cpu)?,
            &Tensor::from_vec(want_a, (frames - 1) * 5, &cpu)?,
        )?;
        assert!(rel < 1e-5, "inverse: abs {abs:.3e} rel {rel:.3e}");
        Ok(())
    }
}
