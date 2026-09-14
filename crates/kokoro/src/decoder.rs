//! The iSTFTNet decoder: encoding, pitch and energy in, waveform out.
//!
//! The last stage predicts a magnitude and a phase spectrum rather than samples, and a
//! 20-point inverse transform does the rest. That is the whole reason this vocoder is
//! cheap: two upsampling convolutions instead of the six a sample-domain vocoder needs.

use crate::blocks::{AdaIn, AdainResBlk, Conv1d, ConvTranspose1d};
use crate::cfg::Config;
use crate::source::{self, Draws, Stft};
use anyhow::Result;
use candle_core::{Device, Tensor};
use tts_nn::Weights;

/// The generator's residual block. Distinct from [`AdainResBlk`]: three dilated pairs with
/// a learned snake between them, and no shortcut convolution.
struct SnakeResBlock {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    adain1: Vec<AdaIn>,
    adain2: Vec<AdaIn>,
    alpha1: Vec<(Tensor, Tensor)>,
    alpha2: Vec<(Tensor, Tensor)>,
}

fn alpha_pair(w: &Weights, name: &str) -> Result<(Tensor, Tensor)> {
    let a = w.get(name)?;
    let recip = a.recip()?;
    Ok((a, recip))
}

impl SnakeResBlock {
    fn load(w: &Weights, prefix: &str, dilations: &[usize]) -> Result<Self> {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        let mut adain1 = Vec::new();
        let mut adain2 = Vec::new();
        let mut alpha1 = Vec::new();
        let mut alpha2 = Vec::new();
        for (i, d) in dilations.iter().enumerate() {
            convs1.push(Conv1d::load(w, &format!("{prefix}.convs1.{i}"), 1, *d)?);
            convs2.push(Conv1d::load(w, &format!("{prefix}.convs2.{i}"), 1, 1)?);
            adain1.push(AdaIn::load(w, &format!("{prefix}.adain1.{i}"))?);
            adain2.push(AdaIn::load(w, &format!("{prefix}.adain2.{i}"))?);
            alpha1.push(alpha_pair(w, &format!("{prefix}.alpha1.{i}"))?);
            alpha2.push(alpha_pair(w, &format!("{prefix}.alpha2.{i}"))?);
        }
        Ok(Self { convs1, convs2, adain1, adain2, alpha1, alpha2 })
    }

    fn apply(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for i in 0..self.convs1.len() {
            let mut t =
                self.adain1[i].apply_snake(&x, s, &self.alpha1[i].0, &self.alpha1[i].1)?;
            t = self.convs1[i].apply(&t)?;
            t = self.adain2[i].apply_snake(&t, s, &self.alpha2[i].0, &self.alpha2[i].1)?;
            t = self.convs2[i].apply(&t)?;
            x = (t + x)?;
        }
        Ok(x)
    }
}

pub struct Generator {
    ups: Vec<ConvTranspose1d>,
    noise_convs: Vec<Conv1d>,
    noise_res: Vec<SnakeResBlock>,
    resblocks: Vec<SnakeResBlock>,
    conv_post: Conv1d,
    source_w: Vec<f32>,
    source_b: f32,
    stft: Stft,
    upsample: usize,
    kernels: usize,
}

impl Generator {
    pub fn load(w: &Weights, cfg: &Config) -> Result<Self> {
        let g = &cfg.istftnet;
        let p = "decoder.generator";
        let mut ups = Vec::new();
        for (i, (rate, kernel)) in
            g.upsample_rates.iter().zip(&g.upsample_kernel_sizes).enumerate()
        {
            ups.push(ConvTranspose1d::load(
                w,
                &format!("{p}.ups.{i}"),
                *rate,
                (kernel - rate) / 2,
                0,
                1,
            )?);
        }
        let mut resblocks = Vec::new();
        for i in 0..ups.len() {
            for (j, d) in g.resblock_dilation_sizes.iter().enumerate() {
                resblocks.push(SnakeResBlock::load(
                    w,
                    &format!("{p}.resblocks.{}", i * g.resblock_kernel_sizes.len() + j),
                    d,
                )?);
            }
        }
        let mut noise_convs = Vec::new();
        let mut noise_res = Vec::new();
        for i in 0..ups.len() {
            if i + 1 < g.upsample_rates.len() {
                let stride: usize = g.upsample_rates[i + 1..].iter().product();
                noise_convs.push(
                    Conv1d::load(w, &format!("{p}.noise_convs.{i}"), stride, 1)?
                        .with_padding((stride + 1) / 2),
                );
            } else {
                noise_convs
                    .push(Conv1d::load(w, &format!("{p}.noise_convs.{i}"), 1, 1)?.with_padding(0));
            }
            noise_res.push(SnakeResBlock::load(
                w,
                &format!("{p}.noise_res.{i}"),
                &[1, 3, 5],
            )?);
        }
        let source_w: Vec<f32> = w.get(&format!("{p}.m_source.l_linear.weight"))?.flatten_all()?.to_vec1()?;
        let source_b = w.get(&format!("{p}.m_source.l_linear.bias"))?.to_vec1::<f32>()?[0];
        Ok(Self {
            ups,
            noise_convs,
            noise_res,
            resblocks,
            conv_post: Conv1d::load(w, &format!("{p}.conv_post"), 1, 1)?,
            source_w,
            source_b,
            stft: Stft::new(g.gen_istft_n_fft, g.gen_istft_hop_size),
            upsample: g.upsample_rates.iter().product::<usize>() * g.gen_istft_hop_size,
            kernels: g.resblock_kernel_sizes.len(),
        })
    }

    /// Magnitude and phase of an excitation waveform, stacked the way the noise
    /// convolutions expect.
    pub fn spectrum(&self, excitation: &[f32], device: &Device) -> Result<Tensor> {
        self.stft.transform_stacked(excitation, device)
    }

    /// The merged harmonic excitation, one value per output sample.
    pub fn excitation(&self, f0: &[f32], draws: &mut (dyn Draws + Send)) -> Vec<f32> {
        let mut upsampled = Vec::with_capacity(f0.len() * self.upsample);
        for v in f0 {
            upsampled.extend(std::iter::repeat(*v).take(self.upsample));
        }
        source::excitation(
            &upsampled,
            &self.source_w,
            self.source_b,
            self.upsample,
            Config::SAMPLE_RATE as f32,
            draws,
        )
    }


    /// `excitation` is the merged harmonic waveform, computed by [`Self::excitation`] —
    /// on another thread, while the decoder's own blocks were running.
    pub fn forward(&self, x: &Tensor, s: &Tensor, excitation: &[f32]) -> Result<Vec<f32>> {
        let device = x.device().clone();
        let t0 = std::time::Instant::now();
        let har = self.spectrum(excitation, &device)?;
        device.synchronize()?;
        let t_source = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let mut x = x.clone();
        let timing = std::env::var("KOKORO_TIMING").is_ok();
        for i in 0..self.ups.len() {
            let ts = std::time::Instant::now();
            x = tts_nn::leaky_relu(&x, 0.1)?;
            let mut source = self.noise_convs[i].apply(&har)?;
            if timing {
                device.synchronize()?;
                eprintln!("      stage{i} nconv {:.3}", ts.elapsed().as_secs_f64());
            }
            let tn = std::time::Instant::now();
            source = self.noise_res[i].apply(&source, s)?;
            if timing {
                device.synchronize()?;
                eprintln!("      stage{i} nres  {:.3}", tn.elapsed().as_secs_f64());
            }
            let tr = std::time::Instant::now();
            x = self.ups[i].apply(&x)?;
            if timing {
                device.synchronize()?;
                eprintln!("      stage{i} up    {:.3}", tr.elapsed().as_secs_f64());
            }
            let tb = std::time::Instant::now();
            if i + 1 == self.ups.len() {
                // ReflectionPad1d((1, 0)): one sample on the left, which is what makes the
                // final length match the excitation's frame count.
                x = Tensor::cat(&[&x.narrow(2, 1, 1)?, &x], 2)?;
            }
            x = (x + source)?;
            let mut sum: Option<Tensor> = None;
            for j in 0..self.kernels {
                let y = self.resblocks[i * self.kernels + j].apply(&x, s)?;
                sum = Some(match sum {
                    Some(acc) => (acc + y)?,
                    None => y,
                });
            }
            x = (sum.unwrap() / self.kernels as f64)?;
            if timing {
                device.synchronize()?;
                eprintln!("      stage{i} res   {:.3}", tb.elapsed().as_secs_f64());
            }
        }
        // The default slope here, not the 0.1 used inside the loop.
        x = tts_nn::leaky_relu(&x, 0.01)?;
        x = self.conv_post.apply(&x)?;
        device.synchronize()?;
        let t_net = t1.elapsed().as_secs_f64();
        let t2 = std::time::Instant::now();
        let bins = self.stft.bins();
        let frames = x.dim(2)?;
        let spec = x.narrow(1, 0, bins)?.exp()?;
        let phase = x.narrow(1, bins, bins)?.sin()?;
        let both = Tensor::cat(&[spec, phase], 1)?;
        let wav = self.stft.inverse_stacked(&both, frames)?;
        let out: Vec<f32> = wav.to_vec1()?;
        if std::env::var("KOKORO_TIMING").is_ok() {
            eprintln!(
                "    source {t_source:.3}  net {t_net:.3}  istft {:.3}",
                t2.elapsed().as_secs_f64()
            );
        }
        Ok(out)
    }

    pub fn post_conv(&self, x: &Tensor) -> Result<Tensor> {
        self.conv_post.apply(&tts_nn::leaky_relu(x, 0.01)?)
    }
}

pub struct Decoder {
    f0_conv: Conv1d,
    n_conv: Conv1d,
    encode: AdainResBlk,
    asr_res: Conv1d,
    decode: Vec<AdainResBlk>,
    pub generator: Generator,
}

impl Decoder {
    pub fn load(w: &Weights, cfg: &Config) -> Result<Self> {
        let dim = cfg.hidden_dim + 2;
        let wide = 1024 + 2 + 64;
        Ok(Self {
            f0_conv: Conv1d::load(w, "decoder.F0_conv", 2, 1)?,
            n_conv: Conv1d::load(w, "decoder.N_conv", 2, 1)?,
            encode: AdainResBlk::load(w, "decoder.encode", dim, false)?,
            asr_res: Conv1d::load(w, "decoder.asr_res.0", 1, 1)?,
            decode: vec![
                AdainResBlk::load(w, "decoder.decode.0", wide, false)?,
                AdainResBlk::load(w, "decoder.decode.1", wide, false)?,
                AdainResBlk::load(w, "decoder.decode.2", wide, false)?,
                AdainResBlk::load(w, "decoder.decode.3", wide, true)?,
            ],
            generator: Generator::load(w, cfg)?,
        })
    }

    /// `asr` is `[1, 512, frames]`; `f0` and `energy` are `[1, 2 * frames]`.
    pub fn forward(
        &self,
        asr: &Tensor,
        f0_curve: &Tensor,
        energy: &Tensor,
        s: &Tensor,
        draws: &mut (dyn Draws + Send),
    ) -> Result<Vec<f32>> {
        // The excitation is host-side and depends on nothing below, so it runs on another
        // thread while these blocks keep the GPU busy. It was 26 ms of an idle device.
        let curve: Vec<f32> = f0_curve.flatten_all()?.to_vec1()?;
        let (x, excitation) = std::thread::scope(|sc| -> Result<(Tensor, Vec<f32>)> {
            let side = sc.spawn(|| self.generator.excitation(&curve, draws));
            let f0 = self.f0_conv.apply(&f0_curve.unsqueeze(1)?)?;
            let n = self.n_conv.apply(&energy.unsqueeze(1)?)?;
            let mut x = self.encode.apply(&Tensor::cat(&[asr, &f0, &n], 1)?, s)?;
            let asr_res = self.asr_res.apply(asr)?;
            for block in &self.decode {
                x = block.apply(&Tensor::cat(&[&x, &asr_res, &f0, &n], 1)?, s)?;
            }
            let excitation = side.join().map_err(|_| anyhow::anyhow!("excitation panicked"))?;
            Ok((x, excitation))
        })?;
        self.generator.forward(&x, s, &excitation)
    }
}

/// Draws from the engine's own seeded generator, so a seed reproduces a render across
/// backends — the same reason `tts_core::rng` exists.
pub struct SeededDraws(tts_core::rng::Rng);

impl SeededDraws {
    pub fn new(seed: u64) -> Self {
        Self(tts_core::rng::Rng::new(seed))
    }
}

impl Draws for SeededDraws {
    fn rand(&mut self, n: usize) -> Vec<f32> {
        let mut out = vec![0f32; n];
        self.0.fill(&mut out);
        out
    }

    fn randn(&mut self, n: usize) -> Vec<f32> {
        // Box-Muller: `tts_core::rng` is uniform-only, and its uniforms are open at both
        // ends, so the logarithm is always finite.
        let mut out = vec![0f32; n];
        let mut i = 0;
        while i < n {
            let (u1, u2) = (self.0.next_f32(), self.0.next_f32());
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            out[i] = r * theta.cos();
            if i + 1 < n {
                out[i + 1] = r * theta.sin();
            }
            i += 2;
        }
        out
    }
}
