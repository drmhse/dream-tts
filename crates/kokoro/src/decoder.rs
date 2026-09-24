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

const NORM_EPS: f64 = 1e-5;

/// The generator's residual block. Distinct from [`AdainResBlk`]: three dilated pairs with
/// a learned snake between them, and no shortcut convolution.
struct SnakeResBlock {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    adain1: Vec<AdaIn>,
    adain2: Vec<AdaIn>,
    alpha1: Vec<(Tensor, Tensor)>,
    alpha2: Vec<(Tensor, Tensor)>,
    padded: Option<Padded>,
    /// The whole padded block as one MPSGraph, where every conv is centred and biased.
    block: Option<tts_nn::mpsblock::BlockKey>,
}

/// The block's six AdaINs in the layout [`SnakeResBlock::apply_padded`] reads, in pair order
/// `adain1[i]`, `adain2[i]`: one style projection for all of them, `[128, 6 * 2C]`, and the
/// learned `[6, 2, C]` (alpha, 1/beta).
struct Padded {
    fc_w: Tensor,
    fc_b: Tensor,
    ab: Tensor,
}

impl Padded {
    /// `None` where an AdaIN has its own affine, which the masked kernel does not apply.
    fn load(w: &Weights, prefix: &str, pairs: usize) -> Result<Option<Self>> {
        let f32 =
            |n: String| -> Result<Tensor> { Ok(w.cpu(&n)?.to_dtype(candle_core::DType::F32)?) };
        let (mut ws, mut bs, mut abs) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..pairs {
            for half in [1, 2] {
                let adain = format!("{prefix}.adain{half}.{i}");
                if w.has(&format!("{adain}.norm.weight")) {
                    return Ok(None);
                }
                ws.push(f32(format!("{adain}.fc.weight"))?);
                bs.push(f32(format!("{adain}.fc.bias"))?);
                let a = f32(format!("{prefix}.alpha{half}.{i}"))?.flatten_all()?;
                abs.push(Tensor::stack(&[&a, &a.recip()?], 0)?);
            }
        }
        let device = w.device();
        Ok(Some(Self {
            fc_w: Tensor::cat(&ws, 0)?.t()?.contiguous()?.to_device(device)?,
            fc_b: Tensor::cat(&bs, 0)?.to_device(device)?,
            ab: Tensor::stack(&abs, 0)?.to_device(device)?,
        }))
    }
}

/// The block's fused-graph key: every conv centred, biased and `C -> C`.
fn block_key(
    convs1: &[Conv1d],
    convs2: &[Conv1d],
    device: &Device,
) -> Option<tts_nn::mpsblock::BlockKey> {
    use tts_nn::mpsblock::ConvShape;
    let mut channels = None;
    let mut shape = |conv: &Conv1d| -> Option<ConvShape> {
        let (w, _, k, dilation) = conv.centred_parts()?;
        let (_, cin, cout) = w.dims3().ok()?;
        if cin != cout || *channels.get_or_insert(cin) != cin {
            return None;
        }
        Some(ConvShape { k, dilation })
    };
    let pairs = convs1
        .iter()
        .zip(convs2)
        .map(|(a, b)| Some((shape(a)?, shape(b)?)))
        .collect::<Option<Vec<_>>>()?;
    tts_nn::mpsblock::key(device, channels?, pairs, NORM_EPS as f32)
}

fn alpha_pair(w: &Weights, name: &str) -> Result<(Tensor, Tensor)> {
    let a = w.cpu(name)?.to_dtype(candle_core::DType::F32)?;
    let recip = a.recip()?;
    Ok((a.to_device(w.device())?, recip.to_device(w.device())?))
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
        let padded = Padded::load(w, prefix, dilations.len())?;
        let block = if padded.is_some() && std::env::var("KOKORO_NO_MPSBLOCK").is_err() {
            block_key(&convs1, &convs2, w.device())
        } else {
            None
        };
        Ok(Self {
            convs1,
            convs2,
            adain1,
            adain2,
            alpha1,
            alpha2,
            padded,
            block,
        })
    }

    fn block_prewarm(&self, len: usize, out: &mut Vec<(tts_nn::mpsblock::BlockKey, usize)>) {
        if let Some(key) = &self.block {
            out.push((key.clone(), len));
        }
    }

    fn mps_specs(&self, len: usize, out: &mut Vec<tts_nn::mpsconv::Spec>) {
        if self.block.is_some() {
            return;
        }
        for (c1, c2) in self.convs1.iter().zip(&self.convs2) {
            out.extend(c1.mps_spec(len, false));
            out.extend(c2.mps_spec(len, true));
        }
    }

    /// [`Self::apply`] on a signal whose first `valid` samples are real: every conv reads
    /// zeros past them, so those samples come out exactly as unpadded. What lies past them
    /// on the way out is garbage for the caller to mask.
    fn apply_padded(&self, x: &Tensor, s: &Tensor, valid: usize) -> Result<Tensor> {
        let p = self.padded.as_ref().expect("checked by Generator::padded");
        let c = x.dim(1)?;
        let gb =
            s.matmul(&p.fc_w)?
                .broadcast_add(&p.fc_b)?
                .reshape((2 * self.convs1.len(), 2, c))?;
        if let Some(key) = &self.block {
            let params = Tensor::cat(&[&gb, &p.ab], 1)?;
            let convs: Vec<(&Tensor, &Tensor)> = self
                .convs1
                .iter()
                .zip(&self.convs2)
                .flat_map(|(a, b)| [a, b])
                .filter_map(|conv| conv.centred_parts().map(|(w, b, _, _)| (w, b)))
                .collect();
            return tts_nn::mpsblock::apply(key, x, &params, &convs, valid);
        }
        let adain = |x: &Tensor, k: usize| -> Result<Tensor> {
            Ok(tts_nn::fused::adain_snake_masked(
                x,
                &gb.get(k)?,
                &p.ab.get(k)?,
                NORM_EPS,
                valid,
            )?)
        };
        let mut x = x.clone();
        for i in 0..self.convs1.len() {
            let t = self.convs1[i].apply(&adain(&x, 2 * i)?)?;
            x = self.convs2[i].apply_residual(&adain(&t, 2 * i + 1)?, Some(&x))?;
        }
        Ok(x)
    }

    fn apply(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for i in 0..self.convs1.len() {
            let mut t = self.adain1[i].apply_snake(&x, s, &self.alpha1[i].0, &self.alpha1[i].1)?;
            t = self.convs1[i].apply(&t)?;
            t = self.adain2[i].apply_snake(&t, s, &self.alpha2[i].0, &self.alpha2[i].1)?;
            x = self.convs2[i].apply_residual(&t, Some(&x))?;
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
        for (i, (rate, kernel)) in g
            .upsample_rates
            .iter()
            .zip(&g.upsample_kernel_sizes)
            .enumerate()
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
                noise_convs.push(Conv1d::load_padded(
                    w,
                    &format!("{p}.noise_convs.{i}"),
                    stride,
                    1,
                    Some(stride.div_ceil(2)),
                )?);
            } else {
                noise_convs.push(Conv1d::load_padded(
                    w,
                    &format!("{p}.noise_convs.{i}"),
                    1,
                    1,
                    Some(0),
                )?);
            }
            noise_res.push(SnakeResBlock::load(
                w,
                &format!("{p}.noise_res.{i}"),
                &[1, 3, 5],
            )?);
        }
        let source_w: Vec<f32> = w
            .get(&format!("{p}.m_source.l_linear.weight"))?
            .flatten_all()?
            .to_vec1()?;
        let source_b = w
            .get(&format!("{p}.m_source.l_linear.bias"))?
            .to_vec1::<f32>()?[0];
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

    /// Whether [`Self::forward_padded`] can run: Metal, and no AdaIN with its own affine.
    fn padded(&self, x: &Tensor) -> bool {
        x.device().is_metal()
            && buckets_per_octave() > 0
            && self
                .noise_res
                .iter()
                .chain(&self.resblocks)
                .all(|b| b.padded.is_some())
    }

    /// `excitation` is the merged harmonic waveform, computed by [`Self::excitation`] —
    /// on another thread, while the decoder's own blocks were running.
    pub fn forward(&self, x: &Tensor, s: &Tensor, excitation: &[f32]) -> Result<Vec<f32>> {
        if self.padded(x) {
            return self.forward_padded(x, s, excitation);
        }
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

    /// Compile, off this thread, the MPSGraph executables a generator input of `len` will
    /// run. Called as soon as the length is known, so the compiles overlap the GPU work
    /// still ahead of the generator rather than stalling it one at a time.
    pub fn prewarm(&self, len: usize, device: &Device) {
        if !device.is_metal() || buckets_per_octave() == 0 {
            return;
        }
        let mut specs = Vec::new();
        let mut blocks = Vec::new();
        let mut at = bucket(len);
        for i in 0..self.ups.len() {
            specs.extend(self.ups[i].mps_spec(at));
            at *= self.ups[i].stride();
            if i + 1 == self.ups.len() {
                at += 1;
            }
            self.noise_res[i].mps_specs(at, &mut specs);
            self.noise_res[i].block_prewarm(at, &mut blocks);
            for j in 0..self.kernels {
                self.resblocks[i * self.kernels + j].mps_specs(at, &mut specs);
                self.resblocks[i * self.kernels + j].block_prewarm(at, &mut blocks);
            }
        }
        specs.extend(self.conv_post.mps_spec(at, false));
        tts_nn::mpsconv::prewarm(device, specs);
        tts_nn::mpsblock::prewarm(device, blocks);
    }

    /// [`Self::forward`] at a bucketed length.
    ///
    /// MPSGraph specialises a graph for every input length it meets, at ~4 ms a graph, and
    /// every utterance has its own length: ~26 graphs made that ~100 ms a segment, a third of
    /// the decoder. Padded to a bucket, a length recurs. The padding is exact rather than
    /// close: moments are taken over the real samples only, and every conv input is masked
    /// to zeros past them, which is what the conv's own padding would have read.
    fn forward_padded(&self, x: &Tensor, s: &Tensor, excitation: &[f32]) -> Result<Vec<f32>> {
        let device = x.device().clone();
        let t0 = std::time::Instant::now();
        let har = self.spectrum(excitation, &device)?;
        let t_source = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let mut valid = x.dim(2)?;
        let mut x = x.pad_with_zeros(2, 0, bucket(valid) - valid)?;
        for i in 0..self.ups.len() {
            x = tts_nn::fused::leaky_masked(&x, 0.1, valid)?;
            x = self.ups[i].apply(&x)?;
            valid *= self.ups[i].stride();
            if i + 1 == self.ups.len() {
                // ReflectionPad1d((1, 0)), as in `forward`.
                x = Tensor::cat(&[&x.narrow(2, 1, 1)?, &x], 2)?;
                valid += 1;
            }
            let source = self.noise_convs[i].apply(&har)?;
            anyhow::ensure!(
                source.dim(2)? == valid,
                "noise conv gives {} samples against {valid}",
                source.dim(2)?
            );
            let source = source.pad_with_zeros(2, 0, x.dim(2)? - valid)?;
            let source = self.noise_res[i].apply_padded(&source, s, valid)?;
            x = (x + source)?;
            let mut sum: Option<Tensor> = None;
            for j in 0..self.kernels {
                let y = self.resblocks[i * self.kernels + j].apply_padded(&x, s, valid)?;
                sum = Some(match sum {
                    Some(acc) => (acc + y)?,
                    None => y,
                });
            }
            x = (sum.unwrap() / self.kernels as f64)?;
        }
        // The default slope here, not the 0.1 used inside the loop.
        x = tts_nn::fused::leaky_masked(&x, 0.01, valid)?;
        x = self
            .conv_post
            .apply(&x)?
            .narrow(2, 0, valid)?
            .contiguous()?;
        device.synchronize()?;
        let t_net = t1.elapsed().as_secs_f64();
        let t2 = std::time::Instant::now();
        let bins = self.stft.bins();
        let spec = x.narrow(1, 0, bins)?.exp()?;
        let phase = x.narrow(1, bins, bins)?.sin()?;
        let both = Tensor::cat(&[spec, phase], 1)?;
        let out: Vec<f32> = self.stft.inverse_stacked(&both, valid)?.to_vec1()?;
        if std::env::var("KOKORO_TIMING").is_ok() {
            eprintln!(
                "    source {t_source:.3}  net {t_net:.3}  istft {:.3}  ({} samples)",
                t2.elapsed().as_secs_f64(),
                x.dim(2)?
            );
        }
        Ok(out)
    }

    pub fn post_conv(&self, x: &Tensor) -> Result<Tensor> {
        self.conv_post.apply(&tts_nn::leaky_relu(x, 0.01)?)
    }
}

/// `KOKORO_BUCKETS`: length buckets per octave for the padded generator, 0 to not pad. Eight
/// wastes ~3% on padding; fewer buckets recur more often and waste more.
fn buckets_per_octave() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("KOKORO_BUCKETS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8)
    })
}

/// `len` rounded up to the next of `buckets_per_octave` steps in its octave.
fn bucket(len: usize) -> usize {
    let n = buckets_per_octave();
    if n == 0 || len < 2 {
        return len;
    }
    let octave = 1usize << (usize::BITS - 1 - len.leading_zeros());
    let step = (octave / n).max(1);
    len.div_ceil(step) * step
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
            let excitation = side
                .join()
                .map_err(|_| anyhow::anyhow!("excitation panicked"))?;
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
