//! nano-edge-tts-standalone (huggingface.co/drmhse/nano-edge-tts-standalone): text to speech in 3.27M
//! parameters with none of Kokoro's networks. From Kokoro it keeps data only: the misaki front end, the
//! phoneme vocabulary and the voice tables.
//!
//! A text side (1.52M) turns phonemes, their words' part-of-speech tags and the voice into per-phoneme
//! features, durations, and F0 and energy at 80 frames/s; [`Nano`] turns those into audio.

use crate::blocks::Linear;
use crate::nano::{MapperCfg, Nano, RendererCfg};
use crate::source::Draws;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use tts_nn::Weights;

pub const WEIGHTS: &str = "nano-tts.safetensors";
const EPS: f64 = 1e-5;
const SPACE: u32 = 16;
const PUNCT: [u32; 13] = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15];

#[derive(Deserialize)]
pub struct TextCfg {
    word_reps: usize,
    phone_reps: usize,
}

#[derive(Deserialize)]
pub struct Config {
    text: TextCfg,
    mapper: MapperCfg,
    renderer: RendererCfg,
    pub vocab: HashMap<String, u32>,
    pub tags: HashMap<String, u32>,
}

/// LayerNorm without affine, then `(1 + a * g)` and `b + bias`, where `(g, bias)` is the one style
/// projection every norm of a stack shares.
struct Norm {
    a: Tensor,
    b: Tensor,
}

impl Norm {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            a: w.get(&format!("{prefix}.a"))?,
            b: w.get(&format!("{prefix}.b"))?,
        })
    }

    fn apply(&self, x: &Tensor, (g, bias): &(Tensor, Tensor)) -> Result<Tensor> {
        let gain = (g.broadcast_mul(&self.a)? + 1.0)?;
        let shift = bias.broadcast_add(&self.b)?;
        Ok(tts_nn::cpu::norm_film(x, &gain, &shift, false, EPS)?)
    }
}

/// Depthwise conv, norm, inverted bottleneck, scaled residual; time-major `[1, n, C]`.
struct Conv {
    taps: Tensor,
    dw_bias: Tensor,
    norm: Norm,
    up: Linear,
    down: Linear,
    scale: Tensor,
}

impl Conv {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        let dw = w.get(&format!("{prefix}.dw.weight"))?; // [C, 1, k]
        Ok(Self {
            taps: dw.squeeze(1)?.t()?.contiguous()?,
            dw_bias: w.get(&format!("{prefix}.dw.bias"))?,
            norm: Norm::load(w, &format!("{prefix}.norm"))?,
            up: Linear::load(w, &format!("{prefix}.up"))?,
            down: Linear::load(w, &format!("{prefix}.down"))?,
            scale: w.get(&format!("{prefix}.scale"))?,
        })
    }

    fn apply(&self, x: &Tensor, st: &(Tensor, Tensor)) -> Result<Tensor> {
        let h = tts_nn::cpu::depthwise_tc(x, &self.taps, &self.dw_bias)?;
        let h = self.norm.apply(&h, st)?;
        let h = self.down.apply(&tts_nn::gelu_exact(&self.up.apply(&h)?)?)?;
        Ok(tts_nn::cpu::scaled_residual(x, &h, &self.scale)?)
    }
}

/// One attention block run several times; a learned row per repeat says which one this is.
struct Attention {
    n1: Norm,
    n2: Norm,
    qkv: Linear,
    out: Linear,
    ff0: Linear,
    ff2: Linear,
    step: Tensor,
}

const HEADS: usize = 4;

impl Attention {
    fn load(w: &Weights) -> Result<Self> {
        Ok(Self {
            n1: Norm::load(w, "text.attn.n1")?,
            n2: Norm::load(w, "text.attn.n2")?,
            qkv: Linear::load(w, "text.attn.qkv")?,
            out: Linear::load(w, "text.attn.out")?,
            ff0: Linear::load(w, "text.attn.ff.0")?,
            ff2: Linear::load(w, "text.attn.ff.2")?,
            step: w.get("text.attn.step.weight")?,
        })
    }

    fn apply(&self, x: &Tensor, st: &(Tensor, Tensor), i: usize) -> Result<Tensor> {
        let x = x.broadcast_add(&self.step.narrow(0, i, 1)?)?;
        let (_, n, d) = x.dims3()?;
        let hd = d / HEADS;
        let qkv = self
            .qkv
            .apply(&self.n1.apply(&x, st)?)?
            .reshape((n, 3, HEADS, hd))?;
        let part = |j: usize| -> Result<Tensor> {
            Ok(qkv
                .narrow(1, j, 1)?
                .squeeze(1)?
                .transpose(0, 1)?
                .contiguous()?) // [H, n, hd]
        };
        let (q, k, v) = (part(0)?, part(1)?, part(2)?);
        let scores = (q.matmul(&k.t()?)? / (hd as f64).sqrt())?;
        let a = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?; // [H, n, hd]
        let a = a.transpose(0, 1)?.contiguous()?.reshape((1, n, d))?;
        let x = (x + self.out.apply(&a)?)?;
        let ff = self.ff2.apply(&tts_nn::gelu_exact(
            &self.ff0.apply(&self.n2.apply(&x, st)?)?,
        )?)?;
        Ok((x + ff)?)
    }
}

/// What the text side hands the decoder.
pub struct Spoken {
    /// `[1, T, 512]`, at 40 frames/s.
    pub feat: Tensor,
    /// `[1, 2T]` each, at 80 frames/s.
    pub f0: Tensor,
    pub energy: Tensor,
    /// Frames (40/s) per phoneme id.
    pub durations: Vec<usize>,
}

pub struct TextSide {
    emb: Tensor,
    tag: Tensor,
    style: Linear,
    convs: Vec<Conv>,
    attn: Attention,
    word_in: Linear,
    word_out: Linear,
    feat: Linear,
    dur: Linear,
    lines: Linear,
    fstyle: Linear,
    fin: Linear,
    frames: Vec<Conv>,
    fout: Linear,
    word_reps: usize,
    phone_reps: usize,
}

/// `[1, 2C]` -> the shared `(g, b)`, each `[1, 1, C]`.
fn style(fc: &Linear, s: &Tensor) -> Result<(Tensor, Tensor)> {
    let gb = fc.apply(s)?.unsqueeze(1)?;
    let c = gb.dim(2)? / 2;
    Ok((gb.narrow(2, 0, c)?, gb.narrow(2, c, c)?))
}

/// Which word each phoneme pools into: every space and punctuation mark is a word of its own.
fn word_ids(ids: &[u32]) -> Vec<usize> {
    let bnd: Vec<bool> = ids
        .iter()
        .map(|i| *i == SPACE || PUNCT.contains(i))
        .collect();
    let (mut a, mut b) = (0usize, 0usize);
    (0..ids.len())
        .map(|i| {
            a += bnd[i] as usize;
            if i > 0 {
                b += bnd[i - 1] as usize;
            }
            a + b
        })
        .collect()
}

impl TextSide {
    fn load(w: &Weights, cfg: &TextCfg) -> Result<Self> {
        let stack = |name: &str| -> Result<Vec<Conv>> {
            (0..)
                .map(|i| format!("text.{name}.{i}"))
                .take_while(|p| w.has(&format!("{p}.scale")))
                .map(|p| Conv::load(w, &p))
                .collect()
        };
        Ok(Self {
            emb: w.get("text.emb.weight")?,
            tag: w.get("text.tag.weight")?,
            style: Linear::load(w, "text.style")?,
            convs: stack("convs")?,
            attn: Attention::load(w)?,
            word_in: Linear::load(w, "text.word_in")?,
            word_out: Linear::load(w, "text.word_out")?,
            feat: Linear::load(w, "text.feat")?,
            dur: Linear::load(w, "text.dur")?,
            lines: Linear::load(w, "text.lines")?,
            fstyle: Linear::load(w, "text.fstyle")?,
            fin: Linear::load(w, "text.fin")?,
            frames: stack("frames")?,
            fout: Linear::load(w, "text.fout")?,
            word_reps: cfg.word_reps,
            phone_reps: cfg.phone_reps,
        })
    }

    /// Phoneme ids and tags (pads at both ends) and the voice's prosody half `[1, 128]`.
    pub fn forward(&self, ids: &[u32], tags: &[u32], s: &Tensor, speed: f32) -> Result<Spoken> {
        let dev = s.device();
        let n = ids.len();
        let st = style(&self.style, s)?;
        let at = |t: &Tensor, rows: &[u32]| -> Result<Tensor> {
            Ok(t.index_select(&Tensor::new(rows, dev)?, 0)?.unsqueeze(0)?)
        };
        let mut x = (at(&self.emb, ids)? + at(&self.tag, tags)?)?;
        for c in &self.convs {
            x = c.apply(&x, &st)?;
        }

        // Words: the mean of each word's phonemes, attended over, added back to every phoneme.
        let wid = word_ids(ids);
        let mut slots: Vec<usize> = wid.clone();
        slots.sort_unstable();
        slots.dedup();
        let rank: Vec<u32> = wid
            .iter()
            .map(|w| slots.binary_search(w).unwrap() as u32)
            .collect();
        let mut mean = vec![0f32; slots.len() * n];
        let mut count = vec![0usize; slots.len()];
        for r in &rank {
            count[*r as usize] += 1;
        }
        for (i, r) in rank.iter().enumerate() {
            mean[*r as usize * n + i] = 1.0 / count[*r as usize] as f32;
        }
        let pool = Tensor::from_vec(mean, (1, slots.len(), n), dev)?;
        let mut w = pool.matmul(&self.word_in.apply(&x)?)?;
        for i in 0..self.word_reps {
            w = self.attn.apply(&w, &st, i)?;
        }
        let back = w
            .squeeze(0)?
            .index_select(&Tensor::new(rank.as_slice(), dev)?, 0)?;
        x = (x + self.word_out.apply(&back.unsqueeze(0)?)?)?;
        for i in 0..self.phone_reps {
            x = self.attn.apply(&x, &st, self.word_reps + i)?;
        }

        let feat = self.feat.apply(&x)?;
        let logd: Vec<f32> = self.dur.apply(&x)?.flatten_all()?.to_vec1()?;
        // torch.round is half to even.
        let durations: Vec<usize> = logd
            .iter()
            .map(|v| ((v.exp() / speed).round_ties_even()).max(1.0) as usize)
            .collect();
        self.prosody(&x, &feat, durations, s)
    }

    /// Pitch and energy drawn per phoneme as lines from a start to an end value, at 80 frames/s, and
    /// a small frame network adds what lines cannot.
    fn prosody(
        &self,
        h: &Tensor,
        feat: &Tensor,
        durations: Vec<usize>,
        s: &Tensor,
    ) -> Result<Spoken> {
        let dev = h.device();
        let mut frame_ph: Vec<u32> = Vec::new();
        let mut starts = Vec::with_capacity(durations.len());
        for (p, d) in durations.iter().enumerate() {
            starts.push(frame_ph.len());
            frame_ph.extend(std::iter::repeat_n(p as u32, *d));
        }
        let t2 = 2 * frame_ph.len();
        let ph2: Vec<u32> = frame_ph.iter().flat_map(|p| [*p, *p]).collect();
        let frac: Vec<f32> = (0..t2)
            .map(|t| {
                let p = ph2[t] as usize;
                let (start, len) = ((2 * starts[p]) as f32, (2 * durations[p]) as f32);
                ((t as f32 + 0.5 - start) / len.max(1.0)).clamp(0.0, 1.0)
            })
            .collect();
        let lines: Vec<f32> = self.lines.apply(h)?.flatten_all()?.to_vec1()?;
        let (mut f0, mut en) = (vec![0f32; t2], vec![0f32; t2]);
        for t in 0..t2 {
            let p = &lines[ph2[t] as usize * 4..];
            f0[t] = p[0] + (p[1] - p[0]) * frac[t];
            en[t] = p[2] + (p[3] - p[2]) * frac[t];
        }
        let cols: Vec<f32> = (0..t2).flat_map(|t| [f0[t], en[t], frac[t]]).collect();
        let idx2 = Tensor::new(ph2.as_slice(), dev)?;
        let x = Tensor::cat(
            &[
                &h.squeeze(0)?.index_select(&idx2, 0)?,
                &Tensor::from_vec(cols, (t2, 3), dev)?,
            ],
            1,
        )?
        .unsqueeze(0)?;
        let st = style(&self.fstyle, s)?;
        let mut x = self.fin.apply(&x)?;
        for c in &self.frames {
            x = c.apply(&x, &st)?;
        }
        let r: Vec<f32> = self.fout.apply(&x)?.flatten_all()?.to_vec1()?;
        let f0: Vec<f32> = (0..t2).map(|t| (f0[t] + r[2 * t]) * 100.0).collect();
        let en: Vec<f32> = (0..t2).map(|t| en[t] + r[2 * t + 1]).collect();
        let feat = feat
            .squeeze(0)?
            .index_select(&Tensor::new(frame_ph.as_slice(), dev)?, 0)?
            .unsqueeze(0)?;
        Ok(Spoken {
            feat,
            f0: Tensor::from_vec(f0, (1, t2), dev)?,
            energy: Tensor::from_vec(en, (1, t2), dev)?,
            durations,
        })
    }
}

pub struct NanoTts {
    pub cfg: Config,
    pub text: TextSide,
    pub decoder: Nano,
    device: Device,
}

impl NanoTts {
    /// From `dir`: `config.json` and `nano-tts.safetensors`, f16 or f32.
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let cfg: Config = serde_json::from_slice(
            &std::fs::read(dir.join("config.json"))
                .with_context(|| format!("reading {}", dir.join("config.json").display()))?,
        )?;
        let path = dir.join(WEIGHTS);
        let w = Weights::load(path.to_str().context("nano path")?, device)?;
        Ok(Self {
            text: TextSide::load(&w, &cfg.text)?,
            decoder: Nano::load(&w, &cfg.mapper, &cfg.renderer, device)?,
            cfg,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Ids, tags and the byte offset each id came from, for one phoneme string and its token tags.
    /// Pads at both ends; trailing whitespace is dropped, as the reference's stripped phonemes are.
    pub fn encode(
        &self,
        phonemes: &str,
        tags: &[tts_phoneme::g2p::TagSpan],
    ) -> (Vec<u32>, Vec<u32>, Vec<Option<usize>>) {
        let sp = self.cfg.tags.get("<sp>").copied().unwrap_or(0);
        let phonemes = phonemes.trim_end();
        let (mut ids, mut tg, mut at) = (vec![0u32], vec![0u32], vec![None]);
        let mut span = 0usize;
        for (offset, c) in phonemes.char_indices() {
            let Some(id) = self
                .cfg
                .vocab
                .get(c.encode_utf8(&mut [0; 4]) as &str)
                .copied()
            else {
                continue;
            };
            while span < tags.len() && tags[span].phonemes.end <= offset {
                span += 1;
            }
            let tag = match tags.get(span) {
                Some(t) if t.phonemes.contains(&offset) => {
                    self.cfg.tags.get(t.tag.as_str()).copied().unwrap_or(0)
                }
                _ => sp,
            };
            ids.push(id);
            tg.push(tag);
            at.push(Some(offset));
        }
        ids.push(0);
        tg.push(0);
        at.push(None);
        (ids, tg, at)
    }

    /// One segment: ids and tags with their pads, a `[1, 256]` voice row -> samples and frames per id.
    pub fn synthesize(
        &self,
        ids: &[u32],
        tags: &[u32],
        voice: &Tensor,
        speed: f32,
        draws: &mut (dyn Draws + Send),
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        let voice = voice.to_dtype(DType::F32)?.to_device(&self.device)?;
        let s_dec = voice.narrow(D::Minus1, 0, 128)?;
        let s_pro = voice.narrow(D::Minus1, 128, 128)?;
        let spoken = self.text.forward(ids, tags, &s_pro, speed)?;
        let audio =
            self.decoder
                .forward(&spoken.feat, &spoken.f0, &spoken.energy, &s_dec, draws)?;
        Ok((audio, spoken.durations))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// As the reference's two cumsums: a boundary and the phoneme after it each open a word.
    #[test]
    fn spaces_and_punctuation_are_words_of_their_own() {
        //          pad  h   i  ' '  t   o   .  pad
        let ids = [0, 50, 51, SPACE, 62, 57, 4, 0];
        let w = word_ids(&ids);
        assert_eq!(w, [0, 0, 0, 1, 2, 2, 3, 4]);
    }
}
