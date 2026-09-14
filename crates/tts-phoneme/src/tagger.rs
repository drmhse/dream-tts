//! spaCy's English POS tagger, ported exactly rather than approximated.
//!
//! The lexicon needs a tag for 790 homograph entries, and misaki's rules need fine Penn
//! tags for acronyms, currency and a handful of function words. A wrong tag is a fluent
//! mispronunciation — "he read it" against "he will read it" — which no audio check finds,
//! so the port is held to tag-for-tag identity with spaCy rather than to an accuracy score.
//!
//! 1.6M parameters: six hashed embedding tables summed, a maxout projection, and four
//! residual window encoders.

use crate::murmur::thinc_hash;
use crate::vocab::Vocab;
use anyhow::{Context, Result};
use safetensors::SafeTensors;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

const EPS: f32 = 1e-8;

#[derive(Deserialize)]
struct Meta {
    attrs: Vec<String>,
    seeds: Vec<u32>,
    rows: Vec<usize>,
    width: usize,
    depth: usize,
    pad: usize,
    labels: Vec<String>,
}

struct Maxout {
    w: Vec<f32>,
    b: Vec<f32>,
    n_o: usize,
    n_p: usize,
    n_i: usize,
}

impl Maxout {
    fn apply(&self, x: &[f32], out: &mut [f32]) {
        for o in 0..self.n_o {
            let mut best = f32::NEG_INFINITY;
            for p in 0..self.n_p {
                let row = &self.w[(o * self.n_p + p) * self.n_i..][..self.n_i];
                let mut acc = self.b[o * self.n_p + p];
                for (v, wt) in x.iter().zip(row) {
                    acc += v * wt;
                }
                best = best.max(acc);
            }
            out[o] = best;
        }
    }
}

struct LayerNorm {
    g: Vec<f32>,
    b: Vec<f32>,
}

impl LayerNorm {
    fn apply(&self, x: &mut [f32]) {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n + EPS;
        let inv = var.powf(-0.5);
        for (i, v) in x.iter_mut().enumerate() {
            *v = (*v - mean) * inv * self.g[i] + self.b[i];
        }
    }
}

pub struct Tagger {
    meta: Meta,
    embed: Vec<Vec<f32>>,
    proj: Maxout,
    proj_ln: LayerNorm,
    enc: Vec<(Maxout, LayerNorm)>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
}

/// What the tagger reads off a token. `norm` is `Some` only when the tokenizer produced
/// this token from a special case that carries its own NORM ("ca" -> "can").
pub struct Tagged<'a> {
    pub text: &'a str,
    pub has_space: bool,
    pub norm: Option<&'a str>,
}

fn f32s(t: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let v = t.tensor(name).with_context(|| format!("missing tensor `{name}`"))?;
    Ok(v.data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn maxout(t: &SafeTensors, prefix: &str) -> Result<Maxout> {
    let view = t.tensor(&format!("{prefix}.W"))?;
    let s = view.shape();
    Ok(Maxout {
        w: f32s(t, &format!("{prefix}.W"))?,
        b: f32s(t, &format!("{prefix}.b"))?,
        n_o: s[0],
        n_p: s[1],
        n_i: s[2],
    })
}

fn layernorm(t: &SafeTensors, prefix: &str) -> Result<LayerNorm> {
    Ok(LayerNorm {
        g: f32s(t, &format!("{prefix}.G"))?,
        b: f32s(t, &format!("{prefix}.b"))?,
    })
}

impl Tagger {
    pub fn load(dir: &Path) -> Result<Self> {
        let meta: Meta = serde_json::from_slice(&std::fs::read(dir.join("tagger.json"))?)?;
        let buf = std::fs::read(dir.join("tagger.safetensors"))?;
        let t = SafeTensors::deserialize(&buf)?;
        let embed = meta
            .attrs
            .iter()
            .map(|a| f32s(&t, &format!("embed.{a}.E")))
            .collect::<Result<Vec<_>>>()?;
        let enc = (0..meta.depth)
            .map(|i| Ok((maxout(&t, &format!("enc.{i}"))?, layernorm(&t, &format!("enc.{i}.ln"))?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed,
            proj: maxout(&t, "proj")?,
            proj_ln: layernorm(&t, "proj.ln")?,
            enc,
            out_w: f32s(&t, "tagger.W")?,
            out_b: f32s(&t, "tagger.b")?,
            meta,
        })
    }

    pub fn labels(&self) -> &[String] {
        &self.meta.labels
    }

    /// The six attribute keys spaCy feeds the embedding tables.
    ///
    /// SPACY and IS_SPACE are the raw flags, not string hashes: hashing them produces a
    /// model that runs and is wrong on every token.
    fn keys(&self, tk: &Tagged, vocab: &Vocab) -> [u64; 6] {
        let text = tk.text;
        let norm = tk.norm.map(str::to_string).unwrap_or_else(|| vocab.norm(text));
        let prefix: String = text.chars().take(1).collect();
        let n = text.chars().count();
        let suffix: String = text.chars().skip(n.saturating_sub(3)).collect();
        [
            vocab.string_id(&norm),
            vocab.string_id(&prefix),
            vocab.string_id(&suffix),
            vocab.string_id(&word_shape(text)),
            tk.has_space as u64,
            (!text.is_empty() && text.chars().all(char::is_whitespace)) as u64,
        ]
    }

    pub fn tag(&self, tokens: &[Tagged], vocab: &Vocab) -> Vec<&str> {
        if tokens.is_empty() {
            return Vec::new();
        }
        let w = self.meta.width;
        let pad = self.meta.pad;
        let n = tokens.len();

        // Embed, then the maxout projection, straight into the padded buffer the residual
        // stack runs over: thinc's with_array pads once around the whole stack, and the pad
        // rows stop being zero after the first layer.
        let mut x = vec![0.0f32; (n + 2 * pad) * w];
        let mut wide = vec![0.0f32; self.proj.n_i];
        for (i, tk) in tokens.iter().enumerate() {
            wide.iter_mut().for_each(|v| *v = 0.0);
            for (j, key) in self.keys(tk, vocab).iter().enumerate() {
                let table = &self.embed[j];
                let (rows, seed) = (self.meta.rows[j], self.meta.seeds[j]);
                let dst = &mut wide[j * w..(j + 1) * w];
                for h in thinc_hash(*key, seed) {
                    let row = &table[(h as usize % rows) * w..][..w];
                    for (d, v) in dst.iter_mut().zip(row) {
                        *d += v;
                    }
                }
            }
            let row = &mut x[(i + pad) * w..][..w];
            self.proj.apply(&wide, row);
            self.proj_ln.apply(row);
        }

        let rows = n + 2 * pad;
        let mut window = vec![0.0f32; 3 * w];
        let mut delta = vec![0.0f32; rows * w];
        for (mx, ln) in &self.enc {
            for i in 0..rows {
                window[..w].copy_from_slice(&zeros_or(&x, i.checked_sub(1), rows, w));
                window[w..2 * w].copy_from_slice(&x[i * w..][..w]);
                window[2 * w..].copy_from_slice(&zeros_or(&x, Some(i + 1), rows, w));
                let out = &mut delta[i * w..][..w];
                mx.apply(&window, out);
                ln.apply(out);
            }
            for (a, d) in x.iter_mut().zip(delta.iter()) {
                *a += d;
            }
        }

        let n_labels = self.meta.labels.len();
        (0..n)
            .map(|i| {
                let h = &x[(i + pad) * w..][..w];
                let mut best = (0, f32::NEG_INFINITY);
                for l in 0..n_labels {
                    let row = &self.out_w[l * w..][..w];
                    let mut acc = self.out_b[l];
                    for (v, wt) in h.iter().zip(row) {
                        acc += v * wt;
                    }
                    if acc > best.1 {
                        best = (l, acc);
                    }
                }
                self.meta.labels[best.0].as_str()
            })
            .collect()
    }
}

fn zeros_or(x: &[f32], i: Option<usize>, rows: usize, w: usize) -> Vec<f32> {
    match i {
        Some(i) if i < rows => x[i * w..][..w].to_vec(),
        _ => vec![0.0; w],
    }
}

/// spaCy's word shape: one character per input character, classed, runs capped at four.
pub fn word_shape(text: &str) -> String {
    if text.chars().count() >= 100 {
        return "LONG".into();
    }
    let mut out = String::new();
    let mut last: Option<char> = None;
    let mut seq = 0;
    for c in text.chars() {
        let s = if c.is_alphabetic() {
            if c.is_uppercase() {
                'X'
            } else {
                'x'
            }
        } else if c.is_ascii_digit() {
            'd'
        } else {
            c
        };
        if Some(s) == last {
            seq += 1;
        } else {
            seq = 0;
            last = Some(s);
        }
        if seq < 4 {
            out.push(s);
        }
    }
    out
}

pub type Labels = HashMap<String, usize>;
