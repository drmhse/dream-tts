//! The StyleTTS2 building blocks: convolutions conditioned on a style vector.
//!
//! Everything here works on `[1, C, T]`. The voice enters only through `AdaIN1d` and
//! `AdaLayerNorm`, which is what makes a voice a 256-float asset rather than a model.

use anyhow::Result;
use candle_core::Tensor;
use tts_nn::Weights;

const NORM_EPS: f64 = 1e-5;
const LEAK: f64 = 0.2;

pub struct Linear {
    w: Tensor,
    b: Option<Tensor>,
}

impl Linear {
    pub fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w: w.get_t(&format!("{prefix}.weight"))?,
            b: w.get_opt(&format!("{prefix}.bias"))?,
        })
    }

    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.broadcast_matmul(&self.w)?;
        Ok(match &self.b {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        })
    }
}

/// A same-padded 1-D convolution. Kokoro's kernels are all odd, so the padding is exact
/// and there is no asymmetric case to get wrong.
pub struct Conv1d {
    weight: ConvWeight,
    b: Option<Tensor>,
    padding: usize,
    stride: usize,
    dilation: usize,
    k: usize,
}

/// One layout per conv, never both: a second copy of every weight was a third of the model.
enum ConvWeight {
    /// `[k, in, out]`, for a centred stride-1 conv: MPSGraph's HWIO, and transposed it is the
    /// `[out, k * in]` the gather GEMM wants.
    Kio(Tensor),
    /// candle's `[out, in, k]`, for everything else.
    Oik(Tensor),
}

/// `[out, in, k]` to `[k, in, out]` by hand, a tap plane per thread: candle's strided copy
/// made loading twice as slow.
fn kio(w: &Tensor) -> Result<Tensor> {
    let (cout, cin, k) = w.dims3()?;
    let src: Vec<f32> = w.flatten_all()?.to_vec1()?;
    let mut dst = vec![0f32; src.len()];
    std::thread::scope(|sc| {
        for (j, plane) in dst.chunks_mut(cin * cout).enumerate() {
            let src = &src;
            sc.spawn(move || {
                for i in 0..cin {
                    for o in 0..cout {
                        plane[i * cout + o] = src[(o * cin + i) * k + j];
                    }
                }
            });
        }
    });
    Ok(Tensor::from_vec(dst, (k, cin, cout), w.device())?)
}

impl Conv1d {
    pub fn load(w: &Weights, prefix: &str, stride: usize, dilation: usize) -> Result<Self> {
        Self::load_padded(w, prefix, stride, dilation, None)
    }

    /// `padding` defaults to the same-length one.
    pub fn load_padded(
        w: &Weights,
        prefix: &str,
        stride: usize,
        dilation: usize,
        padding: Option<usize>,
    ) -> Result<Self> {
        let host = w
            .cpu(&format!("{prefix}.weight"))?
            .to_dtype(candle_core::DType::F32)?;
        let k = host.dim(2)?;
        let padding = padding.unwrap_or((k * dilation - dilation) / 2);
        let centred = padding * 2 == (k - 1) * dilation && stride == 1;
        // Rearranged on the host: on the device the copy would stay in candle's pool.
        let weight = if centred {
            ConvWeight::Kio(kio(&host)?.to_device(w.device())?)
        } else {
            ConvWeight::Oik(host.to_device(w.device())?)
        };
        Ok(Self {
            weight,
            b: w.get_opt(&format!("{prefix}.bias"))?,
            padding,
            stride,
            dilation,
            k,
        })
    }

    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        self.apply_residual(x, None)
    }

    /// `(kio, bias, k, dilation)` for a centred, biased conv — what a fused block graph takes.
    pub fn centred_parts(&self) -> Option<(&Tensor, &Tensor, usize, usize)> {
        match (&self.weight, &self.b) {
            (ConvWeight::Kio(kio), Some(b)) => Some((kio, b, self.k, self.dilation)),
            _ => None,
        }
    }

    /// The MPSGraph conv [`Self::apply_residual`] will run at `len`, if any.
    pub fn mps_spec(&self, len: usize, residual: bool) -> Option<tts_nn::mpsconv::Spec> {
        let ConvWeight::Kio(kio) = &self.weight else {
            return None;
        };
        if std::env::var("KOKORO_NO_MPS").is_ok()
            || !tts_nn::mpsconv::eligible_len(kio.device(), len)
        {
            return None;
        }
        let (k, cin, cout) = kio.dims3().ok()?;
        Some(tts_nn::mpsconv::Spec {
            cin,
            cout,
            k,
            dilation: self.dilation,
            pad_left: self.padding,
            bias: self.b.is_some(),
            residual,
            len,
        })
    }

    /// The conv plus `residual`, which the MPSGraph route adds inside its graph.
    pub fn apply_residual(&self, x: &Tensor, residual: Option<&Tensor>) -> Result<Tensor> {
        let add = |y: Tensor| -> Result<Tensor> {
            Ok(match residual {
                Some(r) => (y + r)?,
                None => y,
            })
        };
        let w = match &self.weight {
            // Centred, stride 1 — the shape every convolution in the decoder has, and the one
            // candle's Metal conv1d is worst at. The centred gather folds both edge pads, so
            // there is no pre-pad in and no re-centring slice out.
            ConvWeight::Kio(kio) if x.dim(0)? == 1 => {
                if std::env::var("KOKORO_NO_MPS").is_err() && tts_nn::mpsconv::eligible(x, kio) {
                    return tts_nn::mpsconv::centered_conv1d_residual(
                        x,
                        kio,
                        self.b.as_ref(),
                        self.dilation,
                        self.padding,
                        residual,
                    );
                }
                let (_, cin, cout) = kio.dims3()?;
                return add(tts_nn::centered_conv1d_gemm(
                    x,
                    &kio.reshape((self.k * cin, cout))?.t()?,
                    self.b.as_ref(),
                    self.k,
                    self.dilation,
                    self.padding,
                )?);
            }
            ConvWeight::Kio(kio) => kio.permute((2, 1, 0))?.contiguous()?,
            ConvWeight::Oik(w) => w.clone(),
        };
        let y = x.conv1d(&w, self.padding, self.stride, self.dilation, 1)?;
        add(match &self.b {
            Some(b) => y.broadcast_add(&b.reshape((1, b.dim(0)?, 1))?)?,
            None => y,
        })
    }
}

pub struct ConvTranspose1d {
    up: tts_nn::upconv::UpConv,
}

impl ConvTranspose1d {
    pub fn load(
        w: &Weights,
        prefix: &str,
        stride: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
    ) -> Result<Self> {
        Ok(Self {
            up: tts_nn::upconv::UpConv::load(w, prefix, stride, padding, output_padding, groups)?,
        })
    }

    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        self.up.apply(x)
    }

    pub fn stride(&self) -> usize {
        self.up.stride()
    }

    pub fn mps_spec(&self, l_in: usize) -> Option<tts_nn::mpsconv::Spec> {
        self.up.mps_spec(l_in)
    }
}

/// LayerNorm over the channel axis of a `[1, C, T]` tensor.
pub struct ChannelNorm {
    gamma: Tensor,
    beta: Tensor,
}

impl ChannelNorm {
    pub fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gamma: w.get(&format!("{prefix}.gamma"))?,
            beta: w.get(&format!("{prefix}.beta"))?,
        })
    }

    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let y = tts_nn::layer_norm(
            &x.transpose(1, 2)?.contiguous()?,
            &self.gamma,
            &self.beta,
            NORM_EPS,
        )?;
        Ok(y.transpose(1, 2)?.contiguous()?)
    }
}

/// Instance norm plus a style-predicted scale and shift.
///
/// The instance norm is declared `affine=True` upstream — a workaround for an old ONNX
/// exporter — but the checkpoint has no weights for it, and the model is loaded with
/// `strict=False`, so those parameters stay at their initialisation and the affine is an
/// identity. Loading them is optional here for exactly that reason: requiring them fails,
/// and inventing them changes every channel's gain.
pub struct AdaIn {
    fc: Linear,
    affine: Option<(Tensor, Tensor)>,
}

impl AdaIn {
    pub fn load(w: &Weights, prefix: &str) -> Result<Self> {
        let affine = match w.get_opt(&format!("{prefix}.norm.weight"))? {
            Some(weight) => Some((weight, w.get(&format!("{prefix}.norm.bias"))?)),
            None => None,
        };
        Ok(Self {
            fc: Linear::load(w, &format!("{prefix}.fc"))?,
            affine,
        })
    }

    /// The style-predicted `(gamma, beta)`, each `[C]`.
    fn scale_shift(&self, s: &Tensor, c: usize) -> Result<(Tensor, Tensor)> {
        let h = self.fc.apply(s)?.reshape((2 * c,))?;
        Ok((h.narrow(0, 0, c)?, h.narrow(0, c, c)?))
    }

    /// This AdaIN with the SnakeBeta that always follows it in the generator, one pass.
    pub fn apply_snake(
        &self,
        x: &Tensor,
        s: &Tensor,
        alpha: &Tensor,
        beta_recip: &Tensor,
    ) -> Result<Tensor> {
        let c = x.dim(1)?;
        if self.affine.is_some() {
            let y = self.apply(x, s)?;
            return Ok(tts_nn::fused::snake_beta(&y, alpha, beta_recip)?);
        }
        let (gamma, beta) = self.scale_shift(s, c)?;
        let m = tts_nn::fused::moments(x)?;
        Ok(tts_nn::fused::adain_snake(
            x,
            &m.narrow(0, 0, 1)?,
            &m.narrow(0, 1, 1)?,
            &gamma,
            &beta,
            &alpha.flatten_all()?,
            &beta_recip.flatten_all()?,
            NORM_EPS,
        )?)
    }

    pub fn apply(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let c = x.dim(1)?;
        // The fused halves below replace four broadcasts and a division with
        // direct per-channel indexing. The affine case is absent from this
        // checkpoint and stays on the composed path.
        if self.affine.is_none() {
            let (gamma, beta) = self.scale_shift(s, c)?;
            let m = tts_nn::fused::moments(x)?;
            let (mean, var) = (m.narrow(0, 0, 1)?, m.narrow(0, 1, 1)?);
            return Ok(tts_nn::fused::adain_apply(
                x, &mean, &var, &gamma, &beta, NORM_EPS,
            )?);
        }
        let h = self.fc.apply(s)?.reshape((1, 2 * c, 1))?;
        let gamma = h.narrow(1, 0, c)?;
        let beta = h.narrow(1, c, c)?;
        let mean = x.mean_keepdim(2)?;
        let centred = x.broadcast_sub(&mean)?;
        let var = centred.sqr()?.mean_keepdim(2)?;
        let mut normed = centred.broadcast_div(&(var + NORM_EPS)?.sqrt()?)?;
        if let Some((weight, bias)) = &self.affine {
            normed = normed
                .broadcast_mul(&weight.reshape((1, c, 1))?)?
                .broadcast_add(&bias.reshape((1, c, 1))?)?;
        }
        Ok((normed.broadcast_mul(&(gamma + 1.0)?)?).broadcast_add(&beta)?)
    }
}

/// LayerNorm over channels, then a style-predicted scale and shift. Unlike [`AdaIn`] the
/// normalisation here has no parameters of its own.
pub struct AdaLayerNorm {
    fc: Linear,
}

impl AdaLayerNorm {
    pub fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc: Linear::load(w, &format!("{prefix}.fc"))?,
        })
    }

    pub fn apply(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let c = x.dim(1)?;
        let h = self.fc.apply(s)?.reshape((1, 2 * c, 1))?;
        let gamma = h.narrow(1, 0, c)?.transpose(1, 2)?;
        let beta = h.narrow(1, c, c)?.transpose(1, 2)?;
        let y = tts_nn::layer_norm_plain(&x.transpose(1, 2)?.contiguous()?, NORM_EPS)?;
        let y = (y.broadcast_mul(&(gamma + 1.0)?)?).broadcast_add(&beta)?;
        Ok(y.transpose(1, 2)?.contiguous()?)
    }
}

/// The residual block the whole decoder is built from.
///
/// `upsample` doubles the length: the residual path uses a grouped transposed convolution
/// and the shortcut a nearest-neighbour repeat, which is not the same operation — the
/// block learns the difference.
pub struct AdainResBlk {
    norm1: AdaIn,
    norm2: AdaIn,
    conv1: Conv1d,
    conv2: Conv1d,
    conv1x1: Option<Conv1d>,
    pool: Option<ConvTranspose1d>,
}

impl AdainResBlk {
    pub fn load(w: &Weights, prefix: &str, dim_in: usize, upsample: bool) -> Result<Self> {
        let conv1x1 = w
            .has(&format!("{prefix}.conv1x1.weight"))
            .then(|| Conv1d::load(w, &format!("{prefix}.conv1x1"), 1, 1))
            .transpose()?;
        let pool = upsample
            .then(|| ConvTranspose1d::load(w, &format!("{prefix}.pool"), 2, 1, 1, dim_in))
            .transpose()?;
        Ok(Self {
            norm1: AdaIn::load(w, &format!("{prefix}.norm1"))?,
            norm2: AdaIn::load(w, &format!("{prefix}.norm2"))?,
            conv1: Conv1d::load(w, &format!("{prefix}.conv1"), 1, 1)?,
            conv2: Conv1d::load(w, &format!("{prefix}.conv2"), 1, 1)?,
            conv1x1,
            pool,
        })
    }

    pub fn apply(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let mut r = tts_nn::leaky_relu(&self.norm1.apply(x, s)?, LEAK)?;
        if let Some(pool) = &self.pool {
            r = pool.apply(&r)?;
        }
        r = self.conv1.apply(&r)?;
        r = tts_nn::leaky_relu(&self.norm2.apply(&r, s)?, LEAK)?;
        r = self.conv2.apply(&r)?;

        let mut shortcut = x.clone();
        if self.pool.is_some() {
            shortcut = tts_nn::upsample_nearest1d(&shortcut, 2)?;
        }
        if let Some(conv) = &self.conv1x1 {
            shortcut = conv.apply(&shortcut)?;
        }
        Ok(((r + shortcut)? * (0.5f64).sqrt())?)
    }
}
