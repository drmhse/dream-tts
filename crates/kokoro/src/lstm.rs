//! A single-layer bidirectional LSTM, which candle does not provide.
//!
//! Six of these carry the prosody path. They are small — 256 hidden units over at most a
//! few hundred steps — so the only optimisation that matters is doing the input projection
//! for the whole sequence in one matmul and leaving the loop with just the recurrent term.

use anyhow::Result;
use candle_core::{Device, Tensor};
use tts_nn::Weights;

struct Direction {
    /// Transposed at load; every use is `x @ wᵀ`.
    w_ih: Tensor,
    /// torch's `[4H, H]`, as [`tts_nn::fused::lstm_seq`] reads it.
    w_hh: Tensor,
    /// torch keeps two bias vectors for symmetry with cuDNN. They are only ever added
    /// together, so they are summed once here.
    bias: Tensor,
}

impl Direction {
    fn load(w: &Weights, prefix: &str, suffix: &str) -> Result<Self> {
        let host = |n: &str| -> Result<Tensor> {
            Ok(w.cpu(&format!("{prefix}.{n}_l0{suffix}"))?
                .to_dtype(candle_core::DType::F32)?)
        };
        let hh = format!("{prefix}.weight_hh_l0{suffix}");
        Ok(Self {
            w_ih: w.get_t(&format!("{prefix}.weight_ih_l0{suffix}"))?,
            w_hh: w.get(&hh)?,
            bias: (host("bias_ih")? + host("bias_hh")?)?.to_device(w.device())?,
        })
    }

    /// `x` is `[T, input]`; returns `[T, hidden]`.
    fn run(&self, x: &Tensor, reverse: bool, hidden: usize, device: &Device) -> Result<Tensor> {
        let t = x.dim(0)?;
        let pre = x.matmul(&self.w_ih)?.broadcast_add(&self.bias)?;
        // `[2, hidden]`: h on row 0, c on row 1. Both rows are contiguous views, so the
        // recurrent matmul and the next step's cell read need no copy, and the whole step
        // is two dispatches — the matmul and `lstm_gates` — where composing candle's ops
        // took eleven. At ~3000 steps per utterance that was the cost of this stage.
        // `[2, hidden]` out of one kernel: the four gates, the cell update and the
        // output, where composing candle's ops took eleven dispatches per timestep. An
        // utterance runs a few thousand of them and they cost more than the arithmetic.
        let mut h = Tensor::zeros((1, hidden), x.dtype(), device)?;
        let mut c = h.clone();
        let mut steps: Vec<Tensor> = Vec::with_capacity(t);
        for i in 0..t {
            let idx = if reverse { t - 1 - i } else { i };
            let hc = tts_nn::fused::lstm_gates(
                &h.matmul(&self.w_hh.t()?)?,
                &pre.narrow(0, idx, 1)?,
                &c,
            )?;
            h = hc.narrow(0, 0, 1)?.contiguous()?;
            c = hc.narrow(0, 1, 1)?.contiguous()?;
            steps.push(h.clone());
        }
        if reverse {
            steps.reverse();
        }
        let out = Tensor::cat(&steps, 0)?;
        Ok(out)
    }
}

pub struct BiLstm {
    forward: Direction,
    backward: Direction,
    hidden: usize,
}

impl BiLstm {
    pub fn load(w: &Weights, prefix: &str) -> Result<Self> {
        let forward = Direction::load(w, prefix, "")?;
        let hidden = forward.w_hh.dim(1)?;
        Ok(Self {
            forward,
            backward: Direction::load(w, prefix, "_reverse")?,
            hidden,
        })
    }

    /// `x` is `[1, T, input]`; returns `[1, T, 2 * hidden]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let device = x.device().clone();
        let x = x.squeeze(0)?;
        if tts_nn::fused::lstm_seq_eligible(&x) {
            let pre = |d: &Direction| -> Result<Tensor> {
                Ok(x.matmul(&d.w_ih)?.broadcast_add(&d.bias)?)
            };
            let out = tts_nn::fused::lstm_seq(
                &pre(&self.forward)?,
                &pre(&self.backward)?,
                &self.forward.w_hh,
                &self.backward.w_hh,
            )?;
            return Ok(out.unsqueeze(0)?);
        }
        let f = self.forward.run(&x, false, self.hidden, &device)?;
        let b = self.backward.run(&x, true, self.hidden, &device)?;
        Ok(Tensor::cat(&[f, b], 1)?.unsqueeze(0)?)
    }
}
