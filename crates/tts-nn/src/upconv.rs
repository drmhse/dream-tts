//! A transposed convolution with stride as polyphase forward convolutions.
//!
//! Candle's Metal `conv_transpose1d` takes its col2im fast route only when
//! `padding == 0 && output_padding == 0`; with padding it falls back to a direct
//! kernel measured at **~0.3 s per call** on Kokoro's two upsamples — half that
//! engine's decoder in two ops, at shapes whose arithmetic is ~25 GFLOP and should
//! take ~12 ms.
//!
//! The polyphase identity instead: output `t = q*s + r` only ever meets kernel taps
//! `k = k1 + m*s` with `k1 = (r + p) mod s`, so each of the `s` phases is an
//! ordinary stride-1 convolution of the input with a `K/s`-tap sub-kernel, and the
//! phases interleave back into the output. Each phase runs as a tap loop of dense
//! GEMMs — no im2col, no zero-stuffed intermediate — through [`matmul_2d`], which is
//! already the fastest shape on this backend. Kokoro's `ups.0` goes from 0.314 s to
//! ~1 ms.
//!
//! The summation order differs from the scatter kernel (taps accumulated per phase
//! rather than contributions accumulated per output), so this is close rather than
//! bit-identical: the tests check it against candle's own `conv_transpose1d` at
//! rel < 1e-5, which is four orders of magnitude below anything the audio gate can
//! see.

use crate::Weights;
use anyhow::Result;
use candle_core::Tensor;

/// One output phase: `out_r[q] = sum_m W[m] @ x[q + c' - m]` (zero outside the
/// input), with `c' = floor((r + p) / s)`.
struct Phase {
    /// `[out, in]` per tap, contiguous.
    taps: Vec<Tensor>,
    /// Left zero-pad on the input.
    pad_left: usize,
    /// `c'`: how far the first tap reaches back.
    shift: usize,
}

pub struct UpConv {
    stride: usize,
    padding: usize,
    output_padding: usize,
    k_size: usize,
    out_ch: usize,
    phases: Vec<Phase>,
    bias: Option<Tensor>,
    /// The original weight and geometry, for the shapes the phases do not cover
    /// (grouped, or phases of uneven length at some input length).
    w: Tensor,
    groups: usize,
}

impl UpConv {
    pub fn load(
        w: &Weights,
        prefix: &str,
        stride: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
    ) -> Result<Self> {
        let weight = w.get(&format!("{prefix}.weight"))?;
        let bias = w.get_opt(&format!("{prefix}.bias"))?;
        Self::from_tensors(weight, bias, stride, padding, output_padding, groups)
    }

    pub fn from_tensors(
        weight: Tensor,
        bias: Option<Tensor>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
    ) -> Result<Self> {
        let (_cin, cout, k_size) = weight.dims3()?;
        let mut phases = Vec::new();
        // Only the ungrouped case decomposes; anything else keeps candle's
        // implementation via the empty phase list.
        if groups == 1 {
            phases.reserve(stride);
            for r in 0..stride {
                let k1 = (r + padding) % stride;
                let shift = (r + padding - k1) / stride;
                let taps_idx: Vec<usize> = (0..)
                    .map(|m| k1 + m * stride)
                    .take_while(|&k| k < k_size)
                    .collect();
                // A phase with no taps (e.g. k=1, s=2) outputs zeros.
                let mut taps = Vec::with_capacity(taps_idx.len());
                for &k in &taps_idx {
                    // `[out, in]`: the window squeezes to `[in, len_p]`.
                    taps.push(weight.narrow(2, k, 1)?.squeeze(2)?.t()?.contiguous()?);
                }
                phases.push(Phase {
                    pad_left: if taps.is_empty() {
                        0
                    } else {
                        // The earliest tap reaches `k_r - 1 - c'` before the input.
                        (taps_idx.len() as isize - 1 - shift as isize).max(0) as usize
                    },
                    taps,
                    shift,
                });
            }
        }
        Ok(Self {
            stride,
            padding,
            output_padding,
            k_size,
            out_ch: cout,
            phases,
            bias,
            w: weight,
            groups,
        })
    }

    /// `weight` is `[in, out, k]`, the transposed-conv layout.

    fn l_out(&self, l_in: usize) -> usize {
        (l_in - 1) * self.stride - 2 * self.padding + (self.k_size - 1) + self.output_padding + 1
    }

    fn fallback(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv_transpose1d(
            &self.w,
            self.padding,
            self.output_padding,
            self.stride,
            1,
            self.groups,
        )?;
        Ok(match &self.bias {
            Some(b) => y.broadcast_add(&b.reshape((1, b.dim(0)?, 1))?)?,
            None => y,
        })
    }

    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _, l_in) = x.dims3()?;
        if self.phases.is_empty() || b != 1 {
            return self.fallback(x);
        }
        let l_out = self.l_out(l_in);
        // Every phase owns the outputs `t = q*s + r < l_out`: all phases must agree
        // on a length, or the interleave below has nothing regular to stack.
        let len_p = if l_out > 0 {
            (l_out - 1) / self.stride + 1
        } else {
            0
        };
        for r in 0..self.stride {
            let len_r = if r < l_out {
                (l_out - 1 - r) / self.stride + 1
            } else {
                0
            };
            if len_r != len_p {
                return self.fallback(x);
            }
        }
        if len_p == 0 {
            return self.fallback(x);
        }
        let mut stacked = Vec::with_capacity(self.stride);
        for ph in &self.phases {
            if ph.taps.is_empty() {
                stacked.push(Tensor::zeros(
                    (self.out_ch, len_p),
                    candle_core::DType::F32,
                    x.device(),
                )?);
                continue;
            }
            // Right pad only when the last window overhangs the input: its top
            // index is `len_p - 1 + c'`, the input's is `l_in - 1`.
            let overhang = (len_p as isize - 1 + ph.shift as isize) - (l_in as isize - 1);
            let right = overhang.max(0) as usize;
            let xp = if ph.pad_left == 0 && right == 0 {
                x.clone()
            } else {
                x.pad_with_zeros(2, ph.pad_left, right)?
            };
            let mut acc: Option<Tensor> = None;
            for (m, w) in ph.taps.iter().enumerate() {
                // Materialise: Metal's matmul validates the strided narrow view
                // against the wrong shape and refuses it.
                let win = xp
                    .narrow(2, ph.pad_left + ph.shift - m, len_p)?
                    .contiguous()?
                    .squeeze(0)?;
                let cin = w.dim(1)?;
                crate::stats::record(
                    self.out_ch,
                    cin,
                    len_p,
                    ((self.out_ch * cin + cin * len_p + self.out_ch * len_p) * 4) as u64,
                );
                let term = w.matmul(&win)?.unsqueeze(0)?;
                acc = Some(match acc {
                    None => term,
                    Some(a) => (a + term)?,
                });
            }
            stacked.push(acc.expect("a phase always has a tap").squeeze(0)?);
        }
        // `[s, out, len_p]` -> `[out, len_p, s]` -> `[out, l_out]`.
        let y = Tensor::stack(&stacked, 0)?
            .permute((1, 2, 0))?
            .contiguous()?
            .reshape((self.out_ch, l_out))?
            .unsqueeze(0)?;
        Ok(match &self.bias {
            Some(b) => y.broadcast_add(&b.reshape((1, b.dim(0)?, 1))?)?,
            None => y,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn check(
        dev: &Device,
        cout: usize,
        cin: usize,
        k: usize,
        stride: usize,
        padding: usize,
        out_pad: usize,
        l_in: usize,
    ) -> anyhow::Result<()> {
        let x = Tensor::randn(0f32, 1.0, (1, cin, l_in), dev)?;
        let w = Tensor::randn(0f32, 0.1, (cin, cout, k), dev)?;
        let b = Tensor::randn(0f32, 1.0, cout, dev)?;
        let want = x
            .conv_transpose1d(&w, padding, out_pad, stride, 1, 1)?
            .broadcast_add(&b.reshape((1, cout, 1))?)?;

        let up = UpConv::from_tensors(w, Some(b), stride, padding, out_pad, 1)?;
        assert!(
            !up.phases.is_empty(),
            "test shape should take the fast path"
        );
        let got = up.apply(&x)?;
        assert_eq!(got.dims(), want.dims());
        let (abs, rel) = crate::abs_and_rel(&got, &want)?;
        assert!(rel < 1e-5, "cout={cout} cin={cin} k={k} s={stride} p={padding} op={out_pad} l={l_in}: abs {abs:.2e} rel {rel:.2e}");
        Ok(())
    }

    #[test]
    fn matches_candle_on_cpu() -> anyhow::Result<()> {
        let dev = Device::Cpu;
        // Kokoro's two upsamples at real lengths, plus edges.
        for &l in &[1usize, 7, 138, 444] {
            check(&dev, 256, 512, 20, 10, 5, 0, l)?;
            check(&dev, 128, 256, 12, 6, 3, 0, l)?;
        }
        // Kernel not a multiple of stride, padding extremes, output padding.
        check(&dev, 32, 16, 7, 4, 2, 0, 53)?;
        check(&dev, 32, 16, 5, 3, 0, 0, 53)?;
        check(&dev, 16, 8, 4, 2, 1, 1, 53)?;
        check(&dev, 8, 8, 1, 2, 0, 0, 53)?;
        Ok(())
    }

    #[test]
    fn matches_candle_on_metal() -> anyhow::Result<()> {
        #[cfg(feature = "metal")]
        let _gpu = crate::gpu_guard();
        let Some(dev) = crate::usable_metal() else {
            return Ok(());
        };
        for &l in &[138usize, 444, 1400] {
            check(&dev, 256, 512, 20, 10, 5, 0, l)?;
            check(&dev, 128, 256, 12, 6, 3, 0, l)?;
        }
        Ok(())
    }

    #[test]
    fn grouped_falls_back() -> anyhow::Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (1, 8, 32), &dev)?;
        let w = Tensor::randn(0f32, 0.1, (8, 1, 3), &dev)?;
        let up = UpConv::from_tensors(w.clone(), None, 2, 1, 1, 8)?;
        assert!(up.phases.is_empty());
        let got = up.apply(&x)?;
        let want = x.conv_transpose1d(&w, 1, 1, 2, 1, 8)?;
        let (abs, rel) = crate::abs_and_rel(&got, &want)?;
        assert!(rel < 1e-6, "fallback differs: abs {abs:.2e} rel {rel:.2e}");
        Ok(())
    }
}
