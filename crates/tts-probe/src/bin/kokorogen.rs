//! Where the Kokoro generator's 85% actually goes.
//!
//! `kokoro-render` puts 0.65 s of a 0.75 s render in the decoder, and inside that,
//! stage 1 (128 ch @ 48240) is 0.45 s of it. This splits one resblock pass at those
//! real shapes into gather, GEMM, AdaIN and snake, so the next change is aimed at
//! whichever one is actually paying.
//!
//! Run: `cargo run -p tts-probe --release --bin kokorogen`

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use tts_bench::Harness;

fn cols(x: &Tensor, k: usize, dil: usize) -> candle_core::Result<Tensor> {
    tts_nn::im2col::im2col_centered(x, k, dil, (k - 1) * dil / 2)
}

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let mut h = Harness::new(&dev, 7)?;

    for (label, c, len) in [
        ("stage0 256ch@8040", 256usize, 8040usize),
        ("stage1 128ch@48240", 128, 48240),
    ] {
        let x = Tensor::randn(0f32, 1.0, (1, c, len), &dev)?;
        let s = Tensor::randn(0f32, 1.0, (1, 128), &dev)?;
        let alpha = Tensor::randn(0f32, 1.0, (1, c, 1), &dev)?.abs()?;
        let recip = alpha.recip()?;

        // Per-kernel conv, and its two halves.
        for k in [3usize, 7, 11] {
            let w_tap = Tensor::randn(0f32, 0.02, (c, k * c), &dev)?;
            let cols_k = cols(&x, k, 1)?;
            let mut variants: Vec<(&str, &mut dyn FnMut() -> candle_core::Result<()>)> = Vec::new();
            let mut a = || {
                tts_nn::centered_conv1d_gemm(&x, &w_tap, None, k, 1, (k - 1) / 2).unwrap();
                Ok(())
            };
            let mut b = || {
                cols(&x, k, 1)?;
                Ok(())
            };
            let mut cc = || {
                w_tap.matmul(&cols_k)?;
                Ok(())
            };
            let mut dd = || {
                tts_nn::tapconv::centered_conv1d_fused_with(
                    &x,
                    &w_tap,
                    None,
                    k,
                    1,
                    (k - 1) / 2,
                    "conv1d_tap_gemm_f32",
                )
                .unwrap();
                Ok(())
            };
            let mut ee = || {
                tts_nn::tapconv::centered_conv1d_fused(&x, &w_tap, None, k, 1, (k - 1) / 2)
                    .unwrap();
                Ok(())
            };
            variants.push(("conv whole", &mut a));
            variants.push(("  im2col only", &mut b));
            variants.push(("  gemm only", &mut cc));
            variants.push(("fused simdgroup", &mut dd));
            variants.push(("fused register tile", &mut ee));
            h.ab(&format!("{label} k={k} d=1"), &mut variants)?;
        }

        // Elementwise stack, one pass each over the same tensor.
        let mean = x.mean_keepdim(2)?;
        let var = tts_nn::fused::sub_sqr(&x, &mean)?.mean_keepdim(2)?;
        let gamma = Tensor::randn(0f32, 1.0, (c,), &dev)?;
        let beta = Tensor::randn(0f32, 1.0, (c,), &dev)?;
        let mut variants: Vec<(&str, &mut dyn FnMut() -> candle_core::Result<()>)> = Vec::new();
        let mut e1 = || {
            x.mean_keepdim(2)?;
            Ok(())
        };
        let mut e2 = || {
            tts_nn::fused::sub_sqr(&x, &mean)?.mean_keepdim(2)?;
            Ok(())
        };
        let mut e3 = || {
            tts_nn::fused::adain_apply(&x, &mean, &var, &gamma, &beta, 1e-5)?;
            Ok(())
        };
        let mut e4 = || {
            tts_nn::fused::snake_beta(&x, &alpha, &recip)?;
            Ok(())
        };
        let mut e5 = || {
            (&x + &x)?;
            Ok(())
        };
        variants.push(("mean_keepdim", &mut e1));
        variants.push(("sub_sqr+mean", &mut e2));
        variants.push(("adain_apply", &mut e3));
        variants.push(("snake_beta", &mut e4));
        let mut e6 = || {
            tts_nn::fused::moments(&x)?;
            Ok(())
        };
        let mut e7 = || {
            tts_nn::fused::adain_snake(
                &x,
                &mean.flatten_all()?,
                &var.flatten_all()?,
                &gamma,
                &beta,
                &alpha.flatten_all()?,
                &recip.flatten_all()?,
                1e-5,
            )?;
            Ok(())
        };
        variants.push(("add", &mut e5));
        variants.push(("moments (vs mean+sub_sqr)", &mut e6));
        variants.push(("adain_snake (vs both)", &mut e7));
        h.ab(&format!("{label} elementwise"), &mut variants)?;

        // GEMM shape study: does widening M (three kernels at once) pay, and does f16?
        let k = 11;
        let cols_k = cols(&x, k, 1)?;
        let w1 = Tensor::randn(0f32, 0.02, (c, k * c), &dev)?;
        let w3 = Tensor::randn(0f32, 0.02, (3 * c, k * c), &dev)?;
        let cols_h = cols_k.to_dtype(DType::F16)?;
        let w1h = w1.to_dtype(DType::F16)?;
        let w3h = w3.to_dtype(DType::F16)?;
        let mut variants: Vec<(&str, &mut dyn FnMut() -> candle_core::Result<()>)> = Vec::new();
        let mut g1 = || {
            w1.matmul(&cols_k)?;
            Ok(())
        };
        let mut g2 = || {
            w3.matmul(&cols_k)?;
            Ok(())
        };
        let mut g3 = || {
            w1h.matmul(&cols_h)?;
            Ok(())
        };
        let mut g4 = || {
            w3h.matmul(&cols_h)?;
            Ok(())
        };
        variants.push(("f32 M=c", &mut g1));
        variants.push(("f32 M=3c (3 kernels)", &mut g2));
        variants.push(("f16 M=c", &mut g3));
        variants.push(("f16 M=3c", &mut g4));
        h.ab(&format!("{label} k=11 gemm shapes"), &mut variants)?;
        let _ = &s;
    }

    h.report_drift()?;
    Ok(())
}
