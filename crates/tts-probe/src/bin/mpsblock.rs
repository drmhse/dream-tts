//! A padded SnakeResBlock: one MPSGraph against the composed route, at the generator's shapes.
//!
//! Run: `cargo run -p tts-probe --release --bin mpsblock`

use anyhow::Result;
use candle_core::{Device, Tensor};
use tts_nn::mpsblock::{self, ConvShape};

#[allow(clippy::needless_range_loop)]
fn main() -> Result<()> {
    let d = Device::new_metal(0)?;
    let eps = 1e-5f32;
    for (c, len) in [
        (256usize, 1024usize),
        (128, 6144),
        (256, 3584),
        (128, 21504),
        (256, 8192),
        (128, 49152),
    ] {
        let valid = len - len / 16;
        for k in [3usize, 7, 11] {
            let dils = [1usize, 3, 5];
            let x = Tensor::randn(0f32, 1., (1, c, len), &d)?;
            let gb = Tensor::randn(0f32, 0.3, (6, 2, c), &d)?;
            let ab = Tensor::rand(0.5f32, 1.5, (6, 2, c), &d)?;
            let params = Tensor::cat(&[&gb, &ab], 1)?;
            let convs: Vec<(Tensor, Tensor)> = (0..6)
                .map(|_| {
                    Ok((
                        Tensor::randn(0f32, 0.02, (k, c, c), &d)?,
                        Tensor::randn(0f32, 0.1, c, &d)?,
                    ))
                })
                .collect::<Result<_>>()?;
            let composed = || -> Result<Tensor> {
                let mut x = x.clone();
                for i in 0..3 {
                    let a = tts_nn::fused::adain_snake_masked(
                        &x,
                        &gb.get(2 * i)?,
                        &ab.get(2 * i)?,
                        eps as f64,
                        valid,
                    )?;
                    let conv =
                        |a: &Tensor, j: usize, dl: usize, r: Option<&Tensor>| -> Result<Tensor> {
                            let (w, b) = &convs[j];
                            if tts_nn::mpsconv::eligible(a, w) {
                                return tts_nn::mpsconv::centered_conv1d_residual(
                                    a,
                                    w,
                                    Some(b),
                                    dl,
                                    (k - 1) * dl / 2,
                                    r,
                                );
                            }
                            let y = tts_nn::centered_conv1d_gemm(
                                a,
                                &w.reshape((k * c, c))?.t()?,
                                Some(b),
                                k,
                                dl,
                                (k - 1) * dl / 2,
                            )?;
                            Ok(match r {
                                Some(r) => (y + r)?,
                                None => y,
                            })
                        };
                    let t = conv(&a, 2 * i, dils[i], None)?;
                    let a = tts_nn::fused::adain_snake_masked(
                        &t,
                        &gb.get(2 * i + 1)?,
                        &ab.get(2 * i + 1)?,
                        eps as f64,
                        valid,
                    )?;
                    x = conv(&a, 2 * i + 1, 1, Some(&x))?;
                }
                Ok(x)
            };
            let pairs = dils
                .iter()
                .map(|&dl| (ConvShape { k, dilation: dl }, ConvShape { k, dilation: 1 }))
                .collect();
            let key = mpsblock::key(&d, c, pairs, eps).unwrap();
            let refs: Vec<(&Tensor, &Tensor)> = convs.iter().map(|(w, b)| (w, b)).collect();
            let fused = || -> Result<Tensor> { mpsblock::apply(&key, &x, &params, &refs, valid) };
            let time = |f: &dyn Fn() -> Result<Tensor>| -> Result<(f64, Tensor)> {
                let mut out = f()?;
                d.synchronize()?;
                let mut best = f64::MAX;
                for _ in 0..3 {
                    let t = std::time::Instant::now();
                    for _ in 0..5 {
                        out = f()?;
                    }
                    d.synchronize()?;
                    best = best.min(t.elapsed().as_secs_f64() / 5.0);
                }
                Ok((best * 1e3, out))
            };
            let (a, ya) = time(&composed)?;
            let (b, yb) = time(&fused)?;
            let diff = (ya.narrow(2, 0, valid)? - yb.narrow(2, 0, valid)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            println!("{c}ch@{len} k={k}: composed {a:.2} ms, one graph {b:.2} ms, {:.2}x, max diff {diff:.2e}", a / b);
        }
    }
    Ok(())
}
