//! What rate does candle's Metal f16 GEMM actually reach at a decode step's shapes?
//!
//! The talker is 62% of a qwen3tts render and, at 48 lanes, achieves 1.53 TFLOP/s in its
//! trunk and 0.88 in its depth predictor against an M4's ~4.3. Neither is bandwidth: the
//! trunk reads 2.8 GB per step, 23 ms on a 120 GB/s bus, and takes 87.6. This asks whether
//! the shortfall is the shape (small M) or candle's GEMM in general, by putting the same
//! kernel on a large square problem where nothing is latency-bound.
//!
//! Run:  cargo run -p tts-probe --release --bin gemm

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use tts_probe::bench::Harness;

const SAMPLES: usize = 5;
const ITERS: usize = 50;
const CHAIN: usize = 500;

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let mut h = Harness::new(&dev, SAMPLES)?;

    // (label, m, k, n)
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("trunk qkv    48x2048x4096", 48, 2048, 4096),
        ("trunk gate   48x2048x6144", 48, 2048, 6144),
        ("trunk down   48x6144x2048", 48, 6144, 2048),
        ("pred  qkv    48x1024x4096", 48, 1024, 4096),
        ("pred  gate   48x1024x3072", 48, 1024, 3072),
        ("pred  down   48x3072x1024", 48, 3072, 1024),
        ("pred  head   48x1024x2048", 48, 1024, 2048),
        ("wide  trunk 512x2048x6144", 512, 2048, 6144),
        ("square     2048x2048x2048", 2048, 2048, 2048),
    ];

    let mut prepared = Vec::new();
    for &(label, m, k, n) in shapes {
        let a = Tensor::randn(0f32, 1.0, (m, k), &dev)?.to_dtype(DType::F16)?;
        let b = Tensor::randn(0f32, 0.02, (k, n), &dev)?.to_dtype(DType::F16)?;
        prepared.push((label, m, k, n, a, b));
    }
    dev.synchronize()?;

    for (label, m, k, n, a, b) in &prepared {
        let mut f = || -> candle_core::Result<()> {
            for _ in 0..ITERS {
                let _ = a.matmul(b)?;
            }
            Ok(())
        };
        let stats = h.ab(
            label,
            &mut [(
                *label,
                &mut f as &mut dyn FnMut() -> candle_core::Result<()>,
            )],
        )?;
        let ms = stats[0].median;
        let per = ms / ITERS as f64;
        let gflop = 2.0 * (*m as f64) * (*k as f64) * (*n as f64) / 1e9;
        let bytes = 2.0 * ((*m * *k + *k * *n + *m * *n) as f64) / 1e9;
        // `per` is milliseconds, so GFLOP/ms is already TFLOP/s.
        println!(
            "  {label}   {per:>8.3} ms   {:>7.2} TFLOP/s   {:>6.1} GB/s",
            gflop / per,
            bytes / per * 1e3,
        );
    }
    // The custom kernel against candle's, at the same shapes, checked before it is timed.
    println!("\n  skinny kernel vs candle (m = 48)");
    for &(label, m, k, n) in shapes {
        if !tts_nn::skinny::eligible(m, k, n) {
            continue;
        }
        let a = Tensor::randn(0f32, 1.0, (m, k), &dev)?.to_dtype(DType::F16)?;
        let b = Tensor::randn(0f32, 0.02, (k, n), &dev)?.to_dtype(DType::F16)?;
        let want = a.matmul(&b)?.to_dtype(DType::F32)?;
        let got = tts_nn::skinny::matmul(&a, &b)?;
        let (abs, rel) = tts_nn::abs_and_rel(&got, &want)?;

        let mut f_ours = || -> candle_core::Result<()> {
            for _ in 0..ITERS {
                let _ = tts_nn::skinny::matmul(&a, &b)?;
            }
            Ok(())
        };
        let mut f_candle = || -> candle_core::Result<()> {
            for _ in 0..ITERS {
                let _ = a.matmul(&b)?;
            }
            Ok(())
        };
        let stats = h.ab(
            label,
            &mut [
                (
                    "skinny",
                    &mut f_ours as &mut dyn FnMut() -> candle_core::Result<()>,
                ),
                ("candle", &mut f_candle),
            ],
        )?;
        let gflop = 2.0 * (m as f64) * (k as f64) * (n as f64) / 1e9;
        let (ours, theirs) = (
            stats[0].median / ITERS as f64,
            stats[1].median / ITERS as f64,
        );
        println!(
            "  {label}  skinny {:>5.2} TFLOP/s   candle {:>5.2}   {:>5.2}x   max|d| {abs:.1e} rel {rel:.1e}",
            gflop / ours,
            gflop / theirs,
            theirs / ours,
        );
    }

    // A decode step is a dependency chain, not a queue of independent ops. `dispatch`
    // measures the issue cost with 4000 independent ops in flight; this measures what one
    // op costs when the next one cannot start until it lands.
    for &(b, n) in &[(48usize, 2048usize), (48, 1024), (48, 6144)] {
        let x = Tensor::randn(0f32, 1.0, (b, 1, n), &dev)?;
        let mut chained = || -> candle_core::Result<()> {
            let mut h = x.clone();
            for _ in 0..CHAIN {
                h = h.sqr()?;
            }
            h.sum_all()?;
            Ok(())
        };
        let mut independent = || -> candle_core::Result<()> {
            for _ in 0..CHAIN {
                let _ = x.sqr()?;
            }
            Ok(())
        };
        let label = format!("chain [{b},1,{n}]");
        let stats = h.ab(
            &label,
            &mut [
                (
                    "dependent",
                    &mut chained as &mut dyn FnMut() -> candle_core::Result<()>,
                ),
                ("independent", &mut independent),
            ],
        )?;
        println!(
            "  [{b},1,{n}]  dependent {:>6.1} us/op   independent {:>6.1} us/op",
            stats[0].median / CHAIN as f64 * 1000.0,
            stats[1].median / CHAIN as f64 * 1000.0,
        );
    }

    h.report_drift()?;
    Ok(())
}

// Appended: dependent-chain latency of a small elementwise op.
