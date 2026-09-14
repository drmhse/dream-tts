//! Live GPU telemetry sidecar: sample the accelerator node and print it.
//!
//! ```text
//! gpumon                      # 2 Hz until interrupted
//! gpumon --hz 4 --count 40    # 4 Hz, 40 samples, then exit (tee-able)
//! ```
//!
//! Unprivileged, unlike `powermetrics`: it reads the AGX node's published
//! `PerformanceStatistics`. Run it beside a render to see whether the GPU stays
//! fed. On machines without an accelerator node it says so and exits.

use anyhow::{Context, Result};
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut hz = 2.0f64;
    let mut count = usize::MAX;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--hz" => {
                hz = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .context("--hz needs a number")?
            }
            "--count" => {
                count = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .context("--count needs a number")?
            }
            "-h" | "--help" => {
                println!("usage: gpumon [--hz 2] [--count N]");
                return Ok(());
            }
            _ => anyhow::bail!("unknown argument {a:?}"),
        }
    }
    anyhow::ensure!((0.1..=20.0).contains(&hz), "--hz in [0.1, 20]");
    let sampler = tts_core::gpumon::Sampler::open().context("no accelerator node to sample")?;
    let period = Duration::from_secs_f64(1.0 / hz);
    let t0 = Instant::now();
    println!("{:>7} {:>4} {:>4} {:>4} {:>8} {:>8}", "t", "gpu", "ren", "til", "alloc", "inuse");
    for _ in 0..count {
        let dt = t0.elapsed();
        match sampler.sample() {
            Some(s) => println!(
                "{:>6.1}s {:>3}% {:>3}% {:>3}% {:>7.1}G {:>7.1}G",
                dt.as_secs_f64(),
                s.device_pct,
                s.renderer_pct,
                s.tiler_pct,
                s.alloc_bytes as f64 / 1e9,
                s.in_use_bytes as f64 / 1e9,
            ),
            None => println!("{:>6.1}s (no sample)", dt.as_secs_f64()),
        }
        std::thread::sleep(period);
    }
    Ok(())
}
