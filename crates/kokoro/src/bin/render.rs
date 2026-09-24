//! Render text to a WAV through the full chain, for listening to what a change did.

use anyhow::{Context, Result};
use candle_core::Device;
use kokoro::decoder::SeededDraws;
use kokoro::model::{Model, Voices};
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |flag: &str, default: &str| -> String {
        match args.iter().position(|a| a == flag) {
            Some(i) => {
                args.remove(i);
                args.remove(i)
            }
            None => default.to_string(),
        }
    };
    let root = PathBuf::from(take("--root", "references/kokoro/weights"));
    let frontend = PathBuf::from(take("--frontend", "references/kokoro/weights/frontend"));
    let voice = take("--voice", "af_heart");
    let out = take("--out", "kokoro.wav");
    let seed: u64 = take("--seed", "1234").parse()?;
    let cpu = args.iter().any(|a| a == "--cpu");
    args.retain(|a| a != "--cpu");
    let text = if args.is_empty() {
        "Hello from Rust.".to_string()
    } else {
        args.join(" ")
    };

    let device = if cpu {
        Device::Cpu
    } else {
        Device::new_metal(0).context("opening the Metal device")?
    };
    let model = Model::load(&root, &device)?;
    let voices = Voices::load(&root.join("voices.safetensors"), &device)?;
    let g2p = tts_phoneme::g2p::G2P::load(&frontend, false)?;

    let (phonemes, unknown) = g2p.phonemize_report(&text);
    println!("{phonemes}");
    if !unknown.is_empty() {
        eprintln!("unpronounceable: {}", unknown.join(", "));
    }
    let ids = model.cfg.encode(&phonemes);
    anyhow::ensure!(
        ids.len() <= 510,
        "{} tokens; split the text first",
        ids.len()
    );

    let style = voices.style(&voice, ids.len() - 2)?;
    let repeat: usize = std::env::var("KOKORO_REPEAT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let mut samples = Vec::new();
    let mut timings = Vec::new();
    let mut started = std::time::Instant::now();
    // Dense-GEMM traffic of the last pass, for the achieved-rate line. The
    // counter is thread-local integers, never a dispatch.
    let mut gemm = (0u64, 0u64);
    for pass in 0..repeat {
        started = std::time::Instant::now();
        tts_nn::stats::take();
        let out = model.synthesize_timed(&ids, &style, 1.0, &mut SeededDraws::new(seed))?;
        gemm = tts_nn::stats::take();
        samples = out.0;
        timings = out.1;
        if repeat > 1 {
            eprintln!("  pass {pass}: {:.3} s", started.elapsed().as_secs_f64());
        }
    }
    let seconds = samples.len() as f64 / model.cfg.sample_rate() as f64;
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "{:.2} s of audio in {elapsed:.2} s — RTF {:.3}",
        seconds,
        elapsed / seconds
    );

    tts_core::wav::write(
        &out,
        &tts_core::Audio {
            samples,
            sample_rate: model.cfg.sample_rate(),
        },
    )
    .with_context(|| format!("writing {out}"))?;
    for (stage, secs) in &timings {
        println!("  {stage:<9} {secs:.3} s  ({:.0}%)", 100.0 * secs / elapsed);
    }
    // Counted dense GEMMs only — direct convs, attention scores and elementwise
    // passes are traffic this does not see, so the GB/s reads low against the
    // bus by construction. What it answers is whether the GEMM shapes are near
    // this backend's ~2.4 TFLOP/s.
    println!(
        "  matmul  {:>7.1} GFLOP {:>6.2} TFLOP/s  {:>7.1} GB {:>6.0} GB/s",
        gemm.0 as f64 / 1e9,
        gemm.0 as f64 / elapsed / 1e12,
        gemm.1 as f64 / 1e9,
        gemm.1 as f64 / elapsed / 1e9,
    );
    println!("wrote {out}");
    Ok(())
}
