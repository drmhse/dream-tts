//! The Kokoro gate: every stage against the PyTorch fixtures, one row per stage.
//!
//! Two tiers. The shape audit reads only the safetensors header, so it runs before any
//! fixture exists and even while the checkpoint is still downloading — that is what caught
//! qwen3tts's mislabelled codebook size on its first run. The stage tier then compares
//! activations.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use kokoro::cfg::Config;
use std::path::PathBuf;
use tts_nn::Weights;

/// Replays the reference's random draws in call order, so the excitation is compared
/// exactly instead of statistically.
struct ReplayDraws {
    values: Vec<Vec<f32>>,
    at: usize,
}

impl ReplayDraws {
    fn new(fx: &Weights, count: usize) -> Result<Self> {
        let values = (0..count)
            .map(|i| Ok(fx.get(&format!("draw.{i}"))?.flatten_all()?.to_vec1()?))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { values, at: 0 })
    }

    fn next(&mut self, n: usize) -> Vec<f32> {
        let v = self.values[self.at].clone();
        self.at += 1;
        assert_eq!(v.len(), n, "draw {} has {} values, wanted {n}", self.at - 1, v.len());
        v
    }
}

impl kokoro::source::Draws for ReplayDraws {
    fn rand(&mut self, n: usize) -> Vec<f32> {
        self.next(n)
    }

    fn randn(&mut self, n: usize) -> Vec<f32> {
        self.next(n)
    }
}

struct Report {
    rows: usize,
    failures: usize,
}

impl Report {
    /// Relative rather than absolute, for a stage whose inputs already carry the
    /// excitation's error: a 20-point transform sums the absolute error up but leaves the
    /// relative error where it was, so an absolute bar here would only be measuring the
    /// transform's gain.
    fn check_rel(&mut self, name: &str, got: &Tensor, want: &Tensor, tol: f32) {
        self.rows += 1;
        match tts_nn::abs_and_rel(got, want) {
            Ok((abs, rel)) if rel <= tol => {
                println!("  {name:<22} ok    rel {rel:.3e}  abs {abs:.3e}");
            }
            Ok((abs, rel)) => {
                self.failures += 1;
                println!("  {name:<22} FAIL  rel {rel:.3e}  abs {abs:.3e}  (tol {tol:.1e})");
            }
            Err(e) => {
                self.failures += 1;
                println!("  {name:<22} FAIL  {e}");
            }
        }
    }

    fn check(&mut self, name: &str, got: &Tensor, want: &Tensor, tol: f32) {
        self.rows += 1;
        let result = (|| -> Result<(f32, f32)> {
            if got.dims() != want.dims() {
                anyhow::bail!("shape {:?} vs {:?}", got.dims(), want.dims());
            }
            tts_nn::abs_and_rel(got, want)
        })();
        match result {
            Ok((abs, rel)) if abs <= tol => {
                println!("  {name:<22} ok    abs {abs:.3e}  rel {rel:.3e}");
            }
            Ok((abs, rel)) => {
                self.failures += 1;
                println!("  {name:<22} FAIL  abs {abs:.3e}  rel {rel:.3e}  (tol {tol:.1e})");
            }
            Err(e) => {
                self.failures += 1;
                println!("  {name:<22} FAIL  {e}");
            }
        }
    }
}

/// The reference's own excitation, put through the same transform, so the comparison is
/// of this port's `SineGen` rather than of its STFT as well.
fn har_reference(fx: &Weights, generator: &kokoro::decoder::Generator) -> Result<Tensor> {
    let merged: Vec<f32> = fx.get("gen_source.0")?.flatten_all()?.to_vec1()?;
    generator.spectrum(&merged, fx.get("gen_source.0")?.device())
}

fn shape_audit(w: &Weights, cfg: &Config) -> Result<usize> {
    let mut problems = 0;
    let mut expect = |name: &str, dims: Vec<usize>| {
        match w.file_shape(name) {
            Some(actual) if actual == dims => {}
            Some(actual) => {
                problems += 1;
                println!("  {name:<58} {actual:?} != {dims:?}");
            }
            None => {
                problems += 1;
                println!("  {name:<58} missing");
            }
        }
    };
    let h = cfg.plbert.hidden_size;
    expect("bert.embeddings.word_embeddings.weight", vec![cfg.n_token, Config::EMBEDDING_SIZE]);
    expect("bert.encoder.embedding_hidden_mapping_in.weight", vec![h, Config::EMBEDDING_SIZE]);
    expect("bert_encoder.weight", vec![cfg.hidden_dim, h]);
    expect("predictor.duration_proj.linear_layer.weight", vec![cfg.max_dur, cfg.hidden_dim]);
    expect("text_encoder.embedding.weight", vec![cfg.n_token, cfg.hidden_dim]);
    expect(
        "decoder.generator.conv_post.weight",
        vec![cfg.istftnet.gen_istft_n_fft + 2, cfg.istftnet.upsample_initial_channel / 4, 7],
    );
    Ok(problems)
}

fn main() -> Result<()> {
    let root = PathBuf::from(
        std::env::args().nth(1).unwrap_or_else(|| "references/kokoro/weights".into()),
    );
    let fixtures = PathBuf::from(
        std::env::args().nth(2).unwrap_or_else(|| "fixtures/kokoro".into()),
    );
    let cfg = Config::load(&root.join("config.json"))?;
    let w = Weights::load(root.join("kokoro.safetensors").to_str().unwrap(), &Device::Cpu)?;

    println!("shape audit ({} tensors)", w.len());
    let problems = shape_audit(&w, &cfg)?;
    println!("  {}", if problems == 0 { "ok" } else { "PROBLEMS" });

    let fx_path = fixtures.join("forward.safetensors");
    if !fx_path.exists() {
        println!("\nno activation fixtures at {} — run references/kokoro/dump_fixtures.py", fx_path.display());
        std::process::exit(if problems == 0 { 0 } else { 1 });
    }
    let fx = Weights::load(fx_path.to_str().unwrap(), &Device::Cpu)
        .context("loading forward fixtures")?;
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixtures.join("forward.json"))?)?;
    let ids: Vec<u32> = meta["input_ids"]
        .as_array()
        .context("input_ids")?
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    println!("\nstages ({} phonemes)", ids.len());
    let mut r = Report { rows: 0, failures: 0 };
    let device = Device::Cpu;

    let bert = kokoro::albert::Albert::load(&w, &cfg)?;
    let bert_out = bert.forward(&ids, &device)?;
    r.check("bert", &bert_out, &fx.get("bert")?, 2e-4);

    let bert_encoder = kokoro::blocks::Linear::load(&w, "bert_encoder")?;
    let d_en = bert_encoder.apply(&bert_out)?;
    r.check("bert_encoder", &d_en, &fx.get("bert_encoder")?, 2e-4);
    let d_en = d_en.transpose(1, 2)?.contiguous()?;

    let ref_s = fx.get("ref_s")?;
    let style_pred = ref_s.narrow(1, 128, 128)?.contiguous()?;
    let style_dec = ref_s.narrow(1, 0, 128)?.contiguous()?;

    let predictor = kokoro::predictor::Predictor::load(&w, &cfg)?;
    let d = predictor.text_encoder.forward(&d_en, &style_pred)?;
    r.check("dur_enc", &d, &fx.get("dur_enc")?, 5e-4);

    let durations = predictor.durations(&d, 1.0)?;
    let want_dur: Vec<usize> =
        fx.get("pred_dur")?.to_vec1::<f32>()?.iter().map(|v| *v as usize).collect();
    r.rows += 1;
    if durations == want_dur {
        println!("  {:<22} ok    {} frames", "pred_dur", durations.iter().sum::<usize>());
    } else {
        r.failures += 1;
        println!("  {:<22} FAIL  {:?} != {:?}", "pred_dur", &durations[..8.min(durations.len())], &want_dur[..8.min(want_dur.len())]);
    }

    let aln = kokoro::predictor::Predictor::alignment(&durations, &device)?;
    let en = d.transpose(1, 2)?.contiguous()?.matmul(&aln)?;
    let (f0, energy) = predictor.f0_and_energy(&en, &style_pred)?;
    r.check("F0_proj", &f0.unsqueeze(1)?, &fx.get("F0_proj")?, 2e-3);
    r.check("N_proj", &energy.unsqueeze(1)?, &fx.get("N_proj")?, 2e-3);

    let text_encoder = kokoro::text_encoder::TextEncoder::load(&w, &cfg)?;
    let t_en = text_encoder.forward(&ids, &device)?;
    r.check("text_encoder", &t_en, &fx.get("text_encoder")?, 2e-4);
    let asr = t_en.matmul(&aln)?;

    let decoder = kokoro::decoder::Decoder::load(&w, &cfg)?;
    let draw_count = meta["draws"].as_u64().unwrap() as usize;


    // The excitation first: it is the only stochastic stage, and a mismatch here would
    // otherwise surface as a wrong waveform with every deterministic stage passing.
    // Driven by the reference's F0, not this port's: `uv = f0 > 10` is a threshold, and a
    // 7e-4 difference in F0 flips it for any frame sitting on the line. That belongs to the
    // predictor's tolerance, not the source module's.
    //
    // The tolerances below are not slack. The reference accumulates the excitation phase
    // unwrapped, reaching 165,000 radians — where one f32 ulp is 0.0156 rad, so `sin` of it
    // is uncertain at 1.6% from the representation alone and no arithmetic ordering
    // reproduces torch. This port computes the same phase wrapped, which is more accurate
    // and not bit-identical. What is checked instead is that the port sits closer to the
    // reference than the reference sits to *itself* under a different noise draw:
    // measured 25.2 dB SNR here against 19.7 and 20.7 dB for two upstream re-runs, and
    // 1.54 dB log-spectral distance against 2.09 and 2.18.
    let ref_curve: Vec<f32> = fx.get("F0_proj")?.flatten_all()?.to_vec1()?;
    let mine = decoder.generator.excitation(&ref_curve, &mut ReplayDraws::new(&fx, draw_count)?);
    let want = fx.get("gen_source.0")?.flatten_all()?;
    r.check(
        "excitation",
        &Tensor::from_vec(mine.clone(), (want.dim(0)?,), &device)?,
        &want,
        3e-3,
    );

    // Compared as a complex spectrum rather than as the magnitude and phase channels the
    // network actually consumes. `atan2` has a branch cut at pi and reports +pi and -pi for
    // the same angle, so the phase channel flips wholesale wherever the imaginary part
    // crosses zero — which says nothing about the signal. The effect those flips do have on
    // the network is what the audio SNR below measures.
    let har = decoder.generator.spectrum(&mine, &device)?;
    let want_har = har_reference(&fx, &decoder.generator)?;
    let bins = har.dim(1)? / 2;
    let complex = |t: &Tensor| -> Result<Tensor> {
        let mag = t.narrow(1, 0, bins)?;
        let phase = t.narrow(1, bins, bins)?;
        Ok(Tensor::cat(&[&(&mag * phase.cos()?)?, &(&mag * phase.sin()?)?], 1)?)
    };
    r.check_rel("har_spectrum", &complex(&har)?, &complex(&want_har)?, 1e-2);

    let audio =
        decoder.forward(&asr, &f0, &energy, &style_dec, &mut ReplayDraws::new(&fx, draw_count)?)?;
    let want_audio: Vec<f32> = fx.get("audio")?.to_vec1()?;
    r.rows += 1;
    if audio.len() != want_audio.len() {
        r.failures += 1;
        println!("  {:<22} FAIL  {} samples != {}", "audio", audio.len(), want_audio.len());
    } else {
        let signal: f64 = want_audio.iter().map(|v| (*v as f64).powi(2)).sum();
        let error: f64 =
            audio.iter().zip(&want_audio).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
        let snr = 10.0 * (signal / error.max(1e-30)).log10();
        // 19.7 dB is what an upstream re-run with a different noise draw scores.
        if snr >= 24.0 {
            println!("  {:<22} ok    SNR {snr:.1} dB  {} samples", "audio", audio.len());
        } else {
            r.failures += 1;
            println!("  {:<22} FAIL  SNR {snr:.1} dB (upstream re-run scores 19.7)", "audio");
        }
    }

    println!("\n{} rows, {} failures", r.rows, r.failures);
    if r.failures > 0 || problems > 0 {
        std::process::exit(1);
    }
    Ok(())
}
