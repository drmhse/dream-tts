//! The `nano` gate: the front end's ids and tags, the text side, the decoder and the whole path against
//! `references/nano/dump_fixtures.py`, with the reference's random draws replayed.
//!
//!     cargo run -p kokoro --release --bin nano-validate [weights] [fixtures] [--metal]

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use kokoro::standalone::NanoTts;
use std::path::PathBuf;
use tts_nn::Weights;

struct Replay(Vec<Vec<f32>>);

impl kokoro::source::Draws for Replay {
    fn rand(&mut self, n: usize) -> Vec<f32> {
        self.randn(n)
    }

    fn randn(&mut self, n: usize) -> Vec<f32> {
        let v = self.0.remove(0);
        assert_eq!(v.len(), n, "a draw of {} values, wanted {n}", v.len());
        v
    }
}

#[derive(Default)]
struct Report {
    rows: usize,
    failures: usize,
}

impl Report {
    fn row(&mut self, name: &str, ok: bool, detail: String) {
        self.rows += 1;
        self.failures += usize::from(!ok);
        println!(
            "  {name:<16} {}  {detail}",
            if ok { "ok  " } else { "FAIL" }
        );
    }

    fn rel(&mut self, name: &str, got: &Tensor, want: &Tensor, tol: f32) {
        let r = (|| -> Result<(f32, f32)> {
            anyhow::ensure!(
                got.dims() == want.dims(),
                "shape {:?} vs {:?}",
                got.dims(),
                want.dims()
            );
            tts_nn::abs_and_rel(&got.to_device(&Device::Cpu)?, want)
        })();
        match r {
            Ok((abs, rel)) => self.row(
                name,
                rel <= tol,
                format!("rel {rel:.2e}  abs {abs:.2e}  (tol {tol:.0e})"),
            ),
            Err(e) => self.row(name, false, e.to_string()),
        }
    }

    fn snr(&mut self, name: &str, got: &[f32], want: &[f32], floor: f64) {
        let signal: f64 = want.iter().map(|v| (*v as f64).powi(2)).sum();
        let error: f64 = got
            .iter()
            .zip(want)
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum();
        let snr = 10.0 * (signal / error.max(1e-30)).log10();
        let ok = got.len() == want.len() && snr >= floor;
        self.row(
            name,
            ok,
            format!(
                "SNR {snr:.1} dB (floor {floor})  {} / {} samples",
                got.len(),
                want.len()
            ),
        );
    }
}

fn ints(t: &Tensor) -> Result<Vec<u32>> {
    Ok(t.flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .map(|v| *v as u32)
        .collect())
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let metal = args.iter().any(|a| a == "--metal");
    args.retain(|a| a != "--metal");
    let root = PathBuf::from(
        args.first()
            .cloned()
            .unwrap_or_else(|| "references/nano/weights".into()),
    );
    let fixtures = PathBuf::from(
        args.get(1)
            .cloned()
            .unwrap_or_else(|| "fixtures/nano".into()),
    );
    let device = if metal {
        Device::new_metal(0)?
    } else {
        Device::Cpu
    };
    let model = NanoTts::load(&root, &device)?;
    let fx_path = fixtures.join("standalone.safetensors");
    if !fx_path.exists() {
        println!(
            "no fixtures at {} — run references/nano/dump_fixtures.py",
            fx_path.display()
        );
        return Ok(());
    }
    let fx = Weights::load(fx_path.to_str().unwrap(), &Device::Cpu)?;
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixtures.join("standalone.json"))?)?;
    let mut r = Report::default();

    println!("front end");
    let frontend = root.join("frontend");
    let (us, gb) = (
        tts_phoneme::g2p::G2P::load(&frontend, false)?,
        tts_phoneme::g2p::G2P::load(&frontend, true)?,
    );
    let (mut same, mut rows, mut first) = (0, 0, None);
    for row in meta["frontend"].as_array().context("frontend rows")? {
        let text = row["text"].as_str().unwrap();
        let g2p = if row["voice"].as_str().unwrap().starts_with('b') {
            &gb
        } else {
            &us
        };
        let tagged = g2p.phonemize_tagged(text);
        let (ids, tags, _) = model.encode(&tagged.phonemes, &tagged.tags);
        let want = |k: &str| -> Vec<u32> {
            row[k]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect()
        };
        rows += 1;
        if ids == want("ids") && tags == want("tags") {
            same += 1;
        } else if first.is_none() {
            first = Some(format!(
                "{text:?}: ids {} tags {}",
                ids == want("ids"),
                tags == want("tags")
            ));
        }
    }
    r.row(
        "ids and tags",
        same == rows,
        format!(
            "{same} / {rows} segments identical{}",
            first
                .map(|f| format!("; first miss {f}"))
                .unwrap_or_default()
        ),
    );

    println!("text side");
    let (ids, tags, style) = (
        ints(&fx.get("in.ids")?)?,
        ints(&fx.get("in.tags")?)?,
        fx.get("in.style")?,
    );
    let style = style.to_device(&device)?;
    let s_pro = style.narrow(1, 128, 128)?;
    let s_dec = style.narrow(1, 0, 128)?;
    let spoken = model.text.forward(&ids, &tags, &s_pro, 1.0)?;
    let want_dur: Vec<usize> = ints(&fx.get("text.dur")?)?
        .iter()
        .map(|v| *v as usize)
        .collect();
    let moved = spoken
        .durations
        .iter()
        .zip(&want_dur)
        .filter(|(a, b)| a != b)
        .count();
    r.row(
        "durations",
        spoken.durations == want_dur,
        format!(
            "{moved} of {} differ, {} frames",
            want_dur.len(),
            want_dur.iter().sum::<usize>()
        ),
    );
    if spoken.durations == want_dur {
        r.rel(
            "features",
            &spoken.feat,
            &fx.get("text.feat")?.transpose(1, 2)?.contiguous()?,
            1e-4,
        );
        r.rel("f0", &spoken.f0, &fx.get("text.f0")?, 1e-4);
        r.rel("energy", &spoken.energy, &fx.get("text.energy")?, 1e-4);
    }

    println!("decoder, from the reference's text side");
    let draws = || -> Result<Replay> {
        Ok(Replay(vec![
            fx.get("draw.0")?.to_vec1()?,
            fx.get("draw.1")?.to_vec1()?,
        ]))
    };
    let on = |t: Tensor| t.to_device(&device);
    let out = model.decoder.render(
        &on(fx.get("text.feat")?.transpose(1, 2)?.contiguous()?)?,
        &on(fx.get("text.f0")?)?,
        &on(fx.get("text.energy")?)?,
        &s_dec,
        &mut draws()?,
    )?;
    r.rel("mapper", &out.mapper, &fx.get("mapper")?, 1e-3);
    // The head reads the pulse train's log spectrum, whose sharpest pulses move by a fraction of a
    // sample (torch's fused f32 upsampling index, matched to 2 ulp).
    r.rel("head", &out.head, &fx.get("head")?, 1e-2);
    let want: Vec<f32> = fx.get("audio")?.to_vec1()?;
    r.snr("audio", &out.audio, &want, 40.0);

    println!("text to audio");
    let (audio, _) = model.synthesize(&ids, &tags, &style, 1.0, &mut draws()?)?;
    if audio.len() == want.len() {
        r.snr("audio", &audio, &want, 30.0);
    } else {
        r.row(
            "audio",
            false,
            format!("{} samples, wanted {}", audio.len(), want.len()),
        );
    }

    println!("\n{} rows, {} failures", r.rows, r.failures);
    if r.failures > 0 {
        std::process::exit(1);
    }
    Ok(())
}
