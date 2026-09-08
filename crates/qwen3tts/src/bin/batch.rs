//! Does batching pay, and where does it saturate? One decode step across batch sizes.
//!
//! Both stages are bandwidth-bound on *weight* reads at batch 1 — the talker reads its 1.4 G
//! parameters once per frame, and the depth predictor reads its 60 M fifteen times. Batching
//! lanes amortises exactly that, but only if the quantized matmul keeps the weights quantized:
//! a path that dequantizes to f32 first turns a 1.06 byte/param read into a 4 byte/param read
//! plus an allocation, and CosyVoice measured a batch-7 q8_0 step at 5.13x a batch-1 step for
//! that reason.
//!
//! Printed as cost *per lane*: flat means the weight read amortised, rising means the batch is
//! paying for something.
//!
//! **Interleaved through [`tts_bench::Harness`], which is why the batch list is an argument.**
//! Every variant in a run holds its own KV cache for the whole run — 47 MB per lane at f16 and
//! this span — so one process cannot hold 1..64. Runs are chained instead: give each group a
//! common anchor batch and compare per-lane cost against it. `scripts/batch-saturation.sh`
//! does that.
//!
//! Run: `cargo run -p qwen3tts --release --bin qwen3tts-batch -- f16 1,4,8,16,24`

use anyhow::{Context, Result};
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor};
use qwen3tts::cfg::{predictor as pk, talker as tk};
use qwen3tts::qwen3::{Geometry, Stack};
use tts_bench::Harness;
use tts_nn::Weight;
use tts_nn::Weights;

const WEIGHTS: &str = "references/qwen3tts/weights/model.safetensors";
const DEFAULT_BATCHES: &str = "1,4,8,16,24";
/// Prompt positions, matching the engine's ICL prompt.
///
/// A realistic span, not a token or two: the ICL prompt is 156 positions and a segment decodes
/// ~100 more, and every decode step reads the whole K and V span per layer. A bench at span 4
/// said batch-8 f32 cost 65 ms/step where the real run saw 167.
const PROMPT: usize = 156;
const DECODE: usize = 256;

/// One timed variant. Aliased because `Harness::ab` wants trait objects and the tuple of
/// (lanes, boxed closure) is otherwise unreadable.
type Lane<'a> = (usize, Box<dyn FnMut() -> candle_core::Result<()> + 'a>);

fn bench(label: &str, h: &mut Harness, stack: &Stack, dim: usize, batches: &[usize]) -> Result<()> {
    let device = h.dev().clone();

    // States and inputs first: a variant closure may not allocate, or the timing measures the
    // allocation and candle's pool growth instead of the step.
    let mut prepared = Vec::with_capacity(batches.len());
    for &b in batches {
        let mut state = stack.new_state_with(b, PROMPT + DECODE)?;
        let prompt = Tensor::zeros((b, PROMPT, dim), DType::F32, &device)?;
        stack.forward(&prompt, &mut state)?;
        let x = Tensor::zeros((b, 1, dim), DType::F32, &device)?;
        prepared.push((b, state, x));
    }
    device.synchronize()?;

    let names: Vec<String> = batches.iter().map(|b| format!("batch {b}")).collect();
    let mut closures: Vec<Lane> = Vec::new();
    for (b, state, x) in prepared.iter_mut() {
        let s = stack;
        let xr = &*x;
        closures.push((
            *b,
            Box::new(move || {
                s.forward(xr, state)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                Ok(())
            }),
        ));
    }
    let mut variants: Vec<(&str, &mut dyn FnMut() -> candle_core::Result<()>)> = names
        .iter()
        .zip(closures.iter_mut())
        .map(|(n, (_, f))| (n.as_str(), f.as_mut() as &mut dyn FnMut() -> _))
        .collect();

    let stats = h.ab(label, &mut variants)?;

    // Per lane is the only column that answers the saturation question.
    let per_lane: Vec<f64> = stats
        .iter()
        .zip(batches)
        .map(|(s, b)| s.median / *b as f64)
        .collect();
    let base = per_lane[0];
    println!(
        "\n  {:<10} {:>11} {:>11} {:>13} {:>13}",
        "batch", "ms/step", "ms/lane", "vs batch 1", "marginal"
    );
    println!("  {}", "-".repeat(62));
    for (i, (&b, &pl)) in batches.iter().zip(per_lane.iter()).enumerate() {
        // Marginal: what each *added* lane cost between this batch and the previous one. The
        // amortisation ratio keeps improving long after the marginal lane stops getting cheaper,
        // so it is the marginal column that shows saturation.
        let marginal = if i == 0 {
            String::from("-")
        } else {
            let d_ms = stats[i].median - stats[i - 1].median;
            let d_lanes = (b - batches[i - 1]) as f64;
            format!("{:.3} ms", d_ms / d_lanes)
        };
        println!(
            "  {:<10} {:>11.2} {:>11.3} {:>12.2}x {:>13}",
            b,
            stats[i].median,
            pl,
            base / pl,
            marginal
        );
    }
    // Memory is not reported here on purpose: RSS excludes Metal's buffers, and the useful
    // figure is `/usr/bin/time -l`'s peak footprint around the whole process.
    println!(
        "\n  {} lanes live in this process",
        batches.iter().sum::<usize>()
    );
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let quant = match args.next().as_deref() {
        Some("f32") => Weight::F32,
        Some("f16") => Weight::F16,
        Some("q4_0") => Weight::Quant(GgmlDType::Q4_0),
        _ => Weight::Quant(GgmlDType::Q8_0),
    };
    let batches: Vec<usize> = args
        .next()
        .unwrap_or_else(|| DEFAULT_BATCHES.to_string())
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().parse::<usize>().context("batch size"))
        .collect::<Result<_>>()?;
    anyhow::ensure!(!batches.is_empty(), "no batch sizes given");
    let samples: usize = std::env::var("QWEN3TTS_SAMPLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    let device = Device::new_metal(0).context("opening the Metal device")?;
    let w = Weights::load(WEIGHTS, &device)?;

    let talker_geo = Geometry {
        dim: tk::DIM,
        layers: tk::LAYERS,
        heads: tk::HEADS,
        n_kv: tk::N_KV,
        head_dim: tk::HEAD_DIM,
        ffn: tk::FFN,
        eps: tk::NORM_EPS,
        rope_base: tk::ROPE_BASE,
        qk_norm: true,
        layer_scale: false,
        window: None,
    };
    let predictor_geo = Geometry {
        dim: pk::DIM,
        layers: pk::LAYERS,
        heads: pk::HEADS,
        n_kv: pk::N_KV,
        head_dim: pk::HEAD_DIM,
        ffn: pk::FFN,
        eps: pk::NORM_EPS,
        rope_base: pk::ROPE_BASE,
        qk_norm: pk::QK_NORM,
        layer_scale: false,
        window: pk::SLIDING_WINDOW,
    };

    println!(
        "quant {quant:?}   batches {batches:?}   span {}",
        PROMPT + DECODE
    );
    let mut h = Harness::new(&device, samples)?;

    let talker = Stack::load(
        &w,
        "talker.model.",
        talker_geo,
        quant,
        PROMPT + DECODE,
        &device,
    )?;
    bench(
        &format!("talker trunk ({} layers @ {})", tk::LAYERS, tk::DIM),
        &mut h,
        &talker,
        tk::DIM,
        &batches,
    )?;
    drop(talker);

    // Skipped when only the trunk is wanted: the predictor doubles the run, and it is the trunk
    // that decides whether a wider batch is worth the memory.
    if std::env::var_os("QWEN3TTS_TRUNK_ONLY").is_none() {
        let predictor = Stack::load(
            &w,
            "talker.code_predictor.model.",
            predictor_geo,
            quant,
            PROMPT + DECODE,
            &device,
        )?;
        bench(
            &format!("depth predictor ({} layers @ {})", pk::LAYERS, pk::DIM),
            &mut h,
            &predictor,
            pk::DIM,
            &batches,
        )?;
    }

    h.report_drift()?;
    Ok(())
}
