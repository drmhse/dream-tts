//! Qwen3-TTS behind the [`Engine`] trait.
//!
//! Two stages: [`crate::talker`] (with its depth predictor) then [`crate::codec`]. Stage
//! timers call `device.synchronize()` first — Metal dispatch is async, and an unsynchronised
//! timer measured enqueue time and misattributed most of CosyVoice's cost.
//!
//! Caller-visible limits: ten languages only ([`cfg::talker::LANGUAGES`]); text advances one
//! token per audio frame, so segmentation behaves differently from CosyVoice (trap 3);
//! streaming is native to the architecture but stays false until the trait has a method for
//! it.

use crate::cfg;
use crate::codec::Codec;
use crate::syllables::Scheme;
use crate::{swahili, syllables};
use crate::talker::{Language, Sampling, Talker};
use anyhow::{Context, Result};
use candle_core::quantized::GgmlDType;
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::{SplitDelimiterBehavior, Tokenizer};
use tts_core::rng::Rng;
use tts_core::{
    text, wav, Audio, Capabilities, Cloning, Engine, EngineConfig, Stats, Synthesis,
    SynthesisRequest,
};
use tts_nn::Weight;

pub const ID: &str = "qwen3tts";

/// Footprint and Metal allocation, for `QWEN3TTS_TIMING`. They differ by the mapped
/// checkpoint, which Metal counts and the footprint does not.
fn mem_line(when: &str, device: &Device) {
    let gb = |b: Option<u64>| b.map_or("?".into(), |b| format!("{:.2} GB", b as f64 / 1e9));
    eprintln!(
        "engine {ID}: {when}: footprint {}, Metal allocated {}",
        gb(tts_core::system::footprint()),
        gb(tts_nn::allocated_bytes(device)),
    );
}

/// Below this many characters a segment's frames-per-character ratio is too noisy to judge.
const SEGMENT_MIN_CHARS: usize = 40;
/// Frames per character above this multiple of the request's median means the talker kept
/// going after its text ran out.
const SEGMENT_RATIO_CEILING: f64 = 1.6;
/// Below this many frames the distinct-code count is not meaningful.
const SEGMENT_MIN_FRAMES: usize = 24;
/// Distinct codebook-0 values per frame below this is degenerate repetition.
const SEGMENT_MIN_VARIETY: f64 = 0.35;

/// Weight formats that load, default first. Quantization covers the talker's and predictor's
/// projections, not the codec decoder, which runs over whole chunks.
///
/// - **`f16` (default)** is the only format that batches. Just a *dense* GEMM shares one weight
///   read across lanes — candle's quantized `mm_t` re-reads per row — so 48 lanes are available
///   here and nowhere else: a 4838-word article renders at RTF 0.164 against q8_0's 0.738.
/// - **`q8_0`** reads half the bytes and therefore wins only where nothing batches, which is a
///   single short passage: 132 words at RTF 0.642 against f16's 0.397. It is **not the small
///   machine's answer** — on that passage it peaks at 11.72 GB against f16's 12.30, so it buys
///   0.58 GB for 62% of the speed. The floor is the codec's activations, not the weights.
/// - **`f32`** is for fixture work. 6.3 GB of projections thrashes a 16 GB machine — measured
///   1994 ms/frame against q8_0's 52, memory pressure rather than arithmetic.
///
/// `docs/reference.md#performance` has the measurements.
const QUANT: &[&str] = &["f16", "q8_0", "f32", "q5_0", "q4_1", "q4_0"];

/// What this engine needs before it starts swapping, from `/usr/bin/time -l` peak footprint:
/// 6.7 GB for one short passage and 9.2 GB for a chapter at 48 lanes, plus what else is running.
const WANTS_MEMORY: u64 = 16 << 30;

/// One segment's decoded frames, with the paragraph index and character count it came from.
type Decoded = (usize, usize, Vec<Vec<u32>>);

/// Lanes per batched decode.
///
/// **48, and the bound is memory rather than diminishing returns.** Per-lane cost was still
/// falling at 64 in the `qwen3tts-batch` sweep — an added lane costs a flat ~0.4 ms from 16
/// upward — so there is no arithmetic saturation to find below the wall. What there is instead
/// is a cliff: three corpora between 1612 and 4763 words render at RTF 0.175-0.192 at 48 lanes,
/// and 56 collapses to 0.701 with 621 s of system time against 17 s, which is the VM compressor
/// rather than compute.
///
/// A lane itself is cheap — 13 MB of peak footprint between 24 and 48 lanes, measured, against
/// the 77 MB its f16 KV cache implies. The 16 GB machine runs out because the *floor* is
/// ~14.5 GB, not because lanes are expensive, so raising this further wants that floor lowered
/// first rather than a bigger number here.
const MAX_BATCH: usize = 48;

/// `MAX_BATCH`, overridable for tuning. Group size trades three things against each other: a
/// wider batch amortises the weight read further, but costs more per step and packs lengths
/// *worse*, since a group runs as long as its longest lane.
fn max_batch() -> usize {
    static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("QWEN3TTS_MAX_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n| *n > 0)
            .unwrap_or(MAX_BATCH)
    })
}

/// How far past this voice's own frames-per-character a lane may run before it is cut.
///
/// Not a tolerance on natural variation — 2x the median is far outside it. It is the point past
/// which the segment is no longer reading its text, and every step it takes after that holds
/// the whole group open behind it.
const SEGMENT_BUDGET_SLACK: f64 = 2.0;

/// A segment whose frames leave [LOW, HIGH] x the voice's own frames-per-character is redrawn:
/// past HIGH it has stopped reading and is babbling or looping (a Swahili adapter ran a 46-char
/// sentence for 40 s), under LOW it dropped text — silently when the text fit inside the
/// reference block, since nothing is left over to count. Good clips never measured under 0.78,
/// a skipped sentence 0.58. The reference clip gives the ratio up front, so this works on a
/// one-sentence request, where `SEGMENT_RATIO_SAMPLE` never fills.
const REDRAW_HIGH: f64 = 1.8;
const REDRAW_LOW: f64 = 0.65;
const REDRAWS: usize = 2;

/// Segments that must have finished before the ratio is trusted to cap anything.
const SEGMENT_RATIO_SAMPLE: usize = 16;

/// Frames a batched lane may reach before the group is redone one segment at a time.
///
/// The KV cache is sized for this, so it cannot be the request's full budget: 4096 positions
/// times eight lanes is 9.4 GB. 512 frames is 41 s of audio from a single segment, which no
/// sentence reaches — and a lane that does hit it is *rerun unbatched with the full budget*
/// rather than truncated, so this bounds memory without bounding output.
const BATCH_FRAME_CAP: usize = 512;

const NO_VOICE: &str = "engine `qwen3tts` requires a voice asset: the talker's prefill \
    carries the reference clip's speaker embedding, and in-context cloning also needs its \
    codes and transcript. Build one with references/qwen3tts/export_voice.py";

pub fn capabilities() -> Capabilities {
    Capabilities {
        id: ID,
        description: "Qwen3-TTS-12Hz-1.7B-Base — Qwen3 talker, 15-step depth transformer, \
                      RVQ codec decoder. 24 kHz, no diffusion. Ten languages only \
                      (en, de, es, zh, ja, fr, ko, ru, it, pt)",
        sample_rate: cfg::SAMPLE_RATE as u32,
        frame_rate: cfg::FRAME_RATE,
        cloning: Cloning::PrecomputedAsset,
        // The architecture streams natively; the trait has no streaming method yet, so
        // claiming it would be a lie a client could act on. See the module docs.
        streaming: false,
        word_timings: false,
        quantization: QUANT,
        languages: Some(cfg::talker::LANGUAGES),
        available: true,
        reason: None,
    }
}

/// Checkpoint paths. Talker at the root, codec under `speech_tokenizer/` — both in the one
/// upstream download, unlike CosyVoice's separate ONNX tokenizer.
pub struct Paths {
    pub talker: PathBuf,
    pub codec: PathBuf,
    pub vocab: PathBuf,
    pub merges: PathBuf,
}

impl Paths {
    pub fn resolve(config: &EngineConfig) -> Self {
        Self {
            talker: config.path("talker", "model.safetensors"),
            codec: config.path("codec", "speech_tokenizer/model.safetensors"),
            // No tokenizer.json in this checkpoint, unlike CosyVoice's — the BPE is built
            // from vocab.json plus merges.txt at load.
            vocab: config.path("vocab", "vocab.json"),
            merges: config.path("merges", "merges.txt"),
        }
    }

    /// Report every missing file at once rather than one per run.
    pub fn check(&self) -> Result<()> {
        let missing: Vec<&Path> = [&self.talker, &self.codec, &self.vocab, &self.merges]
            .into_iter()
            .map(PathBuf::as_path)
            .filter(|p| !p.exists())
            .collect();
        // The command first: on a fresh install this is the only error anyone sees, and the
        // fix is one line. The file list follows for the case where only some are missing.
        anyhow::ensure!(
            missing.is_empty(),
            "engine `{ID}` has no checkpoint yet.\n               Download it:  ./scripts/bootstrap.sh qwen3tts   (~4.3 GB, resumable, curl only)\n               Missing {} file(s): {}",
            missing.len(),
            missing
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        Ok(())
    }
}

/// `auto`, a tag (`italian`), or a blend of tags (`italian:0.7,spanish:0.3`).
/// `adapted`: the language an adapter's scheme names, and its codec row.
fn parse_language(
    spec: &Path,
    adapted: Option<(&str, u32)>,
    row: impl Fn(u32) -> Result<Vec<f32>>,
) -> Result<Language> {
    let text = spec.to_str().with_context(|| format!("non-utf8 language {}", spec.display()))?;
    let lower = text.to_ascii_lowercase();
    if lower == "auto" {
        return Ok(Language::Auto);
    }
    let tag = |name: &str| {
        if let Some((_, id)) = adapted.filter(|(n, _)| *n == name) {
            return Ok(id);
        }
        cfg::talker::language_id(name).with_context(|| {
            format!(
                "engine `{ID}` has no language id for `{name}`; it supports {}, `auto`, or a blend \
                 like `italian:0.7,spanish:0.3`",
                cfg::talker::LANGUAGES.join(", ")
            )
        })
    };
    if !lower.contains(':') {
        return Ok(Language::Tag(tag(&lower)?));
    }
    let mut mix = vec![0f32; cfg::talker::DIM];
    for part in lower.split(',') {
        let (name, w) = part
            .split_once(':')
            .with_context(|| format!("blend term `{part}` is not `name:weight`"))?;
        let w: f32 = w.trim().parse().with_context(|| format!("weight in `{part}`"))?;
        for (m, x) in mix.iter_mut().zip(row(tag(name.trim())?)?) {
            *m += w * x;
        }
    }
    Ok(Language::Vector(mix))
}

fn parse_quant(name: Option<&str>) -> Result<Weight> {
    Ok(match name {
        Some("f32") => Weight::F32,
        None | Some("f16") => Weight::F16,
        Some("q8_0") => Weight::Quant(GgmlDType::Q8_0),
        Some("q5_0") => Weight::Quant(GgmlDType::Q5_0),
        Some("q4_1") => Weight::Quant(GgmlDType::Q4_1),
        Some("q4_0") => Weight::Quant(GgmlDType::Q4_0),
        Some(other) => anyhow::bail!(
            "engine `{ID}` does not support weight format `{other}`; it accepts {}",
            QUANT.join(", ")
        ),
    })
}

/// The gain an adapter was exported at, from its safetensors header (`finetune.py export`
/// records it as `__metadata__.gain`).
fn adapter_export_gain(path: &str) -> Result<f32> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).with_context(|| format!("opening adapter {path}"))?;
    let mut n = [0u8; 8];
    f.read_exact(&mut n)?;
    let mut header = vec![0u8; u64::from_le_bytes(n) as usize];
    f.read_exact(&mut header)?;
    let v: serde_json::Value = serde_json::from_slice(&header).context("adapter header")?;
    v["__metadata__"]["gain"]
        .as_str()
        .and_then(|g| g.parse().ok())
        .with_context(|| format!("adapter {path} records no export gain; adapter_gain cannot be applied"))
}

/// Qwen2's pre-tokenizer split, as transformers' `Qwen2Converter` writes it.
const QWEN2_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Qwen's byte-level BPE from the two files the checkpoint ships, pre-tokenized as transformers'
/// Qwen2 converter does: NFC, Qwen2's split, then byte-level without GPT-2's regex. GPT-2's split
/// alone put punctuation against letters differently ("-K", "P.C.E.A"): 107 of 350,722 pieces in
/// the Swahili training text, English narration included.
fn qwen_tokenizer(vocab: &str, merges: &str) -> Result<Tokenizer> {
    let bpe = BPE::from_file(vocab, merges)
        .build()
        .map_err(|e| anyhow::anyhow!("building the BPE from vocab.json/merges.txt: {e}"))?;
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_normalizer(Some(NFC));
    let split = Split::new(SplitPattern::Regex(QWEN2_SPLIT.into()), SplitDelimiterBehavior::Isolated, false)
        .map_err(|e| anyhow::anyhow!("Qwen2 split regex: {e}"))?;
    tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, false)),
    ])));
    Ok(tokenizer)
}

/// Silences inside a segment shortened to `max_s` and its edges to `EDGE_S`, so the gaps
/// `join_segments` inserts are the pacing. Before the join, so segment timings and the aligner
/// see the final audio. The Swahili adapter learned its data's pauses: 32 over 0.25 s in a
/// 142-word chapter against the base's 15, most at segment edges.
const EDGE_S: f32 = 0.1;

fn cap_pauses(x: &[f32], max_s: f32) -> Vec<f32> {
    let win = cfg::SAMPLE_RATE / 100;
    let peak = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let floor = (peak * 0.05).max(1e-3);
    let quiet: Vec<bool> = x
        .chunks(win)
        .map(|c| c.iter().all(|v| v.abs() < floor))
        .collect();
    let keep = (max_s * cfg::SAMPLE_RATE as f32) as usize / win;
    let mut out = Vec::with_capacity(x.len());
    let mut i = 0;
    while i < quiet.len() {
        let mut j = i;
        while j < quiet.len() && quiet[j] {
            j += 1;
        }
        let run = j - i;
        let edge = (EDGE_S * cfg::SAMPLE_RATE as f32) as usize / win;
        if (i == 0 || j == quiet.len()) && run > edge {
            // Keep the `edge` windows nearest the speech.
            let (a, b) = if i == 0 { (j - edge, j) } else { (i, i + edge) };
            out.extend_from_slice(&x[a * win..(b * win).min(x.len())]);
            i = j;
        } else if run > keep {
            let head = keep / 2;
            out.extend_from_slice(&x[i * win..(i + head) * win]);
            out.extend_from_slice(&x[(j - (keep - head)) * win..(j * win).min(x.len())]);
            i = j;
        } else if run > 0 {
            out.extend_from_slice(&x[i * win..(j * win).min(x.len())]);
            i = j;
        } else {
            out.extend_from_slice(&x[i * win..((i + 1) * win).min(x.len())]);
            i += 1;
        }
    }
    out
}

/// Speech-active RMS: the mean power of 20 ms windows within 30 dB of the loudest.
fn active_rms(x: &[f32]) -> f32 {
    let win = cfg::SAMPLE_RATE / 50;
    let power: Vec<f32> = x.chunks(win).map(|c| c.iter().map(|v| v * v).sum::<f32>() / c.len() as f32).collect();
    let top = power.iter().fold(0f32, |m, &p| m.max(p));
    let active: Vec<f32> = power.into_iter().filter(|&p| p > top * 1e-3).collect();
    if active.is_empty() {
        return 0.0;
    }
    (active.iter().sum::<f32>() / active.len() as f32).sqrt()
}

/// Each segment scaled toward the request's median level, within ±6 dB. Segments are drawn
/// independently and the Swahili adapter's vary audibly in level, heard as a jump at joins.
/// Then, with `absolute`, one gain for the whole request toward that median speech level in dBFS,
/// the peak kept under -1 dBFS: the voice's reference sets the level, and the owner's 16 s clip
/// put a Swahili chapter at -34.6 dB mean against an English post's -23.0. Levelling the clip
/// itself instead cost consistency (ECAPA 0.825 -> 0.795): it raised the recording's noise too.
fn match_levels(pieces: &mut [tts_core::Piece], absolute: Option<f32>) {
    let rms: Vec<f32> = pieces.iter().map(|p| active_rms(&p.samples)).collect();
    let mut sorted: Vec<f32> = rms.iter().copied().filter(|&r| r > 0.0).collect();
    if sorted.len() < 2 {
        return;
    }
    sorted.sort_by(f32::total_cmp);
    let target = sorted[sorted.len() / 2];
    for (p, r) in pieces.iter_mut().zip(rms) {
        if r > 0.0 {
            let g = (target / r).clamp(0.5, 2.0);
            let peak = p.samples.iter().fold(0f32, |m, v| m.max(v.abs())) * g;
            let g = if peak > 0.99 { g * 0.99 / peak } else { g };
            p.samples.iter_mut().for_each(|v| *v *= g);
        }
    }
    let Some(db) = absolute else {
        return;
    };
    let mut now: Vec<f32> = pieces.iter().map(|p| active_rms(&p.samples)).filter(|&r| r > 0.0).collect();
    now.sort_by(f32::total_cmp);
    let peak = pieces.iter().flat_map(|p| p.samples.iter()).fold(0f32, |m, v| m.max(v.abs()));
    if now.is_empty() || peak == 0.0 {
        return;
    }
    let g = (10f32.powf(db / 20.0) / now[now.len() / 2]).clamp(0.1, 10.0).min(0.891 / peak);
    pieces.iter_mut().for_each(|p| p.samples.iter_mut().for_each(|v| *v *= g));
}

/// Distinct codebook-0 values across a segment's frames.
///
/// A talker that has lost the thread repeats one code, and repeated codes render as a
/// metallic buzz — the codec faithfully decodes whatever it is given.
fn distinct_first(frames: &[Vec<u32>]) -> usize {
    let mut seen: Vec<u32> = frames.iter().map(|f| f[0]).collect();
    seen.sort_unstable();
    seen.dedup();
    seen.len()
}

/// Warn about segments that look wrong, against the median of *this* request.
///
/// Two failure modes, and they need opposite tests. CosyVoice only had to catch segments that
/// stopped **early** (a duration ratio below the median). This model can also run **long**:
/// once the text stream is exhausted the talker is fed `tts_pad` forever and will keep emitting
/// frames until it chooses `codec_eos`, so a segment can babble past its text. A median is the
/// right reference because it is measured from this voice and this text rather than assumed.
fn report_segments(stats: &[(usize, usize, usize)]) {
    if stats.len() < 3 {
        return;
    }
    let mut ratios: Vec<f64> = stats
        .iter()
        .map(|(chars, frames, _)| *frames as f64 / (*chars).max(1) as f64)
        .collect();
    let mut sorted = ratios.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    if median <= 0.0 {
        return;
    }
    let verbose = std::env::var_os("QWEN3TTS_SEGMENTS").is_some();
    if verbose {
        eprintln!("engine {ID}: median {median:.2} frames/char");
    }
    for (i, (chars, frames, distinct)) in stats.iter().enumerate() {
        let ratio = ratios[i];
        if verbose {
            eprintln!(
                "  seg {i}: {chars:>4} chars {frames:>4} frames  {ratio:.2} f/c  \
                 {distinct:>3} distinct code0 ({:.2})  {:.2} s",
                *distinct as f64 / (*frames).max(1) as f64,
                *frames as f64 / crate::cfg::FRAME_RATE
            );
        }
        // Frames per character, against the median. Long segments only: a short segment's
        // ratio is naturally noisy.
        if *chars >= SEGMENT_MIN_CHARS && ratio > median * SEGMENT_RATIO_CEILING {
            eprintln!(
                "engine {ID}: segment {i} runs long — {frames} frames for {chars} chars \
                 ({ratio:.2} vs median {median:.2}); the talker kept generating after its text \
                 ran out. Expect audible babble there."
            );
        }
        // Degenerate repetition renders as a metallic buzz.
        let variety = *distinct as f64 / (*frames).max(1) as f64;
        if *frames >= SEGMENT_MIN_FRAMES && variety < SEGMENT_MIN_VARIETY {
            eprintln!(
                "engine {ID}: segment {i} looks degenerate — only {distinct} distinct codebook-0 \
                 values across {frames} frames ({variety:.2}). Repeated codes decode as a \
                 metallic buzz."
            );
        }
    }
    ratios.clear();
}

pub struct Qwen3TtsEngine {
    talker: Talker,
    codec: Codec,
    tokenizer: Tokenizer,
    device: Device,
    /// `--set language=…`; auto when unset.
    language: Option<Language>,
    /// `--set lead=…`: what a segment is prefixed with: every segment under an adapter, a
    /// respelled one only when it opens on a nasal. `lead=off` drops it.
    lead: String,
    /// The first token after the reference is often not spoken: behind "... " a
    /// sentence-initial ng' was kept 11/12 against 0/12 bare; news CER 4.6% -> 4.5%.
    lead_all: bool,
    /// Hyphen-and-lead respelling for Swahili voices; off by default under an adapter, whose
    /// talker was trained on the real spelling. `--set respell=on|off`.
    respell: bool,
    /// Syllable BPE (see [`syllables`]); set by an adapter trained that way.
    syllables: bool,
    /// The adapter's orthography ([`Scheme`]); supersedes `syllables`.
    scheme: Option<Scheme>,
    /// `--set max_pause=S`: cap pauses inside a segment; 0.6 by default under an adapter.
    max_pause: Option<f32>,
    /// Each segment conditioned on the previous one; default under an adapter.
    continuity: bool,
    /// `continuity=paragraph`: the chain restarts at each paragraph.
    per_paragraph: bool,
    /// Segment levels matched ([`match_levels`]); default under an adapter, `--set level=`.
    level: bool,
    /// `--set level_db=-20|off`: the request's median speech level; -20 under an adapter.
    level_db: Option<f32>,
    /// Whether to batch segments through the talker. Dense weights only — see the grouping
    /// code for the measurement.
    batches: bool,
}

impl Qwen3TtsEngine {
    pub fn load(config: &EngineConfig) -> Result<Self> {
        let paths = Paths::resolve(config);
        paths
            .check()
            .with_context(|| format!("loading engine `{ID}`"))?;
        let device = if config.cpu {
            Device::Cpu
        } else {
            Device::new_metal(0).context("opening the Metal device")?
        };
        let quant = parse_quant(config.quant.as_deref())?;
        let s = |p: &Path| -> Result<String> {
            p.to_str()
                .map(str::to_owned)
                .with_context(|| format!("non-utf8 path {}", p.display()))
        };

        let tokenizer = qwen_tokenizer(&s(&paths.vocab)?, &s(&paths.merges)?)?;

        if let Some(total) = tts_core::system::total_memory() {
            if total < WANTS_MEMORY {
                eprintln!(
                    "note: engine `{ID}` peaks at 6.7 GB on a short passage and 9.2 GB on a \
                     chapter, and this machine has {}. It will swap, which reads as the model being \
                     slow rather than as a mistake. `--engine kokoro` peaks at 1.3 GB, without \
                     cloning. Lowering QWEN3TTS_MAX_BATCH below {} saves about 77 MB a lane.",
                    tts_core::system::human_bytes(total),
                    max_batch(),
                );
            }
        }

        // `--set adapter=…`: a LoRA over the checkpoint, e.g. the Swahili fine-tune.
        let adapter = config.overrides.get("adapter").map(|p| s(p)).transpose()?;
        // `--set adapter_gain=G`: the LoRA at strength G, whatever gain the file was exported at.
        // Voices unlike the training speakers can want less (the male English reference: CER 6.0%
        // at 0.6, 5.4% at 0.45 over 36 renders) while the owner's voice wants the export's.
        let adapter_scale = match (config.overrides.get("adapter_gain").and_then(|p| p.to_str()), &adapter) {
            (Some(g), Some(a)) => {
                let want: f32 = g.parse().with_context(|| format!("adapter_gain={g}"))?;
                want / adapter_export_gain(a)?
            }
            (Some(_), None) => anyhow::bail!("adapter_gain needs --set adapter=…"),
            (None, _) => 1.0,
        };
        let talker = Talker::load(&s(&paths.talker)?, adapter.as_deref(), adapter_scale, quant, &device)?;
        if std::env::var_os("QWEN3TTS_TIMING").is_some() {
            mem_line("talker loaded", &device);
        }
        // The orthography an adapter was trained on; it also names the adapter's language row.
        let scheme = talker
            .adapter_bytes("scheme")?
            .map(|b| Scheme::from_json(&b))
            .transpose()
            .context("the adapter's meta::scheme")?;
        let adapted = scheme.as_ref().map(|s| (s.language.as_str(), s.row));
        let language = config
            .overrides
            .get("language")
            .map(|spec| parse_language(spec, adapted, |id| talker.codec_row_f32(id)))
            .transpose()?;
        if let Some(Language::Tag(id)) = language {
            let norm = talker.codec_row_f32(id)?.iter().map(|x| x * x).sum::<f32>().sqrt();
            anyhow::ensure!(
                norm > 0.1,
                "codec row {id} of this talker is untrained (norm {norm:.3}): that language needs \
                 its adapter (--set adapter=…)"
            );
        }
        let respell = match config.overrides.get("respell").and_then(|p| p.to_str()) {
            Some("off" | "0" | "false") => false,
            Some(_) => true,
            None => adapter.is_none(),
        };
        let lead_spec = config.overrides.get("lead").and_then(|p| p.to_str());
        let lead = match lead_spec {
            Some("off") => String::new(),
            Some(l) => format!("{l} "),
            None => "... ".to_string(),
        };
        let lead_all = adapter.is_some() && !lead.is_empty();
        let syllables = talker.adapter_flag("syllables");
        // On by default under an adapter: the Swahili one drifts in timbre between independent
        // segments (window x-vector cosine 0.983, min 0.973) and chaining holds it (0.988, 0.975).
        // Across paragraphs too: restarting at each one was heard as a change of voice and level.
        let spec = config.overrides.get("continuity").and_then(|p| p.to_str());
        let continuity = match spec {
            Some("off" | "0" | "false") => false,
            Some(_) => true,
            None => adapter.is_some(),
        };
        let per_paragraph = spec == Some("paragraph");
        let level = match config.overrides.get("level").and_then(|p| p.to_str()) {
            Some("off" | "0" | "false") => false,
            Some(_) => true,
            None => adapter.is_some(),
        };
        let level_db = match config.overrides.get("level_db").and_then(|p| p.to_str()) {
            Some("off") => None,
            Some(v) => Some(v.parse::<f32>().with_context(|| format!("level_db={v}"))?),
            None => (level && adapter.is_some()).then_some(-20.0),
        };
        let max_pause = match config.overrides.get("max_pause").and_then(|p| p.to_str()) {
            Some("off") => None,
            Some(v) => Some(v.parse::<f32>().with_context(|| format!("max_pause={v}"))?),
            None => adapter.as_ref().map(|_| 0.6),
        };
        let codec = Codec::load(&s(&paths.codec)?, &device)?;
        if std::env::var_os("QWEN3TTS_TIMING").is_some() {
            mem_line("codec loaded", &device);
        }
        Ok(Self {
            talker,
            codec,
            tokenizer,
            device,
            language,
            batches: quant.batches(),
            lead,
            lead_all,
            respell,
            syllables,
            scheme,
            max_pause,
            continuity,
            per_paragraph,
            level,
            level_db,
        })
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let pieces = match &self.scheme {
            Some(s) => s.pieces(text),
            None if self.syllables => syllables::pieces(text),
            None => return self.encode(text),
        };
        let mut ids = Vec::new();
        for piece in pieces {
            ids.extend(self.encode(&piece)?);
        }
        Ok(ids)
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?
            .get_ids()
            .to_vec())
    }
}

impl Engine for Qwen3TtsEngine {
    fn capabilities(&self) -> Capabilities {
        capabilities()
    }

    fn validate(&self, request: &SynthesisRequest) -> Result<()> {
        // Unreachable while `available` is false, but written now so flipping that flag is
        // not also the commit that has to remember this rule.
        tts_core::engine::validate_against(&self.capabilities(), request)?;
        anyhow::ensure!(request.voice.is_some(), "{NO_VOICE}");
        Ok(())
    }

    fn synthesize(&self, request: &SynthesisRequest) -> Result<Synthesis> {
        self.validate(request)?;
        let voice = request.voice.as_ref().expect("validated");
        let spk = self.talker.speaker(voice.get("spk_embedding")?)?;
        // [T, 16] frames-major, the orientation `generate_icl_prompt` indexes.
        let ref_codes = voice.get_rows_u32("ref_codes").unwrap_or_default();
        // A syllable-trained talker saw its references tokenized the same way.
        let ref_text = if (self.syllables || self.scheme.is_some()) && !voice.text.is_empty() {
            self.tokenize(&voice.text)?
        } else {
            voice
                .get_rows_u32("ref_text_tokens")
                .ok()
                .and_then(|r| r.into_iter().next())
                .unwrap_or_default()
        };

        let language = self.language.clone().unwrap_or(Language::Auto);
        let swahili = self.respell && voice.language.as_deref() == Some("swahili");

        let paragraphs = text::segment(&request.text, request.max_chars);
        let flat: Vec<(usize, &String)> = paragraphs
            .iter()
            .enumerate()
            .flat_map(|(pi, para)| para.iter().map(move |s| (pi, s)))
            .collect();
        anyhow::ensure!(!flat.is_empty(), "no text to speak");

        // Prefer this model's own documented defaults over `tts_core::Sampling`'s generic
        // ones, but honour anything the caller actually chose.
        //
        // The generic defaults are temperature 0.7 / top_p 0.9; the reference uses 0.9 / 1.0
        // for both the talker and the sub-talker. A top_p of 0.9 truncates a distribution the
        // reference never truncates, and applying it to the acoustic residuals thins the timbre
        // — audibly metallic, while codebook 0 keeps the words intelligible. Shipping a
        // known-worse default because a shared struct happened to pick it would be wrong.
        //
        // `SynthesisRequest` cannot say whether a field was set or defaulted, so a field equal
        // to the generic default is treated as unset. The cost of that heuristic is a caller who
        // explicitly asks for exactly 0.7/0.9 and silently gets 0.9/1.0; the alternative is
        // every caller getting the worse sound unless they know to override it.
        let generic = tts_core::Sampling::default();
        let req = &request.sampling;
        let pick = |got: f32, generic: f32, reference: f32| {
            if (got - generic).abs() < f32::EPSILON {
                reference
            } else {
                got
            }
        };
        let sampling = Sampling {
            temperature: pick(
                req.temperature,
                generic.temperature,
                cfg::talker::TEMPERATURE,
            ),
            top_p: pick(req.top_p, generic.top_p, cfg::talker::TOP_P),
            top_k: if req.top_k == generic.top_k {
                cfg::talker::TOP_K
            } else {
                req.top_k
            },
            greedy: req.greedy,
            ..Sampling::default()
        };
        let mut rng = Rng::new(request.sampling.seed);
        let mut stats = Stats::default();
        let t0 = Instant::now();
        if std::env::var_os("QWEN3TTS_TIMING").is_some() {
            mem_line("before synthesis", &self.device);
        }

        // Stage 1: the talker, batching every segment whose prompt is the same length.
        //
        // Both transformers are bandwidth-bound on *weight* reads at batch 1 — the trunk reads
        // 1.4 G parameters once a frame and the depth predictor reads its 60 M fifteen times —
        // so a lane costs almost nothing beyond the read that a batch already pays. Measured by
        // `qwen3tts-batch`: dense f32 at batch 8 is **7.45x cheaper per lane**. Quantized is
        // not, at 1.13x: candle's `quantized_matmul_mm_t` re-reads the weights per row, which
        // is why `quant=f32` is the fast configuration here and q8_0 the small one.
        //
        // Grouping by length is what makes this cheap. See `Talker::generate_batch`.
        let budget = request.max_new_tokens.clamp(1, cfg::talker::MAX_NEW_TOKENS);
        let mut prepared: Vec<(usize, usize, Tensor, Tensor)> = Vec::new();
        let mut prepared_ids: Vec<Vec<u32>> = Vec::new();
        let mut shared = usize::MAX;
        for (pi, seg) in &flat {
            let ids = if swahili {
                self.tokenize(&swahili::respell(seg, &self.lead))?
            } else if self.lead_all {
                self.tokenize(&format!("{}{seg}", self.lead))?
            } else {
                self.tokenize(seg)?
            };
            if ids.is_empty() {
                continue;
            }
            let (prompt, trailing, common) = self.talker.build_prompt_shared(
                &ids,
                &ref_text,
                &ref_codes,
                Some(&spk),
                &language,
            )?;
            shared = shared.min(common);
            prepared.push((*pi, seg.chars().count(), prompt, trailing));
            prepared_ids.push(ids);
        }

        // Lanes of equal prompt length, in runs of at most MAX_BATCH; trailing is padded to the
        // group's longest with `tts_pad`, which is what a lane is fed once its text is spent, so
        // the padding changes nothing. Keying on trailing too left every segment whose text
        // outran the reference frames — all of them under syllable BPE — unbatched: RTF 1.3.
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut by_shape: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, (_, _, p, _)) in prepared.iter().enumerate() {
            by_shape.entry(p.dim(1)?).or_default().push(i);
        }
        let mut shapes: Vec<_> = by_shape.into_values().collect();
        shapes.sort_by_key(|g| g[0]);
        // Only dense weights batch. Quantized ones measure *worse* batched (RTF 1.02 against
        // 0.79 on this text) because candle's `quantized_matmul_mm_t` re-reads the weights per
        // row, so the batch pays full price per lane and then wastes steps on finished lanes.
        let lanes = if self.batches { max_batch() } else { 1 };
        for mut g in shapes {
            // **Sort by length before chunking, longest first.** Lanes stop at their own
            // `codec_eos`, so a group runs as long as its longest member. Grouping similar
            // lengths together is what bounds that, and character count is a good enough proxy
            // because frames per character is stable within one voice (it is the same ratio
            // `report_segments` takes a median of).
            //
            // **Longest-first is what makes shedding possible**: `generate_batch` can only drop
            // a contiguous *tail* of finished lanes, since a prefix narrow shares the caches'
            // storage. Ascending order put every early finisher at the head, where nothing can
            // be dropped — 68% of lane-steps useful at 48 lanes, against 47-56% unsorted.
            //
            // Stable, and tie-broken by index, so a render stays reproducible under a seed.
            g.sort_by_key(|&i| (std::cmp::Reverse(prepared[i].1), i));
            for chunk in g.chunks(lanes) {
                groups.push(chunk.to_vec());
            }
        }

        request.notify(tts_core::ProgressEvent::Planned {
            segments: prepared.len(),
        });

        // Frames per character, learned from the groups already decoded. Stable within a voice,
        // which is the same assumption `report_segments` takes a median under.
        let mut seen_ratios: Vec<f64> = Vec::with_capacity(prepared.len());
        let mut ratio: Option<f64> = None;

        let mut out: Vec<Option<Decoded>> = vec![None; prepared.len()];
        let mut unspoken = 0usize;
        let mut talker_done = 0usize;
        let t = Instant::now();
        if self.continuity {
            // Each segment continues the one before it: that segment's text and
            // frames follow the voice's reference. Independent segments each restart the
            // delivery, and a paragraph sounds stitched. Sequential, so nothing batches.
            for i in 0..prepared.len() {
                let (pi, chars, prompt, trailing) = &prepared[i];
                let chained = match i.checked_sub(1).and_then(|p| out[p].as_ref().map(|o| (p, o))) {
                    Some((p, (ppi, _, frames))) if (ppi == pi || !self.per_paragraph) && !frames.is_empty() => {
                        let mut text = ref_text.clone();
                        text.extend_from_slice(&prepared_ids[p]);
                        let mut codes = ref_codes.clone();
                        codes.extend(frames.iter().cloned());
                        Some(self.talker.build_prompt_shared(&prepared_ids[i], &text, &codes, Some(&spk), &language)?)
                    }
                    _ => None,
                };
                let (prompt, trailing) = match &chained {
                    Some((p, t, _)) => (p, t),
                    None => (prompt, trailing),
                };
                let (frames, left, _) = self.talker.generate(prompt, trailing, budget, &sampling, &mut rng)?;
                unspoken += left;
                out[i] = Some((*pi, *chars, frames));
                talker_done += 1;
                request.advanced("talker", talker_done, prepared.len());
                request.check_interrupt(talker_done)?;
            }
        }
        for group in groups.iter().filter(|_| !self.continuity) {
            let mut batched = None;
            let mut padding: Vec<usize> = Vec::new();
            if group.len() > 1 {
                let cap = budget.min(BATCH_FRAME_CAP);
                let prompts: Vec<Tensor> = group.iter().map(|&i| prepared[i].2.clone()).collect();
                let longest = group.iter().map(|&i| prepared[i].3.dim(1)).collect::<Result<Vec<_>, _>>()?;
                let longest = longest.into_iter().max().unwrap_or(1);
                let pad = self.talker.pad_hidden()?;
                let mut trailings: Vec<Tensor> = Vec::with_capacity(group.len());
                for &i in group {
                    let tr = &prepared[i].3;
                    let short = longest - tr.dim(1)?;
                    padding.push(short);
                    trailings.push(if short == 0 {
                        tr.clone()
                    } else {
                        Tensor::cat(&[tr.clone(), pad.repeat((1, short, 1))?], 1)?
                    });
                }
                let prompt = Tensor::cat(&prompts, 0)?.contiguous()?;
                let trailing = Tensor::cat(&trailings, 0)?.contiguous()?;
                // Per-lane budget from the frames-per-character this voice has already shown,
                // doubled. A segment past that has stopped reading its text — the same
                // condition `report_segments` reports, caught while it is still costing steps
                // rather than afterwards. The first group has no ratio yet and runs uncapped.
                let budgets: Vec<usize> = group
                    .iter()
                    .map(|&i| match ratio {
                        Some(r) => ((prepared[i].1 as f64 * r * SEGMENT_BUDGET_SLACK) as usize)
                            .clamp(SEGMENT_MIN_FRAMES, cap),
                        None => cap,
                    })
                    .collect();
                let (frames, left, timing) = self.talker.generate_batch(
                    &prompt, &trailing, cap, &sampling, &mut rng, &budgets, shared,
                )?;
                if std::env::var_os("QWEN3TTS_TIMING").is_some() {
                    eprintln!(
                        "engine {ID}: batch {} — {} steps, {} lane-steps ({} without shedding), \
                         {} frames, {:.0}% of lane-steps useful; prefill {:.2}s, talker {:.2}s, \
                         predictor {:.2}s (stack {:.2}s, heads {:.2}s, read {:.2}s)",
                        group.len(),
                        timing.steps,
                        timing.lane_steps,
                        timing.steps * timing.lanes,
                        timing.frames,
                        timing.frames as f64 / timing.lane_steps.max(1) as f64 * 100.0,
                        timing.prefill_s,
                        timing.talker_s,
                        timing.predictor_s,
                        timing.depth_stack_s,
                        timing.depth_gemm_s,
                        timing.depth_read_s,
                    );
                    mem_line("after the group", &self.device);
                }
                // A lane that filled the cap may have been cut off mid-sentence. Redo the group
                // one segment at a time with the real budget rather than ship truncated audio.
                if frames.iter().any(|f| f.len() >= cap) {
                    eprintln!(
                        "engine {ID}: a batched lane reached {cap} frames; rerunning {} \
                         segment(s) unbatched",
                        group.len()
                    );
                } else {
                    batched = Some((frames, left));
                }
            }
            match batched {
                Some((frames, left)) => {
                    unspoken += left.iter().zip(&padding).map(|(l, p)| l.saturating_sub(*p)).sum::<usize>();
                    for (lane, &i) in group.iter().enumerate() {
                        out[i] = Some((prepared[i].0, prepared[i].1, frames[lane].clone()));
                    }
                }
                None => {
                    for &i in group {
                        let (pi, chars, prompt, trailing) = &prepared[i];
                        let (frames, left, _) = self
                            .talker
                            .generate(prompt, trailing, budget, &sampling, &mut rng)?;
                        unspoken += left;
                        out[i] = Some((*pi, *chars, frames));
                    }
                }
            }
            // Update the ratio from what this group actually produced, before the next one
            // sizes its budgets against it.
            for &i in group {
                if let Some((_, chars, frames)) = &out[i] {
                    if *chars >= SEGMENT_MIN_CHARS {
                        seen_ratios.push(frames.len() as f64 / *chars as f64);
                    }
                }
            }
            if seen_ratios.len() >= SEGMENT_RATIO_SAMPLE {
                let mut sorted = seen_ratios.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite ratios"));
                ratio = Some(sorted[sorted.len() / 2]);
            }

            // Per group, not per lane: a batched group finishes together, and the talker is
            // 0.673 of this engine's 0.846 RTF, so this is the number worth showing.
            talker_done += group.len();
            request.advanced("talker", talker_done, prepared.len());
            request.check_interrupt(talker_done)?;
        }

        // Redraw segments outside the voice's own pace; keep the attempt nearest to it.
        let voice_chars = voice.text.chars().count();
        if !ref_codes.is_empty() && voice_chars >= SEGMENT_MIN_CHARS {
            let per_char = ref_codes.len() as f64 / voice_chars as f64;
            let mut redrawn = 0usize;
            for (i, slot) in out.iter_mut().enumerate() {
                let Some((_, chars, frames)) = slot else { continue };
                if *chars < SEGMENT_MIN_CHARS {
                    continue;
                }
                let expect = *chars as f64 * per_char;
                let off = |n: usize| (n as f64 / expect).ln().abs();
                let bad = |n: usize| (n as f64) > expect * REDRAW_HIGH + 8.0 || (n as f64) < expect * REDRAW_LOW;
                if !bad(frames.len()) {
                    continue;
                }
                let cap = ((expect * REDRAW_HIGH) as usize + 8).min(budget);
                let (_, _, prompt, trailing) = &prepared[i];
                for _ in 0..REDRAWS {
                    let (again, _, _) = self.talker.generate(prompt, trailing, cap, &sampling, &mut rng)?;
                    redrawn += 1;
                    if off(again.len()) < off(frames.len()) {
                        *frames = again;
                    }
                    if !bad(frames.len()) {
                        break;
                    }
                }
            }
            if redrawn > 0 {
                eprintln!("engine {ID}: redrew {redrawn} segment attempt(s) outside the voice's pace");
            }
        }

        let mut spans: Vec<(usize, String, Vec<Vec<u32>>)> = Vec::new();
        // (characters, frames, distinct codebook-0 values) per segment.
        let mut seg_stats: Vec<(usize, usize, usize)> = Vec::new();
        for (i, slot) in out.into_iter().enumerate() {
            let Some((pi, chars, frames)) = slot else {
                continue;
            };
            if !frames.is_empty() {
                seg_stats.push((chars, frames.len(), distinct_first(&frames)));
                // `out` is indexed by the prepared segment, which is indexed by `flat`, so the
                // text is the one at the same position.
                spans.push((pi, flat[i].1.clone(), frames));
            }
        }
        // Metal dispatch is async: without this the stage time is enqueue time and the GPU
        // work is billed to whatever is timed next.
        self.device.synchronize()?;
        stats.add("talker", t.elapsed().as_secs_f64());
        anyhow::ensure!(!spans.is_empty(), "engine {ID} generated no frames");
        report_segments(&seg_stats);

        if unspoken > 0 {
            // Trap 3: text is consumed one token per frame, so a segment that stopped early
            // leaves an exact count behind rather than a ratio to estimate.
            eprintln!(
                "engine {ID}: {unspoken} text position(s) never reached the talker — some text \
                 was not spoken. Lower max_chars."
            );
        }

        // Stage 2: the codec decoder, **once over every segment's frames**, cut afterwards.
        //
        // Not per segment. The decoder is causal and its receptive field is large — pre_conv
        // k=3, a ConvNeXt k=7, then k=7 convs at dilations 1/3/9 through four upsample stages
        // — so a per-segment call opens each segment with zero left context and an audible
        // transient. Segment two onward came out with a different timbre from segment one for
        // exactly that reason.
        //
        // Decoding the concatenation gives every segment real history, and the cut points are
        // *exact* rather than estimated: one frame is `SAMPLES_PER_FRAME` samples, always.
        // Chunking still happens inside `decode`, where it carries its own left context.
        //
        // The reference also prefixes `ref_codes` and cuts it off. The talker continues that
        // clip, so decoding cold clipped every utterance's first consonant ("Mbwa" → "wa").
        let t = Instant::now();
        let context = &ref_codes[ref_codes.len().saturating_sub(cfg::codec::CHUNK_LEFT_CONTEXT)..];
        let all_frames: Vec<Vec<u32>> = context
            .iter()
            .cloned()
            .chain(spans.iter().flat_map(|(_, _, frames)| frames.iter().cloned()))
            .collect();
        let frames_total = all_frames.len() - context.len();
        let mut joined = self.codec.decode(&all_frames)?;
        joined.drain(..(context.len() * cfg::SAMPLES_PER_FRAME).min(joined.len()));
        self.device.synchronize()?;
        stats.add("codec", t.elapsed().as_secs_f64());
        if std::env::var_os("QWEN3TTS_TIMING").is_some() {
            mem_line("after the codec", &self.device);
        }
        // One call for the whole utterance, so there is nothing to count through.
        request.advanced("codec", spans.len(), spans.len());

        let mut pieces: Vec<tts_core::Piece> = Vec::with_capacity(spans.len());
        let mut at = 0usize;
        for (i, (pi, text, frames)) in spans.iter().enumerate() {
            // The last segment takes whatever remains, so rounding cannot drop samples.
            let end = if i + 1 == spans.len() {
                joined.len()
            } else {
                (at + frames.len() * cfg::SAMPLES_PER_FRAME).min(joined.len())
            };
            if end > at {
                let mut samples = joined[at..end].to_vec();
                if let Some(max) = self.max_pause {
                    samples = cap_pauses(&samples, max);
                }
                pieces.push(tts_core::Piece {
                    paragraph: *pi,
                    text: text.clone(),
                    samples,
                });
            }
            at = end;
        }

        if self.level {
            match_levels(&mut pieces, self.level_db);
        }
        let (samples, segments) = tts_core::join_segments(pieces, request.gaps, cfg::SAMPLE_RATE);
        anyhow::ensure!(!samples.is_empty(), "engine {ID} produced no audio");
        stats.segments = spans.len();
        stats.frames = frames_total;
        stats.total_s = t0.elapsed().as_secs_f64();
        Ok(Synthesis {
            audio: Audio {
                samples,
                sample_rate: cfg::SAMPLE_RATE as u32,
            },
            stats,
            segments: Some(segments),
            // No duration predictor and no cross-attention to read, so no word clock — only
            // the segment boundaries, which this engine does know exactly.
            words: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ids from transformers' AutoTokenizer on the checkpoint, which made the training data.
    #[test]
    fn tokenizer_matches_transformers() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../references/qwen3tts/weights");
        let (vocab, merges) = (format!("{dir}/vocab.json"), format!("{dir}/merges.txt"));
        if !Path::new(&vocab).exists() {
            eprintln!("skipped: no checkpoint tokenizer at {dir}");
            return;
        }
        let t = qwen_tokenizer(&vocab, &merges).unwrap();
        let ids = |s: &str| t.encode(s, false).unwrap().get_ids().to_vec();
        let cases: [(&str, &[u32]); 12] = [
            ("P.C.E.A", &[47, 727, 5142, 875]),
            ("Mwenyekiti wa P.C.E.A-Kenya", &[44, 16948, 88, 1225, 12303, 10450, 393, 727, 5142, 875, 15843, 268, 7755]),
            ("-K", &[15843]),
            ("don't stop", &[15007, 944, 2936]),
            ("Mwaka 2026, saa 10:30.", &[44, 86, 13334, 220, 17, 15, 17, 21, 11, 822, 64, 220, 16, 15, 25, 18, 15, 13]),
            ("Hello, world!", &[9707, 11, 1879, 0]),
            ("mbili\nndani", &[3096, 3921, 198, 303, 5559]),
            ("  spaced  out ", &[220, 63828, 220, 700, 220]),
            (" ŋ", &[25917, 233]),
            ("Ng'ombe (wawili)", &[20897, 6, 316, 1371, 320, 86, 672, 3921, 8]),
            ("e.g. naam...", &[68, 1302, 13, 99105, 1112]),
            ("“Habari”—leo", &[2073, 39, 370, 2780, 62650, 81763]),
        ];
        for (text, want) in cases {
            assert_eq!(ids(text), want, "{text:?}");
        }
    }

    #[test]
    fn reads_the_export_gain() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../references/qwen3tts/weights/swahili-adapter.safetensors");
        if !Path::new(path).exists() {
            eprintln!("skipped: no adapter at {path}");
            return;
        }
        assert!((adapter_export_gain(path).unwrap() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn levels_to_an_absolute_target() {
        let tone = |amp: f32, n: usize| -> Vec<f32> {
            (0..n).map(|i| amp * (i as f32 * 0.05).sin()).collect()
        };
        let piece = |samples| tts_core::Piece { paragraph: 0, text: String::new(), samples };
        let mut pieces = vec![piece(tone(0.02, 24000)), piece(tone(0.025, 24000)), piece(tone(0.018, 24000))];
        match_levels(&mut pieces, Some(-20.0));
        let mut db: Vec<f32> = pieces.iter().map(|p| 20.0 * active_rms(&p.samples).log10()).collect();
        db.sort_by(f32::total_cmp);
        assert!((db[1] + 20.0).abs() < 0.2, "{db:?}");
        let mut loud = vec![piece(tone(0.9, 24000)), piece(tone(0.1, 24000))];
        match_levels(&mut loud, Some(-3.0));
        let peak = loud.iter().flat_map(|p| p.samples.iter()).fold(0f32, |m, v| m.max(v.abs()));
        assert!(peak <= 0.892, "{peak}");
    }

    #[test]
    fn caps_pauses_and_edges() {
        let sr = cfg::SAMPLE_RATE;
        let tone = |n: usize| (0..n).map(|i| (i as f32 * 0.1).sin() * 0.5).collect::<Vec<f32>>();
        let mut x = vec![0.0; sr / 2];
        x.extend(tone(sr / 2));
        x.extend(vec![0.0; 2 * sr]);
        x.extend(tone(sr / 2));
        x.extend(vec![0.0; sr / 2]);
        let y = cap_pauses(&x, 0.6);
        let secs = (x.len() - y.len()) as f32 / sr as f32;
        // 1.4 s from the interior run, 0.4 s from each edge.
        assert!((secs - 2.2).abs() < 0.02, "removed {secs} s");
    }

    #[test]
    fn language_specs() {
        let row = |id: u32| Ok(vec![id as f32; cfg::talker::DIM]);
        let parse = |s: &str| parse_language(Path::new(s), None, row);
        assert_eq!(parse("auto").unwrap(), Language::Auto);
        assert_eq!(parse("Italian").unwrap(), Language::Tag(2070));
        let Language::Vector(v) = parse("italian:0.5, spanish:0.5").unwrap() else {
            panic!("a blend is a vector")
        };
        assert_eq!(v[0], 0.5 * 2070.0 + 0.5 * 2054.0);
        assert_eq!(parse("swahili").unwrap(), Language::Tag(cfg::talker::SWAHILI));
        assert!(parse("klingon").is_err());
        let kikuyu = parse_language(Path::new("Kikuyu"), Some(("kikuyu", 2075)), row).unwrap();
        assert_eq!(kikuyu, Language::Tag(2075));
        assert!(parse("italian:x").is_err());
    }

    /// Geometry identities that would otherwise surface as a shape mismatch mid-port.
    #[test]
    fn geometry_is_self_consistent() {
        // 24 kHz at 12.5 Hz is 1920 samples per frame, and the codec's two upsampling
        // stacks have to multiply out to exactly that.
        let ratios: usize = cfg::codec::UPSAMPLING_RATIOS.iter().product();
        let rates: usize = cfg::codec::UPSAMPLE_RATES.iter().product();
        assert_eq!(ratios * rates, cfg::SAMPLES_PER_FRAME);
        assert_eq!(
            cfg::SAMPLE_RATE as f64 / cfg::SAMPLES_PER_FRAME as f64,
            cfg::FRAME_RATE
        );

        // The talker fills codebook 0 and the predictor the rest.
        assert_eq!(cfg::predictor::HEADS_OUT + 1, cfg::CODE_GROUPS);
        assert_eq!(cfg::codec::QUANTIZERS, cfg::CODE_GROUPS);
        assert_eq!(
            cfg::codec::SEMANTIC_QUANTIZERS + cfg::codec::ACOUSTIC_QUANTIZERS,
            cfg::CODE_GROUPS
        );

        // Trap 5: the talker's live range covers each codebook exactly.
        assert_eq!(cfg::talker::CODES, cfg::codec::CODEBOOK);
        assert_eq!(cfg::codec::ENCODER_VALID_QUANTIZERS, cfg::CODE_GROUPS);
        const { assert!(cfg::codec::ENCODER_QUANTIZERS > cfg::codec::ENCODER_VALID_QUANTIZERS) };
        // Every control id sits above the live range, which is what makes a single
        // `id < CODES` test a valid "is this a real code".
        for id in [
            cfg::talker::CODEC_PAD,
            cfg::talker::CODEC_BOS,
            cfg::talker::CODEC_EOS,
            cfg::talker::CODEC_THINK,
            cfg::talker::CODEC_NOTHINK,
            cfg::talker::CODEC_THINK_BOS,
            cfg::talker::CODEC_THINK_EOS,
        ] {
            assert!(id as usize >= cfg::talker::CODES);
            assert!((id as usize) < cfg::talker::VOCAB);
        }

        // Trap 6: the predictor's heads do not tile its hidden size.
        assert_ne!(
            cfg::predictor::HEADS * cfg::predictor::HEAD_DIM,
            cfg::predictor::DIM
        );
        // ...while the talker's do, which is exactly why the mistake is easy.
        assert_eq!(cfg::talker::HEADS * cfg::talker::HEAD_DIM, cfg::talker::DIM);

        // The final conv's width is the decoder dim halved once per rate.
        assert_eq!(
            cfg::codec::OUT_CHANNELS,
            cfg::codec::stage_channels(cfg::codec::UPSAMPLE_RATES.len() - 1)
        );

        // The 1.7B speaker embedding is consumed as one position in the talker's stream, so
        // it has to be exactly that wide.
        assert_eq!(cfg::speaker::ENC_DIM, cfg::talker::DIM);
    }

    #[test]
    fn languages_have_ids_and_the_list_is_closed() {
        for name in cfg::talker::LANGUAGES {
            assert!(
                cfg::talker::language_id(name).is_some(),
                "no codec language id for `{name}`"
            );
        }
        assert_eq!(cfg::talker::LANGUAGES.len(), 10);
        // Swahili is ours, not the checkpoint's, so it stays off the advertised list.
        assert!(!cfg::talker::LANGUAGES.contains(&"swahili"));
        assert!(cfg::talker::language_id("klingon").is_none());
    }

    /// A request with no voice must be refused, not answered with an arbitrary speaker.
    #[test]
    fn refuses_without_a_voice() {
        let caps = capabilities();
        assert!(caps.available);
        assert_eq!(caps.cloning, Cloning::PrecomputedAsset);
        // The capability checks pass; the engine's own voice rule is what rejects this.
        let request = SynthesisRequest::new("Hello.");
        assert!(tts_core::engine::validate_against(&caps, &request).is_ok());
        assert!(NO_VOICE.contains(ID));
    }

    /// f16 is this engine's default because it is the only one that batches, and batching is
    /// worth 4.5x where q8_0's narrower weight read is worth 0.58 GB.
    #[test]
    fn defaults_to_f16() {
        assert_eq!(QUANT[0], "f16");
        assert_eq!(parse_quant(None).unwrap(), Weight::F16);
        assert_eq!(
            parse_quant(Some("q8_0")).unwrap(),
            Weight::Quant(GgmlDType::Q8_0)
        );
        assert_eq!(parse_quant(Some("f32")).unwrap(), Weight::F32);
        assert_eq!(parse_quant(Some("f16")).unwrap(), Weight::F16);
        assert!(parse_quant(Some("nonsense")).is_err());
    }

    /// Only the dense formats batch, which is why the default is one of them. Batching a
    /// quantized weight is worse than not: RTF 1.02 against 0.79 unbatched, because candle's
    /// `mm_t` re-reads the weights per row and the batch pays full price per lane.
    #[test]
    fn only_dense_weights_batch() {
        assert!(parse_quant(None).unwrap().batches());
        assert!(parse_quant(Some("f16")).unwrap().batches());
        assert!(parse_quant(Some("f32")).unwrap().batches());
        assert!(!parse_quant(Some("q8_0")).unwrap().batches());
    }
}
