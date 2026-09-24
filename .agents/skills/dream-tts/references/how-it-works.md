# How it works

## The shape of it

Four engines behind one `Engine` trait, chosen by string id at request time. `qwen3tts` is the
default and carries the project; the other three exist because they are better at one thing each.

| engine | model | output | notes |
|---|---|---|---|
| `qwen3tts` | Qwen3-TTS-12Hz-1.7B-Base | 24 kHz | default. Qwen3 talker, 15-step depth transformer, RVQ codec. No diffusion, no conversion step, ten languages |
| `audio8` | Audio8-TTS-Preview-0.6b | 44.1 kHz | DualAR + RVQ codec. Highest fidelity here; 2.36x its PyTorch reference like for like |
| `cosyvoice` | Fun-CosyVoice3-0.5B | 24 kHz | Qwen2 LLM + DiT flow matching + HiFTGenerator. Unrestricted languages |
| `kokoro` | Kokoro-82M | 24 kHz | The only non-autoregressive one: durations for every phoneme at once, one iSTFTNet pass. 23x realtime in 1.3 GB, English only, 28 built-in voices and no cloning |

`dream-tts engines` prints this live, with the weight formats each accepts and which is the
default.

Everything runs on Metal through candle, with custom kernels in `tts-nn` where candle's
composed ops cost too much. A `--no-default-features` CPU path exists as a portability
guarantee rather than a deployment target: roughly 4x slower.

## qwen3tts, end to end

```
text -> segments -> ICL prompt -> talker -> depth predictor -> codec -> 24 kHz
```

- **Segmentation.** Text is split to a character budget (`max_chars`, default 220) on sentence
  boundaries, with configurable gaps between segments and paragraphs.
- **The ICL prompt** carries the reference clip: its transcript as text embeddings and its codec
  frames, summed position-wise, plus the speaker embedding. This is what makes the clone work,
  and why a voice asset is not just an x-vector.
- **The talker** is a 28-layer Qwen3 trunk at width 2048 that emits codebook 0, one code per
  frame at 12.5 Hz.
- **The depth predictor** fills codebooks 1..15. It is called an "MTP block" upstream, which
  oversells it: it runs **15 sequential 5-layer passes per frame**, and it is 59% of the
  talker's cost. The most important cost fact about this engine.
- **The codec** turns 16 codes per frame into 1920 samples: an 8-layer sliding-window
  transformer, two upsampling stages with ConvNeXt blocks, then four decoder blocks of
  transposed conv plus three residual units each, at 1536 channels narrowing to 96.

## Why it is fast

**Batching across segments, and nothing else comes close.** 48 segments decode as one batch,
sorted longest-first so that finished lanes accumulate at the tail where they can be shed — only
a contiguous tail can be dropped, because a prefix narrow shares the caches' storage. Useful
lane-steps went from 68% to 79–90% when the sort order was fixed.

**f16 weights, and they are the default.** Only a dense GEMM shares one weight read across lanes;
candle's quantized `mm_t` re-reads per row, so q8_0 cannot batch at all. That single default is
worth 4.5x on book-length text and was wrong until v0.2.3.

**Custom kernels where candle composes.** candle performs no fusion, so a five-op elementwise
expression pays five full round-trips to device memory. `tts-nn` ships fused SnakeBeta, an
im2col gather, decode attention that reads the KV cache in place, a channels-last conv that
gathers its taps inside the GEMM, and a GEMM tiled for the shape a decode step has.

Current cost split on a 1612-word chapter: talker 70%, codec 30%, RTF 0.104–0.114, peak 9.2 GB.

## Why it is not faster

**Memory, and specifically candle's Metal buffer pool.** It keys buffers by size and releases
none, so every distinct tensor shape a run touches is permanent for the life of the process.
Most memory wins in this codebase are therefore a *shape removed* rather than bytes shaved —
a uniform codec decode span, prefill in fixed-width windows, shed widths quantised to multiples
of eight.

**The lane count is capped by that.** Per-lane cost is still falling at 64 lanes (1.615 ms at 48
against 1.460, because 64 fills the 8x8 matrix tiles that 48 leaves ragged) but peak footprint
goes 12.7 to 15.8 GB between them and the machine swaps. Four attempts at freeing it are
recorded in `docs/reference.md#what-did-not-work`, including an int8 KV cache that saved no
memory at all and a waveform-stack slicing that was exact and cost more than it returned.

**The codec's activations are the floor.** One 300-frame chunk is 5.5 GB, which is why no weight
format changes what the engine needs.

## Where to read further

- `README.md` — the guide, with the measured numbers and the diagram.
- `docs/reference.md#performance` — every measurement and how it was taken.
- `docs/reference.md#what-did-not-work` — ONNX, CoreML, a custom q8_0 GEMM, f32 weights, int8
  KV, and several allocation schemes. Read this before proposing an optimisation; most obvious
  ideas here have been measured and rejected with numbers.
- `docs/reference.md#porting-traps` — the places this model's reference implementation does
  something other than what its config says.
