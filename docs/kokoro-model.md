# Porting Kokoro-82M

A StyleTTS2 derivative and the only non-autoregressive engine here: duration is predicted
for every phoneme at once, the encoding is stretched by an alignment matrix, and one pass
through an iSTFTNet decoder produces the waveform. There is no sampling loop, so there is no
seed-dependent token stream — but there *is* a stochastic excitation, which is where most of
the difficulty turned out to be.

Gate: `cargo run -p kokoro --release --bin kokoro-validate`. Two tiers, as with qwen3tts —
a shape audit that reads only the safetensors header, then per-stage activations.

## Status

```
bert           ok  abs 1.431e-5   rel 1.233e-6
bert_encoder   ok  abs 2.778e-5   rel 4.704e-6
dur_enc        ok  abs 1.732e-5   rel 5.520e-6
pred_dur       ok  129 frames
F0_proj        ok  abs 7.477e-4   rel 2.390e-6
N_proj         ok  abs 7.361e-6   rel 7.634e-7
text_encoder   ok  abs 2.205e-6   rel 2.207e-6
excitation     ok  abs 1.304e-3   rel 8.666e-3
har_spectrum   ok  rel 6.069e-3   abs 8.259e-3
audio          ok  SNR 25.2 dB    77400 samples
```

Every deterministic stage matches at fp32. The last three cannot be matched bit-for-bit, for
a reason worth stating precisely rather than waving at.

## The excitation phase is not reproducible at fp32, upstream included

`SineGen` accumulates phase unwrapped: a cumulative sum over frames, scaled by `2*pi` and by
the 300x upsample factor. Over a three-second utterance it reaches **165,303 radians**, where
one f32 ulp is **0.0156 radians**. `sin` of that argument is uncertain at 1.6% from the
representation alone.

Four arithmetic orderings were tried against torch's own result and none is bit-identical:
`(1-l)a + lb` and `a + l(b-a)`, each in f32 and f64, all land within three ulp and none on
the mark. That is not a port problem to be solved; it is a property of the computation.

This port therefore keeps the phase **wrapped** — only `sin(phase)` is ever used, so the
running sum is held modulo one turn in f64 and the within-frame interpolation is done on the
per-frame *increment*, which is small. That is strictly more accurate than upstream and
deliberately not identical to it.

What the gate checks instead is that the port sits closer to the reference than the reference
sits to itself:

| | SNR vs reference | log-spectral distance |
|---|---|---|
| upstream, re-run with a different noise draw | 19.7 dB, 20.7 dB | 2.18 dB, 2.09 dB |
| this port | **25.2 dB** | **1.54 dB** |

The random draws themselves are captured in the fixture and replayed by the gate, so the
noise is compared exactly and only the phase is at issue.

## Five traps

1. **Legacy `weight_norm` only materialises `.weight` in a forward pre-hook.** Comparing a
   converted weight against `module.weight` before running a forward pass reports all 89
   convolutions as mismatched, against an uninitialised tensor.
2. **`torch.cumsum` accumulates float input in double** and stores float — verified
   bit-for-bit. A plain f32 running sum drifts by 4e-5, which is invisible until it is
   multiplied by `2*pi*300` and handed to a sine. This one cost a 7x error.
3. **The `InstanceNorm1d` affine parameters do not exist in the checkpoint.** They are
   declared `affine=True` upstream — a comment there blames an old ONNX exporter — but the
   model is loaded with `strict=False`, so they stay at their initialisation and the affine
   is an identity. Requiring them fails to load; inventing them changes every channel's gain.
4. **The noise draw is `[time, channel]` while the phase is channel-major.** Reading the
   noise channel-major gives an excitation that is statistically identical and
   sample-for-sample wrong.
5. **`uv = f0 > 10` is a threshold**, so the 7e-4 difference in this port's F0 flips it for
   any frame sitting on the line. The excitation row is therefore driven by the reference's
   F0: that belongs to the predictor's tolerance, not the source module's.

## Where the time goes

RTF **0.075** on Metal for a 5.5-second utterance (0.42 s wall for 5.55 s audio),
and **0.113** across a 30-minute, 148-chunk blog render with no drift from first
chunk to last. (The blog render predates the fusions below; the engine has only
gotten faster since.) Twice as fast as qwen3tts batched, which is where an 82M
non-autoregressive model should be.

```
source 0.015  net 0.257  istft 0.002  (decoder 0.35 s, 71% of wall)
```

Four changes from the 0.241 above, all gate-covered:

- **The upsamples were the half nobody timed separately.** Candle's Metal
  `conv_transpose1d` takes its col2im fast route only at `padding == 0`; Kokoro's
  two ups use padding 5 and 3 and fell into the direct kernel at ~0.3 s per call.
  `tts_nn::upconv` decomposes the stride-s transpose into s polyphase forward
  convolutions, each a tap loop of dense GEMMs with no im2col and no
  zero-stuffed intermediate: 0.314 s to 0.005 s and 0.298 s to 0.012 s.
  Close rather than bit-identical (taps accumulate per phase, contributions per
  output), checked against candle's own transpose at rel < 1e-5 on CPU and
  Metal; the audio gate sits at SNR 25.2 dB either way.
- **The snake is the fused one.** `snake_full`'s six dispatches are `fused::snake_beta`
  with the same arithmetic in one kernel. Exact on CPU (the fallback is the same
  composition), kernel-checked on Metal.
- **The STFT pair is two dispatches.** The host 20-point transforms cost tens of
  milliseconds per utterance in trig calls alone, and the twiddle math is one
  thread per output with tabled exponentials. `tts_nn::stft` runs the forward
  and the overlap-add on the device (the gather pattern needs no atomics and no
  envelope array); off Metal the same tables drive plain CPU loops, which is
  what the gate checks. `source` 0.031 s to 0.015 s — the remainder is the
  f64 excitation, which stays host-side — and `istft` 0.026 s to 0.001 s.
- **Centred convs gather centred.** The im2col kernel folds the left pad; the
  centred entry point folds both edges, so Kokoro's symmetric convs lost their
  right-pad copy in and their re-centring slice out. The causal contract is
  untouched — its upper bound provably never fires — and its exact-equality
  test still passes. ~12% off the resblocks.
- **AdaIN is two passes, not ten.** Measuring the boring ops showed `broadcast_*`
  against `[1, C, 1]` running ~5x slower than plain ops (0.2 ms against 0.04 ms
  per 2.3 MB pass) — and one AdaIN held four broadcasts plus a division.
  `fused::sub_sqr` takes the centred square the variance needs; `fused::adain_apply`
  takes normalise, scale and shift with direct per-channel indexing and an `rsqrt`.
  The two reductions stay candle's (its `mean` is already at full bandwidth).
  Decoder 0.51 s to 0.38 s at rel < 1e-6 against the composed forms.

Measured, not assumed:

- **The host-side transforms are not the problem.** The 20-point STFT and iSTFT together are
  33 ms of 740. A direct DFT at that width costs less than the dispatch to move it.
- **Nor is it allocation.** Three in-process repeats take 0.587 s,
  0.578 s, 0.583 s — so unlike CosyVoice's vocoder, this is not candle's buffer pool being paid
  cold.
- **Nor the LSTMs.** Six bidirectional LSTMs total 5% of the run.
- **Nor the GEMM shapes.** The old note blamed `[128, 384] x [384, 15481]` at
  80 GFLOP/s and pointed at `tts_nn::skinny`. Isolated at exactly those shapes,
  candle's GEMM runs at 1.3-3.0 TFLOP/s, and a skinny-tiled f16 route measures
  0.35x-0.79x of it (1.35x only at the `[24, 896] x [896, 24000]` conv_post).
  The GEMM was never the bottleneck; the transpose kernel was.
- **What remains is flat.** No single op class holds more than ~15% of the
  decoder: centred convs with their pad/gather/GEMM/bias tail, the two fused
  AdaIN halves, the fused snakes, the polyphase ups. The named micro-wins are
  a centred gather (drop the right-pad copy and the narrowing copy, ~2.5%) and
  folding conv biases into the following AdaIN's beta (~1.5%). The structural
  win is cross-chunk batching — chunks are independent and the decoder has no
  autoregression — but every custom kernel is single-batch today, so that is a
  project, not a patch.

## nano

`--engine nano` (`src/standalone.rs`, `src/nano.rs`, `src/nano_engine.rs`) is
[nano-edge-tts-standalone](https://huggingface.co/drmhse/nano-edge-tts-standalone): 3.27M parameters
distilled from Kokoro-82M, read from `references/nano/weights/`. It keeps Kokoro's data and none of its
networks: the same front end, phoneme vocabulary and voice tables, so it speaks the same 28 voices with the
same lexicon. It replaced an earlier option, `--set decoder=nano`, that swapped only Kokoro's decoder for a
1.75M one and kept Kokoro's 28.5M text side.

- **Text side (1.52M).** Phonemes plus the spaCy part-of-speech tag of the word each came from (the
  front end computes the tags anyway; `G2P::phonemize_tagged` returns them over the phonemes' bytes). Two
  conv blocks, one attention block run four times over words and twice over phonemes, then per-phoneme
  durations, features, and pitch and energy drawn as lines with a small frame network on top.
- **Decoder (1.75M).** A mapper (80 frames/s) and a renderer (160 frames/s) set per-bin gains on a pulse
  train's 25 ms STFT and white noise's 5 ms STFT. Nothing learned runs at audio rate. The harmonic gains
  above 2 kHz are averaged over five bins, and a post-filter deepens the noise's formants above 3 kHz; both
  need whole rows and columns of the head, so the tail runs on the host on every device.

The gate (`nano-validate`, fixtures from `references/nano/dump_fixtures.py` with the draws replayed):
the front end's ids and tags are identical to misaki's on ten segments of awkward text in both accents;
durations are identical, features read 1e-6 relative, mapper 7e-7, head 8e-4; audio from the reference's
text side reads 64.6 dB SNR, and text to audio 42.4 dB on the CPU and 43.1 on Metal. Two things differ from
the PyTorch release by design. `KPipeline` falls back to espeak (GPL) when it is installed, which reads
`9:30` differently; the fixtures are taken without it. And the creak's jitter comes from a fixed torch seed
there, which is not reproduced; a render draws its own from `--seed`.

It runs on the CPU on every machine. `examples/chapter.txt` on an M4, one session, builds interleaved:

| | voice | wall | RTF | peak footprint |
|---|---|---:|---:|---:|
| `nano`, CPU | `af_heart` | 3.5–3.9 s | 0.005 | 418–464 MB |
| `nano`, CPU | `am_michael` | 3.6–4.1 s | 0.005 | 453–502 MB |
| `nano`, Metal | `af_heart` | 6.0 s | 0.008 | |
| `kokoro`, Metal | `af_heart` | 36.0–38.5 s | 0.049–0.052 | 1.95–2.01 GB |

The session was a slow one (Kokoro took 27 s in an earlier one, and the build from before this change read
the same 36–37 s with byte-identical audio), so the ratio is the number to take: ten times the speed on a
quarter of the memory. On Metal each of the small kernels costs a dispatch, and that loses.

### The CPU path (Mac, iPhone, iPad)

On the CPU, candle's own `gemm` crate was 70% of a one-thread render. `tts-nn` turns on candle's
`accelerate` feature for every Apple target, so CPU matmuls go to Accelerate's sgemm (the iOS app already
links the framework). What was left was scalar libm calls and allocating broadcasts, now one-pass kernels
in `tts_nn::cpu` and `fused`:

- nano's decoder: GELU (libm `erff`, half the trunk), the depthwise conv (14 broadcast ops), norm + voice gain and
  bias (8 ops), the scaled residual, and the renderer's head applied on the host straight into the
  inverse STFTs' rows. The creak's brick-wall uses a half-length complex FFT each way (real input).
- Text side: ALBERT's softmax (a scalar `expf` per score), and the LSTM gate op, which was a quarter of
  the render on libm's scalar `exp`/`tanh` and now evaluates them in double precision through a
  vectorisable `exp64`, rounded once (within 0.5 ulp).
- With more than one CPU thread, the nano mapper runs beside its signal preparation and each BiLSTM's
  two directions run side by side. `RAYON_NUM_THREADS=1` keeps everything serial; the gate reads the
  same either way.

The earlier `dream-tts speak --engine kokoro --set decoder=nano` (Kokoro's text side with the 1.75M decoder,
since replaced by `nano`), CPU-only build, five texts (55.4 s of audio), text
in and audio out, quiet M4, three rounds; PyTorch is the released `nano_edge_tts.py` through `KPipeline`
on the same texts, measured between two Rust runs that agree:

| | 1 thread RTF | all cores RTF | peak memory |
|---|---:|---:|---:|
| before | 0.0664 | 0.0404 | 359 MB |
| after | **0.0115-0.0121** | **0.0105** | 338 MB |
| PyTorch | 0.0135 | 0.0128 | 1,267 MB |

Kokoro's own decoder, the same 12.2 s sentence (`kokoro-render --cpu`, builds alternated):

| | 1 thread RTF | all cores RTF |
|---|---:|---:|
| before Accelerate | 0.717 | |
| Accelerate | 0.128-0.141 | 0.123-0.131 |
| now | **0.084-0.088** | **0.064-0.072** |

Its convolutions are cache-sized im2col tiles (512 columns), each multiplied by Accelerate straight into
its slice of the output, read with the weight in place (a transposed GEMM, where `contiguous()` had been
copying every weight on every call); AdaIN + SnakeBeta is one pass with vForce's `vvsinf`. With threads
to spare, tiles, channel rows and statistics split across up to four, and the audio is bit-identical to
one thread. The GEMMs now run near Accelerate's one-thread ceiling on these shapes (1.75 TFLOP/s when the
operands fit in cache, 0.88 at 4096 columns).

The channel statistics AdaIN normalises with were one f32 accumulator over tens of thousands of samples.
Against statistics in double precision that build's audio read 34 dB; eight f32 lanes 42 dB; f64 lanes,
now, exact, at no measurable cost. F0_proj fell from 1.85e-3 to 3.3e-4 against its 2e-3 tolerance: most
of that row's error was the statistics, not the arithmetic around them.

What did not help:

- A whole-sequence CPU `lstm_seq` with hand-vectorised recurrent dot products: no faster in cycles than
  Accelerate's per-step sgemv. Removed.
- Cephes `tanh` in f32 (1.1 ulp): in ALBERT's tanh-GELU it moved F0_proj to 2.14e-3, in the LSTM gates
  to 2.01e-3, both over the 2e-3 tolerance. candle's Accelerate build already runs that GELU on vForce.
- Slicing the LSTM gate rows up front to drop bounds checks: no faster, and summing the two gate inputs
  in f64 instead of f32 moved F0_proj to 2.08e-3.
- A conv as `k` GEMMs, one per tap, from a single padded copy over the whole sequence: 1.47 s against
  1.09 for whole-sequence im2col. Each tap re-reads and re-writes the whole output. Within a 512-column
  tile, where the output slice stays in cache: 0.95 against 0.91.
- An inline Cephes `sin^2` fused into the Snake loop instead of `vvsinf`: 0.96-1.00 s against 0.92, and
  less accurate.

## What is not done

- The `Engine` trait and the registry: this is a library and two binaries, not yet an
  `--engine kokoro`.
- The Metal path is not gate-covered. The gate runs on CPU, where the GEMM route falls back
  to candle's `conv1d`; the two agree on this machine but nothing checks that.
- Text longer than 510 tokens has to be split before it reaches the model.
