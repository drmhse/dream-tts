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

## What is not done

- The `Engine` trait and the registry: this is a library and two binaries, not yet an
  `--engine kokoro`.
- The Metal path is not gate-covered. The gate runs on CPU, where the GEMM route falls back
  to candle's `conv1d`; the two agree on this machine but nothing checks that.
- Text longer than 510 tokens has to be split before it reaches the model.
