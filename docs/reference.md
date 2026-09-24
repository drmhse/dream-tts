# dream-tts reference

Everything beyond the README: setup, architecture, how it validates, how fast it is, the
traps each port hit, and what did not work.

- [Setup](#setup)
- [Architecture](#architecture)
- [Validation](#validation)
- [Performance](#performance)
- [Porting traps](#porting-traps)
- [Serving and narration](#serving-and-narration)
- [What did not work](#what-did-not-work)

---

## Setup

Two entry points. Neither needs anything the machine does not already have except `curl`.

```sh
# No toolchain, no clone: unpack a published release, then download the model.
curl -fsSL https://raw.githubusercontent.com/drmhse/dream-tts/main/install.sh | sh
cd dream-tts && ./scripts/bootstrap.sh
```

```sh
# From a clone.
./scripts/bootstrap.sh                       # qwen3tts, the default, ~4.3 GB
./scripts/bootstrap.sh --all                 # all four engines, ~14 GB
./scripts/bootstrap.sh --list                # the ids, their models, what each costs
./scripts/bootstrap.sh audio8 cosyvoice      # two of them
./scripts/bootstrap.sh kokoro                # the cheap one, ~0.7 GB
./scripts/bootstrap.sh --force audio8        # redo a conversion that already ran
./scripts/bootstrap.sh --prebuilt            # download the binaries rather than build them
```

That is the whole setup. It resolves how to get binaries, downloads and converts the
checkpoints you asked for, fetches the fixtures, and either builds or downloads; every step
is skipped if its output already exists, so re-running costs nothing. The rest of this
section is what it does on your behalf, for when a step fails or you want to do one by hand.

**Why the default is one engine.** `qwen3tts` alone is the only configuration whose setup is
downloads and nothing else — no conversion, so no Python, so no torch venv. `audio8` folds a
`weight_norm` pickle, `cosyvoice` re-serialises a `torch.load` and `kokoro` folds 89 weight
norms, and all three want python >= 3.10. Naming one is what buys that cost. The default used
to be all of them, which meant every first run paid ~13 GB and a 2.5 GB venv to get an engine
that is also the slowest on book-length text.

**Building versus downloading.** `bootstrap.sh` builds when there is a toolchain *and*
sources, and downloads `scripts/fetch-prebuilt.sh`'s archive otherwise; `--build` and
`--prebuilt` force either. `cargo build` rebuilds when the source moved and a downloaded
binary cannot, which is why building wins the tie. For the same reason `fetch-prebuilt.sh`
refuses to install into a git checkout that is dirty or not exactly on the release tag: a
fixture gate run by a binary built from other source is a gate that lies about the code you
are reading. It resolves the release from the checkout's own version, never from "latest".

`./dream-tts` and `./dream-tts-serve` are shims over both paths — `cargo run` when there is
a toolchain and sources, `bin/<name>` when there is not. `DREAM_TTS_PREBUILT=1` forces the
shipped binary; `DREAM_TTS_BIN_DIR` points at one somewhere else. Both resolve their own
symlinks, so a link on PATH works. See [scripts/run-bin.sh](../scripts/run-bin.sh).

Voice assets are already in the repo (`voices/`, ~450 KB) and in the release archive, so
cloning a voice needs no PyTorch — only building a *new* one does. **No engine needs its
upstream repository**: the two CosyVoice artifacts that cannot be derived without one are
fetched as assets.

| level | you get | needs |
|---|---|---|
| 0. Qwen3-TTS | `dream-tts speak`, the default engine | curl, ~4.3 GB |
| 1. Audio8 | `dream-tts speak --engine audio8` | python >= 3.10, ~4 GB |
| 2. CosyVoice | that engine | python >= 3.10, ~4 GB |
| 3. Regenerate fixtures | the gates rebuilt from source | upstream CosyVoice repo, python 3.10 |

Levels 1 and 2 need Rust only if you are building rather than downloading.

Everything assumes macOS on Apple silicon; prebuilt binaries are published for
`aarch64-apple-darwin` alone and target macOS 13 or newer.
`cargo build --no-default-features` drops the Metal kernels for CPU fallbacks and is the
only configuration Linux can build, since candle's `metal` feature does not exist off Apple
platforms — a portability guarantee, not something a release ships.

### 1. Audio8

```sh
./scripts/bootstrap.sh
```

Checks the toolchain, downloads [Audio8-TTS-Preview-0.6b](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b)
(~2.4 GB), creates `references/audio8/.venv`, folds the codec's `weight_norm` into
`codec.safetensors`, fetches the derived assets, and builds. Idempotent.

The one step that is not a download is folding the codec: Audio8 ships `codec.pth`, a 1.35 GB
pickle where every convolution is wrapped in `weight_norm`, so stored parameters are a
magnitude and a direction rather than a weight. Folding once means the Rust side memory-maps
plain weights and does no reparametrisation at runtime.

### 2. The other three checkpoints

**CosyVoice.** Two artifacts it needs are not in the checkpoint and cannot be produced without
the upstream python package: `rand_noise.safetensors` (the CFM decoder builds it once under
`set_all_random_seed(0)` and slices the same tensor every call) and a consolidated
`tokenizer.json` (`CosyVoice3Tokenizer.__init__` registers ~250 special tokens that
`AutoTokenizer.from_pretrained` alone does not). `scripts/fetch-assets.sh` fetches both, so
**running CosyVoice needs no upstream repo** — only its checkpoint and a plain `torch.load`.

```sh
references/audio8/.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B', local_dir='/tmp/Fun-CosyVoice3-0.5B')"

references/audio8/.venv/bin/python references/cosyvoice/convert.py \
    --checkpoints /tmp/Fun-CosyVoice3-0.5B --out references/cosyvoice/weights
```

It prints a tensor inventory — the spec the Rust port was written against. If it does not
match `fixtures/cosyvoice/oracle.json` you have a different model revision and the gates will
fail for a good reason. It also prints `[skip] tokenizer` unless the `cosyvoice` package is
importable; that is expected, since `fetch-assets.sh` already placed the file.

**Qwen3-TTS.** Pip-installable upstream, so no repo and no `PYTHONPATH`:

```sh
mkdir -p references/qwen3tts/weights/speech_tokenizer
B=https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/resolve/main
for f in config.json generation_config.json preprocessor_config.json \
         tokenizer_config.json vocab.json merges.txt model.safetensors; do
  curl -sSL -C - -o "references/qwen3tts/weights/$f" "$B/$f"
done
for f in config.json configuration.json preprocessor_config.json model.safetensors; do
  curl -sSL -C - -o "references/qwen3tts/weights/speech_tokenizer/$f" "$B/speech_tokenizer/$f"
done
```

There is no `tokenizer.json` in this checkpoint, unlike CosyVoice's — the BPE is built from
`vocab.json` plus `merges.txt` at load. Ten languages only (en, de, es, zh, ja, fr, ko, ru, it,
pt). `f16` is its default because it is the only format that batches, and batching is what
this engine is for; `f32` measured 38× slower on a 16 GB machine and `q8_0` gives up 4.5× on a
chapter to save 0.58 GB.

**Kokoro.** The checkpoint is curl and the conversion is torch; the *frontend* is neither,
which is why it is an asset. Exporting it imports spaCy and misaki — a pinned torch of their
own — and produces 18 MB of tables that never change: three tokenizer regexes with 1347
exceptions, the POS tagger as safetensors, and the two lexicons. `fetch-assets.sh` places
them under `references/kokoro/weights/frontend`, and `references/kokoro/export_frontend.py`
regenerates them if you are auditing rather than using.

```sh
scripts/fetch-weights.sh references/kokoro/weights \
    https://huggingface.co/hexgrad/Kokoro-82M/resolve/main \
    config.json kokoro-v1_0.pth voices/af_heart.pt          # …and the other 27
references/audio8/.venv/bin/python references/kokoro/convert.py
```

`convert.py` prints `459 tensors, 81.7M parameters, 89 weight norms folded` and packs every
`voices/*.pt` it finds into one `voices.safetensors`. Only the 28 `a*`/`b*` voicepacks are
fetched: the other 26 are Spanish, French, Hindi, Italian, Japanese, Portuguese and Chinese,
and this frontend is an English lexicon, so they have no faithful path through it.

### 3. Regenerating the fixtures

`fetch-assets.sh` pulls ~130 MB of checksummed ground truth from
[`drmhse/tts-rs-assets`](https://huggingface.co/datasets/drmhse/tts-rs-assets), which is what
lets `./scripts/gates.sh` run with no PyTorch at all. Rebuilding from source is the stronger
move if you are auditing the ports rather than using them — a gate that verifies against
tensors somebody else uploaded is only as trustworthy as the upload.

```sh
cd references/audio8 && .venv/bin/python dump_fixtures.py --weights weights --out ../../fixtures/audio8

cd /path/to/CosyVoice            # needs the upstream repo on PYTHONPATH
PYTHONPATH=.:third_party/Matcha-TTS .venv/bin/python \
    /path/to/dream-tts/references/cosyvoice/dump_fixtures.py \
    --model-dir pretrained_models/Fun-CosyVoice3-0.5B \
    --voice /path/to/dream-tts/voices/cosy-default-cosyvoice \
    --out /path/to/dream-tts/fixtures/cosyvoice

references/qwen3tts/.venv/bin/python references/qwen3tts/dump_fixtures.py \
    --model references/qwen3tts/weights --voice voices/cosy-default-qwen3tts --out fixtures/qwen3tts

cd references/kokoro && .venv/bin/python dump_fixtures.py   # its own venv: kokoro, not torch alone
```

Building a new voice uses the same venvs via each engine's `export_voice.py`. The transcript
matters more than it looks: CosyVoice asserts the prompt text contains `<|endofprompt|>` and
nothing in its frontend adds it.

### Disk budget

| path | size | tracked |
|---|---|---|
| `references/*/weights/` | ~3.4–4.3 GB each, `kokoro` ~0.7 GB | no |
| `fixtures/` | ~130 MB | no — fetched |
| `voices/` | ~200 KB | **yes** |
| `target/` | ~1.8 GB | no |

---

## Architecture

Fifteen crates. Everything shared is `tts-*`; everything engine-specific is named for its
engine and matches the id `--engine` takes.

**Voice assets are the decision that made a second and third engine tractable.** Three of the
four models clone from a reference clip, and in every case turning audio into conditioning
needs machinery the runtime should not carry:

| engine | the clip must become | in-process cost avoided |
|---|---|---|
| `audio8` | `[10, N]` RVQ codes | the codec **encoder** — 126 tensors `convert_codec.py` drops |
| `cosyvoice` | speaker embedding, speech tokens, prompt mel, prompt text tokens | `campplus.onnx` (28 MB) + `speech_tokenizer_v3.onnx` (969 MB), plus an ONNX runtime |
| `qwen3tts` | x-vector, `[T, 16]` RVQ codes, sliced transcript tokens | an ECAPA-TDNN speaker encoder and a Mimi-style RVQ **encoder** |

None of it depends on the text being spoken, so it happens once, offline, in Python, and ships
as a directory of `voice.json` + `voice.safetensors`. `Voice::load` checks the `engine` field
and a mismatch is a hard error rather than a silent substitution.

`kokoro` is the exception and it is `Cloning::None`: its voices are 28 `[510, 256]` style
tables inside the checkpoint, and there is no path from a reference clip to one of them.
`--set voice=<name>` picks one, `--voice` is refused rather than ignored, and
`tts_engines::builtin_voices` reads the names out of the asset so a caller can ask what is
installed. The same fork runs through the CLI and the service: neither loads a voice asset
for an engine that cannot use one.

`Capabilities` is deliberately blunt about what engines genuinely differ on — sample rate,
cloning, streaming, and the quantizations each supports. Neither of the first two models can
use K-quants at all, since those need `k` divisible by 256 and both are 896 wide, so listing
"q8_0" generically would be a lie by omission. Where quantization helps also differs: candle
takes a dedicated matrix-vector kernel only when `dim(-2) == 1`, so quantizing a decode loop
is a 3.35× win while quantizing the DiT — which only runs on full sequences — buys much less.
`cosyvoice` therefore quantizes the LLM and leaves the flow decoder and vocoder dense.

### Adding an engine

1. New crate on `tts-core` + `tts-nn`.
2. `capabilities()` returning `available: false` with a reason — **commit that first**, so the
   identifier exists before the implementation.
3. Register in `tts-engines`.
4. **Before any Rust:** convert weights to safetensors, export a voice asset, dump *per-stage*
   fixtures. This is why Audio8's codec validated at 2.8e-6 and its greedy generation came out
   bit-identical first try, and why CosyVoice's reversed RoPE convention localised to one line
   instead of "the audio sounds wrong". A whole-pipeline mismatch says nothing about which of
   three models is at fault.
5. Implement stage by stage, checking each fixture before starting the next.
6. Flip `available`.
7. Measure through `tts_bench::Harness`, then optimize — and **verify each variant is correct
   before believing its timing**. A 1.25× on the CosyVoice DiT turned out to be computing the
   wrong function.

---

## Validation

```sh
./scripts/gates.sh          # everything below, skipping any tier whose inputs are missing
```

| gate | what it proves |
|---|---|
| `audio8-validate` | 8 checks; greedy generation **bit-identical** to the reference |
| `cosyvoice-validate` | 27 checks; teacher-forced argmax **105/105 identical** |
| `qwen3tts-validate` | 65 rows; argmax codebook 0 identical, predictor **15/15** |
| `kokoro-validate` | 10 rows; every deterministic stage, then the excitation and the audio by SNR |
| `check-phonemes.sh` | the English frontend, **byte-identical to misaki** over 522,542 tokens |
| `cargo test --release` | 32 tests including the `tts_engines` doctest |

Gates compare against per-stage fp32 activations dumped from PyTorch, so a failure localises
to a stage rather than to "the audio sounds wrong". A tier whose inputs are absent is
**skipped and reported**, never silently passed.

Sampled output is deliberately *not* gated on equality: `ras_sampling` draws from torch's
generator, so the sampled sequence is not reproducible across implementations. The gates check
prefill logits and a greedy rollout instead, and quality is checked separately by WER.

`kokoro` has no sampler, so its stages are compared directly — but its excitation is not
reproducible at fp32 *upstream either*: the phase accumulates unwrapped to 165,303 radians,
where one f32 ulp is 0.0156 rad, so `sin` of it is uncertain at 1.6% from the representation
alone. The gate's criterion is therefore set by the reference rather than by eye — the port
must sit closer to the reference than the reference sits to itself under a different noise
draw. It does: **25.3 dB SNR and 1.54 dB log-spectral, against upstream's own 19.7/20.7 dB
and 2.18/2.09 dB.** `docs/kokoro-model.md` has the derivation.

---

## Performance

Two fixtures, both tracked so any figure here can be reproduced:

| fixture | | |
|---|---|---|
| `examples/senior.txt` | 132 words, 7 segments, ~50 s of audio | the short-passage case |
| `examples/chapter.txt` | 1612 words, 100 segments, ~11.6 min of audio | the long-form case, where batching engages |

### Short passage

`examples/senior.txt`, M4 / 16 GB. Median of five samples with the cloning engines interleaved in
one session, taken when `qwen3tts`'s default was `q8_0`; at today's `f16` default the same
passage reads **0.314**, still the wrong case for it.

| engine | reference | this port | spread | |
|---|---|---|---|---|
| `audio8` | 1.307 (PyTorch bf16 MPS, batched) | **0.554** | 0.547–0.562 | 2.36× faster |
| `cosyvoice` | 4.370 (stock PyTorch, CPU-only) | **0.726** | 0.697–0.734 | 6.02× faster |
| `qwen3tts` | — | **0.665** | 0.642–0.687 | the wrong case for it; see below |
| `kokoro` | — | **0.036** | 0.035–0.036 | no loop to fill, so this is its normal rate |

### Chapter, and what batching is actually worth

`examples/chapter.txt`, 100 segments. `f16` is the median of three samples.

| engine | weights | lanes | RTF | talker/LLM | codec |
|---|---|---|---|---|---|
| `qwen3tts` | `q8_0` | 24 | 0.661 | 0.588 | 0.072 |
| `qwen3tts` | `f16` | 24 | 0.260 (0.256–0.261) | 0.187 | 0.073 |
| `qwen3tts` | `f16` | 48 | 0.218 | 0.147 | 0.069 |
| `qwen3tts` | `f16` | 48 | 0.186 | 0.116 | 0.069 |
| `qwen3tts` | `f16` | 48 | 0.158 | 0.117 | 0.040 |
| `qwen3tts` | `f16` | 48 | 0.148 | 0.110 | 0.037 |
| `qwen3tts` | **`f16`** | **48, shipped** | **0.104–0.114** | **0.072–0.081** | 0.033–0.034 |

The last row is the current default — `f16` is what `--quant` resolves to with nothing passed —
and [the section below](#the-next-third-prefill-sampling-three-kernels-and-the-load-path) says
what it adds. The row before it adds the fused channels-last conv on top of the talker's finished-tail shedding and the
codec's uniform decode span; `MAX_BATCH` was 24 when the first two rows were taken. The whole
gain in that step is the codec (0.069 → 0.040) and the render is bit-identical. The last row
adds the transposed convs rewritten as one conv and `tts_nn::skinny`, the decode-shaped GEMM;
those two were measured interleaved against their own alternatives in one thermal state, at
2.14x on the transposed convs and 0.162 → 0.148 end to end. On a 4838-word article the shipped
configuration measured **0.144** against 0.193. Peak footprint was 12.7–13.3 GB then and is
9.2 GB now; `README.md` has the memory analysis — it is candle's buffer pool, not the lane
count.

### The next third: prefill, sampling, three kernels and the load path

A render's stage split had been talker and codec, and the batched path never reported the
talker's *prefill*: 25 s of an 88 s talker stage, a fifth of the render. Every change below was
measured on its own before it went in, and the end-to-end rows are the committed build against
this one, interleaved in one session, two seeds each.

| change | measured | output |
|---|---|---|
| The voice's prompt positions (role, tags, reference transcript: 50 of 157) prefilled once and copied to every lane | a third of prefill | gate row `prefill.shared`, rel 7e-7 |
| Prefill attention: GQA as a reshape of q, the new positions' own k/v, one mask per forward | 535 → 191 ms per 8-lane window | unchanged gate |
| Top-50 by selection instead of a full sort of 2048 logits | 30 → 3 µs a row, 150k rows a chapter | identical picks, `talker::tests` |
| The predictor samples on the device, draws taken up front in the host's order; one read per frame instead of fifteen | ~8 s a chapter | byte-identical WAV to host sampling at the same seed |
| Codec convs with a short reduction through MPSGraph (`tts_nn::mpsnlc`) | codec chunk 899 → 770 ms; 96 → 1 output conv 2.72× | bit-identical in f32 |
| Decode GEMM: split-K to fill the GPU, and 24x32 patches per simdgroup loaded straight from device memory | 1.60–2.52 → 2.42–2.83 TFLOP/s; talker step −9%, predictor −16% | rel ≤ 4e-4 against candle, as before |
| Fused decode attention, K and V read once for both query heads | 26.3 → 15.2 ms per talker step at span 250, 91 GB/s of the bus's 120 | rel < 1e-5 against the CPU path |
| QK-norm, rope and the KV write in one dispatch | talker step −8%, predictor −14%, prefill window −15% | rel < 1e-5 against candle's ops |
| SnakeBeta and the residual folded into a residual unit's k=1 conv, up to 384 channels | residual units −9.5%; codec alone 5.29 → 3.0 GB | rel < 1e-5 |
| Weights cast, transposed and concatenated on the host, uploaded once; text embeddings read from the mapping | talker allocation 6.76 → 3.37 GB, peak 12.5 → 9.2 GB | bit-identical weights |

| `examples/chapter.txt` | seed 1 | seed 2 |
|---|---|---|
| committed | RTF 0.151 · 1m 45s · 12.49 GB | 0.147 · 1m 42s · 12.49 GB |
| this build | **0.104 · 1m 12s · 9.22 GB** | **0.114 · 1m 19s · 9.21 GB** |

**Quality**, on the audio because a sampled model takes a different path through the same text
the moment a logit moves: WER 0.008 against the committed build's 0.013 on seed 1, and median F0 and LTAS cosine against the cloned clip
179.8 Hz and 0.9973, against 177.8 Hz and 0.9973 — the clip itself is 179.8 Hz. `references/cosyvoice/wer.py` and `references/audio8/verify_voice.py`.

**The memory finding that made the last row possible.** candle 0.10.2 keeps two buffer maps:
every op's output goes into one it never trims, rounded up to a power of two, and a buffer
uploaded from host data goes into the other, exact and released when dropped. So a cast on the
device leaves its input and output resident for good. Loading the talker and dropping it again
left 6.64 GB allocated; loading it through the host leaves 0.00. Anything computed once at load
belongs on the host.

**Still open.** 64 lanes now fits — 12.4 GB, where it used to thrash at 15.6 — but runs at 0.107
against 48's 0.100, for two reasons worth fixing: `skinny`'s tile is 48 rows, so 64 falls back to
candle's GEMM, and a 64-lane KV layer is 87.8 MB, which the pool rounds to 128 MiB, 2.2 GB of
padding across the cache. The decode GEMMs reach 2.4–2.8 TFLOP/s of a ~3.6 ceiling and attention
76% of the bus; prefill's projections are 2.5 against their GEMMs' 3.3.

**`q8_0` gains nothing from 14× more segments** — 0.665 on seven, 0.661 on a hundred. That is
candle's quantized `mm_t` padding to a large row tile, so batch 8 costs 8× batch 1, measured
end to end rather than inferred from op timings. `f16` gets **2.54×**, and all of it is in the
talker (0.588 → 0.187); the codec is unchanged at 0.072, as it must be, since the weight format
being varied is the talker's.

This is why the narration path uses `f16` and why `qwen3tts` is the default for a book despite
being the slowest of the cloning engines on a short passage. Reproduce with:

```sh
./dream-tts speak --engine qwen3tts \
    --voice voices/cosy-default-qwen3tts --quant f16 \
    --text-file examples/chapter.txt --out chapter.wav
```

An RTF of **0.253** was previously quoted for this configuration with no fixture behind it and
no end-to-end render supporting it. The measurement above is what replaced it; the claim turned
out to be close to right, which is luck rather than evidence.

### Chapter, the other three engines

Same fixture, same machine, one voice each. `kokoro` is here rather than above because it is
the only engine whose short-passage and long-form numbers are the same figure.

| engine | voice | RTF | wall | audio | peak footprint |
|---|---|---|---|---|---|
| `kokoro` | `af_heart` | **0.036** | 26.9 s | 12:15 | 2.01 GB |
| `kokoro` | `am_michael` | **0.034** | 27.8 s | 13:32 | 2.33 GB |
| `audio8` | cloned female / male | 0.536 / 0.527 | 6m 12s / 5m 47s | 11:34 / 10:59 | — |
| `cosyvoice` | cloned female / male | 0.718 / 0.703 | 9m 12s / 8m 15s | 12:48 / 11:44 | — |

The stage split under `kokoro` is flat across both: decoder 78–80%, prosody 7.6%, bert 6.4%,
duration 4.3%, encoder 1.5%. A segment is one forward pass, so there is no batch to fill and
nothing that rewards length — 0.036 on seven segments against 0.036 on a hundred is the whole
story, and it is why the engine is worth having on a machine that cannot hold `qwen3tts`.
Peak footprint is 1.57 GB on the short passage, against 6.7 for the default. Those are the build
after the [third pass](#kokoro-third-pass-one-graph-per-resblock-and-albert-s-small-passes),
with the GPU otherwise idle; after the second it was 0.038 and 28.5 s, and before it 0.043.

### Word error rate, all eight demo renders

Whisper `small.en` through `faster-whisper` (batched, int8), the same normalisation for every
file: `references/cosyvoice/wer.py --text-file examples/chapter.txt <files>`. The reference is
1612 words.

| engine | female / first voice | male / second voice |
|---|---|---|
| `kokoro` | **0.009** (15 errors) | 0.042 (68) |
| `qwen3tts` | 0.011 (18) | 0.050 (80) |
| `audio8` | 0.012 (19) | 0.011 (17) |
| `cosyvoice` | 0.019 (31) | 0.017 (28) |

Read this as renders compared against each other, not as a publishable WER: the normalisation
is blunt and the transcriber is the same for every row, which is the only property that
matters here. Two things it does say. The spread *within* an engine is larger than the spread
between engines — `qwen3tts` covers 0.011 to 0.050 across two voices of the same model — so a
voice is at least as much of the quality story as an engine is. And the six cloned files were
rendered at an earlier revision than the two `kokoro` ones; the text and the measurement are
identical, the engine revisions are not.

### Where the time goes

On `examples/senior.txt` at `q8_0`, against the same split when each port was first measured.

| | stage | RTF | share | previously |
|---|---|---|---|---|
| `audio8` | AR loop (batched) | 0.340 | 61% | 0.341 |
| | codec | **0.214** | 39% | **0.158** |
| `cosyvoice` | LLM (batched) | 0.183 | 26% | 0.187 |
| | flow decoder | 0.477 | 67% | 0.474 |
| | vocoder | **0.049** | 7% | **0.037** |
| `qwen3tts` | talker | 0.576 | 89% | 0.695 |
| | codec | 0.068 | 11% | 0.168 |

**Open: the two convolution stages regressed and the transformer stages did not.** AR, LLM and
flow all land within 1% of their original measurements; the codec is 35% slower and the vocoder
32%. The suspect is the channels-last conv path added in `tts-nn: decode attention,
channels-last convs, and dense f16 projections`, which was never benchmarked end to end. This
is why `audio8` moved from 0.499 to 0.554. `tts-probe`'s `convgemm` and `upsconv` would isolate
it.

One caveat on the tables. CosyVoice's reference is CPU-only because upstream's
`CosyVoiceModel` hardcodes `cuda if available else cpu` and has no MPS path at all; the adapted
MPS service reaches ~0.76, which this beats by 5% rather than by 6×. And model load is
1.0–1.5 s for Audio8 including quantizing 417 M params, against 15–17 s for the PyTorch
service's reload path.

### Kokoro's decoder, and what is left in it

A 10 s render on an M4 (canary 60 ms, so a cool machine) went **0.075 -> 0.038 RTF**. The
decoder was 86% of it at the start and is 78% now. That figure is one utterance through
`kokoro-render`; through the CLI a chapter measures 0.043, and the difference is the frontend
and the per-segment launches, not the decoder.

| | before | after |
|---|---|---|
| bert / duration / prosody / encoder | 110 ms | 87 ms |
| stage 0 (256 ch @ 8040) | 126 ms | 88 ms |
| stage 1 (128 ch @ 48240) | 323 ms | 165 ms |
| pre-generator, excitation, iSTFT | 96 ms | 48 ms |
| **total render** | **755 ms** | **385 ms** |

Five changes, each measured with `tts-probe`'s `kokorogen` or `mpsconv` before being wired in.

**1. MPSGraph for the generator's convolutions — the big one, worth 1.29x.** See
`tts-nn`'s `mpsconv`. candle does not use MPSGraph at all, and MPSGraph's `convolution2D`
does not build an im2col matrix: 1.7x on the shape that dominates. The `tapconv` note below
is what pointed at it.

**2. The centred conv stopped chunking (1.18x).** `TTS_GEMM_COL_BUDGET` was 64 M elements
and stage 1's six `k=11` convolutions needed 67.9 M — they missed by 578 columns and split
in two, and a split costs far more than its copies suggest: 15.3 ms against 10.3 ms whole.

**3. `moments` and `adain_snake` (1.07x).** An AdaIN was `mean` + `sub_sqr` + `mean` + apply
— three passes over the signal and a full-size intermediate for what one read can compute —
and a SnakeBeta always follows it. At `128ch @ 48240`: 1.57 ms -> 0.54 and 1.61 -> 0.78,
36 times per render.

**4. The excitation overlaps the decoder (1.07x).** It is host-side and depends on nothing
in the decode blocks, so it runs on another thread while they keep the GPU busy. It was
26 ms of an idle device.

**5. `lstm_gates` (1.08x).** Eleven dispatches per timestep became two, over ~3000 timesteps.
Read the trap below before touching it.

**The trap: a recurrence multiplies its kernel's rounding.** Metal compiles with fast math
on by default. With the fast `exp` and `tanh`, `lstm_gates` matched the composed form to
1e-7 over one step and had diverged to **4e-1 after sixty** — which moved the predicted
durations and turned a 10.05 s render into 4.17 s. Every fixture row still passed: the
fixtures are 50 phonemes, 52 timesteps, and at that length the gates saturate and the two
paths agree bit-for-bit. Comparing against candle's composed form does not catch it either,
because both are equally wrong. What catches it is checking the kernel against a
double-precision host reference across the input range, to a couple of ulp —
`lstm_gates_is_accurate_to_double_precision`. Any future recurrent kernel needs the same.

**What did not work, so it is not worth retrying.** f16 GEMM is 1.10x at these shapes.
Batching the three MRF kernels into one wider GEMM is 0.35x — exactly 3x the work, because
the block is diagonal and `M = 128` is already wide enough. Low-rank factorisation of the
generator's convolutions saves nothing: rank at 99% of the energy is 95% of full rank, so
those weights are dense. And there is no per-utterance fixed cost to amortise by batching
sentences — RTF is flat from 1.3 s of audio upward.

**Writing the convolution by hand loses to MPS, and that is what pointed at MPSGraph.**
The gather was a third of the generator's convolution time and every byte of it was written
only to be read straight back, so a kernel staging the input tile in threadgroup memory
should have won. Two of them, in `tts-nn`'s `tapconv` and timed by `kokorogen`: simdgroup
matrices reached 0.73 TFLOP/s and a classical 128x128 register tile 0.40, against MPS's
~3.0 on the same GEMM and an M4's ~4.3 peak. Widening the K block to cut barriers made the
first *worse* — the staging buffer costs more occupancy than the barriers cost time — and
the register tile spills 64 accumulators per thread. M3/M4 have no matrix unit in the GPU,
so `simdgroup_multiply_accumulate` is a scheduled ALU sequence and Apple schedules it
better. The right conclusion was not "tune harder" but "use more of Apple's code", which is
change 1.

**What was left then**, and what became of it in the next section: the transposed
convolutions (now one stacked conv each), the MPSGraph syncs (still there — candle does not
expose its command buffer), and `bert` (still 30 ms; neither fused QKV nor fused attention moved
it).

### Kokoro, second pass: the per-length compile, the recurrences, the load path

The first pass was measured the way it was built: one utterance rendered four times. A chapter
is a hundred utterances of a hundred lengths, and that hid the largest cost left. Interleaved
against the committed build in one session, M4 / 16 GB, with a browser open (so both columns
read high against the quiet-machine 0.043 above; the ratio is the finding):

| | before | after |
|---|---|---|
| `examples/chapter.txt`, three rounds | RTF 0.052 / 0.054 / 0.058 | **0.038 / 0.040 / 0.043** |
| `examples/senior.txt`, three rounds | 0.057 / 0.058 / 0.052 | **0.039 / 0.038 / 0.038** |
| one sentence, fresh process: synthesis / wall | 0.4 s / 0.61 s | **0.2 s / 0.51 s** |
| load | 0.29 s | **0.12 s** |
| Metal allocated after load | 0.65 GB | **0.38 GB** |
| peak footprint, short passage / chapter | 1.36 / 1.93 GB | 1.27 / 1.87 GB |

Quality: the chapter renders to the same length, 49 dB apart. Whisper (`--backend openai`)
reads 14 errors in both, WER 0.009; median F0 198.3 Hz in both; LTAS cosine 1.00000. The
batched `faster` backend read the committed build's file as 0.047 by dropping a 62-word span;
the audio does not differ there. With `TTS_NN_LSTM_SEQ=0` the two builds are 89 dB apart,
which is 16-bit quantisation: the LSTM's summation order is the only numeric change, and it is
the more accurate of the two (below). The fixture gate is 10 rows, 0 failures.

| change | what it was | effect |
|---|---|---|
| **Length buckets + executables compiled ahead** | MPSGraph specialises a graph for every input length, ~4 ms a graph, on the caller's thread with the GPU idle; the generator runs ~26 graphs, so ~100 ms a segment and a third of the decoder on a chapter. The generator now pads its input to one of eight lengths per octave, and `mpsconv::prewarm` compiles the executables that length needs on four threads as soon as the durations are known, behind the prosody and decoder blocks | the bulk of the chapter gain |
| **Exact padding** | `adain_snake_masked` takes moments over the real samples only and writes zeros past them, `leaky_masked` likewise, so every conv reads the zeros its own padding would have. 86.6 dB against the unpadded path — 16-bit quantisation | ~3% of padded work |
| **`lstm_seq`** | one encoder of per-step dispatches, both directions per dispatch, a simdgroup per hidden unit summing H/32 terms of all four gates; the step form was a gemv, a gates kernel and copies per direction | 30 → 7 µs a step; duration 22 → 8 ms and prosody 33 → 24 on an 11 s utterance. 1.9e-7 max error against an f64 recurrence over 1200 steps, where the step form is 4.4e-7 |
| **Transposed convs as one conv** | the `s` phases share an input and differ in which of three offsets they read, so they stack into one `[kw, in, s * out]` conv and an interleave, where the tap loop was ~20 matmuls, copies and adds | 36 → 17 ms a segment; MPSGraph for `ups.1`, the GEMM for `ups.0`, whose input is short |
| **Residual inside the conv graph** | `convs2`'s output plus the block input, added in MPSGraph rather than as a pass | ~2% of the decoder |
| **One weight layout per conv** | `[k, in, out]` is MPSGraph's HWIO, and transposed it is the gather GEMM's `[out, k * in]`; every conv held both candle's layout and the tap-major one, each rearranged on the device into candle's pool | load allocation 0.65 → 0.38 GB |
| **Host rearranging, done fast** | the rearranging moved to the host, where candle's strided copy of a transposed view doubled load time; a tiled transpose on threads (`host_layout`, shared with every engine) and a tap-plane permute fixed it | load 0.29 → 0.12 s |
| **The frontend a segment ahead** | G2P ran twice per segment, on the synthesis thread; it runs once, on its own thread | ~3% of a chapter |
| **`leaky_relu` in one pass** | relu, sub, mul, add | bit-identical |

**Next:** the SnakeResBlock as one MPSGraph, done in the next section.

### Kokoro, third pass: one graph per resblock, and ALBERT's small passes

Measured interleaved against 96565e2, M4 / 16 GB, the GPU otherwise idle (`gpumon` reads 97%
busy through a chapter on both builds, so what was left was work, not waiting):

| | 96565e2 | now |
|---|---|---|
| `examples/chapter.txt`, `af_heart`, three rounds | RTF 0.039 / 0.038 / 0.039 | **0.036 / 0.036 / 0.036** |
| `examples/chapter.txt`, `am_michael` | 0.036 | **0.034** |
| `examples/senior.txt`, two rounds | 0.038 / 0.039 | **0.035 / 0.036** |
| one sentence (4.5 s), fresh process | 0.49 s | **0.45 s** |
| one sentence through a warm server: first / median / best | 0.25 / 0.16 / 0.10 s | **0.21 / 0.15 / 0.10 s** |
| `/v1/batch`, `senior.txt`'s paragraphs | 2.17 s | **1.95 s** |
| peak footprint, short passage / chapter | 1.28 / 1.87 GB | 1.57 / 2.01 GB |

Quality: the fixture gate is 10 rows, 0 failures, with every stage's error unchanged. The chapter
renders to the same length, median F0 198.3 Hz in both, LTAS cosine 1.00000; faster-whisper
`small.en` reads 16 errors against 17, and its differences go both ways (the new render loses
"someone" and gains "invariants"), which is the transcriber flipping on near-identical audio. The
two renders are 45 dB apart. The resblock graph alone is 87 dB apart — 16-bit rounding — and the
rest is the layer norm: a 1e-6 difference in ALBERT's output moves F0 by as little, and the sine
source integrates that into a phase that drifts over a segment without changing how it sounds.

| change | what it was | effect |
|---|---|---|
| **One MPSGraph per SnakeResBlock** (`tts_nn::mpsblock`) | each of the six AdaIN-snake-conv steps was a moments pass, an AdaIN pass and an MPSGraph conv with its commit-and-wait. Now the moments over the real samples, the AdaIN and snake, the masking, the convs, biases and residuals of a block are one executable, prewarmed like the convs were | 1.07-1.13× at a chapter's 48k samples, 1.3-2.0× at a sentence's; eight of these per segment |
| **Graphs in a six-entry LRU** | a graph keeps every run's intermediates, ~90 MB at 128 channels and 48k samples, until it is dropped; with no bound a chapter peaked at 6.65 GB. A segment needs six (the noise blocks share the resblocks' shapes), and three recompiled every segment (RTF 0.053) | +0.14 GB on a chapter's peak |
| **A two-pass layer norm kernel** | `tts_nn::layer_norm` was the eight-pass composed reference, 108 µs on ALBERT's `[60, 768]`. candle's fused kernel takes the variance in one pass, which cancelled on ALBERT's activations and moved F0 past its gate tolerance (2.7e-3 against 2.0e-3). The kernel takes two passes and sums about the row's first value: 3.9e-6 from an f64 reference where the composed passes are 1.4e-5 | ALBERT 14.0 → 11.3 ms at 60 tokens |
| **ALBERT's QKV as one projection** | one GEMM, one bias add and one head split for all three instead of three of each; the split copies and broadcast adds cost more than the GEMMs at a sentence's length | 11.3 → 10.1 ms |
| **Biases broadcast once per forward** | twelve layers share one set of weights, so each bias is broadcast to `[1, t, n]` once and added plainly: a broadcast add is 33 µs at 60 tokens and a plain one 4 | 10.1 → 8.8 ms; 20.4 ms at 190 tokens against 30.7 |
| **The attention scale in the queries** | 1/√64 is a power of two, so folding it into the query weights is exact and drops a pass | — |

**Next:** ALBERT is now GEMM-bound at a sentence's length — candle's f32 matmul runs at 0.3 TFLOP/s
for `[60, 768] × [768, 768]` against 2.2 at 190 rows — and the generator's convs are at
MPSGraph's f32 ceiling. What is left in f32 is small; the larger lever is precision, which this
pass did not spend.

### How to measure without fooling yourself

An M4 under sustained GPU load drifts **~2×**. This was caught by accident: `dilation.rs`
reported 119.57 ms for a conv that `convopt.rs` had measured at 59.78 ms twenty minutes
earlier. Absolute timings taken on a cool machine early in a session are not comparable to
anything taken later.

1. Interleave variants **in the same run**, never A-then-B-much-later.
2. Report **median and spread of ≥5 samples**, not a 3-iteration mean.
3. Include a **fixed canary workload** so the run's thermal state is recorded with the result.
4. Prefer **ratios within a run** over absolute numbers across runs.
5. Idle-cool between runs, or report the drift.

`crates/tts-bench/src/lib.rs` (`Harness`) implements all five, and
`references/audio8/bench_ar.py` mirrors it so Rust and torch runs land on the same thermal
scale. Two further ways to get it wrong, both learned here: an **unsynchronised stage timer**
once misattributed 13% of a pipeline to the wrong stage, and a **warm A/B loop cannot see
first-touch allocation cost**.

### Live GPU telemetry

Two surfaces, both unprivileged. `powermetrics` needs root; these do not.

`cargo run -p tts-probe --bin gpumon` samples the accelerator node's published
`PerformanceStatistics` at 2 Hz (`--hz`, `--count`): device/renderer/tiler
utilization plus allocated and in-use system memory. Run it beside a render to
see whether the GPU stays fed — 0% idle against 97-100% under a Kokoro render
on this M4, back to idle when the process exits. The counters update about once
a second, so this is a duty-cycle check, not a kernel profiler.

`kokoro-render` (and any engine wired to `tts_nn::stats`) prints a `matmul` line
with the render: counted dense-GEMM FLOPs and bytes over wall time. Coverage is
the dense matmuls only — direct convs, attention scores and elementwise passes
are not counted — so the GB/s reads low against the ~120 GB/s bus by
construction. What it answers is whether the GEMM shapes are near this
backend's ~2.4 TFLOP/s: Kokoro's decoder reports ~0.7 TFLOP/s over the whole
synthesis, i.e. the GEMMs are fine and the time is in elementwise passes and
dispatch, which is what the next optimization has to attack.

There is no unprivileged system-DRAM GB/s on macOS — only residency and
utilization are published; byte counters need `powermetrics --samplers
bandwidth` as root. Per-kernel GB/s stays in `tts-probe` microbenchmarks, where
the traffic is exact rather than sampled.

### Memory, and quantization quality

Memory is **not** characterised. An earlier claim that it was "flat by construction" was wrong
— that was an argument, not a measurement. Two attempts to measure it both saturated:
`/usr/bin/time -l` reports RSS, which cannot exceed what is resident, and pinned at 3.59 GB
across a 6× range of input; `phys_footprint_peak` returned exactly 13.00 GB for three
configurations that should differ enormously. What can be said: one process renders 16 minutes
of audio on a 16 GB machine without the system struggling, and two concurrent engines do not
fit. One real find along the way: an unbounded LLM batch at 101 MB per lane, now capped.

`q8_0` costs nothing audible. Token identity is the wrong metric for this — sampled sequences
diverge for reasons unrelated to weight precision, so the question has to be asked of the audio
(WER and speaker similarity), not of the codes.

---

## Porting traps

The traps below cost real time. They are listed because each produced output that was
*plausible but wrong*, which is the worst failure mode a port has.

### Audio8, ranked by cost

1. **RoPE is interleaved, not half-split.** `_apply_rope` reshapes the last dim into adjacent
   `(real, imag)` pairs; candle's default `rotary_emb::rope` uses the half-split convention.
   You need `rope_i`.
2. **RoPE tables are built in bfloat16, then applied in fp32.** Replicate the bf16 round-trip
   on the table rather than computing fresh fp32 sin/cos.
3. **`_fast_step(hidden, 0)`'s result is discarded.** It exists only to prime the fast KV cache
   at position 0. Skip it and codebooks 1..9 are garbage.
4. **The top-k/top-p filter softmaxes *before* temperature** — the opposite of the conventional
   order. Replicate as written.
5. **Residual codebooks are size 1024, not 4096.** The fast head emits 4096-way logits but the
   quantizer clamps rows 1..9 to `0..1023`. Measured: 0 of 216 residual codes exceeded 1023, so
   the clamp is a safety net, not load-bearing — port it anyway, it is two `min` calls.
6. **Sampling is Gumbel-max, not multinomial**: `argmax(softmax(s) / -log(u))`.
7. **RAS draws twice per step**, in that order, substituting the second when the first repeats
   within a 10-token window.
8. **The RAS window initialises to zeros**, so the first 10 steps compare against token id 0
   and never trigger.

Also: **Audio8's reference sampler is broken under its own default dtype.** `_sample` draws
Gumbel noise at `dtype=probabilities.dtype`, so in bfloat16 the uniforms have ~256 distinct
values and output is unintelligible and never reaches EOS. Both implementations here draw in
f32.

### CosyVoice

1. **RoPE reaches head 0 only** in the reference.
2. **The two engines' RoPE conventions are opposite** — HF's Qwen2 uses `rotate_half`.
3. **The flow's initial noise is a fixed tensor, not a draw.** A port that samples its own
   looks correct and sounds different.
4. **`<|endofprompt|>` is required and nothing in the frontend adds it.** The LLM asserts on it.
5. **The upstream safetensors are the wrong weights.** 0 of 290 tensors match the `llm.model.*`
   tensors inside `llm.pt`, max relative difference 1.82. `llm.pt` is the fine-tune and the only
   correct source — checked rather than assumed.
6. **The tokenizer needs ~280 special tokens added at construction.**
7. **The leaky-ReLU before `conv_post` has slope 0.01, not 0.1.**
8. **The NSF noise is not in the checkpoint and is not negligible.** `SineGen2` holds it as a
   plain attribute, not a registered buffer, so it is redrawn at construction and only
   reproducible because the config calls `torch.manual_seed(1986)` first. Zeroing it moves the
   waveform by max 0.164 against a signal of rms 0.078.
9. **The harmonic phase is numerically degenerate in f32.** The reference accumulates phase to
   1.7e7 radians, where one f32 ulp is a full radian. This port accumulates on the host in f64
   modulo one cycle — the one place it is deliberately *more* accurate than its reference, and
   it moves *closer* to the reference's output, not further.

### Qwen3-TTS

Params come from the safetensors headers, so it reads **2.10 B** rather than the 1.7 B in its
name, which counts only the language backbone. The port was written from the reference and the
checkpoint's tensor shapes *before* any model code — the same idea as fixtures-first, applied
one step earlier, to the traps.

---

## Serving and narration

`dream-tts-serve` replaces the Python FastAPI service and speaks the same protocol, verified side by
side on the same request: identical WAV format (`RIFF`, PCM, mono, 16-bit), identical
`X-Audio-Seconds` / `X-Wall-Seconds` / `X-RTF` / `X-Audio-Format` / `Content-Disposition`
headers, identical auth (`Bearer` or `X-API-Key`, constant-time compare, `503` when no key is
configured), and errors as `{"detail": …}`. The audio differs, as it must — different RNG
streams.

| route | |
|---|---|
| `POST /tts`, `POST /tts/stream`, `POST /v1/batch`, `GET /health`, `GET /v1/capabilities`, `GET /` | served |
| `/v1/tts-jobs`, `/v1/alignment-jobs`, `/v1/artifacts/…` | **501 with an explanation**, not 404 |

`mode=instruct`, `mode=cross_lingual`, `speed != 1.0` and `instruct_text` also return 501.
**Refusing rather than ignoring is the rule**: returning speed-1.0 audio to a client that asked
for 1.5 would report the request as honoured when it was not. Two optional additions the Python
schema lacks: a per-request `voice`, and a `seed`. An extra `X-Stages` header carries the
per-stage split so a client can see where time went without a second request.

One GPU, so synthesis is serialised behind a semaphore — two requests interleaving on one Metal
queue make both slower and neither faster — and runs on `spawn_blocking` so it never occupies
an async worker. `/tts/stream` is buffered, not incremental.

**`GET /` has two audiences.** JSON to a client, a self-contained HTML page to a browser,
negotiated on `Accept` and overridable with `?format=json` / `?format=html`. A wildcard
`*/*` — what curl sends — counts as a machine and gets JSON; only a client that names
`text/html` gets the page. The page carries the routes, the request body, the response
headers, the auth rule, and this process's live engine, voice, port and settings file. It
loads nothing from the network: a local service that cannot explain itself offline is broken
in exactly the situation someone is most likely to be reading it. Everything interpolated
into it is a number, an engine-supplied `&'static str`, or an HTML-escaped path.

**`POST /tts` takes `text/plain` as well as JSON.** The body is the text; `X-Seed` and
`X-Voice` carry what the JSON fields would. It exists because `narrate-book.sh` built its
request body with a `python3 -c "import json…"` per chapter — a Python dependency on the
critical path of the one feature that is supposed to run with nothing but curl, and quoting
arbitrary prose into JSON from bash is a bug waiting for an apostrophe. JSON remains the
default for any other content type, including none, so an existing client is unaffected.

With that gone, `narrate-book.sh` needs **ffmpeg and nothing else** outside the binaries;
`--no-align` drops the last optional interpreter. Verified by running a chapter end to end
with `python3` replaced by a stub that exits 127.

`DREAM_TTS_API_KEY` is checked first and `TTS_API_KEY` still works, because an existing
deployment's environment is part of the wire compatibility this claims. The default port
stays `3003` (via `PORT`, then `serve.port`, then that) for the same reason; a port already
in use is answered with the `lsof` line that names the holder rather than a bare
"Address already in use".

### Narrating a long document

```sh
scripts/narrate-book.sh --book path/to/document --out narration
scripts/verify-narration.py narration/*.webm
```

One engine load for the whole run, resumable per *stage* (a section with a WAV master is never
re-synthesised), deterministic under a seed. WebM/Opus at 48 kbps for delivery, because
Safari's Opus-in-Ogg support is unreliable and the failure mode is silence.

**Timings are derived from recognising the audio and matching it to the source text**, not from
placing known words into assumed windows. That shortcut measured a median error of **4.8 s per
word** while reporting 99.4% of words "aligned". Every section reports the share of words
carrying a measured time, the longest run it could not measure, and how many cue boundaries
land on a silence `ffmpeg` detected independently.

Text preparation carries most of the quality — stripping what is not speech and rewriting what
the voice reads wrong. Defects were found by aggregating alignment manifests rather than by
listening at random: a word the voice mangles is never recognised, so it shows up as
interpolated in every occurrence, which turns a 146,000-word book into a short list of suspects
for free.

---

## Settings and storage

Everything in this section exists because the thing stopped being a repository you build and
started being an application someone installs.

### The settings file

`dream-tts.json` beside the install, or `~/.config/dream-tts/config.json`.
[`dream-tts.example.json`](../dream-tts.example.json) is the annotated copy.

| | |
|---|---|
| `engine`, `voice`, `quant` | what to synthesize with |
| `data_dir` | where the 4-13 GB lives |
| `max_chars`, `gaps` | segmentation |
| `gpu_lock` | the advisory lock below |
| `serve.host`, `serve.port`, `serve.max_chars`, `serve.segment_chars` | the service |

Precedence, written down once in `tts_core::config` and honoured by both binaries: **flag >
environment > `dream-tts.json` > user config > built-in default**. Every key is optional.
Keys beginning with `//` are comments, stripped before deserialization; every *other*
unknown key is a hard error, because a misspelled key that silently does nothing is the
worst outcome available — the user believes they configured something.

`dream-tts config` prints what resolved and what decided it. Without that, a config file
being ignored — wrong directory, `DREAM_TTS_CONFIG` set in a shell profile — is
indistinguishable from one whose values happen to match the defaults.

### Two roots, deliberately distinct

**root** is the installation: binaries, `voices/`, `scripts/`. Small, replaced wholesale by
an upgrade. `DREAM_TTS_ROOT`, which the shims export, else the working directory.

**data_dir** is the downloads: checkpoints and fixtures. Survives an upgrade, and is the
thing someone wants on an external disk. `DREAM_TTS_DATA_DIR`, else `data_dir`, else the
root — which is exactly what every path in this repo meant before the setting existed, so
relocation is opt-in and nothing changes for someone who never writes a config.

Paths the *user* supplies are resolved leniently: as typed if that exists, otherwise against
the install root. The working directory wins, so nothing shipped can shadow a local file —
but `--voice voices/…`, the form every doc uses, keeps working when `dream-tts` is invoked
from elsewhere through a symlink on PATH. And `--voice` may be omitted entirely: every engine
here clones from an asset, so the shipped one is used rather than failing on a decision the
caller has no information to make.

### Downloading

`scripts/fetch-weights.sh` HEADs every file before fetching any, so the progress bar knows
the total byte count up front. Three properties the plain `curl -fsSL` loop it replaced did
not have:

- **Byte progress across the whole set.** A per-file percentage is useless when file 7 of 11
  is 4 GB and the other ten are 2 KB each.
- **Resume that is accounted for.** `curl -C -` already resumed, silently, so a resumed 4 GB
  download looked identical to a stalled one. Bytes on disk now count as progress and the
  summary says how many were skipped.
- **Verification.** Hugging Face returns `x-linked-etag` for LFS-backed files and it is the
  content's sha256, so the large files are checked rather than merely counted. Small files
  have no such header and are checked by length. A failure deletes the file rather than
  leaving a plausible-looking truncation.

Verified digests go in `<data_dir>/references/<engine>/weights/.weights-manifest`, so a
re-run costs three HEAD requests instead of re-hashing gigabytes. Delete it to force full
re-verification.

### One engine on the GPU

A single render peaks well above what a 16 GB machine can spare alongside a second engine,
and two resident engines drive it into swap — which looks like the models getting slower
rather than like a mistake. `tts_core::lock` takes an advisory `flock` on
`<data_dir>/.dream-tts-gpu.lock` for the lifetime of the render, or of the service process.

Advisory is the right strength: it is per-weights rather than per-machine, two independent
installs do not contend, and anything that declines to take it is simply uncovered rather
than blocked. It is taken *before* the weights load, because the load is itself most of the
memory pressure. A second process is refused immediately, naming the first — never made to
wait silently, since these operations run for minutes. `--no-gpu-lock` opts out.

This replaces a `pgrep` against `target/release/dream-tts` in the narration scripts, a
heuristic that broke the moment binaries could also live in `bin/` and that never saw a
process started any other way. The `pgrep` remains as belt, because it produces a better
message before a 3 s model load than after one.

### Terminal output

Decoration is never part of the data. A pipe, a file, a CI log, `NO_COLOR=1`, `CLICOLOR=0`
and `TERM=dumb` all get the same plain text, so `dream-tts storage | grep` and
`dream-tts config > issue.txt` behave; `CLICOLOR_FORCE` overrides in the other direction.
Escape codes leaking into a redirect is the classic way a pretty CLI becomes an unusable
one. Progress goes to stderr and results to stdout, so redirecting either one still makes
sense on its own.

Engines report progress as **segments completed per stage** (`tts_core::ProgressEvent`) —
the one unit every engine has, and the same unit the caller's text was split into. Each
stage counts from zero, so a three-stage engine reports three passes rather than one merged
fiction; that matches the stage breakdown printed at the end and is honest about where the
time goes. A batched stage steps by the group size rather than interpolating inside a call
it cannot see into.

Progress lines carry **position only, never a duration.** A stage whose work happens inside
one call reports a single event *after* it — the qwen3tts codec decodes a whole utterance at
once — so timing from that event measures nothing and would print `0.0s` for 3.5 seconds of
work. The synchronised breakdown is the only honest source for time, and progress does not
compete with it. The live bar's ETA is derived from the running stage's own observed rate,
which is legitimate while it is still running; extrapolating across stages would be a guess,
since an engine's stages differ in cost per segment by an order of magnitude. See
[How to measure without fooling yourself](#how-to-measure-without-fooling-yourself) for why
this repo is careful here.

### Namespacing

`dream-tts`, `dream-tts-serve`, `dream-tts.json`, `DREAM_TTS_*`, `~/.config/dream-tts/`,
`.dream-tts-gpu.lock`. `tts` alone belongs to whoever installed it first — coqui-TTS ships a
binary by that name. The internal crates keep their `tts-*` names: they are path
dependencies, never published, and can collide with nothing.

### Uninstalling

```sh
./scripts/uninstall.sh                     # lists every category with its size; deletes nothing
./scripts/uninstall.sh --weights qwen3tts  # one engine's checkpoint
./scripts/uninstall.sh --all               # weights, venvs, fixtures, build output
./scripts/uninstall.sh --everything --yes  # and the directory itself
```

It asks `dream-tts config` where `data_dir` actually points, so it finds checkpoints that
were relocated. It refuses any path outside the install and the data directory, and refuses
to run without a TTY unless `--yes` is passed — a piped invocation has no informed consent
to give. A script rather than a subcommand on purpose: the thing that deletes an hour of
downloading should be readable before it is run, and should keep working when the binary is
the part that is broken.

## Documents and narration

### One stage in front of four that already worked

`narrate-book.sh` takes its chapter structure from the *filesystem* — `chapter-NNN.md`, one
per file — and four stages hang off that: verbalisation, WAV master, delivery encode,
alignment manifest. So document import is a stage in front and changes none of them:

```text
document  ->  chapter-001.md, chapter-002.md, ...  ->  [the existing pipeline]
```

Which means the work is not extracting text; it is **splitting one monolithic document into
chapters**. That is what decides which formats are easy, and it is why markdown is the
intermediate rather than plain text: `tts-narrate` is built for markdown, and a heading is
what produces the 320 ms paragraph gap a listener hears as a section break.

| format | chapter signal | notes |
|---|---|---|
| EPUB | the OPF spine, in reading order | explicit; `linear="no"` front matter is skipped |
| DOCX | `w:pStyle w:val="Heading1"` | `Heading1`, `heading 2`, `Title`, `Subtitle` all occur |
| ODT | `text:h` with `text:outline-level` | explicit |
| HTML | `<h1>`…`<h6>` | `script`, `style` and `head` contribute nothing |
| Markdown | the shallowest heading level *present* | a book of `##` under one `#` still splits |
| PDF | `PDFOutline`, else one chapter | see below |

DOCX, ODT and EPUB are all a zip of XML, so one pair of dependencies — `zip` and `quick-xml`
— covers three formats and pulls no C library, which is what keeps `bin/` free of
non-system dylibs. macOS `textutil` reads DOCX too and is free, but it was measured against
the same file and **flattens every heading to a styled `<p>`**; losing the headings loses the
chapter boundaries the whole pipeline hangs off, so it is not used.

### Why PDF goes through PDFKit

PDF is the one format with no structure to read: it describes glyphs at positions, not
paragraphs. Two consequences.

**Extraction is delegated.** `PDFPage.string` applies Quartz's layout analysis — reading
order, column detection, de-hyphenation — which is a large body of work no pure-Rust
extractor matches. PDFKit is in `/System/Library/Frameworks`, so it costs nothing to install
and stays inside the release audit that refuses any non-system dynamic dependency; the
shipped `dream-tts` links Foundation, Metal, PDFKit and three `/usr/lib` dylibs, and nothing
else.

**Chapters come from the outline, or from nothing.** `PDFOutline` is the table of contents
the file itself declares, naming both the chapter and the page it starts on. Without one this
yields a single chapter rather than inferring headings from font size, because a wrong split
is worse than none: it desynchronises the per-chapter resume the pipeline is built on. Only
the top level is used — a nested outline describes sections within a chapter, and splitting
on those gives a narration file per subsection, each too short for the engine to batch.

Two details that are not obvious. Quartz returns a newline per *visual* line, and a newline
is a paragraph break to the converter — so a chapter would arrive as several hundred one-line
paragraphs, each getting the 320 ms gap, which reads as a stammer; lines are rejoined unless
the previous one ends a sentence *and* is short enough to be the last line of one. And a
hyphen at a line break is ambiguous — PDF hyphenates words and also wraps after a compound's
own hyphen — so the hyphen is **kept**: a wrongly-kept hyphen narrates as separate words,
while a wrongly-dropped one fuses two words into a non-word, and a fused non-word is
precisely the input that sends the model into a repetition loop.

**Marks are placed at the heading, not at the page.** An outline destination names a page,
which is too coarse for anything but a book of chapter-per-page: a real paper puts "Abstract"
and "1 Introduction" on page 0, and slicing by page has to drop one of them and gives the
whole page to whichever survives. Measured on a 14-page VLDB paper, page-granular splitting
lost **five of its thirteen chapters**. So each mark is placed where its own heading text
appears within its page — matched with whitespace collapsed and case folded, since PDF layout
breaks a heading across lines and prints it in capitals — and falls back to the page boundary
only when the label does not match the printed heading, where merging is the honest outcome.
The printed heading is then dropped from the body, because the chapter title already carries
it and the voice would otherwise announce every section twice, in capitals, which it spells
letter by letter.

A scanned document has no text to find. That is reported with the fix named rather than
returning an empty chapter; this extracts text and does not perform OCR.

**What a paper still costs.** A research PDF is the hardest input this has: its figures,
tables and algorithm blocks have no marker distinguishing them from prose, so a figure's axis
labels and a comparison table's cells arrive as sentences. `dream-tts narrate` reports each
one — surviving markup, and any sentence over 400 characters — and on the paper above that is
six warnings across fourteen chapters, every one of them a figure or a table. There is no
signal in extracted PDF text to fix this automatically; the warnings name the passages to
check.

### The narration port, and how it is checked

`crates/tts-narrate` is a port of `scripts/md-to-narration.py`. The Python stays in the tree
for two jobs: it is the long-form record of *why* each of its ~forty rules exists, and it is
the reference the port is checked against.

```sh
./scripts/check-narrate.sh              # every function
./scripts/check-narrate.sh clean_inline # one of them
```

Both implementations run over the same corpus and must produce byte-identical output for
`speak_code`, `speak_math`, `speak_numbers`, `clean_inline`, `convert`, `page_text` and the
word-alignment map. The corpus has three sources, because each catches a different class of
mistake: one hand-written case per rule, so a mistranslated rule fails on its own line and
says which; the real chapter in `prep-handbook/`, which contains combinations nobody would
think to write; and seeded random recombinations of the fragments, because **most of the traps
recorded in the Python are interactions between rules** — the currency rule consuming the
number the magnitude rule wanted — and only recombination reaches those. 6190 lines and 621
documents, in about seven seconds.

That harness earned its place immediately. It found three bugs in the port that no
hand-written case had caught:

- **A Rust raw string does not process a backslash-newline continuation.** The currency
  pattern, written across four lines the way the Python has it, kept the backslashes and
  never matched — so every `$` was left for the inline-maths rule, which then paired two
  prices and ran `speak_math` over the prose between them. `concat!` instead.
- **`$$` is an escaped dollar in a replacement string.** `"$${1}"` emitted the literal text
  `${1}` rather than a dollar and the group.
- **The harness itself was wrong before the port was.** Python's `json.dumps` defaults to
  `", "` separators and `\uXXXX` escapes and serde_json does neither, so a byte comparison
  reported every case as differing on formatting alone.

`crates/tts-narrate/tests/reference.rs` ports the Python's own suite case for case, so the
crate is verified with nothing else installed; `check-narrate.sh` is the stronger evidence
and needs python3, and `gates.sh` reports it as skipped when there is none.

One deliberate transliteration rather than a rewrite: `align::Matcher` reproduces Python's
`difflib.SequenceMatcher.get_matching_blocks`, including its tie-breaking and its recursive
split. A generic LCS diff disagrees with it on real input, and the word map this feeds is
consumed by a player that highlights the wrong word when they do.

## Runs, and watching them

### Why there is a server

Everything in this section exists because a thousand-page reference book is hours of
synthesis, and that length changes the requirements rather than just the runtime.

A run has to survive its terminal closing, so the state cannot live only in the process
doing the work. It has to be discoverable by a process that did not start it, or the second
thing anyone does is start a second run on top of the first. And progress has to be fine
enough to be worth watching, which means something must be emitting it while the work
happens.

All three want the work to outlive the command that asked for it. `dream-tts-serve` already
held the engine resident, serialised GPU work behind a semaphore and held the GPU flock for
its lifetime — so the queue went there rather than into a second daemon that would have
fought it for the device. Its `/v1/tts-jobs` route had answered `501 "durable job queue"`
since before any of this; the slot was reserved for exactly this.

### Why the records are files, not the server's memory

`crates/tts-jobs` writes one JSON file per run under `data_dir/jobs/`. The reason is the
discovery question: **"is something already narrating?" has to be answerable when the
service is not running**, because that is precisely the moment someone is about to start
one. A database, or state held in the service, would make the service the only thing that
can answer.

That is also what makes the extensibility real rather than claimed. `dream-tts jobs` is one
client; a status bar, a menu-bar app or a shell script reading the same files is another, and
neither needs the HTTP API or to be trusted with its key.

Every write is a write-and-rename. A reader is often a progress display refreshing several
times a second, so a reader catching a half-written file would be the normal case rather than
a rare one.

### Identity is the document, so resume is the same command

A job's id is a SHA-256 prefix over the text that will be spoken and the settings it will be
spoken with. Resubmitting the same book therefore *is* the resume path — no `--resume`, no
remembered id, just the command the user was going to type anyway.

Settings are in the hash because they change the audio: narrating the same book in a
different voice is a different run and must not adopt the first one's chapters. So is the
text, which matters more — **editing a chapter changes the id**, because resuming onto audio
of the old wording would be silent damage.

Adoption is conservative in one further way: a finished chapter is only inherited if its WAV
is still on disk. Deleting a WAV to force a re-render has to actually cause one, which is the
documented way to redo a chapter.

Resume needs no "where we were" marker, because finished work is the marker:
`Job::next_chapter` is the first chapter that is not `Done`. A *failed* chapter is next again
rather than skipped — the failure may have been the machine, and skipping would ship a book
with a hole in it.

### Pause, and what it costs

Synthesis is minutes per chapter. "Stop" therefore has to mean something sooner than "when
it finishes", and there are only two honest options:

| | when | cost |
|---|---|---|
| `--control pause` | after the chapter in flight | none; its work is kept |
| `--control pause --now` | within seconds | that chapter is discarded and redone on resume |

`--now` works because `SynthesisRequest` carries an `Interrupt` — a `Fn() -> bool` the engines
ask **between segments**, at the same boundary where they already report progress. That is
where an engine's state is consistent; mid-segment there is a partly generated waveform that
is not audio yet, so there is nothing to keep. An interrupted chapter is left `Pending`, not
`Failed`: a caller stopped it, which is a run to resume rather than a fault to report.

A queued pause is shown as **`pausing`**, in the CLI and on the page. A control that appears
to have done nothing is worse than one that refuses, and on a twenty-minute chapter the gap
between asking and its taking effect is long enough to look broken.

### The estimate, and economies of scale

A flat words-per-second rate is the obvious model and it is wrong by a factor of two and a
half, in both directions.

These engines have strong economies of scale. `qwen3tts` batches across segments and that
only engages once a chapter has enough of them, so the figures are RTF **0.397** on
`examples/senior.txt` (132 words) and **0.148** on `examples/chapter.txt` (1612 words) — a
2.7× spread (0.314 and 0.109 today, the same shape) in seconds per word for the same voice on the same machine. A rate learned from a
short chapter and applied to a long one is wrong by about that much.

It goes the other way too, and that was the first bug: a book's front matter is 27 words of
title page, almost all fixed cost, and extrapolating from it put a real run's estimate at
**2h 09m against an actual hour**.

So the model is `wall ≈ fixed + marginal × words`, fitted by least squares over the chapters
that have finished and applied to each remaining chapter *individually*. A long chapter
amortises the fixed part the way it really does; a short one still pays it. Degenerate fits —
one sample, chapters all the same length, or noise making longer chapters look cheaper in
total — fall back to a flat rate, which is the honest answer when there is no slope to find.

No estimate at all until there are two finished chapters or five percent of the words. A
confident wrong number on an eight-hour run is worse than none, because the direction it errs
in is the one that makes someone abandon a run that would have finished.

### Stale runs

A service killed mid-run leaves a record saying `Running` for ever. Each job carries the pid
that owns it, `Job::is_stale` asks `kill(pid, 0)`, and both the service at startup and the
CLI before it reports anything turn stale runs into **paused** ones. Paused, not failed:
nothing went wrong with the work, and every finished chapter is still on disk.

### What the machine can hold

One resident engine is gigabytes and two on a 16 GB machine swap — which presents as the
models getting slower rather than as a mistake. But refusing a second engine on a 64 GB
machine would be wrong, so `tts_core::system` reports rather than decides: total memory, a
measured per-engine footprint, and therefore roughly how many fit. `dream-tts book` prints
it alongside anything already in flight and lets the user choose.

### Incremental synthesis

`POST /tts/stream` used to be buffered, and said so in an `x-streaming: buffered` header. It
is now genuinely incremental: raw PCM per segment, chunked, first audio after one segment
instead of after all of them. Measured on the same text, **2.7s to first audio against 5.5s
buffered**. Even unbatched the engine stays well under realtime — 0.314 on the short-passage
fixture — so the stream outpaces playback and a live listener never runs dry once it has
started.

It is **slower overall, deliberately.** `Engine::synthesize` batches across segments and that
is worth 2.7× on book-length text; one segment at a time gives it up. What a realtime caller
needs is not throughput but a short time to first audio, and those are different quantities —
which is why the job runner does not use this path. A listener waiting on a live response
cares about the first second; a book cares about the last hour.

Segmenting outside the engine means reproducing what it does *between* segments — 90 ms
within a paragraph, 320 ms between them. Without that, the same text would render differently
depending on which URL was called.

The chunk queue is two deep on purpose: a client that stops reading should stop the work
rather than let the server run a chapter ahead into memory.

## What did not work

| | why |
|---|---|
| **ONNX Runtime** | op coverage gaps, and no path that beat candle on Metal |
| **CoreML** | 19–26 s compilations, and coverage holes on the ops that mattered |
| **A custom q8_0 GEMM** | amortised up to 5.2× where candle's does 1.0×, validated against candle at real widths for m = 1..8 — but no configuration beat MPS f16 in absolute terms |
| **f32 weights for `qwen3tts`** | 7.45× per lane in isolation but RTF **6.25** in a real render at 6.97 GB resident |
| **Slicing the codec's waveform stack** | exact — every conv below the pre-transformer is causal, so a slice with real left context reproduces the whole-signal pass — and worth **-1.08 GB for +9% time**. It does not buy the lane count it was for: 64 lanes still thrashed at RTF 0.692. Padding the *front* of a slice is also wrong, the convs carrying biases |
| **One cache shape for every group** | allocating at the batch width and narrowing, to stop a 64 + 36 split leaving two permanent caches: +0.4 GB at 48 lanes and 64 still thrashed |
| **An int8 KV cache** | groups of 32 with inline scales: RTF 0.164 → 0.158 and **no memory saved at all** (13.32 → 13.36 GB peak), because the peak is the codec's activations. Fails the gate at `step1.hidden` — 4.52e-1 for K alone against a 5.0e-3 tolerance, since a key error moves a score before the softmax |
| **One KV state for the whole render** | reused across groups instead of allocated per group: **17.10 GB peak against 13.33** at no change in RTF. A cache sized for the widest group is pinned while the codec runs, where per-group caches are buffers the pool hands on |
| **Device-side sampling, per step** | moved less than the transfer it saved, because the step still synchronised. Sampling all fifteen depth steps on the device with draws taken up front — one read per frame — is what shipped |
| **f16 codec decoder** | quality loss. It is no longer "without a speed win": through MPSGraph an f16 residual unit is 2.66× today's f32 path, so quality alone is what rules it out |
| **MPSGraph for every codec conv** | 1.02× in sum at f32. It wins while the reduction is short (96 → 1 at k=7 is 2.72×) and loses once it is long (0.84× at 768 → 1920, 0.50× on a 96-channel k=1); `tts_nn::mpsnlc::eligible` takes only the winners |
| **64 lanes, retried after prefix sharing and device sampling** | 15.6 GB peak and RTF 0.790 — still the VM compressor. Peak at 48 fell 0.74 GB, not the 4 GB 64 needs |
| **An unfilled KV cache** (`Tensor::empty` for `Tensor::zeros`) | identical audio, since attention reads only written positions — but peak footprint rose 11.74 → 13.84 GB. Skipping the fill does not leave a Metal buffer's pages uncommitted |
| **KV sized to the group's frame budgets** | candle rounds every buffer up to a power of two, and a 48-lane layer cache stays in the 64 MiB class until capacity drops below ~340 positions |
| **Staged variants of the decode GEMM** | double-buffered threadgroup tiles 5-8% slower, a 48x128 tile and 64-deep chunks slower too, any patch past 12 accumulators per simdgroup spills (0.42 TFLOP/s at 24, 1.4 with a 2x K unroll). What won was the opposite direction: no staging at all |
| **Loading a conv's operands straight from device memory** | the move that won for the decode GEMM loses 3-12x on the codec: with `M` in the hundreds of thousands, every 32-position simdgroup re-reads the weights — ~4.8 GB per conv at 96 channels. Staging is what amortises them |
| **Fusing residual add, RMS norm and the f16 cast** | one dispatch for four, and no measurable change in a decode step (69.8 against 69.5 ms). The QK-norm/rope fusion paid because it replaced slow 4-D norm and rope kernels, not because it removed dispatches |
| **SnakeBeta fused into a 768-channel conv** | the kernel reloads its input once per 32 output channels, so the sin runs 24 times per element and the saved pass buys nothing; up to 384 channels it is a 9.5% win and shipped |
| **Padded decode attention** | superseded by fused decode attention |
| **`kokoro`: an LSTM in one threadgroup per direction** | all steps in one dispatch, h and gates in threadgroup memory: 31 µs a step against the step form's ~16 per direction. One core cannot pull `w_hh`'s 1 MB a step; spreading the rows over threadgroups, with the dispatch boundary as the barrier, is what shipped |
| **`kokoro`: a thread per gate row, looping over H** | 33 µs a step: 256 dependent loads per thread, latency-bound. A simdgroup per unit, H/32 independent loads a lane, is 7 |
| **`kokoro`: fused QKV and candle's `sdpa` in ALBERT** | 30 ms before and after. The GEMMs run at under 1 TFLOP/s at M = 189, and neither change touches that |
| **`kokoro`: MPSGraph below 4096 samples** | `TTS_MPS_MIN_LEN=512`, to take the decoder's 1090-channel convs: prosody 23 → 28 ms, decoder no better |
| **`kokoro`: MPSGraph at `Level0`** | a new length still cost ~3 ms a graph against ~4; compiling ahead is what removed it |
| **`kokoro`: f16 generator convs through MPSGraph** | 1.12-1.18× over f32 at 128 channels, not enough to spend quality on |
| **`kokoro`: candle's fused layer norm** | as fast as the kernel that shipped, but its one-pass variance cancels on ALBERT's activations: F0 2.7e-3 from the fixture against a 2.0e-3 tolerance, where the composed passes give 1.3e-3 |
| **`kokoro`: the resblock graph encoded onto an `MPSCommandBuffer`** | the hope was that intermediates would come from the buffer's heap and be released with it. They grew by the same ~90 MB a length; only dropping the graph frees them |
| **`kokoro`: draining an autorelease pool around graph runs** | no change to the footprint; the memory is held by the graph, not by autoreleased objects |
| **`kokoro`: ALBERT's linears as 2-D matmuls** | on the suspicion that `broadcast_matmul` copied the weight per call: no change, it does not |

The one that keeps paying: **candle is 1.70–2.23× slower than torch on this codec**, stable
across every thermal state, which is what motivated the custom Metal kernels in `tts-nn`.
