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
./scripts/bootstrap.sh --all                 # all three engines, ~13 GB
./scripts/bootstrap.sh --list                # the ids, their models, what each costs
./scripts/bootstrap.sh audio8 cosyvoice      # two of them
./scripts/bootstrap.sh --force audio8        # redo a conversion that already ran
./scripts/bootstrap.sh --prebuilt            # download the binaries rather than build them
```

That is the whole setup. It resolves how to get binaries, downloads and converts the
checkpoints you asked for, fetches the fixtures, and either builds or downloads; every step
is skipped if its output already exists, so re-running costs nothing. The rest of this
section is what it does on your behalf, for when a step fails or you want to do one by hand.

**Why the default is one engine.** `qwen3tts` alone is the only configuration whose setup is
downloads and nothing else — no conversion, so no Python, so no torch venv. `audio8` folds a
`weight_norm` pickle and `cosyvoice` re-serialises a `torch.load`, and both want python
>= 3.10. Naming either is what buys that cost. The default used to be all three, which meant
every first run paid ~13 GB and a 2.5 GB venv to get an engine that is also the slowest of
the three on book-length text.

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

### 2. The other two checkpoints

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
pt), and `q8_0` is its default because the checkpoint is bf16 and f32 measured 38× slower on a
16 GB machine.

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
```

Building a new voice uses the same venvs via each engine's `export_voice.py`. The transcript
matters more than it looks: CosyVoice asserts the prompt text contains `<|endofprompt|>` and
nothing in its frontend adds it.

### Disk budget

| path | size | tracked |
|---|---|---|
| `references/*/weights/` | ~3.4–4.3 GB each | no |
| `fixtures/` | ~130 MB | no — fetched |
| `voices/` | ~200 KB | **yes** |
| `target/` | ~1.8 GB | no |

---

## Architecture

Ten crates. Everything shared is `tts-*`; everything engine-specific is named for its engine
and matches the id `--engine` takes.

**Voice assets are the decision that made a second and third engine tractable.** All three
models clone from a reference clip, and in every case turning audio into conditioning needs
machinery the runtime should not carry:

| engine | the clip must become | in-process cost avoided |
|---|---|---|
| `audio8` | `[10, N]` RVQ codes | the codec **encoder** — 126 tensors `convert_codec.py` drops |
| `cosyvoice` | speaker embedding, speech tokens, prompt mel, prompt text tokens | `campplus.onnx` (28 MB) + `speech_tokenizer_v3.onnx` (969 MB), plus an ONNX runtime |
| `qwen3tts` | x-vector, `[T, 16]` RVQ codes, sliced transcript tokens | an ECAPA-TDNN speaker encoder and a Mimi-style RVQ **encoder** |

None of it depends on the text being spoken, so it happens once, offline, in Python, and ships
as a directory of `voice.json` + `voice.safetensors`. `Voice::load` checks the `engine` field
and a mismatch is a hard error rather than a silent substitution.

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
| `cargo test --release` | 32 tests including the `tts_engines` doctest |

Gates compare against per-stage fp32 activations dumped from PyTorch, so a failure localises
to a stage rather than to "the audio sounds wrong". A tier whose inputs are absent is
**skipped and reported**, never silently passed.

Sampled output is deliberately *not* gated on equality: `ras_sampling` draws from torch's
generator, so the sampled sequence is not reproducible across implementations. The gates check
prefill logits and a greedy rollout instead, and quality is checked separately by WER.

---

## Performance

Two fixtures, both tracked so any figure here can be reproduced:

| fixture | | |
|---|---|---|
| `examples/senior.txt` | 132 words, 7 segments, ~50 s of audio | the short-passage case |
| `examples/chapter.txt` | 1612 words, 100 segments, ~11.6 min of audio | the long-form case, where batching engages |

### Short passage

`examples/senior.txt`, M4 / 16 GB, `q8_0`. Median of five samples with the three engines
interleaved in one session.

| engine | reference | this port | spread | |
|---|---|---|---|---|
| `audio8` | 1.307 (PyTorch bf16 MPS, batched) | **0.554** | 0.547–0.562 | 2.36× faster |
| `cosyvoice` | 4.370 (stock PyTorch, CPU-only) | **0.726** | 0.697–0.734 | 6.02× faster |
| `qwen3tts` | — | **0.665** | 0.642–0.687 | the wrong case for it; see below |

### Chapter, and what batching is actually worth

`examples/chapter.txt`, 100 segments. `f16` is the median of three samples.

| engine | weights | RTF | talker/LLM | codec |
|---|---|---|---|---|
| `qwen3tts` | `q8_0` | 0.661 | 0.588 | 0.072 |
| `qwen3tts` | **`f16`** | **0.260** (0.256–0.261) | 0.187 | 0.073 |

**`q8_0` gains nothing from 14× more segments** — 0.665 on seven, 0.661 on a hundred. That is
candle's quantized `mm_t` padding to a large row tile, so batch 8 costs 8× batch 1, measured
end to end rather than inferred from op timings. `f16` gets **2.54×**, and all of it is in the
talker (0.588 → 0.187); the codec is unchanged at 0.072, as it must be, since the weight format
being varied is the talker's.

This is why the narration path uses `f16` and why `qwen3tts` is the default for a book despite
being the slowest of the three on a short passage. Reproduce with:

```sh
./dream-tts speak --engine qwen3tts \
    --voice voices/cosy-default-qwen3tts --quant f16 \
    --text-file examples/chapter.txt --out chapter.wav
```

An RTF of **0.253** was previously quoted for this configuration with no fixture behind it and
no end-to-end render supporting it. The measurement above is what replaced it; the claim turned
out to be close to right, which is luck rather than evidence.

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
| `POST /tts`, `POST /tts/stream`, `GET /health`, `GET /v1/capabilities`, `GET /` | served |
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
only engages once a chapter has enough of them, so the documented figures are RTF **0.665**
on `examples/senior.txt` (132 words) and **0.260** on `examples/chapter.txt` (1612 words) —
0.287 s/word against 0.112 s/word for the same voice on the same machine. A rate learned
from a short chapter and applied to a long one predicts 463 s where the truth is 181 s.

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
buffered**; at `qwen3tts`'s RTF of 0.67 the stream outpaces playback, so a live listener
never runs dry once it has started.

It is **slower overall, deliberately.** `Engine::synthesize` batches across segments and that
is worth 2x on book-length text; one segment at a time gives it up. What a realtime caller
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
| **f32 weights for `qwen3tts`** | 7.45× per lane in isolation but RTF **6.25** in a real render at 6.97 GB resident. This is the wall that made `q8_0` the default |
| **Device-side sampling** | moved less than the transfer it saved |
| **f16 codec decoder** | quality loss without a speed win |
| **Padded decode attention** | superseded by fused decode attention |

The one that keeps paying: **candle is 1.70–2.23× slower than torch on this codec**, stable
across every thermal state, which is what motivated the custom Metal kernels in `tts-nn`.
