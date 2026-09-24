# dream-tts Repository Guide

Offline text-to-speech in Rust with Metal kernels: four engines behind one CLI, no Python at
runtime. `qwen3tts` is the default and carries the project.

Start with the **`dream-tts` skill** in `.agents/skills/dream-tts/` — it covers setup, narrating
a page or a book, and how the engine works. This file is only the rules that hold regardless of
the task.

## Where the truth lives

- `README.md` — the guide, with the measured numbers and the pipeline diagram.
- `docs/reference.md` — setup detail, every measurement and how it was taken, the porting traps,
  and **what did not work**.
- The code's own comments. Constants carry the measurement that chose them, and kernels carry
  the versions that lost. Read them before changing a number.

## Build, run, check

```sh
./scripts/bootstrap.sh          # checkpoint, assets, build. ~4.3 GB, resumable
cargo build --release           # NOT `-p qwen3tts`: `dream-tts` lives in tts-cli
./scripts/gates.sh              # fixture gates for all four engines, tests, HTTP smoke test
cargo test --release -p tts-nn  # the kernels
./scripts/check-phonemes.sh     # Kokoro's English frontend, against spaCy and misaki
cargo run -p kokoro --release --bin kokoro-validate
```

`kokoro` is the fourth engine and the only non-autoregressive one. It cannot clone — its
voices are 28 style tables inside the checkpoint, picked with `--set voice=<name>` — and its
English frontend is a lexicon with no espeak fallback, so an unknown word is named in a lint
rather than guessed. `docs/kokoro-frontend.md` and `docs/kokoro-model.md` carry its traps and
its measurements. On Metal its generator runs padded to length buckets (`KOKORO_BUCKETS`, 0 to
disable), so an A/B of a generator change has to render several lengths, not one utterance
repeated: a repeated utterance never pays MPSGraph's per-length compile. Each resblock runs as
one MPSGraph (`KOKORO_NO_MPSBLOCK` to compose it instead); a graph holds its intermediates until
dropped, so they live in a six-entry LRU, and an A/B must watch peak footprint as well as RTF.
`TTS_NN_LN=0` puts `tts_nn::layer_norm` back on the composed passes.

`cargo build -p <crate>` does not relink the binaries, and `cargo clippy --workspace` leaves them
built **without Metal**. Rebuild with a plain `cargo build --release` before measuring anything.

## The rules

**The gates decide, not the ears and not the diff.** `./scripts/gates.sh` must pass before a
change is committed. The qwen3tts gate compares **absolute** difference against a per-row
tolerance, and it decodes a **single lane** — so it does not cover the batched decode path or
the GEMM that path uses. A change to those needs its own test.

**Quality is checked on the audio, never on the codes.** Token identity means nothing for a
sampled model: any change that moves the logits produces different codes and the same speech.
Use `references/cosyvoice/wer.py` and `references/audio8/verify_voice.py`, and say which of the
two a change was checked with. A change that claims to be exact should be shown to be
bit-identical rather than asserted.

**Measure by interleaving, in one thermal state.** This machine drifts by more than most changes
are worth — the harness canary moved 59 to 65 ms across one afternoon, which is larger than a
1.2x kernel. A/B inside one process run, or behind an env switch, and quote the canary. Never
compare a number against one taken in an earlier session.

**Record what did not work, with its numbers.** Half the value in `docs/reference.md` is the
refuted list, and most obvious optimisations here have already been measured and rejected. Read
it before proposing one; add to it when something loses. A kernel that lost belongs in the
kernel's comments so it is not written twice.

**Memory is candle's Metal buffer pool.** It keys buffers by size and releases none, so every
distinct tensor shape a run touches is permanent. Memory wins are shapes removed, not bytes
shaved, and a change that adds a shape costs more than its size suggests.

## Style

Comments earn their place by recording *why* — a constraint, a trap, a rejected alternative — in
a line or two. No narration of what the code already says, no section essays, no restating a
signature. Match the terse end of the codebase.
