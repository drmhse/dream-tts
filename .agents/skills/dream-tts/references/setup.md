# Setting it up

## The fast path

`curl` is the only prerequisite. The default engine's checkpoint is a plain download with no
conversion step, and a `v*` tag publishes prebuilt arm64 binaries.

```sh
curl -fsSL https://raw.githubusercontent.com/drmhse/dream-tts/main/install.sh | sh
cd dream-tts && ./scripts/bootstrap.sh       # ~4.3 GB, resumable, sha256-verified
./dream-tts speak --text "Hello from a fresh install." --out hello.wav
```

`install.sh` unpacks a release and stops; it prints the weights command rather than pulling
4.3 GB unannounced. `DREAM_TTS_SETUP=1` makes it run that itself, `DREAM_TTS_DIR` moves the
install, `DREAM_TTS_VERSION` pins a release.

From a git clone the same script does everything, and builds instead of downloading when it
finds a toolchain and sources:

```sh
./scripts/bootstrap.sh
```

Every step is skipped when its output exists, so re-running after an interruption is cheap.

## The other three engines

```sh
./scripts/bootstrap.sh --list              # ids, models, what each costs
./scripts/bootstrap.sh audio8 cosyvoice
./scripts/bootstrap.sh kokoro              # ~0.7 GB
./scripts/bootstrap.sh --all               # ~14 GB
```

Each converts its checkpoint with PyTorch, so they want python >= 3.10 and a torch venv from
`references/<engine>/requirements.txt`. That cost is paid only by whoever asks for them.

## What it needs

| engine | peak footprint | RTF on a short passage | reach for it when |
|---|---|---|---|
| `qwen3tts` | 12.3 GB | 0.397 | the default: best quality, and the only cloning engine practical for books |
| `audio8` | 9.7 GB | 0.544 | 44.1 kHz output |
| `cosyvoice` | 5.0 GB | 0.716 | widest language coverage |
| `kokoro` | 1.3 GB | 0.044 | fastest and smallest by far, English only, cannot clone: `--set voice=<name>` |

**16 GB for the default engine.** Most of its peak is the codec decoder's activations rather
than weights — one 300-frame chunk is 5.5 GB — so no weight format moves the floor. Under 16 GB
the engine says so on load and keeps going; it will swap, and swapping reads as the model being
slow rather than as a mistake.

`qwen3tts` speaks ten languages only: en, de, es, zh, ja, fr, ko, ru, it, pt. Anything else has
no faithful path through it, and the CLI says so when it was chosen by default rather than named.

## Where things live

```sh
./dream-tts config      # every setting, and which file or variable decided it
./dream-tts storage     # what is on disk, biggest first, and what removes each part
./dream-tts engines     # what is installed, what each supports, which is the default
```

`data_dir` holds the 4–13 GB of checkpoints and fixtures, separate from the install so it
survives an upgrade and can sit on an external disk:

```json
{ "data_dir": "/Volumes/ssd/dream-tts" }
```

Settings resolve flag, then environment, then `dream-tts.json` beside the install, then
`~/.config/dream-tts/config.json`, then built-in defaults. Copy `dream-tts.example.json` to
start. Keys beginning with `//` are comments; any other unknown key is an error rather than a
silent no-op. `dream-tts config` prints what actually resolved.

Symlinking works: `ln -s "$PWD/dream-tts" ~/.local/bin/`. Both binaries resolve their own
symlinks, so asset paths stay pinned to the install while `--out` and `--text-file` stay
relative to wherever the command was typed.

## Voices

A voice is a directory holding `voice.json` and `voice.safetensors`, and the repo ships several.
Building one takes about ten seconds of clean audio and its exact transcript:

```sh
references/qwen3tts/.venv/bin/python references/qwen3tts/export_voice.py \
    --model references/qwen3tts/weights --audio clip.wav \
    --text "the exact words spoken in the clip" \
    --name my-voice --out voices/my-voice

./dream-tts voice voices/my-voice     # what the asset holds, without synthesising
```

The transcript has to match the clip: it is the reference text half of the in-context prompt,
and a wrong one degrades the clone rather than failing. Voices are per engine — a
`cosy-default-qwen3tts` asset is not interchangeable with the `cosyvoice` one of the same name.

Export is the only runtime-adjacent step that wants Python, and it runs once per voice. The
speaker encoders stay in `references/`; the engine loads exported conditioning and never carries
an encoder.

## Checking the install

```sh
./scripts/gates.sh
```

Fixture gates for all four engines, the unit tests, and an HTTP smoke test. The qwen3tts gate
alone is 65 rows comparing tensors against dumps from the reference implementation. It compares
**absolute** difference against a per-row tolerance, and it decodes a single lane — so it does
not exercise the batched decode path or the GEMM that path uses.
