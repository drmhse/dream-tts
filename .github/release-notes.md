## Install

```sh
curl -fsSL https://raw.githubusercontent.com/drmhse/dream-tts/main/install.sh | sh
cd dream-tts && ./scripts/bootstrap.sh
./dream-tts speak --text "Hello from a fresh install." --out hello.wav
```

Apple silicon, macOS 13 or newer. **Needs `curl` and nothing else** — no Rust toolchain and
no Python. That works because the default engine, `qwen3tts`, has no conversion step: its
checkpoint is a plain download.

The installer stops after unpacking rather than pulling 4.3 GB unasked; `bootstrap.sh` is
what downloads the model. `TTS_SETUP=1` does both in one go, and `TTS_DIR` picks the
install location.

Add the other engines whenever you want them:

```sh
./scripts/bootstrap.sh audio8 cosyvoice
```

Those two convert their checkpoints with PyTorch, so they want python >= 3.10 and a torch
venv — which is exactly why they are not the default.

## Verify

```sh
shasum -a 256 -c SHA256SUMS
./scripts/gates.sh            # the fixture gates, on the prebuilt binaries
```

`gates.sh` reports its source-only tiers (unit tests, lints, the CPU build) as *skipped* in
a binary install, because there is no toolchain to run them with. The fixture gates
themselves run.

A browser download quarantines the archive and `curl` does not. If you fetched it by hand,
`xattr -dr com.apple.quarantine <dir>`.
