#!/usr/bin/env bash
#
# Run one of this workspace's binaries, built or prebuilt.
#
#   scripts/run-bin.sh dream-tts speak --text "hello" --out out.wav
#   scripts/run-bin.sh --which dream-tts-serve   # print the path, building it if needed
#
# The point is that a checkout without a Rust toolchain still works. A release archive
# unpacks its binaries into `bin/`, and this finds them there.
#
# Resolution order, and the reason for it:
#
#   1. $DREAM_TTS_BIN_DIR       explicit wins, for testing a release archive in place
#   2. cargo, if present  `cargo run` rebuilds when the source moved; a bare exec of a
#                         stale target/release/dream-tts would silently run old code, which is
#                         the one failure mode worth paying a no-op cargo check to avoid
#   3. bin/<name>         the prebuilt, for a machine with no cargo
#
# Set DREAM_TTS_PREBUILT=1 to skip step 2 when you have cargo but want the shipped binary.
#
# Asset paths inside the binaries are repo-relative (`voices/…`, `references/…`), so run
# these from the repo root, as every documented command does.
set -euo pipefail

# Symlink-resolved, for the same reason the shims resolve themselves: this may be reached
# through a link on PATH, and the install root is what every asset path hangs off.
self="${BASH_SOURCE[0]}"
hops=0
while [ -L "$self" ]; do
  hops=$((hops + 1))
  [ "$hops" -le 40 ] || { echo "run-bin.sh: symlink loop" >&2; exit 1; }
  link="$(readlink "$self")"
  case "$link" in
    /*) self="$link" ;;
    *)  self="$(dirname "$self")/$link" ;;
  esac
done
ROOT="$(cd "$(dirname "$self")/.." && pwd)"

# The binaries resolve `voices/…` and `references/…` against this, not against the caller's
# working directory, so `dream-tts` works from anywhere. Paths the *user* passes — `--out`,
# `--text-file` — stay relative to where they typed them, which is what they mean. An
# already-set value is respected: someone who exported it meant it.
export DREAM_TTS_ROOT="${DREAM_TTS_ROOT:-$ROOT}"

# `--which` exists for callers that need a path rather than a child process: gates.sh
# backgrounds the service and later kills it, and `exec cargo run` would put cargo in the
# pid it kills while the service kept running as cargo's child.
WHICH=""
if [ "${1:-}" = "--which" ]; then WHICH=1; shift; fi

[ $# -gt 0 ] || { printf 'usage: %s [--which] <binary> [args...]\n' "$0" >&2; exit 2; }
BIN="$1"; shift

# Which crate owns each binary. `cargo run` needs the package, not just the bin name.
case "$BIN" in
  dream-tts)          CRATE=tts-cli   ;;
  dream-tts-serve)    CRATE=tts-serve ;;
  audio8-validate)    CRATE=audio8    ;;
  cosyvoice-validate) CRATE=cosyvoice ;;
  qwen3tts-validate)  CRATE=qwen3tts  ;;
  narrate-diff)       CRATE=tts-narrate ;;
  *) printf 'error: unknown binary `%s`\n' "$BIN" >&2; exit 2 ;;
esac

if [ -n "${DREAM_TTS_BIN_DIR:-}" ]; then
  if [ -n "$WHICH" ]; then printf '%s\n' "$DREAM_TTS_BIN_DIR/$BIN"; exit 0; fi
  exec "$DREAM_TTS_BIN_DIR/$BIN" "$@"
fi

# A release archive carries binaries and no sources, so having cargo is not enough — the
# workspace manifest has to be there too, or `cargo run` fails on a machine that has both
# a toolchain and an unpacked release.
if [ -z "${DREAM_TTS_PREBUILT:-}" ] && [ -f "$ROOT/Cargo.toml" ] && command -v cargo >/dev/null; then
  if [ -n "$WHICH" ]; then
    cargo build --quiet --release \
        --manifest-path "$ROOT/Cargo.toml" -p "$CRATE" --bin "$BIN" >&2 || exit 1
    printf '%s\n' "${CARGO_TARGET_DIR:-$ROOT/target}/release/$BIN"
    exit 0
  fi
  exec cargo run --quiet --release \
      --manifest-path "$ROOT/Cargo.toml" -p "$CRATE" --bin "$BIN" -- "$@"
fi

if [ -x "$ROOT/bin/$BIN" ]; then
  if [ -n "$WHICH" ]; then printf '%s\n' "$ROOT/bin/$BIN"; exit 0; fi
  exec "$ROOT/bin/$BIN" "$@"
fi

printf 'error: no `%s` to run.\n' "$BIN" >&2
printf '  Either install Rust (https://rustup.rs) and re-run, or fetch a prebuilt\n' >&2
printf '  release: ./scripts/bootstrap.sh --prebuilt\n' >&2
exit 1
