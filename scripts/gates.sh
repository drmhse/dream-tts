#!/usr/bin/env bash
#
# Everything that can say "this port is still correct", in one command.
#
# Three tiers, cheapest first:
#   unit tests    — no model weights needed
#   fixture gates — need converted weights and dumped fixtures (docs/reference.md#setup)
#   renders       — need a voice asset; produce audio for the quality scripts
#
# A tier whose inputs are missing is *skipped and reported*, never silently passed.
# Exit status is non-zero if anything that ran failed.
set -uo pipefail

cd "$(dirname "$0")/.."

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
skip() { printf '\033[33m--- skipped: %s\033[0m\n' "$*"; SKIPPED=$((SKIPPED + 1)); }
fail=0
SKIPPED=0

run() { scripts/run-bin.sh "$@"; }

# A checkout unpacked from a release archive has the binaries and neither a toolchain nor
# the sources. The source tiers below cannot run there, and that is a missing input like
# any other: skipped and reported, never silently passed. Both halves are checked, because
# a developer machine can have cargo and still be sitting in an unpacked release.
if command -v cargo >/dev/null && [ -f Cargo.toml ]; then HAVE_CARGO=1; else HAVE_CARGO=""; fi

# Whether an engine's converted weights are present. The gates and the renders both need
# this, and with the default bootstrap installing one engine it is the common case that
# three of the four are legitimately absent.
engine_ready() {
  case "$1" in
    qwen3tts)  [ -f references/qwen3tts/weights/model.safetensors ] ;;
    audio8)    [ -f references/audio8/weights/codec.safetensors ] ;;
    cosyvoice) [ -f references/cosyvoice/weights/llm.safetensors ] ;;
    kokoro)    [ -f references/kokoro/weights/kokoro.safetensors ] ;;
    *) return 1 ;;
  esac
}

say "Unit tests"
if [ -z "$HAVE_CARGO" ]; then
  skip "unit tests: no cargo in this checkout"
elif cargo test --release --quiet 2>&1 | tail -20; then
  echo "tests ok"
else
  echo "tests FAILED"; fail=1
fi

say "Narration reference tests"
# The Python's own suite. It is the reference `tts-narrate` is checked against, so it has to
# keep passing on its own terms — a reference that has drifted proves nothing.
if ! command -v python3 >/dev/null; then
  skip "md-to-narration reference: no python3"
elif python3 scripts/test_md_to_narration.py >/dev/null 2>&1; then
  echo "md-to-narration reference ok"
else
  python3 scripts/test_md_to_narration.py 2>&1 | tail -20
  echo "md-to-narration reference FAILED"; fail=1
fi

say "Clippy and formatting (advisory)"
if [ -z "$HAVE_CARGO" ]; then
  skip "lints: no cargo in this checkout"
else
  if cargo fmt --check >/dev/null 2>&1; then
    echo "formatting clean"
  else
    echo "note: \`cargo fmt\` would reformat some files"
  fi
  lints=$(cargo clippy --release --all-targets 2>&1 | grep -cE "^warning: [a-z]" || true)
  echo "clippy: $lints lint(s)"
fi

say "CPU-only build (the portable configuration)"
if [ -z "$HAVE_CARGO" ]; then
  skip "CPU-only build: no cargo in this checkout"
elif cargo build --release --no-default-features --quiet 2>&1 | tail -5; then
  echo "no-default-features builds"
else
  echo "no-default-features FAILED"; fail=1
fi

say "Narration port matches its reference"
# The one tier that compares two implementations rather than checking one. `tts-narrate` is
# a port of a thousand regexes and this is the evidence it agrees with the Python; it needs
# both a toolchain (for narrate-diff) and python3 (the reference itself).
if [ -z "$HAVE_CARGO" ]; then
  skip "narration port: no cargo, so narrate-diff cannot be built"
elif ! command -v python3 >/dev/null; then
  skip "narration port: no python3, so there is no reference to compare against"
elif ./scripts/check-narrate.sh >/dev/null 2>&1; then
  echo "tts-narrate matches md-to-narration.py"
else
  ./scripts/check-narrate.sh 2>&1 | tail -20
  echo "narration port FAILED"; fail=1
fi

say "Audio8 fixture gate"
if [ -f fixtures/audio8/oracle.safetensors ] && engine_ready audio8; then
  run audio8-validate || fail=1
else
  skip "fixtures/audio8/oracle.safetensors or references/audio8/weights/codec.safetensors missing"
fi

say "CosyVoice fixture gate"
if [ -f fixtures/cosyvoice/oracle.safetensors ] && engine_ready cosyvoice; then
  run cosyvoice-validate || fail=1
else
  skip "fixtures/cosyvoice/oracle.safetensors or references/cosyvoice/weights missing"
fi

say "Qwen3-TTS gate"
# Two tiers inside one bin: a shape audit that reads only the checkpoint header, then per-stage
# numerics against fixtures/qwen3tts. The numerics tier reports itself as skipped when the
# fixtures are absent. Gated on the talker checkpoint, which is what the audit reads.
if engine_ready qwen3tts; then
  run qwen3tts-validate || fail=1
else
  skip "references/qwen3tts/weights/model.safetensors missing"
fi

say "Kokoro gate"
# A shape audit over the 459 tensors, then every deterministic stage against
# fixtures/kokoro. The last row is the excitation, which is stochastic and is judged by SNR
# against the reference rather than by tolerance — docs/kokoro-model.md says why.
if [ -f fixtures/kokoro/forward.safetensors ] && engine_ready kokoro; then
  run kokoro-validate || fail=1
else
  skip "fixtures/kokoro/forward.safetensors or references/kokoro/weights missing"
fi

say "End-to-end renders"
mkdir -p target/gate
# The second field is the whole voice argument, not a path: kokoro cannot clone, and picks
# one of its built-in style tables instead of loading an asset.
for spec in "qwen3tts:--voice voices/cosy-default-qwen3tts" "kokoro:--set voice=af_heart" \
           "audio8:--voice voices/cosy-default" "cosyvoice:--voice voices/cosy-default-cosyvoice"; do
  id="${spec%%:*}"; voice="${spec##*:}"
  asset="${voice#--voice }"
  if ! engine_ready "$id"; then
    skip "$id: weights not installed (./scripts/bootstrap.sh $id)"
  elif [ "$asset" != "$voice" ] && [ ! -d "$asset" ]; then
    skip "$id: voice asset $asset missing"
  # shellcheck disable=SC2086
  elif ! run dream-tts speak --engine "$id" $voice \
      --text-file examples/senior.txt --out "target/gate/$id.wav"; then
    echo "render FAILED for $id"; fail=1
  fi
done

say "HTTP service smoke test"
# Whichever engine is installed, in catalogue order. Pinning this to cosyvoice made the
# tier skip on every checkout that took the default bootstrap.
smoke_engine=""
for id in qwen3tts kokoro audio8 cosyvoice; do
  if engine_ready "$id"; then smoke_engine="$id"; break; fi
done
smoke_bin=""
if [ -n "$smoke_engine" ]; then
  smoke_bin=$(scripts/run-bin.sh --which dream-tts-serve) || smoke_bin=""
fi
if [ -n "$smoke_bin" ] && [ -x "$smoke_bin" ]; then
  # Not `./dream-tts-serve`: the shim execs cargo, so the pid killed below would be cargo's
  # while the service kept running as its child.
  DREAM_TTS_API_KEY=gate-smoke-key "$smoke_bin" --engine "$smoke_engine" --port 3099 \
      >target/gate/serve.log 2>&1 &
  serve_pid=$!
  for _ in $(seq 1 60); do
    curl -sf -o /dev/null http://127.0.0.1:3099/health && break
    sleep 1
  done
  if curl -sf -o /dev/null http://127.0.0.1:3099/health; then
    code=$(curl -s -o target/gate/http.wav -w '%{http_code}' -X POST http://127.0.0.1:3099/tts \
      -H 'content-type: application/json' -H 'X-API-Key: gate-smoke-key' \
      -d '{"text":"Gate smoke test."}')
    unauth=$(curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:3099/tts \
      -H 'content-type: application/json' -d '{"text":"no key"}')
    if [ "$code" = 200 ] && [ "$unauth" = 401 ] && [ -s target/gate/http.wav ]; then
      echo "POST /tts 200, unauthenticated 401, wav non-empty"
    else
      echo "http smoke FAILED (tts=$code unauth=$unauth)"; fail=1
    fi
  else
    echo "http smoke FAILED: /health never came up"; fail=1
  fi
  kill "$serve_pid" 2>/dev/null
  wait "$serve_pid" 2>/dev/null
elif [ -z "$smoke_engine" ]; then
  skip "http smoke: no engine weights installed"
else
  skip "http smoke: no dream-tts-serve binary (no cargo, and bin/dream-tts-serve absent)"
fi

say "Summary"
[ "$SKIPPED" -gt 0 ] && echo "$SKIPPED tier(s) skipped for missing inputs — see docs/reference.md#setup"
if [ "$fail" -eq 0 ]; then
  echo "everything that ran passed"
else
  echo "FAILURES above"
fi
exit "$fail"
