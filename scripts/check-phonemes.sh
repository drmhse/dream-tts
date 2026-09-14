#!/usr/bin/env bash
# The frontend gate: the Rust tokenizer and POS tagger against the pinned spaCy, and the
# G2P against misaki, over the narration corpus plus the hand-written trap cases.
#
# Staged rather than end-to-end: a phoneme mismatch alone cannot say whether the split, the
# tag or the lexicon rule moved, and those are three unrelated ports.
set -euo pipefail
cd "$(dirname "$0")/.."

REF=references/kokoro
VENV="$REF/.venv/bin/python"
FRONTEND="$REF/weights/frontend"
WORK="${TMPDIR:-/tmp}/dream-tts-phonemes"
mkdir -p "$WORK"

[ -x "$VENV" ] || { echo "missing $VENV — see $REF/requirements.txt"; exit 1; }
[ -f "$FRONTEND/tokenizer.json" ] || { echo "missing $FRONTEND — run $REF/export_frontend.py"; exit 1; }

corpus="$WORK/corpus.txt"
if [ ! -s "$corpus" ]; then
  # Real narration output is the input distribution the engine actually sees: numbers,
  # units and currency have already been verbalised by tts-narrate before this stage.
  cat narration*/*.txt prep-handbook/*.txt 2>/dev/null \
    | tr -s ' ' | sed '/^$/d' | awk 'length($0)>0 && length($0)<2000' > "$corpus"
  cat "$REF/traps.txt" >> "$corpus"
fi

oracle="$WORK/oracle.jsonl"
if [ ! -s "$oracle" ] || [ "$corpus" -nt "$oracle" ]; then
  echo "dumping oracle over $(wc -l < "$corpus" | tr -d ' ') lines"
  "$VENV" "$REF/dump_g2p.py" "$corpus" -o "$oracle"
fi

# The exhaustive number sweep. The committed fixture only samples it, so that `cargo test`
# needs no venv; here the Python is available, so check the whole thing.
"$VENV" "$REF/dump_numbers.py" "$WORK/numbers.json" --full
DREAM_TTS_NUMBERS="$WORK/numbers.json" cargo test --release -p tts-phoneme

cargo build --release -p tts-phoneme
# Capture rather than pipe: this binary restores the default SIGPIPE handler, so a reader
# that leaves early kills it and `set -o pipefail` fails a run that actually passed.
out="$(./target/release/phoneme-validate "$oracle" "$FRONTEND")"
echo "$out"
