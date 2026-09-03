#!/usr/bin/env bash
#
# Prove `crates/tts-narrate` matches `scripts/md-to-narration.py`, byte for byte.
#
#   scripts/check-narrate.sh              every function over the built-in corpus
#   scripts/check-narrate.sh clean_inline just that one
#
# A port of a thousand regexes is only as good as the evidence that it agrees with the
# original, and hand-written cases only cover what the author thought to write. So the corpus
# is deliberately hostile: every construct each rule targets, plus the real chapter in
# prep-handbook, plus randomised recombinations of the constructs so the interactions between
# rules are exercised too — most of the traps recorded in the Python are interactions.
#
# The Python stays in the tree as the reference. This is what lets it.
set -euo pipefail

cd "$(dirname "$0")/.."

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

command -v python3 >/dev/null || die "python3 not found; the oracle is the Python implementation"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

BIN="$(scripts/run-bin.sh --which narrate-diff)" || die \
  "could not build narrate-diff. It is a test binary and is not shipped in a release
   archive, so this check needs a source checkout with a toolchain."

FUNCTIONS="${*:-speak_code speak_math speak_numbers clean_inline convert page_text}"

python3 scripts/narrate-corpus.py "$WORK" || die "corpus generation failed"

fail=0
for fn in $FUNCTIONS; do
  cases="$WORK/$fn.json"
  [ -f "$cases" ] || cases="$WORK/lines.json"
  [ "$fn" = convert ] && cases="$WORK/documents.json"
  [ "$fn" = page_text ] && cases="$WORK/documents.json"

  python3 scripts/narrate-oracle.py "$fn" <"$cases" >"$WORK/$fn.py.json"
  "$BIN" "$fn" <"$cases" >"$WORK/$fn.rs.json"

  if cmp -s "$WORK/$fn.py.json" "$WORK/$fn.rs.json"; then
    n=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$cases")
    printf '  %-14s %4d case(s) identical\n' "$fn" "$n"
  else
    printf '\033[31m  %-14s DIFFERS\033[0m\n' "$fn"
    python3 - "$WORK/$fn.py.json" "$WORK/$fn.rs.json" "$cases" <<'PY'
import json, sys
py, rs, cases = (json.load(open(p)) for p in sys.argv[1:4])
shown = 0
for i, (a, b) in enumerate(zip(py, rs)):
    if a != b:
        shown += 1
        if shown > 8:
            print("    ... and more")
            break
        print(f"    case {i}: {cases[i]!r}")
        print(f"      python {a!r}")
        print(f"      rust   {b!r}")
PY
    fail=1
  fi
done

# The word mapping, which is the alignment port rather than a rule port.
python3 - "$WORK" <<'PY'
import json, subprocess, sys
from pathlib import Path
work = Path(sys.argv[1])
docs = json.load(open(work / "documents.json"))
sys.path.insert(0, "scripts")
import importlib.util
spec = importlib.util.spec_from_file_location("m", "scripts/md-to-narration.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
pairs = []
for d in docs:
    spoken = [m.normalize_word(w) for w in m.WORD.findall(m.convert(d))]
    page = [m.normalize_word(w) for w in m.WORD.findall(m.page_text(d))]
    pairs.append([spoken, page])
(work / "align.json").write_text(json.dumps(pairs))
PY
python3 scripts/narrate-oracle.py align <"$WORK/align.json" >"$WORK/align.py.json"
"$BIN" align <"$WORK/align.json" >"$WORK/align.rs.json"
if cmp -s "$WORK/align.py.json" "$WORK/align.rs.json"; then
  printf '  %-14s %4d document(s) identical\n' "align" \
    "$(python3 -c "import json;print(len(json.load(open('$WORK/align.json'))))")"
else
  printf '\033[31m  %-14s DIFFERS\033[0m\n' "align"
  fail=1
fi

[ "$fail" -eq 0 ] || die "the Rust port does not match the Python reference"
say "tts-narrate matches md-to-narration.py"
