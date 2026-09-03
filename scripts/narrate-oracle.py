#!/usr/bin/env python3
"""The reference implementation, exposed for differential testing.

`crates/tts-narrate` is a port of `md-to-narration.py`, and a port of a thousand regexes is
only as good as the evidence that it agrees with the original. This applies one function of
the Python to a JSON array of cases on stdin and writes a JSON array of results, so
`scripts/check-narrate.sh` can compare the two implementations byte for byte.

    echo '["7B parameters"]' | scripts/narrate-oracle.py clean_inline

Not part of the runtime. The Rust is what narrates; this exists to prove it matches.
"""
import importlib.util
import json
import sys


def emit(value):
    """Compact and non-escaping, to match `serde_json::to_string` byte for byte.

    Python defaults to `", "` separators and `\\uXXXX` escapes; serde_json does neither, and
    `check-narrate.sh` compares bytes. Without this the harness reports every case as
    differing on formatting alone.
    """
    print(json.dumps(value, separators=(",", ":"), ensure_ascii=False))
from pathlib import Path
spec = importlib.util.spec_from_file_location("m", Path("scripts/md-to-narration.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
fn = sys.argv[1]
data = json.loads(sys.stdin.read())
if fn == "speak_code":
    emit([m.speak_code(x) for x in data])
elif fn == "clean_inline":
    emit([m.clean_inline(x) for x in data])
elif fn == "convert":
    emit([m.convert(x) for x in data])
elif fn == "page_text":
    emit([m.page_text(x) for x in data])
elif fn == "align":
    emit([m.align_tokens(a, b) for a, b in data])
elif fn == "speak_math":
    emit([m.speak_math(x) for x in data])
elif fn == "speak_numbers":
    emit([m.speak_numbers(x) for x in data])
