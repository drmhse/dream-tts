#!/usr/bin/env python3
"""Number-word ground truth.

Committed, unlike the activation fixtures: these are facts about English rather than about
the pinned checkpoint, so `cargo test` can check them with no venv and no download. Kept to
the structurally interesting values plus a small seeded sample — the exhaustive sweep runs
in `scripts/check-phonemes.sh`, where the Python is available to say what differs.
"""
import json, random, sys
from num2words import num2words

def main(out, full=False):
    # Every carry boundary, every scale word, and the two-digit and three-digit joins.
    vals = set(range(0, 130)) | set(range(95, 1101))
    vals |= {10 ** k for k in range(3, 16)}
    vals |= {10 ** k + d for k in range(3, 13) for d in (1, 7, 100, 999, 1000)}
    vals |= {v * 10 ** k for v in (2, 9, 21, 101, 110) for k in range(3, 12)}
    random.seed(20260914)
    vals |= {random.randrange(0, 10 ** 15) for _ in range(3000 if full else 250)}
    vals |= {-v for v in sorted(vals)[:60]}
    floats = ['3.5', '0.25', '12.30', '2.50', '0.05', '1.000', '1234.5678', '0.5', '100.75',
              '-3.25', '7.0', '0.0', '.5', '12.05']
    records = []
    for v in sorted(vals):
        rec = {'n': v, 'card': num2words(v)}
        if v >= 0:
            rec['ord'] = num2words(v, to='ordinal')
            rec['year'] = num2words(v, to='year')
        records.append(rec)
    records += [{'s': s, 'dec': num2words(float(s))} for s in floats]
    with open(out, 'w') as f:
        json.dump(records, f)
    print(f'{len(vals)} integers + {len(floats)} decimals -> {out}', file=sys.stderr)

if __name__ == '__main__':
    argv = [a for a in sys.argv[1:] if a != '--full']
    main(argv[0] if argv else 'fixtures/kokoro/numbers.json', full='--full' in sys.argv)
