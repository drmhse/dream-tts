#!/usr/bin/env python3
"""Ground truth for the phonemizer gate: spaCy's tokens, spaCy's tags, misaki's phonemes.

Three stages per line rather than one, because a single end-to-end string cannot say
whether a mismatch came from the tokenizer, the tagger or the lexicon — and those are
three separate ports with three separate failure modes.
"""
import argparse, json, sys
from misaki import en

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus', help='one input per line, blank lines skipped')
    ap.add_argument('-o', '--out', default='-')
    ap.add_argument('--british', action='store_true')
    args = ap.parse_args()

    g = en.G2P(trf=False, british=args.british, fallback=None, unk='❓')
    out = sys.stdout if args.out == '-' else open(args.out, 'w')
    n = 0
    for line in open(args.corpus):
        line = line.rstrip('\n')
        if not line.strip():
            continue
        ps, tokens = g(line)
        # spaCy's own view, before misaki's retokenize splits anything further.
        doc = g.nlp(en.G2P.preprocess(line)[0])
        rec = {
            'text': line,
            'spacy': [{'t': t.text, 'tag': t.tag_, 'ws': t.whitespace_} for t in doc],
            'phonemes': ps,
            'tokens': [{'t': t.text, 'tag': t.tag, 'ws': t.whitespace,
                        'ps': t.phonemes, 'rating': t._.rating} for t in tokens],
        }
        out.write(json.dumps(rec, ensure_ascii=False) + '\n')
        n += 1
    if out is not sys.stdout:
        out.close()
    print(f'{n} records -> {args.out}', file=sys.stderr)

if __name__ == '__main__':
    main()
