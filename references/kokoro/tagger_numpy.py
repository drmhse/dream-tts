#!/usr/bin/env python3
"""A dependency-free reimplementation of spaCy's tagger, from the exported files.

This exists to pin the semantics before any Rust is written: it is the executable spec for
the port, and `--check` proves it against spaCy itself. Everything here — the shape string,
the two different Murmur variants, the maxout window encoder — is a place the Rust can go
quietly wrong, so each is reproduced in the smallest code that can be read against spaCy's.
"""
import argparse, json, os
import numpy as np

MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1


def murmur64a(data: bytes, seed: int = 1) -> int:
    """MurmurHash64A — spaCy's `hash_string`, which produces the StringStore key."""
    m, r = 0xc6a4a7935bd1e995, 47
    h = (seed ^ ((len(data) * m) & MASK64)) & MASK64
    n = len(data) - len(data) % 8
    for i in range(0, n, 8):
        k = int.from_bytes(data[i:i + 8], 'little')
        k = (k * m) & MASK64
        k ^= k >> r
        k = (k * m) & MASK64
        h ^= k
        h = (h * m) & MASK64
    tail = data[n:]
    if tail:
        h ^= int.from_bytes(tail + b'\x00' * (8 - len(tail)), 'little')
        h = (h * m) & MASK64
    h ^= h >> r
    h = (h * m) & MASK64
    h ^= h >> r
    return h


def thinc_hash(key: int, seed: int):
    """thinc's `ops.hash` — four buckets from one 64-bit key.

    Named x86_128 in thinc, but it is neither: no blocks, no tail, 64-bit constants and
    fmix64. Reimplementing the published MurmurHash3 here produces plausible-looking
    numbers that are simply wrong, which is why this is transcribed rather than recalled.
    """
    c1, c2 = 0x87c37b91114253d5, 0x4cf5ad432745937f

    def fmix64(h):
        h ^= h >> 33
        h = (h * 0xff51afd7ed558ccd) & MASK64
        h ^= h >> 33
        h = (h * 0xc4ceb9fe1a85ec53) & MASK64
        return h ^ (h >> 33)

    h1 = (key * c1) & MASK64
    h1 = ((h1 << 31) | (h1 >> 33)) & MASK64
    h1 = (h1 * c2) & MASK64
    h1 ^= seed
    h1 ^= 8
    h2 = (seed ^ 8) & MASK64
    h1 = (h1 + h2) & MASK64
    h2 = (h2 + h1) & MASK64
    h1, h2 = fmix64(h1), fmix64(h2)
    h1 = (h1 + h2) & MASK64
    h2 = (h2 + h1) & MASK64
    return [h1 & MASK32, h1 >> 32, h2 & MASK32, h2 >> 32]


def shape(text: str) -> str:
    """spaCy's word shape: character classes, with runs capped at four."""
    if len(text) >= 100:
        return 'LONG'
    out = []
    run = 0
    prev = ''
    for c in text:
        if c.isalpha():
            s = 'X' if c.isupper() else 'x'
        elif c.isdigit():
            s = 'd'
        else:
            s = c
        run = run + 1 if s == prev else 1
        prev = s
        if run <= 4:
            out.append(s)
    return ''.join(out)


def norm_of(text: str, exc_norm, lexeme_norm):
    if exc_norm is not None:
        return exc_norm
    lower = text.lower()
    return lexeme_norm.get(lower, lower)


class Tagger:
    def __init__(self, frontend_dir):
        from safetensors.numpy import load_file
        self.meta = json.load(open(os.path.join(frontend_dir, 'tagger.json')))
        self.t = load_file(os.path.join(frontend_dir, 'tagger.safetensors'))
        tok = json.load(open(os.path.join(frontend_dir, 'tokenizer.json')))
        self.lexeme_norm = tok['lexeme_norm']
        self.symbols = tok['symbols']

    def string_id(self, s: str) -> int:
        sym = self.symbols.get(s)
        return sym if sym is not None else murmur64a(s.encode())

    def attr_ids(self, tokens):
        """One uint64 key per (token, attr). SPACY and IS_SPACE are the raw flags, not
        hashes — feeding them through the string hash gives a working model that is
        wrong on every token."""
        out = np.zeros((len(tokens), 6), dtype=np.uint64)
        for i, tk in enumerate(tokens):
            text, ws, exc_norm = tk['t'], tk['ws'], tk.get('norm')
            out[i, 0] = self.string_id(norm_of(text, exc_norm, self.lexeme_norm))
            out[i, 1] = self.string_id(text[:1])
            out[i, 2] = self.string_id(text[-3:])
            out[i, 3] = self.string_id(shape(text))
            out[i, 4] = 1 if ws else 0
            out[i, 5] = 1 if text.isspace() else 0
        return out

    def embed(self, ids):
        cols = []
        for j, attr in enumerate(self.meta['attrs']):
            E = self.t[f'embed.{attr}.E']
            nV, seed = self.meta['rows'][j], self.meta['seeds'][j]
            out = np.zeros((ids.shape[0], E.shape[1]), dtype=np.float32)
            for i, key in enumerate(ids[:, j].tolist()):
                for h in thinc_hash(int(key), seed):
                    out[i] += E[h % nV]
            cols.append(out)
        return np.concatenate(cols, axis=1)

    @staticmethod
    def maxout(X, W, b):
        nO, nP, nI = W.shape
        Y = X @ W.reshape(nO * nP, nI).T + b.reshape(nO * nP)
        return Y.reshape(-1, nO, nP).max(axis=2)

    @staticmethod
    def layernorm(X, G, b):
        mu = X.mean(axis=1, keepdims=True)
        var = X.var(axis=1, keepdims=True) + 1e-8
        return ((X - mu) * var ** -0.5) * G + b

    @staticmethod
    def expand_window(X):
        pad = np.zeros((1, X.shape[1]), dtype=X.dtype)
        prev = np.concatenate([pad, X[:-1]], axis=0)
        nxt = np.concatenate([X[1:], pad], axis=0)
        return np.concatenate([prev, X, nxt], axis=1)

    def tok2vec(self, tokens):
        X = self.embed(self.attr_ids(tokens))
        X = self.layernorm(self.maxout(X, self.t['proj.W'], self.t['proj.b']),
                           self.t['proj.ln.G'], self.t['proj.ln.b'])
        # thinc's with_array wraps the whole residual stack in `pad` zero rows and strips
        # them at the end. The rows start at zero but the first layer's bias makes them
        # non-zero, so from layer 1 on the edge tokens see a real neighbour. Padding each
        # layer independently instead is wrong only at the two ends — and then diverges
        # across the stack, which reads as a precision problem and is not one.
        pad = self.meta['pad']
        P = np.zeros((pad, X.shape[1]), dtype=X.dtype)
        X = np.concatenate([P, X, P], axis=0)
        for i in range(self.meta['depth']):
            Y = self.maxout(self.expand_window(X), self.t[f'enc.{i}.W'], self.t[f'enc.{i}.b'])
            X = X + self.layernorm(Y, self.t[f'enc.{i}.ln.G'], self.t[f'enc.{i}.ln.b'])
        return X[pad:len(X) - pad]

    def __call__(self, tokens):
        logits = self.tok2vec(tokens) @ self.t['tagger.W'].T + self.t['tagger.b']
        return [self.meta['labels'][i] for i in logits.argmax(axis=1)], logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus')
    ap.add_argument('--frontend', default='weights/frontend')
    ap.add_argument('--limit', type=int, default=200)
    args = ap.parse_args()

    import spacy
    nlp = spacy.load('en_core_web_sm', enable=['tok2vec', 'tagger'])
    tagger = Tagger(args.frontend)

    n = tags_ok = tags_n = 0
    worst_vec = 0.0
    norm_derived_ok = norm_n = 0
    for line in open(args.corpus):
        line = line.strip()
        if not line or n >= args.limit:
            continue
        doc = nlp(line)
        tokens = [{'t': t.text, 'ws': t.whitespace_, 'norm': t.norm_} for t in doc]
        mine, _ = tagger(tokens)
        for t, m in zip(doc, mine):
            tags_n += 1
            tags_ok += t.tag_ == m
        # Does NORM need the exception table, or is lookup+lower enough?
        for t in doc:
            norm_n += 1
            norm_derived_ok += norm_of(t.text, None, tagger.lexeme_norm) == t.norm_
        ref = nlp.get_pipe('tok2vec').model.predict([doc])[0]
        worst_vec = max(worst_vec, float(np.abs(ref - tagger.tok2vec(tokens)).max()))
        n += 1

    print(f'{n} lines, {tags_n} tokens')
    print(f'  tags identical   {tags_ok}/{tags_n}  ({100*tags_ok/tags_n:.3f}%)')
    print(f'  tok2vec max diff {worst_vec:.3e}')
    print(f'  NORM without the exception table {norm_derived_ok}/{norm_n}')
    return 0 if tags_ok == tags_n else 1


if __name__ == '__main__':
    raise SystemExit(main())
