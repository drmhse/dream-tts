#!/usr/bin/env python3
"""Export the English frontend — tokenizer rules, POS tagger, misaki lexicon — as data.

The rules are *exported*, not retyped. spaCy's tokenizer is three compiled regexes plus an
exception table, and its tagger is 1.6M parameters; transcribing either by hand would drift
from the pinned model with no gate able to say when.

Writes into weights/frontend/:
  tokenizer.json   prefix/suffix/infix/url patterns, exceptions, lexeme norms
  tagger.json      labels, dims, hashembed seeds and table sizes
  tagger.safetensors
  lexicon-us.json  misaki's gold+silver, already grown (capitalised variants materialised)
"""
import argparse, json, os
import numpy as np

ATTRS = ['NORM', 'PREFIX', 'SUFFIX', 'SHAPE', 'SPACY', 'IS_SPACE']


def export_tokenizer(nlp, out):
    import srsly
    d = srsly.msgpack_loads(nlp.tokenizer.to_bytes())
    # Exception values keep only ORTH and NORM: the tagger reads NORM, and every other
    # attribute in the table is unused downstream.
    # msgpack keys are spaCy's numeric attribute ids, not names.
    ORTH, NORM = 65, 67
    exceptions = {
        k: [{'orth': t[ORTH], 'norm': t.get(NORM)} for t in v]
        for k, v in d['exceptions'].items()
    }
    norms = dict(nlp.vocab.lookups.get_table('lexeme_norm'))
    # StringStore resolves a symbol name to its small integer id, not to a hash. Shapes
    # like "X" and every empty string land here, so leaving the table out corrupts the
    # embedding lookup for a minority of tokens and nothing says so.
    from spacy.symbols import IDS
    symbols = dict(IDS)
    # NORM has three sources and lower() as the floor. BASE_NORMS folds currency symbols
    # and smart quotes onto one representative; lexeme_norm is keyed by the *hash* of the
    # orth, not by the word, so a table keyed by string silently never hits.
    from spacy.lang.norm_exceptions import BASE_NORMS
    base_norms = dict(BASE_NORMS)
    blob = {
        'prefix_search': d['prefix_search'],
        'suffix_search': d['suffix_search'],
        'infix_finditer': d['infix_finditer'],
        'url_match': d['url_match'],
        'token_match': d['token_match'],
        'faster_heuristics': d['faster_heuristics'],
        'exceptions': exceptions,
        'lexeme_norm': norms,
        'symbols': symbols,
        'base_norms': base_norms,
    }
    with open(os.path.join(out, 'tokenizer.json'), 'w') as f:
        json.dump(blob, f, ensure_ascii=False)
    return len(exceptions), len(norms), len(symbols), len(base_norms)


def _maxout(m):
    return {'W': m.get_param('W'), 'b': m.get_param('b')}


def _layernorm(m):
    return {'G': m.get_param('G'), 'b': m.get_param('b')}


def export_tagger(nlp, out):
    from safetensors.numpy import save_file
    t2v = nlp.get_pipe('tok2vec').model
    embed_stack, proj_block = t2v.layers[0].layers[2].layers[0], t2v.layers[0].layers[3]
    encoders = t2v.layers[1].layers[0]

    tensors, meta = {}, {'attrs': ATTRS, 'seeds': [], 'rows': [], 'width': t2v.get_dim('nO')}
    for attr, branch in zip(ATTRS, embed_stack.layers):
        he = branch.layers[1]
        tensors[f'embed.{attr}.E'] = np.ascontiguousarray(he.get_param('E'), dtype=np.float32)
        meta['seeds'].append(int(he.attrs['seed']))
        meta['rows'].append(int(he.get_dim('nV')))

    mx = proj_block.layers[0].layers[0].layers[0]
    ln = proj_block.layers[0].layers[0].layers[1]
    tensors['proj.W'] = np.ascontiguousarray(mx.get_param('W'), dtype=np.float32)
    tensors['proj.b'] = np.ascontiguousarray(mx.get_param('b'), dtype=np.float32)
    tensors['proj.ln.G'] = np.ascontiguousarray(ln.get_param('G'), dtype=np.float32)
    tensors['proj.ln.b'] = np.ascontiguousarray(ln.get_param('b'), dtype=np.float32)

    for i, res in enumerate(encoders.layers):
        inner = res.layers[0]           # expand_window >> maxout >> layernorm >> dropout
        mx, ln = inner.layers[1].layers[0].layers[0], inner.layers[1].layers[0].layers[1]
        tensors[f'enc.{i}.W'] = np.ascontiguousarray(mx.get_param('W'), dtype=np.float32)
        tensors[f'enc.{i}.b'] = np.ascontiguousarray(mx.get_param('b'), dtype=np.float32)
        tensors[f'enc.{i}.ln.G'] = np.ascontiguousarray(ln.get_param('G'), dtype=np.float32)
        tensors[f'enc.{i}.ln.b'] = np.ascontiguousarray(ln.get_param('b'), dtype=np.float32)
    meta['depth'] = len(encoders.layers)
    # with_array pads the residual stack, and the pad rows stop being zero after the first
    # layer. Exported rather than assumed equal to depth.
    meta['pad'] = int(t2v.layers[1].attrs['pad'])

    tagger = nlp.get_pipe('tagger')
    sm = tagger.model.layers[1].layers[0]
    tensors['tagger.W'] = np.ascontiguousarray(sm.get_param('W'), dtype=np.float32)
    tensors['tagger.b'] = np.ascontiguousarray(sm.get_param('b'), dtype=np.float32)
    meta['labels'] = list(tagger.labels)

    save_file(tensors, os.path.join(out, 'tagger.safetensors'))
    with open(os.path.join(out, 'tagger.json'), 'w') as f:
        json.dump(meta, f)
    return {k: list(v.shape) for k, v in tensors.items()}


def export_lexicon(out, british=False):
    import importlib.resources
    from misaki import data
    blob = {}
    for tier in ('gold', 'silver'):
        # Exported ungrown: `grow_dictionary` is deterministic and doubles the asset, so the
        # Rust side derives the capitalised variants at load instead of downloading them.
        with importlib.resources.open_text(data, f"{'gb' if british else 'us'}_{tier}.json") as r:
            blob[tier] = json.load(r)
    name = 'lexicon-gb.json' if british else 'lexicon-us.json'
    with open(os.path.join(out, name), 'w') as f:
        json.dump(blob, f, ensure_ascii=False)
    return {k: len(v) for k, v in blob.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default='weights/frontend')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import spacy
    nlp = spacy.load('en_core_web_sm', enable=['tok2vec', 'tagger'])
    print('tokenizer:', export_tokenizer(nlp, args.out), '(exceptions, lexeme norms, symbols, base norms)')
    for k, v in export_tagger(nlp, args.out).items():
        print(f'  {k:<20} {v}')
    print('lexicon us:', export_lexicon(args.out, british=False))
    print('lexicon gb:', export_lexicon(args.out, british=True))


if __name__ == '__main__':
    main()
