# Kokoro's English frontend, in Rust

Kokoro does not take text. It takes IPA phoneme ids produced by `misaki`, which is spaCy's
tokenizer, spaCy's POS tagger, a 90k-entry lexicon and ~700 lines of rules — and for words
outside the lexicon, espeak-ng. espeak-ng is GPL-3.0 and would not survive the release
audit, so the port is lexicon-first and reports what it cannot say rather than guessing.

The frontend is three ports with three unrelated failure modes, so `scripts/check-phonemes.sh`
gates them separately. An end-to-end phoneme diff says a line is wrong and never says which
of the three moved.

Current state: tokenizer, tagger and phonemes are all **identical to the pinned Python over
522,542 tokens** of real narration plus the trap cases.

## The rules are exported, not retyped

`references/kokoro/export_frontend.py` writes `weights/frontend/`: three regexes and 1,347
exception entries for the tokenizer, 1.6M parameters for the tagger, and misaki's lexicons.
Transcribing spaCy's tokenizer rules by hand would produce something that tokenizes English
plausibly and disagrees with the tagger's training distribution in ways no test names.

`references/kokoro/tagger_numpy.py` is the executable spec — the same model in numpy, from
the same exported files, proved against spaCy with `--check`. It exists because every trap
below is easier to find in twenty lines of numpy than in Rust.

## Five traps, each of which runs and is wrong

1. **thinc's `ops.hash` is not MurmurHash3_x86_128**, despite the name. No blocks, no tail,
   64-bit constants, fmix64. Implementing the published algorithm gives plausible numbers
   and the wrong embedding row for every token.
2. **`with_array` pads the residual stack, once, around the whole thing** (`pad: 4`). The pad
   rows start at zero but the first layer's bias makes them non-zero, so from layer 1 the
   edge tokens see a real neighbour. Padding each window layer independently instead is
   wrong only at the two ends — and then diverges through the stack, which presents as a
   precision problem (max diff 1.6 on a 96-wide vector) and is not one. Tags were 99.2%
   right, which is the worst possible symptom: good enough to look like rounding.
3. **StringStore resolves symbol names to small integer ids, not hashes.** Every empty
   string and any shape that collides with a symbol name — `X` is `101` — takes this path.
4. **`lexeme_norm` is keyed by the hash of the orth, not by the word.** A table keyed by
   string compiles, loads, and never hits. NORM has four sources in order: the tokenizer
   exception's own NORM, `BASE_NORMS` (currency symbols and smart quotes folded together,
   which is why `£` norms to `$`), `lexeme_norm`, then `lower()`.
5. **Affix splitting is not the whole tokenizer.** `id.` splits to `i d .` because `id` is
   the apostrophe-less contraction rule; a second pass then matches special keys against the
   *split* stream and puts `d.` back together. Skipping it costs 23 lines in 12,719 — all of
   them the same abbreviation — which is few enough to dismiss as noise and is a missing
   algorithm.

`SPACY` and `IS_SPACE` are raw flags, not string hashes. Hashing them gives a model that
runs and is wrong on every token.

## Out-of-lexicon words

There is no espeak fallback and there will not be one. What there is instead is a derivation
stage, kept **separate from the misaki-identical core** so the byte-identity gate keeps
meaning: it only ever runs where misaki returned nothing, and `phoneme-validate` checks
exactly that by comparing the strict path and the derived path separately.

On a technical corpus — the worst case for a lexicon — it takes **2,037 unknown occurrences
down to 612** out of 522,542 tokens. Four rule families do it: `-ied` and friends,
comparatives, productive prefixes, and compound splitting. Two things that had to be right:

- **Inflection sits outside the compound.** "namespaces" is name+space+s. Splitting the
  inflected form directly finds names+paces, which is two real words and the wrong two.
- **Prefixes come from a table, not from the lexicon.** Its `re` is the musical note (`ɹˌA`)
  and its `inter` is the verb (`ɪntˈɜɹ`), so looking them up yields a word that is
  confidently and audibly wrong. Compound splits prefer the most balanced, then the shorter
  head, which is what makes "namespace" name+space rather than names+pace.

What remains needs letter-to-sound, not more rules, and is reported by name rather than
guessed at. The lexicon itself is training data for that, if it is ever worth doing.

## What is not done

- Throughput. 5,250 tokens/s, dominated by the tagger's unbatched maxout. Kokoro's model is
  fast enough that the frontend can become the bottleneck, so this will need the projections
  batched into one GEMM per layer.
