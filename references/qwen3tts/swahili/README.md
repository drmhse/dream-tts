# Swahili for qwen3tts

Qwen3-TTS has ten languages and Swahili is not one. This teaches the talker Swahili with a LoRA
and a trained language row (codec id 2074), shipped as an adapter the Rust engine overlays at
load. Adapters before run 12 carry an untrained row, the mean of the ten language rows: the prefix
it sits in ran without gradient, which also left the LoRA's effect on the reference encoding
unoptimised. `SPLIT=0` trains through the whole prompt.

```sh
dream-tts speak --engine qwen3tts --voice voices/mc-swahili-qwen3tts \
  --set adapter=references/qwen3tts/weights/swahili-adapter.safetensors --set language=swahili \
  --text-file chapter.md --text-language swahili --out chapter.wav
```

## Rebuilding it

```sh
PY=references/qwen3tts/.venv/bin/python        # plus pyarrow and torchaudio
D=/path/to/scratch
export SCHEME=references/qwen3tts/langs/swahili.json
# 1. broadcast news, any language: see langs/harvest/README.md -> clips-spk.jsonl
$PY references/qwen3tts/swahili/data.py encode $HARVEST_DIR/clips-spk.jsonl $D/news.pt
# 2. studio prompt reads, kept only where they flow (<= 0.10 mid-phrase pauses a word)
$PY references/qwen3tts/swahili/data.py fetch $D && $PY references/qwen3tts/swahili/data.py waxal $D
$PY references/qwen3tts/swahili/data.py align $D
$PY references/qwen3tts/swahili/data.py encode $D/waxal/chunks.jsonl $D/waxal.pt
$PY references/qwen3tts/swahili/fluency.py $D/waxal/chunks.jsonl $D/waxal/fluency.jsonl
$PY references/qwen3tts/swahili/data.py fluent $D/waxal.pt $D/waxal/chunks.jsonl $D/waxal/fluency.jsonl $D/waxal-fl.pt
# 3. train from the base on MLX (finetune.py is the torch reference, same env and checkpoints)
SPLIT=0 CP_LR=0 PAIR_SIM=0.6 SUBW=1 ROW_LR=1 ROW_PULL=0.01 STEPS=2000 LR=5e-5 EXCLUDE=<episode> \
  FOCUS="(?i)ng['’]=4;(^|[.!?] )(M[bv]|N[dgjz])=2" \
  $PY references/qwen3tts/swahili/finetune_mlx.py train $D/news.pt $D/waxal-fl.pt
$PY references/qwen3tts/swahili/finetune.py export swahili-lora-2000.pt \
  references/qwen3tts/weights/swahili-adapter.safetensors 0.75
```

`LEVEL_DB=-20` at encode and `langs/harvest/annotate.py` (ECAPA vector and channel per clip) feed
`PAIR_SIM`: a reference must share the target's channel and match it acoustically.

The export embeds the orthography (`meta::scheme`), so the engine tokenizes as training did.

## What mattered, in order

1. **Flowing speech.** The first adapter (studio prompt reads plus scripture) read word by word:
   12.4 mid-phrase pauses per 100 words on unseen news against the original speaker's 6.8. Its corpus
   paused mid-phrase 15-27 times per 100 words. Trained on 11.5 h of broadcast presenters and reporters
   (500+ speakers) plus the fluent quarter of the prompt reads: 4.3, CER 5.2% -> 4.6%.
2. **Orthography** (`langs/swahili.json`): open syllables; prenasal onsets split off
   (`' m'+'bi'` read "mibili"); j split from its vowel; ng' -> ŋ, one token everywhere.
3. **The engine's lead-in.** Sentence-initial ng' was dropped 15/20 whatever the data; behind
   `... ` 3/20. 71 clips cut to open on ng' (`langs/harvest/onsets.py`) did not fix it and cost
   flow (8.4 pauses per 100 words).
4. **One voice per passage.** Sentences rendered separately sounded like new speakers or rooms:
   level SD 3.7 dB against the base model's 1.35, ECAPA pairwise mean 0.746 against 0.822. News
   pairs mixed studio and field recordings of one reporter, clips differed in level, the code
   predictor learned many channels, and the untrained row sat out of distribution. Run 12
   (`PAIR_SIM=0.6 LEVEL_DB=-20 CP_LR=0 SPLIT=0`): 1.1 dB and 0.825; 0.818 against 0.581 in
   `male-qwen3tts`. An octave-band EQ toward the passage median moved ECAPA 0.735 -> 0.736: the
   drift is voice quality, not coloration, so it is fixed in training, not after.
5. **Gain 0.6** for run 12 (0.75 before the row trained): CER 1.6/5.1/8.3% in the three voices
   against 1.9/10.1/8.8% at 0.75 and 3.2/6.0/11.1% at 1.0. Over full paragraphs the two voices
   with English references read slightly worse than the previous adapter (4.1 and 5.5% against
   3.5 and 2.4%). Over 36 renders the male voice is 6.0% against the previous adapter's 4.1%.
6. **What did not recover it** (both resumed from run 12, measured the same way):
   - `SUB_CHANNEL=studio` (code predictor retrained on studio targets only): CER 2.7/5.5/7.6%,
     ECAPA 0.813/0.723/0.777 against 0.825/0.699/0.818; drift came back, accuracy barely moved.
   - `XREF=0.4` with 418 same-speaker English and code-switched clips as references
     (`langs/harvest/assign_refs.py`): male CER 6.5% over 36 renders, and one sentence at 0.322 to
     the voice's centroid. Cross-lingual pairs sound less alike, so they teach departing from
     the reference: the consistency problem again.
   - `PAIR_SIM=0` (random same-speaker pairs again): male 6.6%, and a 0.254 outlier sentence in
     `cosy-default-qwen3tts`. Nor was it the row (`language=auto` 6.9%, the ten-row mean 7.7%)
     or pace (149 against 145 wpm). Gain 0.45 gives the male voice 5.4%.
   - It is specialisation: run 12's own checkpoints give the male voice 4.7% at step 500, 6.5%
     at 1000, 7.1% at 1500 and 6.0% at 2000, while the owner's voice goes 3.1 -> 1.6% and flow
     7.1 -> 5.0. The earlier checkpoints lose on everything else.
   - 150 Common Voice speakers, loss on codebook 0 only (`cb0_only` rows): male 5.0%,
     `cosy-default` 3.2%, but mid-phrase pauses 9.6 against 5.0 and a 0.302 outlier sentence.
     Read prompts teach halting delivery even through codebook 0.
   Within one passage, as narration renders, run 12 is at base-model consistency in all three
   voices (ECAPA over the passage's sentences 0.782/0.699/0.672 against 0.747/0.707/0.684).
7. **Voices from another language.** Cloning continues the reference clip, accent included:
   with an English reference the adapter spoke Swahili with an English accent, and training could
   not undo it (items 5 and 6). `clone=xvector` (speaker embedding only) takes the accent from the
   adapter: `male-qwen3tts` CER 2.4% over 36 renders against 6.0%.
8. **Generic voices** (`voices/sw-male-qwen3tts`, `voices/sw-female-qwen3tts`): designed with
   Qwen3-TTS VoiceDesign from a text description, spoken in Swahili through the adapter with
   `clone=xvector`, and the most fluent take kept as the voice's own reference. They clone from
   a Swahili clip, so nothing English is inherited: over three paragraphs, sentence-to-sentence
   ECAPA 0.782 and 0.818 with CER 2.6 and 3.1%, against 0.672 and 0.699 for the English-reference
   voices continuing their clips. On the 12 held-out sentences rendered apart: CER 1.6 and 1.7%,
   ECAPA 0.885 and 0.885, level SD 0.96 and 0.67 dB. A chapter aligns 142/142 and 141/142 words.
9. **Loudness.** The model copies its reference's level: the owner's 16 s clip rendered chapters
   at -34.6 dB mean. The engine now lifts a request's median speech level to -20 dBFS after
   synthesis (`level_db`); a levelled re-export of the clip cost consistency (0.825 -> 0.795).

Evaluation: `langs/harvest` holds out one episode; its presenter's sentences are rendered in the
target voice and scored for mid-phrase pauses (MMS alignment), rate and Gemini CER. Sound-level
checks (ng' kept, dropped or turned to k) score the audio against competing spellings under
MMS; on real speech the check agrees 35/36 mid-sentence and 35/40 at a sentence start.

## Data

| source | used | speakers | notes |
|---|---|---|---|
| publicly available TV broadcasts | 11.5 h of ~56 h | 500+ (ECAPA clusters per episode) | studio and field; Gemini transcripts checked by MMS |
| google/WaxalNLP `swa_tts` | 3.8 h of 14.5 | 7 | only clips as fluent as the presenters |

Superseded: OpenBible (Tanzanian, scripture cadence), FLEURS (81 wpm), Common Voice (noisy).

## Speed and memory

`finetune_mlx.py` against `finetune.py` on the same real pairs: losses within 2.1e-3 relative,
gradient norm ratio 1.004; 0.581 against 1.070 s an example (1.84x). With `SPLIT=0` (prefix with
gradient) MLX runs 2.9 s a step at `ACCUM=4`, against torch's 4.2 with the prefix split. Batching
buys little: the trunk at batch 4 is 1.19x per example, for 4x the activations. `encode` peaks at
2.2 GB. Never run two of these at once on 16 GB.
