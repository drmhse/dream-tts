# Harvesting a language from broadcast audio

Stages, each resumable and on CPU, so the GPU stays free for the LoRA:

```sh
export HARVEST=references/qwen3tts/langs/swahili-harvest.json HARVEST_DIR=/path/to/data
PY=/path/to/venv/bin/python   # silero-vad, speechbrain, torchaudio, soundfile
$PY fetch.py                  # searches -> raw/*.m4a
$PY vad.py $HARVEST_DIR/raw/*.m4a      # 2-15 s chunks cut at pauses, 6 s an episode
BATCH=4 THINK=low python3 verify.py    # Gemini: transcript + labels; keys in $GEMINI_KEYS_DIR/gemini_api_key*.txt
$PY mms.py                    # CTC check of each text against its audio
python3 pick.py               # filters -> clips.jsonl, and disagree.txt for a second pass
$PY speakers.py               # ECAPA clusters within an episode -> clips-spk.jsonl
python3 part.py 0             # fully processed episodes -> part-0.jsonl, for data.py encode
```

Then `SCHEME=langs/<language>.json data.py encode part-0.jsonl out.pt` (swahili/data.py).

What each check is worth, measured on Swahili TV broadcasts:

- **Gemini in batches.** 15 clips a request shifted texts by one clip in runs (31% of clips);
  4 a request with low thinking recovered 30 of 40 rejects. Quota is per key *and* model, so
  `verify.py` runs a worker per pair.
- **MMS gap** (best-path minus forced-path log-prob per frame): correct texts p95 0.39, another
  clip's text min 0.82; gate 0.5. 0.5 s a clip on CPU.
- **Speakers.** Qwen's x-vector does not separate voices (different-speaker median cosine
  0.954); SpeechBrain ECAPA does (same 0.70, different p95 0.45); threshold 0.5.
- **Fluency.** Prompt-read corpora pause mid-phrase 15-27 times per 100 words; broadcast presenters 10.
  A talker trained on the former reads word by word (`swahili/fluency.py`).

A new language needs a harvest config (sources, prompt names, accent to keep; the channel,
title prefix and queries go in a git-ignored `<language>-harvest.local.json`) and an
orthography scheme (`langs/<language>.json`, see `orthography.py`).
