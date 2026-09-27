"""vad.py EPISODE.m4a... — Silero VAD chunks of 2-15 s cut at pauses, into candidates.jsonl (no draft).

The CPU replacement for segment.py's Whisper pass (6 s an episode against 4 min on the GPU the
LoRA needs). Text comes from verify.py; mms.py then checks it against the audio.
"""
import sys, os, json, subprocess
import numpy as np, soundfile as sf, torch
from silero_vad import load_silero_vad, get_speech_timestamps

from common import D, load
m = load_silero_vad()


def spans(x, a0=0.0, silence=250):
    ts = get_speech_timestamps(torch.from_numpy(x), m, sampling_rate=16000, min_silence_duration_ms=silence,
                               speech_pad_ms=60, return_seconds=True)
    out = []
    for t in ts:
        a, b = t["start"] + a0, t["end"] + a0
        if b - a > 15 and silence > 100:
            out += spans(x[int((a - a0) * 16000): int((b - a0) * 16000)], a, 100)
        else:
            out.append((a, b))
    return out


done = set()
if os.path.exists(f"{D}/candidates.jsonl"):
    done = {json.loads(l)["episode"] for l in open(f"{D}/candidates.jsonl")}
out = open(f"{D}/candidates.jsonl", "a")
for path in sys.argv[1:]:
    ep = os.path.basename(path).rsplit(".", 1)[0]
    if ep in done:
        continue
    x16, x24 = load(path, 16000), load(path, 24000)
    chunks, cur = [], []
    for a, b in spans(x16):
        if cur and (a - cur[-1][1] > 0.35 or b - cur[0][0] > 15):
            chunks.append((cur[0][0], cur[-1][1])); cur = []
        cur.append((a, b))
    if cur:
        chunks.append((cur[0][0], cur[-1][1]))
    n = 0
    for a, b in chunks:
        if not 2.0 <= b - a <= 15.0:
            continue
        fn = f"{D}/chunks/{ep}_{n:04d}.wav"
        sf.write(fn, x24[max(0, int((a - 0.05) * 24000)): int((b + 0.1) * 24000)], 24000)
        out.write(json.dumps({"episode": ep, "wav": fn, "start": round(a, 2), "end": round(b, 2), "draft": "",
                              "src": "vad"}) + "\n")
        n += 1
    out.flush()
    print(ep, f"{len(x16) / 16000 / 60:.0f} min ->", n, "chunks", flush=True)
