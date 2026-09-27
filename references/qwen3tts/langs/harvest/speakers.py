"""speakers.py — clips.jsonl -> clips-spk.jsonl: speaker = episode + ECAPA cluster.

ICL pairs must share a voice and an episode mixes anchors, reporters and interviewees. Qwen's
own x-vector cannot tell them apart (different-speaker median cosine 0.954); SpeechBrain ECAPA
can (same 0.70, different p95 0.45, on WAXAL and OpenBible).
"""
import os, json, collections
import numpy as np, soundfile as sf, torch, torchaudio
from speechbrain.inference.speaker import EncoderClassifier

from common import D
THRESH, MIN_CLIPS = 0.5, 4
m = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=os.path.expanduser("~/.cache/speechbrain-ecapa"),
                                   run_opts={"device": "cpu"})
cache_f = f"{D}/ecapa.pt"
cache = torch.load(cache_f) if os.path.exists(cache_f) else {}
rows = [json.loads(l) for l in open(f"{D}/{os.environ.get('IN', 'clips.jsonl')}")]
for r in rows:
    if r["wav"] not in cache:
        w, sr = sf.read(r["wav"], dtype="float32")
        with torch.no_grad():
            cache[r["wav"]] = m.encode_batch(torchaudio.functional.resample(torch.tensor(w), sr, 16000)[None]).reshape(-1)
torch.save(cache, cache_f)


def cluster(X):
    """Average-linkage agglomerative on cosine, stopping below THRESH."""
    groups = [[i] for i in range(len(X))]
    C = X @ X.T
    while len(groups) > 1:
        best, pair = -1, None
        for a in range(len(groups)):
            for b in range(a):
                s = C[np.ix_(groups[a], groups[b])].mean()
                if s > best:
                    best, pair = s, (a, b)
        if best < THRESH:
            break
        a, b = pair
        groups[b] += groups.pop(a)
    return groups


by_ep = collections.defaultdict(list)
for r in rows:
    by_ep[r["speaker"]].append(r)
out, sizes = [], []
for ep, rs in by_ep.items():
    X = torch.nn.functional.normalize(torch.stack([cache[r["wav"]] for r in rs]), dim=1).numpy()
    for k, g in enumerate(sorted(cluster(X), key=len, reverse=True)):
        if len(g) < MIN_CLIPS:
            continue
        sizes.append(len(g))
        out += [{**rs[i], "speaker": f"{ep}:{k}"} for i in g]
with open(f"{D}/{os.environ.get('OUT', 'clips-spk.jsonl')}", "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{len(rows)} clips -> {len(out)} in {len(sizes)} speakers ({sum(r['seconds'] for r in out) / 3600:.2f} h); "
      f"largest {sorted(sizes, reverse=True)[:8]}")
