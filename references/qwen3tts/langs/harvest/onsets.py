"""onsets.py IN.jsonl OUT.jsonl REGEX — clips cut to start at a word matching REGEX, for rare onsets.

Utterance-initial sounds are learned only from clips that open on them: 3 of 6,089 Swahili
training clips began with ng', and the adapter dropped a sentence-initial ng' 12 times in 12
while getting it right mid-sentence. MMS alignment finds each matching word inside longer
clips; the audio from its first letter (less 30 ms) is a new clip whose text starts there.
"""
import sys, os, re, json
import soundfile as sf, torch, torchaudio
from torchaudio.pipelines import MMS_FA
from common import romanize

MIN_S = 1.5
model = MMS_FA.get_model(with_star=False).eval()
vocab, aligner = MMS_FA.get_dict(star=None), MMS_FA.get_aligner()
src, dst, pattern = sys.argv[1:4]
word_re = re.compile(pattern, re.I)
out, n = open(dst, "w"), 0
for r in map(json.loads, open(src)):
    words = r["text"].split()
    keep = [i for i, w in enumerate(words) if re.sub(r"[^a-z]", "", romanize(w))]
    hits = [k for k, i in enumerate(keep) if i > 0 and word_re.match(words[i])]
    if not hits:
        continue
    x, sr = sf.read(r["wav"], dtype="float32")
    x16 = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000)
    toks = [[vocab[c] for c in re.sub(r"[^a-z]", "", romanize(words[i])) if c in vocab] for i in keep]
    with torch.inference_mode():
        em = model(x16[None])[0][0]
    if sum(map(len, toks)) > em.shape[0]:
        continue
    spans, spf = aligner(em, toks), len(x16) / em.shape[0]
    for k in hits:
        t = max(0.0, spans[k][0].start * spf / 16000 - 0.03)
        if len(x) / sr - t < MIN_S:
            continue
        i = keep[k]
        text = " ".join(words[i:])
        text = text[0].upper() + text[1:]
        wav = r["wav"].rsplit(".", 1)[0] + f"_on{i}.wav"
        sf.write(wav, x[int(t * sr):], sr)
        out.write(json.dumps({**r, "wav": wav, "text": text, "seconds": round(len(x) / sr - t, 2), "onset": True},
                             ensure_ascii=False) + "\n")
        n += 1
print(n, "onset clips", dst)
