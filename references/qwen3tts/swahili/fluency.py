"""fluency.py IN.jsonl OUT.jsonl — per clip: pauses >= 0.15 s mid-phrase (no punctuation before) and
at punctuation, via MMS forced alignment. WAXAL's prompt reads pause mid-phrase 15-27 times per
100 words against broadcast presenters' 10, and an adapter trained on them read word by word."""
import sys, os, re, json, torch, torchaudio, soundfile as sf
from torchaudio.pipelines import MMS_FA

torch.set_num_threads(int(os.environ.get("THREADS", "4")))
model = MMS_FA.get_model(with_star=False).eval()
vocab, aligner = MMS_FA.get_dict(star=None), MMS_FA.get_aligner()
src, dst = sys.argv[1:3]
done = {json.loads(l)["wav"] for l in open(dst)} if os.path.exists(dst) else set()
out = open(dst, "a")
for r in map(json.loads, open(src)):
    if r["wav"] in done:
        continue
    x, sr = sf.read(r["wav"], dtype="float32")
    x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000)
    words = [w for w in r["text"].split() if re.sub(r"[^a-z]", "", w.lower())]
    toks = [[vocab[c] for c in re.sub(r"[^a-z]", "", w.lower()) if c in vocab] for w in words]
    with torch.inference_mode():
        em = model(x[None])[0][0]
    mid = punct = 0
    if words and sum(map(len, toks)) <= em.shape[0]:
        spf = len(x) / em.shape[0]
        sp = aligner(em, toks)
        for i in range(1, len(sp)):
            if (sp[i][0].start - sp[i - 1][-1].end) * spf / 16000 >= 0.15:
                punct += bool(re.search(r"[.,;:!?]$", words[i - 1]))
                mid += not re.search(r"[.,;:!?]$", words[i - 1])
    out.write(json.dumps({"wav": r["wav"], "words": len(words), "mid": mid, "punct": punct,
                          "seconds": round(len(x) / 16000, 2)}) + "\n")
