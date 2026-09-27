"""mms.py — CTC check of each verified text against its audio: mms.jsonl {wav: gap}.

gap = (best-path log-prob - forced-path log-prob) per frame under MMS_FA: near 0 when the text is
what was said, large when Gemini attached another clip's text. CPU; the GPU is the LoRA's.
"""
import os, re, json, sys
import soundfile as sf, torch, torchaudio
from torchaudio.pipelines import MMS_FA

from common import D, romanize
torch.set_num_threads(int(os.environ.get("THREADS", "4")))
model = MMS_FA.get_model(with_star=False).eval()
vocab = MMS_FA.get_dict(star=None)


def gap(wav, text):
    x, sr = sf.read(wav, dtype="float32")
    x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000)
    letters = [vocab[c] for c in re.sub(r"[^a-z' ]", "", romanize(text).replace("’", "'")).replace(" ", "") if c in vocab]
    with torch.inference_mode():
        em = torch.log_softmax(model(x[None])[0], -1)
    T = em.shape[1]
    # CTC needs a blank between repeated letters; a text longer than its audio is not its text.
    if not letters or len(letters) + sum(x == y for x, y in zip(letters, letters[1:])) > T:
        return 99.0
    _, scores = torchaudio.functional.forced_align(em, torch.tensor([letters], dtype=torch.int32), blank=0)
    return round(float((em[0].max(-1).values.sum() - scores[0].sum()) / T), 4)


def rows():
    for f in ("verified.jsonl", "verified-redo.jsonl"):
        if os.path.exists(f"{D}/{f}"):
            yield from (json.loads(l) for l in open(f"{D}/{f}"))


if __name__ == "__main__":
    path = f"{D}/mms.jsonl"
    done = {(r["wav"], r["text"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    out = open(path, "a")
    n = 0
    for r in rows():
        if (r["wav"], r["text"]) in done:
            continue
        done.add((r["wav"], r["text"]))
        out.write(json.dumps({"wav": r["wav"], "text": r["text"], "gap": gap(r["wav"], r["text"])}, ensure_ascii=False) + "\n")
        n += 1
        if n % 200 == 0:
            out.flush(); print(n, flush=True)
    print("scored", n)
