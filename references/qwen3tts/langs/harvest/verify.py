"""verify.py [N] — Gemini transcripts + labels for candidates.jsonl, BATCH clips per request.

Appends to verified.jsonl; resumable. Keys rotate across the trial accounts; a 429 parks a key.
"""
import sys, os, glob, json, base64, subprocess, time, urllib.request, urllib.error
from common import CFG, D

# GEMINI_KEYS_DIR holds one key per gemini_api_key*.txt; quota is per key and per model.
KEY_DIR = os.path.expanduser(os.environ.get("GEMINI_KEYS_DIR", "~/.config/gemini"))
KEYS = [open(f).read().strip() for f in sorted(glob.glob(f"{KEY_DIR}/gemini_api_key*.txt"))]
# Quota is per key and per model, so every (key, model) pair is its own worker.
MODELS = os.environ.get("GEM_MODELS", "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash,gemini-3-flash-preview").split(",")
GAP_S = 7.5  # ~8 requests a minute per pair
BATCH = int(os.environ.get("BATCH", "15"))
PROMPT = """You are preparing a {region} {language} text-to-speech corpus from broadcast audio.
For each of the {n} clips below (Clip 1 .. Clip {n}, in order) return one JSON object:
- "clip": its number
- "text": an exact verbatim transcript in standard {language} orthography, every word actually spoken, nothing added.
  Write numbers, dates and currency the way they are spoken, in {language} words.
  Keep words from other languages as spoken, in their usual spelling. Punctuate sentences normally. {notes}
- "lang": "{code}" if entirely {language}, "mixed" if any words of another language, "en" or "other" otherwise
- "channel": "studio" (anchor or voice-over, clean mic), "field" (reporter or interviewee on location),
  "phone", or "noisy" (music, crowd, overlap or heavy background under the speech)
- "speakers": number of distinct voices
- "gender": "m" or "f" of the main voice
- "accent": one of {accents}
- "cut": true if the clip starts or ends mid-word
Return a JSON array of {n} objects only."""


def ogg(path):
    return subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "16000", "-c:a", "libopus",
                           "-b:a", "32k", "-f", "ogg", "pipe:1"], capture_output=True, check=True).stdout


def ask(key, model, clips):
    parts = [{"text": PROMPT.format(n=len(clips), language=CFG["language"], region=CFG.get("region", ""), code=CFG["code"],
                                   notes=CFG.get("notes", ""), accents=", ".join(f'"{a}"' for a in CFG["accents"]))}]
    for k, c in enumerate(clips, 1):
        parts += [{"text": f"Clip {k}:"}, {"inline_data": {"mime_type": "audio/ogg", "data": base64.b64encode(ogg(c["wav"])).decode()}}]
    body = {"contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                 "thinkingConfig": {"thinkingLevel": os.environ.get("THINK", "minimal")}}}
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", "x-goog-api-key": key})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    return json.loads(r["candidates"][0]["content"]["parts"][0]["text"])


import fcntl
lock = open(f"{D}/.{os.environ.get('OUT', 'verified.jsonl')}.lock", "w")
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(0)
cands = [json.loads(l) for l in open(f"{D}/candidates.jsonl")]
# REDO=list.txt OUT=verified-redo.jsonl: a second pass over what pick.py rejected as disagreeing.
OUT = f"{D}/{os.environ.get('OUT', 'verified.jsonl')}"
if os.environ.get("REDO"):
    want = set(open(os.environ["REDO"]).read().split())
    cands = [c for c in cands if c["wav"] in want]
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["wav"] for l in open(OUT)}
todo = [c for c in cands if c["wav"] not in done][: int(sys.argv[1]) if len(sys.argv) > 1 else None]
out = open(OUT, "a")
import threading
batches = [todo[i: i + BATCH] for i in range(0, len(todo), BATCH)]
mu = threading.Lock()


def worker(k, model):
    while True:
        with mu:
            if not batches:
                return
            batch = batches.pop(0)
        t = time.time()
        try:
            res = ask(KEYS[k], model, batch)
        except Exception as e:
            code = getattr(e, "code", None)
            with mu:
                batches.append(batch)
            print("key", k, model, code or repr(e)[:120], flush=True)
            time.sleep({429: 1800, 503: 60, 404: 10 ** 6}.get(code, 30))
            continue
        if isinstance(res, list):
            got = {r.get("clip"): r for r in res if isinstance(r, dict)}
            with mu:
                for n, c in enumerate(batch, 1):
                    if n in got:
                        out.write(json.dumps({**c, "batch": batch[0]["wav"], "model": model,
                                              **{x: y for x, y in got[n].items() if x != "clip"}}, ensure_ascii=False) + "\n")
                out.flush()
                print(f"key {k} {model} {len(batch)} clips {time.time() - t:.0f}s, {len(batches)} left", flush=True)
        time.sleep(max(0, GAP_S - (time.time() - t)))


threads = [threading.Thread(target=worker, args=(k, m), daemon=True) for k in range(len(KEYS)) for m in MODELS]
[t.start() for t in threads]
while any(t.is_alive() for t in threads) and (batches or any(t.is_alive() for t in threads)):
    time.sleep(5)
    with mu:
        if not batches:
            break
time.sleep(60)  # let in-flight requests land
