"""pick.py — verified.jsonl -> clips.jsonl (the encode input), with a report of what each filter cut.

Gemini sometimes shifts texts by a clip within a batch, so each text is re-matched to the batch
draft it agrees with best and kept only if that agreement is good.
"""
import os, json, re, collections, difflib
from common import CFG, D

MAX_CER = float(os.environ.get("MAX_CER", "0.35"))
CHANNELS = set(os.environ.get("CHANNELS", "studio,field").split(","))
# REF_LANGS=en,mixed: clips in these languages are kept as references only (never targets), so a
# voice cloned from a clip in another language has been seen conditioning this one.
REF_LANGS = set(filter(None, os.environ.get("REF_LANGS", "").split(",")))
OUT = os.environ.get("OUT", "clips.jsonl")


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w' ]", " ", t.lower())).strip()


def cer(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 1.0
    return 1 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


rows = [json.loads(l) for l in open(f"{D}/verified.jsonl")]
if os.path.exists(f"{D}/verified-redo.jsonl"):
    redo = {r["wav"]: r for r in map(json.loads, open(f"{D}/verified-redo.jsonl"))}
    rows = [redo.get(r["wav"], r) for r in rows]
disagree = []
MAX_GAP = float(os.environ.get("MAX_GAP", "0.5"))
gaps = {}
if os.path.exists(f"{D}/mms.jsonl"):
    gaps = {(r["wav"], r["text"]): r["gap"] for r in map(json.loads, open(f"{D}/mms.jsonl"))}
by_batch = collections.defaultdict(list)
for r in rows:
    by_batch[r["batch"]].append(r)
cut = collections.Counter()
out, rematched, decisions = [], 0, []
for batch in by_batch.values():
    texts = [(r["text"], r) for r in batch]
    for r in batch:
        if not r["draft"]:
            # VAD chunks have no draft; the MMS gap is the check, and nothing to re-match against.
            g = gaps.get((r["wav"], r["text"]))
            if g is None or g > MAX_GAP:
                cut["unscored" if g is None else "mms"] += 1
                decisions.append({"wav": r["wav"], "decision": "unscored" if g is None else "mms"})
                continue
            best, src = 0.0, r
        else:
            best, src = min(((cer(r["draft"], t), s) for t, s in texts), key=lambda x: x[0])
        if src is not r:
            rematched += 1
        r2 = {**r, **{k: src[k] for k in ("text", "lang", "channel", "speakers", "gender", "accent", "cut")}, "agree": 1 - best}
        target = r2["lang"] == CFG["code"] or r2["lang"] == "mixed" and r2["channel"] == "studio"
        ref_only = not target and r2["lang"] in REF_LANGS
        why = ("disagree" if best > MAX_CER else
               "lang" if not (target or ref_only) else
               "channel" if r2["channel"] not in CHANNELS else
               "speakers" if r2["speakers"] != 1 else
               "cut" if r2["cut"] else
               "accent" if r2["accent"] != CFG["keep_accent"] else
               "digits" if re.search(r"\d", r2["text"]) else
               "rate" if not 8 <= len(norm(r2["text"])) / (r2["end"] - r2["start"]) <= 22 else None)
        if why == "disagree":
            disagree.append(r["wav"])
        decisions.append({"wav": r["wav"], "decision": why or ("reference" if ref_only else "selected"),
                          "text": r2["text"].strip()})
        if why:
            cut[why] += 1
            continue
        out.append(dict(wav=r["wav"], text=r2["text"].strip(), speaker=f"{CFG['code']}:{r['episode']}", gender=r2["gender"],
                        channel=r2["channel"], seconds=round(r["end"] - r["start"], 2), **({"ref_only": True} if ref_only else {})))
with open(f"{D}/{OUT}", "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
open(f"{D}/disagree.txt", "w").write("\n".join(disagree))
# Each clip's fate and, where it was re-matched within its batch, the text it was given.
with open(f"{D}/decisions.jsonl", "w") as f:
    f.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in decisions)
h = sum(r["seconds"] for r in out) / 3600
print(f"{len(rows)} verified -> {len(out)} kept ({h:.2f} h), rematched {rematched}; cut {dict(cut)}")
print("by channel", collections.Counter(r["channel"] for r in out), "reference-only", sum("ref_only" in r for r in out))
