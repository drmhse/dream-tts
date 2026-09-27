"""part.py N — clips-spk.jsonl rows from episodes fully verified and scored, not in an earlier part -> part-N.jsonl."""
import sys, json, glob, os, collections
from common import D
cands = collections.defaultdict(set)
for r in map(json.loads, open(f"{D}/candidates.jsonl")):
    cands[r["episode"]].add(r["wav"])
ver = {r["wav"]: r for r in map(json.loads, open(f"{D}/verified.jsonl"))}
scored = {r["wav"] for r in map(json.loads, open(f"{D}/mms.jsonl"))}
taken = {json.loads(l)["speaker"].split(":")[1] for f in glob.glob(f"{D}/part-*.jsonl") for l in open(f)}
ready = {ep for ep, ws in cands.items() if ep not in taken and all(w in ver and (ver[w]["draft"] or w in scored) for w in ws)}
rows = [r for r in map(json.loads, open(f"{D}/clips-spk.jsonl")) if r["speaker"].split(":")[1] in ready]
with open(f"{D}/part-{sys.argv[1]}.jsonl", "w") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
print(len(ready), "episodes,", len(rows), "clips,", round(sum(r["seconds"] for r in rows) / 3600, 2), "h")
