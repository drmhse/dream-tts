"""assign_refs.py XREF.jsonl SPK.jsonl OUT.jsonl — reference-only clips joined to existing speakers.

Each ref-only clip (pick.py REF_LANGS) goes to the nearest target speaker of its episode by ECAPA
cosine to that speaker's centroid, if at least MIN_SIM; the target clusters are left as they were.
Clustering targets and references together instead merged speakers through the references (a
277-clip cluster where 46 had been the largest).
"""
import sys, os, json, collections, torch

D = os.environ["HARVEST_DIR"]
MIN_SIM = float(os.environ.get("MIN_SIM", "0.55"))
xref, spk, dst = sys.argv[1:4]
cache = torch.load(f"{D}/ecapa.pt")
refs = [r for r in map(json.loads, open(xref)) if r.get("ref_only")]
by = collections.defaultdict(list)
for r in map(json.loads, open(spk)):
    by[r["speaker"]].append(torch.nn.functional.normalize(cache[r["wav"]], dim=0))
cent = {k: torch.nn.functional.normalize(torch.stack(v).mean(0), dim=0) for k, v in by.items()}
per_ep = collections.defaultdict(list)
for k in cent:
    per_ep[k.rsplit(":", 1)[0]].append(k)
out = []
for r in refs:
    e = cache.get(r["wav"])
    cands = per_ep.get(r["speaker"], [])
    if e is None or not cands:
        continue
    e = torch.nn.functional.normalize(e, dim=0)
    best = max(cands, key=lambda k: float(e @ cent[k]))
    if float(e @ cent[best]) >= MIN_SIM:
        out.append({**r, "speaker": best})
with open(dst, "w") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
print(f"{len(refs)} reference-only clips -> {len(out)} assigned to {len({r['speaker'] for r in out})} speakers "
      f"({sum(r['seconds'] for r in out) / 3600:.2f} h)")
