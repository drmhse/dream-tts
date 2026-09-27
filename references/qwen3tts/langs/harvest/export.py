"""export.py OUT — a harvest as a Hugging Face dataset folder, ready to push; nothing is uploaded.

    OUT/README.md                    dataset card
    OUT/data/{train,heldout}-*.parquet  every verified clip: FLAC audio and all labels
    OUT/raw/epNNN.m4a                the episodes (APFS clones where possible: no extra disk)
    OUT/encoded/                     training tensors, speakers renamed to the neutral ids
    OUT/provenance/                  neutral id -> source id; leave out of an upload to keep it

Episodes get neutral ids (ep000...), so the tables name no source. HOLDOUT=<source id> puts that
episode in the heldout split. ENCODED=a.pt,b.pt copies training tensors.
"""
import sys, os, io, json, glob, shutil, subprocess, collections
import numpy as np, soundfile as sf, pyarrow as pa, pyarrow.parquet as pq, torch
from common import CFG, D

OUT = sys.argv[1]
HOLDOUT = os.environ.get("HOLDOUT", "")
SHARD_BYTES = 256 << 20
for sub in ("data", "raw", "encoded", "provenance"):
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)

load = lambda f: [json.loads(l) for l in open(f"{D}/{f}")] if os.path.exists(f"{D}/{f}") else []
verified = load("verified.jsonl")
redo = {r["wav"]: r for r in load("verified-redo.jsonl")}
verified = [redo.get(r["wav"], r) for r in verified]
decision = {r["wav"]: r for r in load("decisions.jsonl")}
gap = {(r["wav"], r["text"]): r["gap"] for r in load("mms.jsonl")}
target = {r["wav"]: r["speaker"] for r in load("clips-spk.jsonl")}
reference = {r["wav"]: r["speaker"] for r in load("refs.jsonl")}
ecapa = torch.load(f"{D}/ecapa.pt") if os.path.exists(f"{D}/ecapa.pt") else {}

sources = sorted({r["episode"] for r in verified})
ep = {s: f"ep{i:03d}" for i, s in enumerate(sources)}
# Episodes downloaded but never cut keep ids after the processed ones, so those stay stable.
for s in sorted({os.path.basename(f).rsplit(".", 1)[0] for f in glob.glob(f"{D}/raw/*")} - set(sources)):
    ep[s] = f"ep{len(ep):03d}"
neutral = lambda spk: None if spk is None else ":".join([ep.get(spk.split(":")[1], spk.split(":")[1])] + spk.split(":")[2:])


def flac(path):
    x, sr = sf.read(path, dtype="int16")
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="FLAC", subtype="PCM_16")
    return buf.getvalue(), sr, len(x) / sr


FEATURES = {
    "id": "string", "episode": "string", "split": "string", "start": "float32", "end": "float32",
    "duration": "float32", "text": "string", "draft": "string", "segmenter": "string", "lang": "string",
    "channel": "string", "speakers": "int32", "gender": "string", "accent": "string", "cut": "bool",
    "mms_gap": "float32", "transcriber": "string", "decision": "string", "role": "string", "speaker": "string",
}
hf_features = {k: {"dtype": v, "_type": "Value"} for k, v in FEATURES.items()}
hf_features["audio"] = {"sampling_rate": 24000, "_type": "Audio"}
hf_features["ecapa"] = {"feature": {"dtype": "float32", "_type": "Value"}, "_type": "Sequence"}
schema = pa.schema([pa.field("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())]))]
                   + [pa.field(k, getattr(pa, {"string": "string", "float32": "float32", "int32": "int32",
                                                   "bool": "bool_"}[v])()) for k, v in FEATURES.items()]
                   + [pa.field("ecapa", pa.list_(pa.float32()))],
                   metadata={"huggingface": json.dumps({"info": {"features": hf_features}})})

rows = {"train": [], "heldout": []}
written = collections.Counter()
stats = collections.Counter()


def flush(split, final=False):
    batch = rows[split]
    if not batch or (not final and sum(len(r["audio"]["bytes"]) for r in batch) < SHARD_BYTES):
        return
    pq.write_table(pa.Table.from_pylist(batch, schema=schema), f"{OUT}/data/{split}-{written[split]:05d}.parquet")
    written[split] += 1
    rows[split] = []


for r in sorted(verified, key=lambda r: r["wav"]):
    if not os.path.exists(r["wav"]):
        continue
    audio, sr, dur = flac(r["wav"])
    d = decision.get(r["wav"], {})
    text = d.get("text") or r.get("text", "")
    split = "heldout" if r["episode"] == HOLDOUT else "train"
    cid = ep[r["episode"]] + "_" + os.path.basename(r["wav"]).rsplit("_", 1)[1].rsplit(".", 1)[0]
    role = "target" if r["wav"] in target else "reference" if r["wav"] in reference else None
    e = ecapa.get(r["wav"])
    rows[split].append({
        "audio": {"bytes": audio, "path": f"{cid}.flac"}, "id": cid, "episode": ep[r["episode"]], "split": split,
        "start": r["start"], "end": r["end"], "duration": dur, "text": text, "draft": r.get("draft") or "",
        "segmenter": "vad" if r.get("src") == "vad" or not r.get("draft") else "whisper",
        "lang": r.get("lang"), "channel": r.get("channel"), "speakers": r.get("speakers"),
        "gender": r.get("gender"), "accent": r.get("accent"), "cut": bool(r.get("cut")),
        "mms_gap": gap.get((r["wav"], text)), "transcriber": r.get("model") or "gemini-3.5-flash",
        "decision": d.get("decision"), "role": role, "speaker": neutral(target.get(r["wav"]) or reference.get(r["wav"])),
        "ecapa": None if e is None else [float(v) for v in torch.nn.functional.normalize(e.float(), dim=0)],
    })
    stats[(split, d.get("decision"), role)] += dur
    flush(split)
for split in rows:
    flush(split, final=True)
for split, n in written.items():  # the n-of-N names Hugging Face expects
    for i in range(n):
        os.rename(f"{OUT}/data/{split}-{i:05d}.parquet", f"{OUT}/data/{split}-{i:05d}-of-{n:05d}.parquet")

for s, e in ep.items():
    src = next((f for f in glob.glob(f"{D}/raw/{s}.*")), None)
    if src and not os.path.exists(f"{OUT}/raw/{e}{os.path.splitext(src)[1]}"):
        subprocess.run(["cp", "-c", src, f"{OUT}/raw/{e}{os.path.splitext(src)[1]}"], check=True)

for p in filter(None, os.environ.get("ENCODED", "").split(",")):
    t = torch.load(p)
    for r in t:
        r["speaker"] = neutral(r["speaker"]) if r["speaker"].count(":") >= 2 else r["speaker"]
    torch.save(t, f"{OUT}/encoded/{os.path.basename(p)}")

with open(f"{OUT}/provenance/episodes.jsonl", "w") as f:
    for s, e in ep.items():
        f.write(json.dumps({"episode": e, "source_id": s, "url": f"https://www.youtube.com/watch?v={s}",
                            "heldout": s == HOLDOUT, "processed": s in sources}) + "\n")
if os.path.exists(os.environ.get("HARVEST", "")):
    shutil.copy(os.environ["HARVEST"], f"{OUT}/provenance/harvest-config.json")
json.dump({f"{k[0]}|{k[1]}|{k[2]}": round(v / 3600, 3) for k, v in sorted(stats.items(), key=str)},
          open(f"{OUT}/provenance/hours-by-split-decision-role.json", "w"), indent=1)
print({s: n for s, n in written.items()}, "shards;", len(ep), "episodes;",
      f"{sum(stats.values()) / 3600:.2f} h")
