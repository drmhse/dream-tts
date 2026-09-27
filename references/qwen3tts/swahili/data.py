"""Swahili training data for the qwen3tts talker: download, cut, encode.

    data.py fetch   DIR            # WAXAL swa_tts (24 h, 7 speakers) + two OpenBible shards
    data.py waxal   DIR            # parquet -> rows/*.wav + rows.jsonl
    data.py align   DIR            # rows -> <=15 s sentence chunks, MMS forced alignment
    data.py bible   DIR            # OpenBible verses -> trimmed wavs + verses.jsonl
    data.py cv      DIR            # Common Voice sw: 40 clips from each of 150 speakers
    data.py fluent  IN.pt CHUNKS.jsonl FLUENCY.jsonl OUT.pt [limit]   # drop halting reads
    data.py encode  IN.jsonl OUT.pt  # codec codes, x-vector, text ids — what finetune.py eats;
                                     # SCHEME=../langs/swahili.json: that orthography (see
                                     # langs/orthography.py); SYLLABLES=1 the older plain syllables;
                                     # LEVEL_DB=-20: every clip to one speech level

Every clip is trimmed to speech plus 0.12 s: FLEURS, untrimmed, averaged 81 wpm and taught
the talker to crawl (dev pace 144 -> 95 wpm).
"""
import sys, os, re, json, glob, subprocess

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")
SR = 24000
MAX_S, PAD_S = 15.0, 0.12
WAXAL = "https://huggingface.co/datasets/google/WaxalNLP/resolve/main/data/TTS/swa"
BIBLE = "https://huggingface.co/datasets/multilingual-tts/open-bible/resolve/main/Swahili"
CV = "https://huggingface.co/datasets/fsicoli/common_voice_17_0/resolve/main"


def pieces(text):
    """`syllables::pieces` in the engine: cut after every vowel a letter follows."""
    out, cur = [], ""
    for i, c in enumerate(text):
        if c.isspace() and cur.strip():
            out.append(cur); cur = ""
        cur += c
        if c in "aeiouAEIOU" and i + 1 < len(text) and text[i + 1].isalpha():
            out.append(cur); cur = ""
    if cur:
        out.append(cur)
    return out


def scheme_pieces():
    """SCHEME=langs/<language>.json: that orthography's pieces; else SYLLABLES=1's plain open syllables."""
    if os.environ.get("SCHEME"):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "langs"))
        from orthography import Scheme
        return Scheme(os.environ["SCHEME"]).pieces
    return pieces


def level(x, sr, db):
    """Scaled so 20 ms windows within 30 dB of the loudest average `db` dBFS, peak kept under -1 dBFS."""
    import numpy as np
    w = sr // 50
    p = np.array([np.mean(x[i:i + w] ** 2) for i in range(0, max(1, len(x) - w), w)])
    act = p[p > p.max() * 1e-3]
    if not len(act):
        return x
    g = 10 ** ((db - 10 * np.log10(act.mean())) / 20)
    g = min(g, 0.89 / max(1e-6, np.abs(x).max()))
    return (x * g).astype(np.float32)


def decode(b: bytes):
    import numpy as np
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
                         input=b, capture_output=True, check=True).stdout
    return np.frombuffer(pcm, np.float32)


def trim(x, a=0, b=None):
    """[a, b) tightened to where speech energy starts and ends, padded by PAD_S."""
    import numpy as np
    b = len(x) if b is None else b
    seg = x[a:b]
    win = SR // 100
    if len(seg) < 10 * win:
        return a, b
    e = np.sqrt(np.convolve(seg ** 2, np.ones(win) / win, mode="same"))
    idx = np.where(e > max(e.max() * 0.02, 1e-4))[0]
    if len(idx) == 0:
        return a, b
    pad = int(PAD_S * SR)
    return max(a, a + idx[0] - pad), min(b, a + idx[-1] + pad)


def fetch(d):
    os.makedirs(f"{d}/waxal", exist_ok=True)
    os.makedirs(f"{d}/bible", exist_ok=True)
    for f in ("swa-train-00000", "swa-train-00001", "swa-validation-00000", "swa-test-00000"):
        subprocess.run(["curl", "-fLC", "-", "-o", f"{d}/waxal/{f}.parquet", f"{WAXAL}/{f}.parquet"], check=True)
    for i in ("00", "01"):
        f = f"train-000{i}-of-00016.parquet"
        subprocess.run(["curl", "-fLC", "-", "-o", f"{d}/bible/{f}", f"{BIBLE}/{f}"], check=True)
    os.makedirs(f"{d}/cv", exist_ok=True)
    for u in ("transcript/sw/train.tsv", "audio/sw/train/sw_train_0.tar", "audio/sw/train/sw_train_1.tar"):
        subprocess.run(["curl", "-fLC", "-", "-o", f"{d}/cv/{os.path.basename(u)}", f"{CV}/{u}"], check=True)


def waxal(d):
    import pyarrow.parquet as pq, soundfile as sf
    os.makedirs(f"{d}/waxal/rows", exist_ok=True)
    with open(f"{d}/waxal/rows.jsonl", "w") as out:
        for p in sorted(glob.glob(f"{d}/waxal/*.parquet")):
            for r in pq.read_table(p).to_pylist():
                wav = decode(r["audio"]["bytes"])
                fn = f"{d}/waxal/rows/{r['id']}.wav"
                sf.write(fn, wav, SR)
                out.write(json.dumps({"wav": fn, "text": r["text"], "speaker": f"waxal:{r['speaker_id']}",
                                      "gender": r["gender"], "seconds": round(len(wav) / SR, 2)}, ensure_ascii=False) + "\n")


def bible(d):
    import pyarrow.parquet as pq, soundfile as sf
    os.makedirs(f"{d}/bible/wav", exist_ok=True)
    with open(f"{d}/bible/verses.jsonl", "w") as out:
        for p in sorted(glob.glob(f"{d}/bible/*.parquet")):
            for i, r in enumerate(pq.read_table(p).to_pylist()):
                if not 1.0 <= float(r["duration_seconds"]) <= MAX_S + 1.5:
                    continue
                x = decode(r["audio"]["bytes"])
                a, b = trim(x)
                if not 1.0 <= (b - a) / SR <= MAX_S + 0.5:
                    continue
                fn = f"{d}/bible/wav/{os.path.basename(p)[:11]}_{i:05d}.wav"
                sf.write(fn, x[a:b], SR)
                out.write(json.dumps({"wav": fn, "text": r["text"], "speaker": f"bible:{r['speaker_id']}", "gender": "",
                                      "seconds": round((b - a) / SR, 2)}, ensure_ascii=False) + "\n")


def cv(d, per_client="40"):
    """Speaker count, not hours: 14 speakers let the LoRA memorise voices, and an unseen
    reference then drifted toward them between segments (x-vector cosine 0.985 -> 0.971)."""
    import csv, tarfile, random, collections, soundfile as sf
    rows = list(csv.DictReader(open(f"{d}/cv/train.tsv"), delimiter="\t", quoting=csv.QUOTE_NONE))
    random.Random(0).shuffle(rows)
    take, n = {}, collections.Counter()
    for r in rows:
        if n[r["client_id"]] < int(per_client) and int(r["up_votes"] or 0) > int(r["down_votes"] or 0):
            n[r["client_id"]] += 1
            take[r["path"]] = r
    os.makedirs(f"{d}/cv/wav", exist_ok=True)
    with open(f"{d}/cv/clips.jsonl", "w") as out:
        for tar in sorted(glob.glob(f"{d}/cv/*.tar")):
            with tarfile.open(tar) as t:
                for m in t:
                    r = take.get(os.path.basename(m.name))
                    if r is None:
                        continue
                    x = decode(t.extractfile(m).read())
                    a, b = trim(x)
                    if not 1.0 <= (b - a) / SR <= MAX_S:
                        continue
                    fn = f"{d}/cv/wav/{os.path.basename(m.name)[:-4]}.wav"
                    sf.write(fn, x[a:b], SR)
                    out.write(json.dumps({"wav": fn, "text": r["sentence"], "speaker": f"cv:{r['client_id'][:12]}",
                                          "gender": r["gender"], "seconds": round((b - a) / SR, 2)}, ensure_ascii=False) + "\n")


def align(d):
    """Known transcripts, so one CTC pass replaces recognition. Takes past 150 s are aligned in
    240 s windows given ~170 s of text each (surplus audio aligns to blanks): an hour-long
    take in one trellis was killed for memory."""
    import numpy as np, soundfile as sf, torch, torchaudio
    from torchaudio.pipelines import MMS_FA

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MMS_FA.get_model(with_star=False).to(dev).eval()
    vocab = MMS_FA.get_dict(star=None)
    aligner = MMS_FA.get_aligner()
    norm = lambda w: re.sub(r"[^a-z]", "", w.lower())

    def emission(x16):
        frames, hop, ctx = [], 16000 * 30, 16000
        with torch.inference_mode():
            for s in range(0, len(x16), hop):
                a, b = max(0, s - ctx), min(len(x16), s + hop + ctx)
                e = model(torch.from_numpy(x16[a:b]).to(dev)[None])[0][0].float().cpu()
                per = e.shape[0] / (b - a)
                frames.append(e[int(round((s - a) * per)): int(round((min(s + hop, len(x16)) - a) * per))])
        return torch.cat(frames), len(x16) / sum(f.shape[0] for f in frames)

    def times(sents, x16):
        words, owner = [], []
        for si, s in enumerate(sents):
            for w in s.split():
                n = norm(w)
                if n and all(c in vocab for c in n):
                    words.append(n); owner.append(si)
        if not words:
            return {}
        em, spf = emission(x16)
        got = {}
        for spans, si in zip(aligner(em, [[vocab[c] for c in w] for w in words]), owner):
            a, b = spans[0].start * spf / 16000, spans[-1].end * spf / 16000
            got[si] = (got.get(si, (a, b))[0], b)
        return got

    os.makedirs(f"{d}/waxal/chunks", exist_ok=True)
    path = f"{d}/waxal/chunks.jsonl"
    done = {os.path.basename(json.loads(l)["wav"]).rsplit("_", 1)[0] for l in open(path)} if os.path.exists(path) else set()
    out = open(path, "a")
    for line in open(f"{d}/waxal/rows.jsonl"):
        r = json.loads(line)
        stem = os.path.splitext(os.path.basename(r["wav"]))[0]
        if stem in done:
            continue
        wav, _ = sf.read(r["wav"], dtype="float32")
        x16 = torchaudio.functional.resample(torch.from_numpy(wav), SR, 16000).numpy()
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", r["text"].strip()) if s.strip()]
        total = len(x16) / 16000
        span = {}
        try:
            if total <= 150:
                span = times(sents, x16)
            else:
                rate = sum(len(norm(w)) for w in r["text"].split()) / total
                i, t0 = 0, 0.0
                while i < len(sents):
                    j, budget = i, 0
                    while j < len(sents) and budget + len(norm(sents[j])) < rate * 170:
                        budget += len(norm(sents[j])); j += 1
                    j = max(j, i + 1)
                    got = times(sents[i:j], x16[int(t0 * 16000): int(min(total, t0 + 240) * 16000)])
                    keep = j if j == len(sents) else max(i + 1, j - 2)
                    last = None
                    for k in range(i, keep):
                        if k - i in got:
                            a, b = got[k - i]
                            span[k] = (t0 + a, t0 + b); last = t0 + b
                    if last is None:
                        break
                    i, t0 = keep, last
        except Exception as ex:
            print("skip", stem, ex, flush=True)
            continue
        chunks, cur, c0 = [], [], None
        for si in range(len(sents)):
            if si not in span:
                cur, c0 = [], None
                continue
            a, b = span[si]
            if c0 is not None and b - c0 > MAX_S:
                chunks.append((c0, span[cur[-1]][1], cur)); cur, c0 = [], None
            if c0 is None:
                c0 = a
            if b - c0 > MAX_S:
                cur, c0 = [], None
                continue
            cur.append(si)
        if cur:
            chunks.append((c0, span[cur[-1]][1], cur))
        for k, (a, b, idx) in enumerate(chunks):
            a0, b0 = trim(wav, max(0, int((a - 0.3) * SR)), min(len(wav), int((b + 0.3) * SR)))
            if not 1.0 <= (b0 - a0) / SR <= MAX_S + 1:
                continue
            fn = f"{d}/waxal/chunks/{stem}_{k:03d}.wav"
            sf.write(fn, wav[a0:b0], SR)
            out.write(json.dumps({"wav": fn, "text": " ".join(sents[i] for i in idx), "speaker": r["speaker"],
                                  "gender": r["gender"], "seconds": round((b0 - a0) / SR, 2)}, ensure_ascii=False) + "\n")
        out.flush()
        print(stem, f"{total:.0f}s ->", len(chunks), flush=True)
        if dev.type == "mps":
            torch.mps.empty_cache()


def encode(src, dst):
    """2.2 GB peak: loaded bf16 (as stored) with the talker dropped, the codec reloaded fp32
    (fp32 on disk; bf16 changed ~15% of codes)."""
    import gc, soundfile as sf, torch, librosa
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

    tts = Qwen3TTSModel.from_pretrained(WEIGHTS, device_map="cpu", dtype=torch.bfloat16)
    m = tts.model
    m.talker = None
    gc.collect()
    m.speaker_encoder.float()
    codec = Qwen3TTSTokenizer.from_pretrained(f"{WEIGHTS}/speech_tokenizer", device_map="cpu", dtype=torch.float32)
    sr_spk = m.speaker_encoder_sample_rate
    out, split = [], scheme_pieces()
    for line in open(src):
        r = json.loads(line)
        wav, sr = sf.read(r["wav"], dtype="float32")
        if not 1.0 <= len(wav) / sr <= MAX_S + 1:
            continue
        if os.environ.get("LEVEL_DB"):
            # One speech level for every clip, reference and target alike, so a pair never
            # teaches that loudness may drift from the reference.
            wav = level(wav, sr, float(os.environ["LEVEL_DB"]))
        with torch.no_grad():
            spk = m.extract_speaker_embedding(audio=librosa.resample(wav, orig_sr=sr, target_sr=sr_spk), sr=sr_spk).reshape(-1)
            codes = codec.encode(wav, sr=sr).audio_codes[0].to(torch.int32)
        if os.environ.get("SYLLABLES") or os.environ.get("SCHEME"):
            tok = tts.processor.tokenizer
            ids = torch.tensor([i for p in split(r["text"]) for i in tok(p, add_special_tokens=False)["input_ids"]],
                               dtype=torch.int32)
        else:
            ids = tts.processor(text=f"<|im_start|>assistant\n{r['text']}<|im_end|>\n<|im_start|>assistant\n",
                                return_tensors="pt")["input_ids"][0, 3:-5].to(torch.int32)
        out.append(dict(text=r["text"], spk=spk.float(), codes=codes, ids=ids, speaker=r["speaker"], gender=r.get("gender", "")))
        if len(out) % 100 == 0:
            print(len(out), flush=True)
    torch.save(out, dst)
    print("saved", len(out), dst)


def retok(src, dst):
    """Recompute text ids under the current SCHEME/SYLLABLES without re-encoding audio."""
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(WEIGHTS)
    rows, split = torch.load(src), scheme_pieces()
    for r in rows:
        r["ids"] = torch.tensor([i for p in split(r["text"]) for i in tok(p, add_special_tokens=False)["input_ids"]],
                                dtype=torch.int32)
    torch.save(rows, dst)
    print("saved", len(rows), dst)


def fluent(src, chunks, scores, dst, limit="0.10"):
    """Rows whose clip pauses mid-phrase at most `limit` times a word (fluency.py); rows carry
    no wav, so they are matched to chunks.jsonl by speaker and text."""
    import torch
    wav = {(r["speaker"], r["text"]): r["wav"] for r in map(json.loads, open(chunks))}
    rate = {r["wav"]: r["mid"] / max(1, r["words"]) for r in map(json.loads, open(scores))}
    rows = torch.load(src)
    keep = [r for r in rows if rate.get(wav.get((r["speaker"], r["text"])), 1.0) <= float(limit)]
    torch.save(keep, dst)
    print(f"kept {len(keep)} of {len(rows)} ({sum(r['codes'].shape[0] for r in keep) / 12.5 / 3600:.2f} h)", dst)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"fetch": fetch, "waxal": waxal, "bible": bible, "cv": cv, "align": align, "encode": encode, "retok": retok, "fluent": fluent}[cmd](*args)
