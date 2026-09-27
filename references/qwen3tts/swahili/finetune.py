"""LoRA-teach the qwen3tts talker Swahili, with a new language row (codec id 2074).

    finetune.py train DATA.pt [DATA.pt ...]      # env: STEPS LR RANK ACCUM OUT WEIGHTS RESUME
                                                 #      SUBW ROW_LR ROW_PULL CP_LR EXCLUDE FOCUS PAIR_SIM SPLIT
                                                 #      SUB_CHANNEL XREF
    finetune.py export CKPT.pt ADAPTER.safetensors [gain]    # env: SCHEME=../langs/<language>.json

Examples are ICL, as inference is: reference = another utterance by the same speaker, loss only
on the target's frames. The prompt mirrors `Talker::build_prompt_shared`, which also avoids
the upstream SFT script's bugs (double-shifted labels, leaked sub-talker state, no
text_projection). Run with `--set adapter=ADAPTER --set language=swahili`.
"""
import os, sys, re, json, math, random, time
import torch, torch.nn as nn, torch.nn.functional as F

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")
SWAHILI = 2074  # an untrained codec row (norm at init), inside the language range; a SCHEME names its own
dev = torch.device(os.environ.get("DEV", "mps"))
DT = torch.bfloat16
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.a.to(x.dtype).T @ self.b.to(x.dtype).T) * self.scale


def wrap(module: nn.Module, r: int, prefix: str, found: dict):
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and name in TARGETS:
            lora = LoRA(child, r, 2 * r)
            setattr(module, name, lora)
            found[full] = lora
        else:
            wrap(child, r, full, found)


def load(rank):
    from qwen_tts import Qwen3TTSModel
    tts = Qwen3TTSModel.from_pretrained(MODEL, device_map="cpu", dtype=DT)
    m = tts.model
    m.speech_tokenizer = None
    m.speaker_encoder = None
    talker = m.talker.to(dev).eval()
    for p in talker.parameters():
        p.requires_grad_(False)
    loras = {}
    wrap(talker.model, rank, "talker.model", loras)
    wrap(talker.code_predictor, max(4, rank // 2), "talker.code_predictor", loras)
    return m, talker, loras


def build(m, talker, row, ref, tgt):
    """Embeddings for one ICL example, and (start, n) of the positions predicting tgt."""
    cfg, tc = m.config, m.config.talker_config
    E, CP = talker.get_input_embeddings(), talker.code_predictor.get_input_embeddings()
    text = lambda ids: talker.text_projection(talker.get_text_embeddings()(torch.tensor([ids], device=dev)))
    crow = lambda i: E(torch.tensor([[i]], device=dev))
    frames = lambda c: E(c[:, :1].T) + sum(CP[i - 1](c[:, i:i + 1].T) for i in range(1, 16))
    pad, bos, eos = text([cfg.tts_pad_token_id, cfg.tts_bos_token_id, cfg.tts_eos_token_id]).chunk(3, 1)

    codec = torch.cat([crow(tc.codec_think_id), crow(tc.codec_think_bos_id), row.to(DT).view(1, 1, -1),
                       crow(tc.codec_think_eos_id), ref["spk"].to(dev, DT).view(1, 1, -1),
                       crow(tc.codec_pad_id), crow(tc.codec_bos_id)], 1)
    L = codec.shape[1]
    prefix = torch.cat([pad.expand(-1, L - 2, -1), bos], 1) + codec[:, :-1]
    role = text([151644, 77091, 198])

    t_emb = torch.cat([text(ref["ids"].tolist() + tgt["ids"].tolist()), eos], 1)
    rc = ref["codes"].to(dev).long()
    c_emb = torch.cat([crow(tc.codec_bos_id), frames(rc)], 1)
    tl, cl = t_emb.shape[1], c_emb.shape[1]
    if tl > cl:
        icl, trailing = t_emb[:, :cl] + c_emb, t_emb[:, cl:]
    else:
        icl, trailing = torch.cat([t_emb, pad.expand(-1, cl - tl, -1)], 1) + c_emb, pad
    tc_ = tgt["codes"].to(dev).long()
    T = tc_.shape[0]
    tr = trailing.shape[1]
    txt = torch.cat([trailing[:, :T]] + ([pad.expand(-1, T - tr, -1)] if T > tr else []), 1)
    steps = frames(tc_) + txt
    seq = torch.cat([role, prefix, icl, steps], 1)
    start = role.shape[1] + prefix.shape[1] + icl.shape[1] - 1
    return seq, start, tc_


SUBFRAC = float(os.environ.get("SUBFRAC", "0.25"))


def hidden(talker, seq, start, split=True):
    """Trunk states from `start` on. The prefix carries no loss, so with `split` it runs
    without grad into the KV cache and backward covers only the target span."""
    if not split:
        return talker.model(inputs_embeds=seq, use_cache=False).last_hidden_state[0, start:]
    with torch.no_grad():
        cache = talker.model(inputs_embeds=seq[:, :start], use_cache=True).past_key_values
    return talker.model(inputs_embeds=seq[:, start:], past_key_values=cache, use_cache=True).last_hidden_state[0]


def loss_of(m, talker, row, ref, tgt, split=True, subfrac=None):
    seq, start, codes = build(m, talker, row, ref, tgt)
    T = codes.shape[0]
    h = hidden(talker, seq, start, split)[: T + 1]
    labels = torch.cat([codes[:, 0], torch.tensor([m.config.talker_config.codec_eos_token_id], device=dev)])
    ce0 = F.cross_entropy(talker.codec_head(h).float(), labels)
    frac = SUBFRAC if subfrac is None else subfrac
    pick = torch.randperm(T, device=dev)[: max(1, int(T * frac))] if frac < 1 else torch.arange(T, device=dev)
    logits, _ = talker.forward_sub_talker_finetune(codes[pick], h[:T][pick])
    sub = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), codes[pick][:, 1:].reshape(-1))
    return ce0, sub


def pairs(data):
    by = {}
    for d in data:
        by.setdefault(d["speaker"], []).append(d)
    return {k: v for k, v in by.items() if len(v) >= 2}


def train(paths):
    rank = int(os.environ.get("RANK", "16"))
    steps = int(os.environ.get("STEPS", "3000"))
    lr = float(os.environ.get("LR", "1e-4"))
    out = os.environ.get("OUT", "swahili-lora.pt")
    accum = int(os.environ.get("ACCUM", "4"))
    m, talker, loras = load(rank)
    tc = m.config.talker_config
    E = talker.get_input_embeddings()
    seed_row = torch.stack([E.weight[i] for i in tc.codec_language_id.values()]).float().mean(0)
    row = nn.Parameter(seed_row.clone().to(dev))
    first = 1
    if os.environ.get("RESUME"):
        ck = torch.load(os.environ["RESUME"])
        row.data = ck["row"].to(dev).float()
        for k, (a, b, _) in ck["loras"].items():
            loras[k].a.data, loras[k].b.data = a.to(dev), b.to(dev)
        # RESTART=1: keep the weights, start a fresh schedule (a new data mix, say).
        first = 1 if os.environ.get("RESTART") else ck["step"] + 1
        print("resumed at", first, flush=True)
    data = [d for p in paths for d in torch.load(p)]
    data = [d for d in data if 20 <= d["codes"].shape[0] <= 200]
    # EXCLUDE=ep1,ep2: held-out episodes (speaker ids contain them), kept for evaluation.
    held = [e for e in os.environ.get("EXCLUDE", "").split(",") if e]
    data = [d for d in data if not any(e in d["speaker"] for e in held)]
    # Reference-only rows (langs/harvest/pick.py REF_LANGS) are never targets, nor in validation,
    # which stays the targets' own first 40 so runs remain comparable.
    refs = [d for d in data if d.get("ref_only")]
    extra = [d for d in data if d.get("cb0_only") and not d.get("ref_only")]
    data = [d for d in data if not d.get("ref_only") and not d.get("cb0_only")]
    random.Random(0).shuffle(data)
    val, tr = data[:40], data[40:]
    groups_tr, groups_val = pairs(tr + extra + refs), pairs(val + tr)
    tr = tr + extra
    # WEIGHTS="waxal:8=2,waxal:2=2": speakers drawn in proportion to size times weight.
    wmap = dict((k, float(v)) for k, v in (kv.split("=") for kv in os.environ.get("WEIGHTS", "").split(",") if kv))
    keys = list(groups_tr)
    kw = [len(groups_tr[k]) * wmap.get(k, 1.0) for k in keys]
    # FOCUS="ng['’]=4;(^|[.!?] )(M[bv]|N[dgjz])=2": targets matching a pattern drawn that much more
    # often within their speaker. Rare sounds (ng' is 3% of clips) were dropped ("ombe").
    focus = [(re.compile(p), float(w)) for p, w in (f.rsplit("=", 1) for f in os.environ.get("FOCUS", "").split(";") if f)]
    tw = {k: [0.0 if d.get("ref_only") else max([1.0] + [w for p, w in focus if p.search(d["text"])]) for d in g]
          for k, g in groups_tr.items()}
    keys = [k for k in keys if sum(tw[k])]
    kw = [len(groups_tr[k]) * wmap.get(k, 1.0) for k in keys]
    # XREF=0.3: that share of targets whose speaker has reference-only clips (another language)
    # get one as the reference: a voice cloned from an English clip must still speak Swahili.
    xref = float(os.environ.get("XREF", "0"))
    # PAIR_SIM=0.6: a reference must share the target's channel and be this close under ECAPA
    # (langs/harvest/annotate.py). Studio/field pairs taught that the recording may change
    # from the reference, heard as a new microphone per sentence.
    # SPLIT=0: the prefix with gradient, so the language row (inside it) trains too; ~1.8x a step.
    split = os.environ.get("SPLIT", "1") != "0"
    # SUB_CHANNEL=studio: the sub-talker (codebooks 1-15: timbre and channel) learns only from
    # studio targets, so the code predictor's LoRA gains Swahili detail without field acoustics.
    sub_channel = os.environ.get("SUB_CHANNEL", "")
    pair_sim = float(os.environ.get("PAIR_SIM", "0"))
    sims = {}
    if pair_sim:
        for k, g in groups_tr.items():
            if all("ecapa" in d for d in g):
                E = torch.stack([d["ecapa"].float() for d in g])
                ch = [d.get("channel") for d in g]
                sims[k] = (E @ E.T, ch)
        print(f"PAIR_SIM {pair_sim}: {len(sims)} of {len(groups_tr)} speakers annotated", flush=True)
    print(f"{len(tr)} train utts in {len(keys)} speakers, {len(val)} val", flush=True)
    # Official SFT weighs the sub-talker 0.3; a shared row at 10x lr soaked up the average voice.
    subw = float(os.environ.get("SUBW", "0.3"))
    pull = float(os.environ.get("ROW_PULL", "0.01"))
    cp_lr = float(os.environ.get("CP_LR", "1"))
    seed_dev = seed_row.to(dev)
    main = [p for k, l in loras.items() if "code_predictor" not in k for p in (l.a, l.b)]
    cp = [p for k, l in loras.items() if "code_predictor" in k for p in (l.a, l.b)]
    for p in cp:
        p.requires_grad_(cp_lr > 0)
    params = main + (cp if cp_lr > 0 else [])
    groups = [{"params": main, "lr": lr, "weight_decay": 0.0},
              {"params": [row], "lr": lr * float(os.environ.get("ROW_LR", "2")), "weight_decay": 0.0}]
    if cp_lr > 0:
        groups.append({"params": cp, "lr": lr * cp_lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 100) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    rng = random.Random(1)

    def evaluate():
        tot0 = tots = 0.0
        with torch.no_grad():
            for t in val:
                g = [d for d in groups_val.get(t["speaker"], []) if d is not t]
                ref = g[0] if g else t
                a, b = loss_of(m, talker, row, ref, t, subfrac=1.0)
                tot0 += a.item(); tots += b.item()
        return round(tot0 / len(val), 4), round(tots / len(val), 4)

    for _ in range(first - 1):
        sched.step()
    if first == 1:
        print("val before", evaluate(), flush=True)
    t0 = time.time()
    for s in range(first, steps + 1):
        for _ in range(accum):
            k = rng.choices(keys, kw)[0]
            i = rng.choices(range(len(groups_tr[k])), tw[k])[0]
            tgt = groups_tr[k][i]
            # An onset clip (langs/harvest/onsets.py) is its source's tail: never each other's reference.
            g = groups_tr[k]
            ok = [j for j in range(len(g)) if j != i and tgt["text"] not in g[j]["text"] and g[j]["text"] not in tgt["text"]]
            ok = ok or [j for j in range(len(g)) if j != i]
            xr = [j for j in ok if g[j].get("ref_only")]
            if xr and rng.random() < xref:
                if k in sims:
                    S, ch = sims[k]
                    xr = sorted([j for j in xr if ch[j] == ch[i]] or xr, key=lambda j: -float(S[i, j]))[:3]
                ref = g[rng.choice(xr)]
            else:
                ok = [j for j in ok if not g[j].get("ref_only")] or ok
                if k in sims:
                    S, ch = sims[k]
                    close = [j for j in ok if ch[j] == ch[i] and S[i, j] >= pair_sim]
                    ok = close or sorted(ok, key=lambda j: -float(S[i, j]))[:3]
                ref = g[rng.choice(ok)]
            a, b = loss_of(m, talker, row, ref, tgt, split=split)
            w = subw if not sub_channel or tgt.get("channel") == sub_channel else 0.0
            # cb0_only rows (many-speaker, band-limited): pronunciation from many voices, no timbre or channel.
            w = 0.0 if tgt.get("cb0_only") else w
            ((a + w * b + pull * (row - seed_dev).pow(2).sum()) / accum).backward()
        torch.nn.utils.clip_grad_norm_(params + [row], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if dev.type == "mps":
            torch.mps.empty_cache()
        if s % 25 == 0:
            print(f"step {s} ce0 {a.item():.3f} sub {b.item():.3f} {(time.time() - t0) / (s - first + 1):.1f}s/step", flush=True)
        if s % 250 == 0 or s == steps:
            print(f"val step {s}", evaluate(), flush=True)
            torch.save({"step": s, "rank": rank, "cp_lr": cp_lr, "row": row.detach().cpu(), "loras": {k: (l.a.detach().cpu(), l.b.detach().cpu(), l.scale) for k, l in loras.items()}}, out.replace(".pt", f"-{s}.pt"))


def export(adapter, out, gain=1.0):
    """The adapter file `Weights::with_adapter` reads: LoRA pairs, plus the codec table."""
    # Without its orthography the engine tokenizes whole words, which the adapter never saw.
    if not (os.environ.get("SCHEME") or os.environ.get("SYLLABLES")):
        sys.exit("export: set SCHEME=langs/<language>.json (or SYLLABLES=1 for a pre-scheme checkpoint)")
    from safetensors import safe_open
    from safetensors.torch import save_file
    ad = torch.load(adapter)
    t = {}
    for name, (a, b, scale) in ad["loras"].items():
        # TALKER_ONLY=1 leaves the code predictor (codebooks 1-15, where timbre lives) as shipped.
        if (os.environ.get("TALKER_ONLY") or ad.get("cp_lr") == 0) and "code_predictor" in name:
            continue
        t[f"{name}.weight::lora_a"] = a.float().contiguous()
        t[f"{name}.weight::lora_b"] = (b.float() * scale * gain).contiguous()
    emb = "talker.model.codec_embedding.weight"
    with safe_open(f"{MODEL}/model.safetensors", "pt") as f:
        table = f.get_tensor(emb).clone()
    scheme = json.load(open(os.environ["SCHEME"])) if os.environ.get("SCHEME") else None
    row = int(scheme["row"]) if scheme else SWAHILI
    table[row] = ad["row"].to(table.dtype)
    t[f"{emb}::replace"] = table
    if os.environ.get("SYLLABLES"):
        # Trained on syllable BPE; the engine then tokenizes that way, reference included.
        t["meta::syllables"] = torch.ones(1)
    if scheme:
        # The engine tokenizes by this (`syllables::Scheme`) and resolves language=<name> to its row.
        t["meta::scheme"] = torch.frombuffer(bytearray(json.dumps(scheme, ensure_ascii=False).encode()), dtype=torch.uint8).clone()
    save_file(t, out, metadata={"language": scheme["language"] if scheme else "swahili", "row": str(row), "gain": str(gain), "step": str(ad.get("step"))})
    print("wrote", out, len(ad["loras"]), "loras")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2:])
    else:
        export(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 1.0)
