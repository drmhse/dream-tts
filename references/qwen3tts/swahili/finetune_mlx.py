"""finetune.py's `train` on MLX: same env, pairing, logs, and torch checkpoints `finetune.py export` reads.

    finetune_mlx.py train DATA.pt [DATA.pt ...]      # env as finetune.py train; DEV=cpu for the CPU
    finetune_mlx.py export CKPT.pt ADAPTER.safetensors [gain]    # finetune.py export, unchanged

Needs mlx and torch in one env (data and checkpoints stay torch .pt).
"""
import os, sys, re, json, math, random, time
import numpy as np
import torch
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_map
from finetune import pairs, export

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")
DT = mx.bfloat16
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SUBFRAC = float(os.environ.get("SUBFRAC", "0.25"))
PICK = np.random.default_rng(0)  # torch draws the sub-talker frames on the MPS generator, not reproducible here
if os.environ.get("DEV") == "cpu":
    mx.set_default_device(mx.cpu)


class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        out, inp = base.weight.shape
        self.a = mx.random.normal((r, inp)) / math.sqrt(inp)
        self.b = mx.zeros((out, r))
        self.scale = alpha / r

    def __call__(self, x):
        return self.base(x) + (x @ self.a.astype(x.dtype).T @ self.b.astype(x.dtype).T) * self.scale


swiglu = mx.compile(lambda g, u: nn.silu(g) * u, shapeless=True)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, self.heads, self.kv = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"]
        self.hd, self.theta = c.get("head_dim") or d // self.heads, float(c["rope_theta"])
        self.q_proj = nn.Linear(d, self.heads * self.hd, bias=False)
        self.k_proj = nn.Linear(d, self.kv * self.hd, bias=False)
        self.v_proj = nn.Linear(d, self.kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.heads * self.hd, d, bias=False)
        self.q_norm = nn.RMSNorm(self.hd, c["rms_norm_eps"])
        self.k_norm = nn.RMSNorm(self.hd, c["rms_norm_eps"])

    def __call__(self, x, pos, past):
        B, L, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(B, L, self.heads, self.hd)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, L, self.kv, self.hd)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.kv, self.hd).transpose(0, 2, 1, 3)
        # The talker's M-RoPE gets t = h = w for every token: plain half-split RoPE (cfg.rs, MROPE_SECTION).
        q = mx.fast.rope(q, self.hd, traditional=False, base=self.theta, scale=1.0, offset=pos)
        k = mx.fast.rope(k, self.hd, traditional=False, base=self.theta, scale=1.0, offset=pos)
        kv = (k, v)
        if past is not None:
            k, v = mx.concatenate([past[0], k], 2), mx.concatenate([past[1], v], 2)
        # "causal" aligns the last query with the last key: the span after a prefix, as with a KV cache.
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.hd ** -0.5, mask="causal")
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1)), kv


class MLP(nn.Module):
    def __init__(self, d, inner):
        super().__init__()
        self.gate_proj = nn.Linear(d, inner, bias=False)
        self.up_proj = nn.Linear(d, inner, bias=False)
        self.down_proj = nn.Linear(inner, d, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = Attention(c)
        self.mlp = MLP(c["hidden_size"], c["intermediate_size"])
        self.input_layernorm = nn.RMSNorm(c["hidden_size"], c["rms_norm_eps"])
        self.post_attention_layernorm = nn.RMSNorm(c["hidden_size"], c["rms_norm_eps"])

    def __call__(self, x, pos, past):
        a, kv = self.self_attn(self.input_layernorm(x), pos, past)
        x = x + a
        return x + self.mlp(self.post_attention_layernorm(x)), kv


class Trunk(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.layers = [Layer(c) for _ in range(c["num_hidden_layers"])]
        self.norm = nn.RMSNorm(c["hidden_size"], c["rms_norm_eps"])

    def __call__(self, x, pos=0, past=None):
        kvs = []
        for i, layer in enumerate(self.layers):
            x, kv = layer(x, pos, past[i] if past else None)
            kvs.append(kv)
        return self.norm(x), kvs


class ResizeMLP(nn.Module):
    def __init__(self, inp, inner, out):
        super().__init__()
        self.linear_fc1, self.linear_fc2 = nn.Linear(inp, inner), nn.Linear(inner, out)

    def __call__(self, x):
        return self.linear_fc2(nn.silu(self.linear_fc1(x)))


class Talker(nn.Module):
    """The talker's training path, under the checkpoint's parameter names (less `talker.`)."""

    def __init__(self, tc):
        super().__init__()
        cc, d = tc["code_predictor_config"], tc["hidden_size"]
        self.model = Trunk(tc)
        self.model.codec_embedding = nn.Embedding(tc["vocab_size"], d)
        self.model.text_embedding = nn.Embedding(tc["text_vocab_size"], tc["text_hidden_size"])
        self.text_projection = ResizeMLP(tc["text_hidden_size"], tc["text_hidden_size"], d)
        self.codec_head = nn.Linear(d, tc["vocab_size"], bias=False)
        self.code_predictor = nn.Module()
        self.code_predictor.model = Trunk(cc)
        self.code_predictor.model.codec_embedding = [nn.Embedding(cc["vocab_size"], d) for _ in range(cc["num_code_groups"] - 1)]
        self.code_predictor.lm_head = [nn.Linear(cc["hidden_size"], cc["vocab_size"], bias=False) for _ in range(cc["num_code_groups"] - 1)]
        self.code_predictor.small_to_mtp_projection = nn.Linear(d, cc["hidden_size"])

    def text(self, ids):
        return self.text_projection(self.model.text_embedding(mx.array([ids])))

    def frames(self, c):
        # Summed in finetune.py's order, so bf16 rounds the same way.
        E, CP = self.model.codec_embedding, self.code_predictor.model.codec_embedding
        return (E(c[:, 0]) + sum(CP[i - 1](c[:, i]) for i in range(1, 16)))[None]

    def sub(self, codes, h):
        """`forward_sub_talker_finetune`'s logits: [h, codebooks 0-14] in, codebooks 1-15 out."""
        cp = self.code_predictor
        x = mx.concatenate([h[:, None], self.model.codec_embedding(codes[:, :1])]
                           + [cp.model.codec_embedding[i - 1](codes[:, i:i + 1]) for i in range(1, 15)], 1)
        y = cp.model(cp.small_to_mtp_projection(x))[0]
        return mx.stack([cp.lm_head[i - 1](y[:, i]) for i in range(1, 16)], 1)


def wrap(trunk, r, prefix, found):
    for i, layer in enumerate(trunk.layers):
        for part in ("self_attn", "mlp"):
            mod = layer[part]
            for name in TARGETS:
                if name in mod:
                    mod[name] = found[f"{prefix}.layers.{i}.{part}.{name}"] = LoRA(mod[name], r, 2 * r)


def load(rank, cfg=None, weights=None):
    cfg = cfg or json.load(open(f"{MODEL}/config.json"))
    talker = Talker(cfg["talker_config"])
    if weights is None:
        weights = {k[7:]: v for k, v in mx.load(f"{MODEL}/model.safetensors").items() if k.startswith("talker.")}
    talker.load_weights(list(weights.items()))
    mx.random.seed(0)
    loras = {}
    wrap(talker.model, rank, "talker.model", loras)
    wrap(talker.code_predictor.model, max(4, rank // 2), "talker.code_predictor.model", loras)
    talker.freeze()
    for l in loras.values():
        l.unfreeze(keys=["a", "b"], recurse=False)
    mx.eval(talker.parameters())
    return cfg, talker, loras


def build(cfg, talker, ref, tgt):
    """Embeddings for one ICL example, and (start, n) of the positions predicting tgt."""
    tc = cfg["talker_config"]
    E = talker.model.codec_embedding
    crow = lambda i: E(mx.array([[i]]))
    codes = lambda d: mx.array(d["codes"].numpy())
    pad, bos, eos = mx.split(talker.text([cfg["tts_pad_token_id"], cfg["tts_bos_token_id"], cfg["tts_eos_token_id"]]), 3, 1)

    codec = mx.concatenate([crow(tc["codec_think_id"]), crow(tc["codec_think_bos_id"]), talker.row.astype(DT).reshape(1, 1, -1),
                            crow(tc["codec_think_eos_id"]), mx.array(ref["spk"].float().numpy()).astype(DT).reshape(1, 1, -1),
                            crow(tc["codec_pad_id"]), crow(tc["codec_bos_id"])], 1)
    L = codec.shape[1]
    prefix = mx.concatenate([mx.repeat(pad, L - 2, 1), bos], 1) + codec[:, :-1]
    role = talker.text([151644, 77091, 198])

    t_emb = mx.concatenate([talker.text(ref["ids"].tolist() + tgt["ids"].tolist()), eos], 1)
    c_emb = mx.concatenate([crow(tc["codec_bos_id"]), talker.frames(codes(ref))], 1)
    tl, cl = t_emb.shape[1], c_emb.shape[1]
    if tl > cl:
        icl, trailing = t_emb[:, :cl] + c_emb, t_emb[:, cl:]
    else:
        icl, trailing = mx.concatenate([t_emb, mx.repeat(pad, cl - tl, 1)], 1) + c_emb, pad
    tc_ = codes(tgt)
    T = tc_.shape[0]
    tr = trailing.shape[1]
    txt = mx.concatenate([trailing[:, :T]] + ([mx.repeat(pad, T - tr, 1)] if T > tr else []), 1)
    steps = talker.frames(tc_) + txt
    seq = mx.concatenate([role, prefix, icl, steps], 1)
    start = role.shape[1] + prefix.shape[1] + icl.shape[1] - 1
    return seq, start, tc_


def prefix_kv(talker, seq, start):
    """KV of the loss-free prefix, made outside any gradient as finetune.py's no-grad cache is.
    The language row sits in it, so under SPLIT=1 the row's only gradient is ROW_PULL's."""
    kv = talker.model(seq[:, :start])[1]
    mx.eval(kv)
    return kv


def hidden(talker, seq, start, past=None):
    past = prefix_kv(talker, seq, start) if past is None else past
    return talker.model(seq[:, start:], start, past)[0][0]


def picks(T, subfrac=None):
    frac = SUBFRAC if subfrac is None else subfrac
    return mx.array(PICK.permutation(T)[: max(1, int(T * frac))], mx.int32) if frac < 1 else mx.arange(T)


def losses(cfg, talker, seq, start, codes, past, pick):
    """`past` None: one pass with gradient through the prefix too (SPLIT=0)."""
    T = codes.shape[0]
    h = (talker.model(seq)[0][0, start:] if past is None else hidden(talker, seq, start, past))[: T + 1]
    labels = mx.concatenate([codes[:, 0], mx.array([cfg["talker_config"]["codec_eos_token_id"]], mx.int32)])
    ce0 = nn.losses.cross_entropy(talker.codec_head(h).astype(mx.float32), labels, reduction="mean")
    logits = talker.sub(codes[pick], h[:T][pick])
    sub = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]).astype(mx.float32), codes[pick][:, 1:].reshape(-1), reduction="mean")
    return ce0, sub


def loss_of(cfg, talker, ref, tgt, subfrac=None):
    seq, start, codes = build(cfg, talker, ref, tgt)
    return losses(cfg, talker, seq, start, codes, prefix_kv(talker, seq, start), picks(codes.shape[0], subfrac))


def checkpoint(path, step, rank, cp_lr, talker, loras):
    t = lambda x: torch.from_numpy(np.array(x.astype(mx.float32)))
    torch.save({"step": step, "rank": rank, "cp_lr": cp_lr, "row": t(talker.row),
                "loras": {k: (t(l.a), t(l.b), l.scale) for k, l in loras.items()}}, path)


def train(paths):
    rank = int(os.environ.get("RANK", "16"))
    steps = int(os.environ.get("STEPS", "3000"))
    lr = float(os.environ.get("LR", "1e-4"))
    out = os.environ.get("OUT", "swahili-lora.pt")
    accum = int(os.environ.get("ACCUM", "4"))
    cfg, talker, loras = load(rank)
    tc = cfg["talker_config"]
    E = talker.model.codec_embedding.weight
    lang = mx.stack([E[i] for i in tc["codec_language_id"].values()]).astype(mx.float32)
    # torch's mean bit for bit (mx.mean's 1/n is not): off by an ulp, a torch RESUME's row feeds
    # ROW_PULL a gradient that Adam scales up to lr-sized steps.
    seed_row = lang.sum(0) / lang.shape[0]
    talker.row = seed_row
    first = 1
    if os.environ.get("RESUME"):
        ck = torch.load(os.environ["RESUME"])
        talker.row = mx.array(ck["row"].float().numpy())
        for k, (a, b, _) in ck["loras"].items():
            loras[k].a, loras[k].b = mx.array(a.float().numpy()), mx.array(b.float().numpy())
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
    # SPLIT=0: the prefix with gradient, so the language row (inside it) trains, and the LoRA
    # learns how it encodes the reference; ~1.8x a step.
    split = os.environ.get("SPLIT", "1") != "0"
    # SUB_CHANNEL=studio: the sub-talker (codebooks 1-15: timbre and channel) learns only from
    # studio targets, so the code predictor's LoRA gains Swahili detail without field acoustics.
    sub_channel = os.environ.get("SUB_CHANNEL", "")
    pair_sim = float(os.environ.get("PAIR_SIM", "0"))
    sims = {}
    if pair_sim:
        for k, g in groups_tr.items():
            if all("ecapa" in d for d in g):
                # torch, so the similarities, and so the pairs, are finetune.py's bit for bit.
                S = torch.stack([d["ecapa"].float() for d in g])
                ch = [d.get("channel") for d in g]
                sims[k] = (S @ S.T, ch)
        print(f"PAIR_SIM {pair_sim}: {len(sims)} of {len(groups_tr)} speakers annotated", flush=True)
    print(f"{len(tr)} train utts in {len(keys)} speakers, {len(val)} val", flush=True)
    # Official SFT weighs the sub-talker 0.3; a shared row at 10x lr soaked up the average voice.
    subw = float(os.environ.get("SUBW", "0.3"))
    pull = float(os.environ.get("ROW_PULL", "0.01"))
    cp_lr = float(os.environ.get("CP_LR", "1"))
    talker.unfreeze(keys=["row"], recurse=False)
    if not cp_lr > 0:
        for k, l in loras.items():
            if "code_predictor" in k:
                l.freeze()
    # torch's AdamW: bias-corrected, a fresh state on RESUME, and LambdaLR's lr(s - 1) at step s.
    adam = lambda: optim.AdamW(0.0, weight_decay=0.0, bias_correction=True)
    groups = [(adam(), lr * float(os.environ.get("ROW_LR", "2")), lambda p, _: p == "row")]
    if cp_lr > 0:
        groups.append((adam(), lr * cp_lr, lambda p, _: "code_predictor" in p))
    groups.append((adam(), lr, None))
    opt = optim.MultiOptimizer([o for o, _, _ in groups], [f for _, _, f in groups[:-1]])
    lam = lambda s: min(1, s / 100) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
    rng = random.Random(1)

    def evaluate():
        tot0 = tots = 0.0
        for t in val:
            g = [d for d in groups_val.get(t["speaker"], []) if d is not t]
            ref = g[0] if g else t
            a, b = loss_of(cfg, talker, ref, t, subfrac=1.0)
            tot0 += a.item(); tots += b.item()
        mx.clear_cache()
        return round(tot0 / len(val), 4), round(tots / len(val), 4)

    if first == 1:
        print("val before", evaluate(), flush=True)
    t0 = time.time()
    for s in range(first, steps + 1):
        acc = None
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
            if split:
                seq, start, codes = build(cfg, talker, ref, tgt)
                past = prefix_kv(talker, seq, start)
            else:
                codes, past = mx.array(tgt["codes"].numpy()), None
            pick = picks(codes.shape[0])

            def objective():
                # Built inside the gradient when unsplit: the row enters through the prompt.
                s_, st_, c_ = (seq, start, codes) if split else build(cfg, talker, ref, tgt)
                a, b = losses(cfg, talker, s_, st_, c_, past, pick)
                w = subw if not sub_channel or tgt.get("channel") == sub_channel else 0.0
                # cb0_only rows (many-speaker, band-limited): pronunciation from many voices, no timbre or channel.
                w = 0.0 if tgt.get("cb0_only") else w
                return (a + w * b + pull * ((talker.row - seed_row) ** 2).sum()) / accum, (a, b)

            (_, (a, b)), grads = nn.value_and_grad(talker, objective)()
            acc = grads if acc is None else tree_map(mx.add, acc, grads)
            mx.eval(acc, a, b)
        acc, _ = optim.clip_grad_norm(acc, 1.0)
        for o, base, _ in groups:
            o.learning_rate = base * lam(s - 1)
        opt.update(talker, acc)
        mx.eval(talker.trainable_parameters(), opt.state)
        mx.clear_cache()
        if s % 25 == 0:
            print(f"step {s} ce0 {a.item():.3f} sub {b.item():.3f} {(time.time() - t0) / (s - first + 1):.1f}s/step", flush=True)
        if s % 250 == 0 or s == steps:
            print(f"val step {s}", evaluate(), flush=True)
            checkpoint(out.replace(".pt", f"-{s}.pt"), s, rank, cp_lr, talker, loras)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2:])
    else:
        export(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 1.0)
