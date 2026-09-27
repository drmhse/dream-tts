"""Full-size check of finetune_mlx.py against finetune.py: ce0/sub and gradients on real pairs, then
seconds per training example. One framework per process, never both resident.

    parity_mlx.py DATA.pt [DATA.pt ...]    # ~8 GB peak (the torch half): run with the GPU otherwise idle
"""
import os, sys, re, random, subprocess, tempfile, time
import numpy as np, torch

SUBW, PULL = 0.3, 0.01
N_PARITY, N_WARM, N_TIME = 3, 2, 10


def examples(paths):
    """(ref, tgt) pairs drawn much as train draws them (no FOCUS, no PAIR_SIM)."""
    from finetune import pairs
    data = [d for p in paths for d in torch.load(p)]
    data = [d for d in data if 20 <= d["codes"].shape[0] <= 200]
    random.Random(0).shuffle(data)
    groups = pairs(data[40:])
    keys, rng, out = list(groups), random.Random(1), []
    for _ in range(N_PARITY + N_WARM + N_TIME):
        g = groups[rng.choices(keys, [len(groups[k]) for k in keys])[0]]
        i, j = rng.sample(range(len(g)), 2)
        out.append((g[j], g[i]))
    return out


def run_torch(tmp, paths):
    import finetune as ft
    m, talker, loras = ft.load(16)
    gen = torch.Generator().manual_seed(0)
    for l in loras.values():
        l.b.data = (0.002 * torch.randn(l.b.shape, generator=gen)).to(ft.dev)  # the shipped adapters' b std
    E = talker.get_input_embeddings()
    seed = torch.stack([E.weight[i] for i in m.config.talker_config.codec_language_id.values()]).float().mean(0)
    row = torch.nn.Parameter(seed.clone())
    params = [p for l in loras.values() for p in (l.a, l.b)] + [row]
    ex = examples(paths)
    with torch.no_grad():
        losses = [[x.item() for x in ft.loss_of(m, talker, row, r, t, subfrac=1.0)] for r, t in ex[:N_PARITY]]
    a, b = ft.loss_of(m, talker, row, *ex[0], subfrac=1.0)
    (a + SUBW * b + PULL * (row - seed).pow(2).sum()).backward()
    grads = {k: (l.a.grad.float().cpu(), l.b.grad.float().cpu()) for k, l in loras.items()}
    sec = []
    for n, (r, t) in enumerate(ex[N_PARITY:], 1):
        torch.mps.synchronize(); t0 = time.time()
        for p in params:
            p.grad = None
        a, b = ft.loss_of(m, talker, row, r, t)
        (a + SUBW * b + PULL * (row - seed).pow(2).sum()).backward()
        if n % 4 == 0:
            torch.mps.empty_cache()
        torch.mps.synchronize(); sec.append(time.time() - t0)
    torch.save({"loras": {k: (l.a.detach().float().cpu(), l.b.detach().float().cpu()) for k, l in loras.items()},
                "row": row.detach().cpu(), "losses": losses, "grads": grads, "sec": sec[N_WARM:]}, f"{tmp}/torch.pt")


def run_mlx(tmp, paths):
    import mlx.core as mx, mlx.nn as nn
    from mlx.utils import tree_flatten
    import finetune_mlx as fm
    cfg, talker, loras = fm.load(16)
    ck = torch.load(f"{tmp}/torch.pt")
    for k, (a, b) in ck["loras"].items():
        loras[k].a, loras[k].b = mx.array(a.numpy()), mx.array(b.numpy())
    seed = mx.array(ck["row"].numpy())
    talker.row = seed
    talker.unfreeze(keys=["row"], recurse=False)
    ex = examples(paths)
    losses = [[x.item() for x in fm.loss_of(cfg, talker, r, t, subfrac=1.0)] for r, t in ex[:N_PARITY]]

    def grads_of(ref, tgt, subfrac=None):
        seq, start, codes = fm.build(cfg, talker, ref, tgt)
        past, pick = fm.prefix_kv(talker, seq, start), fm.picks(codes.shape[0], subfrac)

        def objective():
            a, b = fm.losses(cfg, talker, seq, start, codes, past, pick)
            return a + SUBW * b + PULL * ((talker.row - seed) ** 2).sum(), (a, b)

        (_, (a, b)), g = nn.value_and_grad(talker, objective)()
        mx.eval(g, a, b)
        return g

    g = dict(tree_flatten(grads_of(*ex[0], subfrac=1.0)))
    t = lambda x: torch.from_numpy(np.array(x.astype(mx.float32)))
    grads = {k: (t(g[f"{k[7:]}.a"]), t(g[f"{k[7:]}.b"])) for k in loras}
    del g
    mx.clear_cache(); mx.reset_peak_memory()
    sec = []
    for n, (r, tg) in enumerate(ex[N_PARITY:], 1):
        t0 = time.time()
        grads_of(r, tg)
        if n % 4 == 0:
            mx.clear_cache()
        sec.append(time.time() - t0)
    torch.save({"losses": losses, "grads": grads, "sec": sec[N_WARM:], "peak": mx.get_peak_memory()}, f"{tmp}/mlx.pt")


def phase(name, tmp, paths):
    p = subprocess.run(["/usr/bin/time", "-l", sys.executable, os.path.abspath(__file__), name, tmp, *paths],
                       stderr=subprocess.PIPE, text=True)
    foot = re.search(r"(\d+)\s+peak memory footprint", p.stderr)
    if p.returncode:
        sys.exit(f"{name} phase failed:\n{p.stderr}")
    return int(foot.group(1)) / 1e9 if foot else float("nan")


def main(paths):
    busy = subprocess.run(["pgrep", "-fl", r"finetune(_mlx)?\.py train"], capture_output=True, text=True).stdout.strip()
    if busy:
        sys.exit(f"a training run holds the GPU, not starting:\n{busy}")
    with tempfile.TemporaryDirectory() as tmp:
        ft_peak, mx_peak = phase("torch", tmp, paths), phase("mlx", tmp, paths)
        T, M = torch.load(f"{tmp}/torch.pt"), torch.load(f"{tmp}/mlx.pt")
    ex = examples(paths)
    worst = 0.0
    for n, ((r, t), lt, lm) in enumerate(zip(ex, T["losses"], M["losses"])):
        d = [abs(x - y) / abs(x) for x, y in zip(lt, lm)]
        worst = max(worst, *d)
        print(f"pair {n} ({r['codes'].shape[0]} ref + {t['codes'].shape[0]} target frames): "
              f"ce0 torch {lt[0]:.4f} mlx {lm[0]:.4f} ({d[0]:.1e}), sub torch {lt[1]:.4f} mlx {lm[1]:.4f} ({d[1]:.1e})")
    print(f"losses: worst relative difference {worst:.1e} -> {'PASS' if worst < 1e-2 else 'FAIL'} at 1e-2")
    gt, gm = T["grads"], M["grads"]
    rel = sorted(float((gm[k][i] - gt[k][i]).norm() / gt[k][i].norm()) for k in gt for i in (0, 1) if gt[k][i].norm() > 0)
    ft_all = torch.cat([gt[k][i].flatten() for k in gt for i in (0, 1)])
    mx_all = torch.cat([gm[k][i].flatten() for k in gt for i in (0, 1)])
    print(f"grads over {len(gt)} LoRAs (pair 0, all frames): cosine {float(torch.nn.functional.cosine_similarity(ft_all, mx_all, 0)):.5f}, "
          f"|g| mlx/torch {float(mx_all.norm() / ft_all.norm()):.4f}, per-tensor relative error median {rel[len(rel) // 2]:.1e} worst {rel[-1]:.1e}")
    frames = np.mean([r["codes"].shape[0] + t["codes"].shape[0] for r, t in ex[N_PARITY + N_WARM:]])
    ts, ms = np.mean(T["sec"]), np.mean(M["sec"])
    print(f"speed, {N_TIME} examples after {N_WARM} warm-up (mean {frames:.0f} ref+target frames, SUBFRAC as trained): "
          f"torch {ts:.3f} s/example, mlx {ms:.3f} s/example, {ts / ms:.2f}x")
    print(f"peak footprint: torch {ft_peak:.1f} GB, mlx {mx_peak:.1f} GB (mlx allocator peak while timing {M['peak'] / 1e9:.1f} GB)")


if __name__ == "__main__":
    if sys.argv[1] in ("torch", "mlx"):
        {"torch": run_torch, "mlx": run_mlx}[sys.argv[1]](sys.argv[2], sys.argv[3:])
    else:
        main(sys.argv[1:])
