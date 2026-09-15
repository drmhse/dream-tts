#!/usr/bin/env python3
"""kokoro-v1_0.pth -> safetensors, with weight normalisation folded away.

Folding at conversion rather than at load is the point: `weight_g`/`weight_v` are a
training-time reparameterisation with no inference meaning, and carrying them into Rust
would put a norm-and-scale on the critical path of every convolution for nothing.
"""
import argparse, json, os, re
import torch
from safetensors.torch import save_file

def fold_weight_norm(sd):
    """weight = g * v / ||v||, norm taken over every dimension but the output channel."""
    out, folded = {}, 0
    for k, v in sd.items():
        if k.endswith('weight_v'):
            g = sd[k[:-1] + 'g']
            dims = tuple(range(1, v.dim()))
            norm = v.norm(2, dim=dims, keepdim=True)
            out[k[:-2]] = (g * v / norm).contiguous()
            folded += 1
        elif k.endswith('weight_g'):
            continue
        else:
            out[k] = v.contiguous()
    return out, folded

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='weights/kokoro-v1_0.pth')
    ap.add_argument('--voices', default='weights/voices')
    ap.add_argument('-o', '--out', default='weights')
    args = ap.parse_args()

    sd = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    tensors, folded_total = {}, 0
    for section, part in sd.items():
        part = {re.sub(r'^module\.', '', k): v for k, v in part.items()}
        part, folded = fold_weight_norm(part)
        folded_total += folded
        for k, v in part.items():
            tensors[f'{section}.{k}'] = v.to(torch.float32)
    save_file(tensors, os.path.join(args.out, 'kokoro.safetensors'))
    total = sum(v.numel() for v in tensors.values())
    print(f'{len(tensors)} tensors, {total/1e6:.1f}M parameters, {folded_total} weight norms folded')

    # One file for every voice: 522 KB of style vectors each is not worth a download apiece,
    # and the engine has to be able to list them without fetching them.
    voices = {}
    for name in sorted(os.listdir(args.voices)):
        if not name.endswith('.pt'):
            continue
        v = torch.load(os.path.join(args.voices, name), map_location='cpu', weights_only=True)
        voices[name[:-3]] = v.squeeze(1).contiguous().to(torch.float32)
    if voices:
        save_file(voices, os.path.join(args.out, 'voices.safetensors'))
        shape = list(next(iter(voices.values())).shape)
        print(f'{len(voices)} voices, each {shape}')

if __name__ == '__main__':
    main()
