#!/usr/bin/env python3
"""Ground truth for the `nano` engine (drmhse/nano-edge-tts-standalone): one sentence through the
released `nano_tts.py` with every stage kept, and the phoneme ids and tags misaki gives for a few more.

The renderer's two draws (noise, then the creak jitter) are kept as draw.0 and draw.1 and replayed
by the Rust gate, so the audio is compared exactly.

    python dump_fixtures.py --nano DIR    (DIR holds nano_tts.py, config.json, nano-tts.safetensors)
"""
import argparse, json, os, sys
import torch
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument('--nano', required=True)
ap.add_argument('--voices', default='references/kokoro/weights/voices')
ap.add_argument('--out', default='fixtures/nano')
ap.add_argument('--voice', default='af_heart')
ap.add_argument('--text', default='The lighthouse keeper climbed the stairs, and lit the lamp before the first ships came into view.')
a = ap.parse_args()
sys.path.insert(0, a.nano)
import nano_tts as nt

tts = nt.load(a.nano)
# KPipeline falls back to espeak (GPL) when it is installed; dream-tts has no espeak, so its front end is
# compared with misaki's own.
from kokoro import KPipeline
for lang in 'ab':
    tts._g2p[lang] = KPipeline(lang_code=lang, model=False, repo_id=nt.KOKORO)
    tts._g2p[lang].g2p.fallback = None
tts._voices[a.voice] = torch.load(os.path.join(a.voices, f'{a.voice}.pt'), weights_only=True)
[(ids, tags)] = tts.phonemize(a.text, a.voice[0])
style = tts._voices[a.voice][len(ids) - 3]

out, draws, randn = {}, [], torch.randn


def recorded(*args, **kw):
    x = randn(*args, **kw)
    draws.append(x.reshape(-1).clone())
    return x


text_forward = tts.text.forward
tts.text.forward = lambda *x: out.__setitem__('text', text_forward(*x)) or out['text']
tts.text.dur.register_forward_hook(lambda m, i, o: out.__setitem__('dur', o))
tts.mapper.register_forward_hook(lambda m, i, o: out.__setitem__('mapper', o))
tts.renderer.head.register_forward_hook(lambda m, i, o: out.__setitem__('head', o))
nt.torch.randn = recorded
try:
    audio = tts.synth(ids, tags, style)
finally:
    nt.torch.randn = randn
assert len(draws) == 2, len(draws)
feat, f0, energy = out['text']
dur = torch.round(out['dur'].squeeze(-1).exp()).clamp(min=1)
tensors = {
    'in.ids': torch.tensor(ids, dtype=torch.float32), 'in.tags': torch.tensor(tags, dtype=torch.float32),
    'in.style': style,
    'text.feat': feat, 'text.f0': f0, 'text.energy': energy, 'text.dur': dur,
    'draw.0': draws[0], 'draw.1': draws[1],
    'mapper': out['mapper'].transpose(1, 2), 'head': out['head'], 'audio': audio,
}
os.makedirs(a.out, exist_ok=True)
save_file({k: v.float().contiguous() for k, v in tensors.items()}, os.path.join(a.out, 'standalone.safetensors'))

# The front end's half: ids and tags as the released code builds them, for the Rust frontend to match.
texts = [
    a.text,
    'He read it yesterday; will you read it tomorrow?',
    'Dr. Smith paid $3.50 for 2 coffees at 9:30 a.m.',
    '"Well," she said, "that\'s the long-awaited end — finally!"',
    'The quick brown fox jumps over the lazy dog.',
]
rows = []
for voice in ('af_heart', 'bf_emma'):
    for t in texts:
        for ids_, tags_ in tts.phonemize(t, voice[0]):
            rows.append({'voice': voice, 'text': t, 'ids': ids_, 'tags': tags_})
json.dump({'text': a.text, 'voice': a.voice, 'seconds': audio.shape[-1] / 24000, 'frontend': rows},
          open(os.path.join(a.out, 'standalone.json'), 'w'), indent=1)
print({k: tuple(v.shape) for k, v in tensors.items()})
