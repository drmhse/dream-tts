#!/usr/bin/env python3
"""Ground truth for the nano decoder port: Kokoro's decoder inputs for one sentence, nano's
mapper output, pulse train, head output and audio, and the two random draws it made.

The draws (noise, then the creak jitter) are captured as draw.0 and draw.1 and replayed by the Rust
gate, so the audio is compared exactly.

    python dump_nano.py --nano DIR    (DIR holds nano_edge_tts.py from drmhse/nano-edge-tts)
"""
import argparse, json, os, sys
import torch
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument('--nano', required=True)
ap.add_argument('--weights', default='references/kokoro/weights')
ap.add_argument('--out', default='fixtures/kokoro')
ap.add_argument('--voice', default='af_heart')
ap.add_argument('--text', default='The lighthouse keeper climbed the stairs, and lit the lamp before the first ships came into view.')
a = ap.parse_args()
sys.path.insert(0, a.nano)
import nano_edge_tts as ne
from kokoro import KModel, KPipeline

torch.manual_seed(0)
model = KModel(repo_id='hexgrad/Kokoro-82M', config=os.path.join(a.weights, 'config.json'),
               model=os.path.join(a.weights, 'kokoro-v1_0.pth')).eval()
inputs = {}


class Capture(torch.nn.Module):
    def forward(self, asr, F0, N, s):
        inputs.update(asr=asr, f0=F0, energy=N, style=s)
        return torch.zeros(1, F0.shape[-1] * 300)


model.decoder = Capture()
pipe = KPipeline(lang_code=a.voice[0], model=False, repo_id='hexgrad/Kokoro-82M')
phonemes = next(r.phonemes for r in pipe(a.text) if r.phonemes)
pack = torch.load(os.path.join(a.weights, 'voices', f'{a.voice}.pt'), weights_only=True)
with torch.no_grad():
    model(phonemes, pack[len(phonemes) - 1])

nano = ne.load_decoder(os.path.join(a.weights, 'nano')).eval()
draws, randn = [], torch.randn


def recorded(*args, **kw):
    x = randn(*args, **kw)
    draws.append(x.reshape(-1).clone())
    return x


out = {}
nano.mapper.register_forward_hook(lambda m, i, o: out.__setitem__('mapper', o))
nano.renderer.head.register_forward_hook(lambda m, i, o: out.__setitem__('head', o))
ne.torch.randn = recorded
try:
    with torch.no_grad():
        audio = nano(inputs['asr'], inputs['f0'], inputs['energy'], inputs['style'])
finally:
    ne.torch.randn = randn
assert len(draws) == 2, len(draws)
f0 = inputs['f0']
tensors = {
    'in.asr': inputs['asr'], 'in.f0': f0, 'in.energy': inputs['energy'], 'in.style': inputs['style'],
    'draw.0': draws[0], 'draw.1': draws[1],
    'mapper': out['mapper'].transpose(1, 2), 'head': out['head'],
    'pulse': ne.pulse_train(f0)[0], 'creak': ne.creak_train(f0, nano.renderer.creak)[0],
    'audio': audio[0],
}
os.makedirs(a.out, exist_ok=True)
save_file({k: v.float().contiguous() for k, v in tensors.items()}, os.path.join(a.out, 'nano.safetensors'))
json.dump({'text': a.text, 'phonemes': phonemes, 'voice': a.voice, 'seconds': audio.shape[-1] / 24000},
          open(os.path.join(a.out, 'nano.json'), 'w'), indent=1)
print({k: tuple(v.shape) for k, v in tensors.items()})
