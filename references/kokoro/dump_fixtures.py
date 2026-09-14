#!/usr/bin/env python3
"""Per-stage ground truth for the Kokoro port.

Stage-by-stage rather than audio-only, for the reason every other engine in this repo is
gated that way: a waveform that is wrong says nothing about which of thirty modules moved.

The source module draws random phase and noise. Those draws are captured here and replayed
by the Rust gate, so the whole path is compared exactly instead of the deterministic part
being compared and the stochastic part being hoped about.
"""
import argparse, json, os
import torch
from safetensors.torch import load_file, save_file
from kokoro import KModel

CAPTURE = {
    'bert': 'bert',
    'bert_encoder': 'bert_encoder',
    'dur_enc': 'predictor.text_encoder',
    'dur_lstm': 'predictor.lstm',
    'duration_proj': 'predictor.duration_proj',
    'shared': 'predictor.shared',
    'F0_proj': 'predictor.F0_proj',
    'N_proj': 'predictor.N_proj',
    'text_encoder': 'text_encoder',
    'dec_F0_conv': 'decoder.F0_conv',
    'dec_N_conv': 'decoder.N_conv',
    'dec_encode': 'decoder.encode',
    'dec_asr_res': 'decoder.asr_res',
    'dec_decode_0': 'decoder.decode.0',
    'dec_decode_1': 'decoder.decode.1',
    'dec_decode_2': 'decoder.decode.2',
    'dec_decode_3': 'decoder.decode.3',
    'gen_source': 'decoder.generator.m_source',
    'gen_up_0': 'decoder.generator.ups.0',
    'gen_up_1': 'decoder.generator.ups.1',
    'gen_noise_conv_0': 'decoder.generator.noise_convs.0',
    'gen_noise_conv_1': 'decoder.generator.noise_convs.1',
    'gen_noise_res_0': 'decoder.generator.noise_res.0',
    'gen_noise_res_1': 'decoder.generator.noise_res.1',
    'gen_conv_post': 'decoder.generator.conv_post',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phonemes', default='ðə kwˈɪk bɹˈWn fˈɑks jˈʌmps ˌOvəɹ ðə lˈAzi dˈɔɡ.')
    ap.add_argument('--voice', default='af_heart')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('-o', '--out', default='../../fixtures/kokoro')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = KModel(repo_id='hexgrad/Kokoro-82M', config='weights/config.json',
                   model='weights/kokoro-v1_0.pth').eval()

    captured, draws = {}, []

    def hook(name):
        def fn(_mod, _inp, out):
            if isinstance(out, tuple):
                for i, o in enumerate(out):
                    if torch.is_tensor(o):
                        captured[f'{name}.{i}'] = o.detach().float().contiguous()
            elif torch.is_tensor(out):
                captured[name] = out.detach().float().contiguous()
        return fn

    by_name = dict(model.named_modules())
    for label, path in CAPTURE.items():
        by_name[path].register_forward_hook(hook(label))

    # Record every random draw in call order; the gate replays them.
    torch.manual_seed(args.seed)
    real_rand, real_randn_like = torch.rand, torch.randn_like

    def rand(*a, **kw):
        t = real_rand(*a, **kw)
        draws.append(t.detach().float().contiguous())
        return t

    def randn_like(t, *a, **kw):
        r = real_randn_like(t, *a, **kw)
        draws.append(r.detach().float().contiguous())
        return r

    torch.rand, torch.randn_like = rand, randn_like
    try:
        voices = load_file('weights/voices.safetensors')
        ids = [0] + [model.vocab[p] for p in args.phonemes if p in model.vocab] + [0]
        ref_s = voices[args.voice][len(ids) - 3].unsqueeze(0)
        with torch.no_grad():
            audio, pred_dur = model.forward_with_tokens(torch.LongTensor([ids]), ref_s, 1.0)
    finally:
        torch.rand, torch.randn_like = real_rand, real_randn_like

    captured['input_ids'] = torch.LongTensor([ids]).float()
    captured['ref_s'] = ref_s.float()
    captured['pred_dur'] = pred_dur.float()
    captured['audio'] = audio.detach().float()
    for i, d in enumerate(draws):
        captured[f'draw.{i}'] = d

    save_file(captured, os.path.join(args.out, 'forward.safetensors'))
    meta = {
        'phonemes': args.phonemes,
        'voice': args.voice,
        'seed': args.seed,
        'input_ids': ids,
        'draws': len(draws),
        'shapes': {k: list(v.shape) for k, v in sorted(captured.items())},
    }
    with open(os.path.join(args.out, 'forward.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'{len(captured)} tensors, {len(draws)} random draws, audio {list(audio.shape)}')
    for k in sorted(captured):
        if not k.startswith('draw.'):
            print(f'  {k:22s} {list(captured[k].shape)}')


if __name__ == '__main__':
    main()
