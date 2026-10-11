#!/usr/bin/env python3
"""Export g2p_en's model as weights/frontend/g2p.safetensors: Kokoro's guess at a word no
lexicon has.

Kyubyong Park's g2p_en (Apache-2.0) is a GRU encoder and decoder trained on CMUdict, letters
in and ARPAbet out, 3 MB. It stands where misaki would call espeak-ng, whose GPL the
release cannot carry.

Standard library only: the pinned wheel comes from PyPI and is checked against its hash,
and the .npy arrays inside it are plain little-endian float32 behind a short header.
"""
import argparse, ast, hashlib, io, json, os, struct, urllib.request, zipfile

WHEEL = ('https://files.pythonhosted.org/packages/d7/d9/'
         'b77dc634a7a0c0c97716ba97dd0a28cbfa6267c96f359c4f27ed71cbd284/'
         'g2p_en-2.1.0-py3-none-any.whl')
SHA256 = '2a7aabf1fc7f270fcc3349881407988c9245173c2413debbe5432f4a4f31319f'
NAMES = ['enc_emb', 'enc_w_ih', 'enc_w_hh', 'enc_b_ih', 'enc_b_hh',
         'dec_emb', 'dec_w_ih', 'dec_w_hh', 'dec_b_ih', 'dec_b_hh', 'fc_w', 'fc_b']


def npy(raw):
    assert raw[:6] == b'\x93NUMPY', 'not an .npy array'
    major = raw[6]
    size, start = (struct.unpack('<H', raw[8:10])[0], 10) if major == 1 else (struct.unpack('<I', raw[8:12])[0], 12)
    header = ast.literal_eval(raw[start:start + size].decode('latin1'))
    assert header['descr'] == '<f4' and not header['fortran_order'], header
    return list(header['shape']), raw[start + size:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'weights/frontend/g2p.safetensors'))
    args = ap.parse_args()

    wheel = urllib.request.urlopen(WHEEL).read()
    assert hashlib.sha256(wheel).hexdigest() == SHA256, 'the wheel is not the one pinned'
    wheel = zipfile.ZipFile(io.BytesIO(wheel))
    npz = zipfile.ZipFile(io.BytesIO(wheel.read('g2p_en/checkpoint20.npz')))

    header, blobs, offset = {}, [], 0
    for name in NAMES:
        shape, data = npy(npz.read(f'{name}.npy'))
        header[name] = {'dtype': 'F32', 'shape': shape, 'data_offsets': [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    header['__metadata__'] = {'source': 'g2p_en 2.1.0 checkpoint20.npz', 'license': 'Apache-2.0'}
    text = json.dumps(header, separators=(',', ':')).encode()
    text += b' ' * (-len(text) % 8)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        f.write(struct.pack('<Q', len(text)) + text + b''.join(blobs))
    # Apache-2.0 travels with the weights, and every bundle copies this directory whole.
    with open(os.path.splitext(args.out)[0] + '.LICENSE', 'wb') as f:
        f.write(wheel.read('g2p_en-2.1.0.dist-info/LICENSE.txt'))
    print(f'wrote {args.out}: {offset / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
