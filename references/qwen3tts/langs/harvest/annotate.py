"""annotate.py IN.pt CLIPS.jsonl OUT.pt — add each encoded row's SpeechBrain ECAPA vector and channel.

finetune.py's PAIR_SIM pairs a target only with references as close as this: a reporter in the
studio and the same reporter in the field is a pair that teaches the talker that microphone and
room may change from the reference, heard as a new recording environment per sentence.
Rows carry no wav, so they are matched to CLIPS.jsonl by speaker and text.
"""
import sys, os, json, torch, torchaudio, soundfile as sf
from speechbrain.inference.speaker import EncoderClassifier

src, clips, dst = sys.argv[1:4]
m = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                   savedir=os.path.expanduser("~/.cache/speechbrain-ecapa"), run_opts={"device": "cpu"})
meta = {(r["speaker"], r["text"]): r for r in map(json.loads, open(clips))}
rows, hit = torch.load(src), 0
for r in rows:
    c = meta.get((r["speaker"], r["text"]))
    if c is None:
        continue
    x, sr = sf.read(c["wav"], dtype="float32")
    with torch.no_grad():
        e = m.encode_batch(torchaudio.functional.resample(torch.from_numpy(x), sr, 16000)[None]).reshape(-1)
    r["ecapa"] = torch.nn.functional.normalize(e, dim=0)
    r["channel"] = c.get("channel", "studio")
    hit += 1
torch.save(rows, dst)
print(f"annotated {hit} of {len(rows)}", dst)
