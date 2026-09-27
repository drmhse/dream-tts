"""Text -> the pieces a language's talker is trained on, from a scheme file (langs/<language>.json).

Each piece is BPE-encoded alone. Mirrored by `syllables::Scheme` in the engine, which reads the
same JSON from the adapter's `meta::scheme`, so training and inference tokenize alike.

- vowels: a piece ends after each vowel a letter follows (open syllables; whole-word BPE split
  Swahili "bunifu" as b+unifu, read "bimoni")
- apostrophes: folded to ' first
- map: an onset replaced by its own piece; text streams a token a frame, so Swahili ng' is ŋ,
  or the talker has committed to [ŋg] on "ng" before the apostrophe arrives
- isolate_map: a mapped onset is always the same lone lowercase token, its leading space a piece
  of its own; 'Ŋ', ' ŋ' and 'ŋ' are three different tokens, and sentence-initial 'Ŋ', the rarest,
  was dropped 8 times in 12 against 0 in 18 mid-sentence
- onsets: split off as their own piece, longest first: " mbi" is ' mb'+'i' (one Qwen token)
  rather than ' m'+'bi', the shape of syllabic m, read "mibili"; "ja" is German's /ja/
- syllabic: a nasal before a consonant (not a glide) is a piece alone
"""
import json


class Scheme:
    def __init__(self, spec):
        s = json.load(open(spec)) if isinstance(spec, str) else spec
        self.language, self.row = s["language"], int(s["row"])
        self.vowels = set(s["vowels"] + s["vowels"].upper())
        self.fold = str.maketrans({c: "'" for c in s.get("apostrophes", "")})
        self.map = sorted(s.get("map", {}).items(), key=lambda kv: -len(kv[0]))
        self.onsets = sorted(s.get("onsets", []), key=len, reverse=True)
        self.syllabic, self.glides = s.get("syllabic", ""), s.get("glides", "")
        self.isolate = bool(s.get("isolate_map"))
        self.spec = s

    def syllables(self, text):
        out, cur = [], ""
        for i, c in enumerate(text):
            if c.isspace() and cur.strip():
                out.append(cur); cur = ""
            cur += c
            if c in self.vowels and i + 1 < len(text) and text[i + 1].isalpha():
                out.append(cur); cur = ""
        if cur:
            out.append(cur)
        return out

    def split(self, p):
        body = p.lstrip()
        ws, low = p[: len(p) - len(body)], body.lower()
        for k, v in self.map:
            if low.startswith(k):
                head = ([ws] if ws else []) + [v] if self.isolate else [ws + (v.upper() if body[0].isupper() else v)]
                return head + ([body[len(k):]] if len(body) > len(k) else [])
        for o in self.onsets:
            if low.startswith(o) and len(body) > len(o):
                return [ws + body[: len(o)], body[len(o):]]
        if (len(body) > 1 and low[0] in self.syllabic and body[1].isalpha()
                and low[1] not in self.vowels and low[1] not in self.glides):
            return [ws + body[0], body[1:]]
        return [p]

    def pieces(self, text):
        return [q for p in self.syllables(text.translate(self.fold)) for q in self.split(p)]


if __name__ == "__main__":
    # orthography.py parity SCHEME.json TEXTS.jsonl OUT.jsonl — pieces for the engine's parity test.
    import sys
    _, _, spec, src, dst = sys.argv
    sch = Scheme(spec)
    with open(dst, "w") as f:
        for line in open(src):
            t = json.loads(line)["text"]
            f.write(json.dumps({"scheme": sch.spec, "text": t, "pieces": sch.pieces(t)}, ensure_ascii=False) + "\n")
