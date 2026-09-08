#!/usr/bin/env python3
"""Renders the two lane-sweep charts in docs/ from the measurements below.

Data is inline and dated because a chart nobody can regenerate is a chart nobody
can correct. Both sweeps: one M4 / 16 GB, f16 weights, canary 59.4 ms (cool).
"""

import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs"

# End to end, `ck` pixel-watch article: 4763 words, 201 segments, 28m 31s of audio.
# (lanes, RTF, swapping)
E2E = [(24, 0.235, False), (48, 0.200, False), (56, 0.701, True), (64, 0.647, True)]
# Talker trunk alone, span 412 positions, n=5 interleaved. (batch, ms/lane, swapping)
PER_LANE = [
    (8, 5.466, False), (16, 4.661, False), (24, 3.222, False), (32, 2.520, False),
    (48, 1.816, False), (64, 1.470, False), (96, 17.450, True), (128, 25.496, True),
]

W, H = 620, 320
L, R, T, B = 62, 18, 52, 52
PW, PH = W - L - R, H - T - B

# Light values live on the bare classes, dark under prefers-color-scheme, per the
# palette's two-scope rule minus the data-theme scope: an SVG in an <img> has no host root.
STYLE = """
  text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  .surface { fill: #fcfcfb; }
  .title   { fill: #0b0b0b; font-size: 15px; font-weight: 600; }
  .sub     { fill: #52514e; font-size: 11.5px; }
  .axis    { fill: #898781; font-size: 11px; }
  .value   { fill: #0b0b0b; font-size: 11.5px; font-weight: 600; }
  .note    { fill: #52514e; font-size: 11px; }
  .grid    { stroke: #e1e0d9; stroke-width: 1; }
  .base    { stroke: #c3c2b7; stroke-width: 1; }
  .ok      { fill: #2a78d6; }
  .okline  { stroke: #2a78d6; fill: none; stroke-width: 2; }
  .bad     { fill: #d03b3b; }
  .badline { stroke: #d03b3b; fill: none; stroke-width: 2; }
  .ring    { stroke: #fcfcfb; stroke-width: 2; }
  .onbar   { fill: #fcfcfb; font-size: 11px; }
  .ref     { stroke: #52514e; stroke-width: 1.5; stroke-dasharray: 5 4; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; }
    .title   { fill: #ffffff; }
    .sub, .note { fill: #c3c2b7; }
    .grid    { stroke: #2c2c2a; }
    .base    { stroke: #383835; }
    .value   { fill: #ffffff; }
    .ok      { fill: #3987e5; }
    .okline  { stroke: #3987e5; }
    .bad     { fill: #e66767; }
    .badline { stroke: #e66767; }
    .ring    { stroke: #1a1a19; }
    .ref     { stroke: #c3c2b7; }
  }
"""


def head(title, sub):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" role="img" aria-label="{title}. {sub}">',
        f"<style>{STYLE}</style>",
        f'<rect class="surface" width="{W}" height="{H}" rx="6"/>',
        f'<text class="title" x="{L}" y="24">{title}</text>',
        f'<text class="sub" x="{L}" y="41">{sub}</text>',
    ]


def legend():
    x, y = W - R - 128, 24
    return [
        f'<circle class="ok" cx="{x}" cy="{y - 4}" r="4"/>',
        f'<text class="note" x="{x + 9}" y="{y}">renders</text>',
        f'<circle class="bad" cx="{x + 66}" cy="{y - 4}" r="4"/>',
        f'<text class="note" x="{x + 75}" y="{y}">swaps</text>',
    ]


def top_rounded(x, y, w, h, r=4):
    r = min(r, h)
    return (
        f'M{x},{y + h} L{x},{y + r} Q{x},{y} {x + r},{y} L{x + w - r},{y} '
        f'Q{x + w},{y} {x + w},{y + r} L{x + w},{y + h} Z'
    )


def rtf_chart():
    ymax = 0.75
    y = lambda v: T + PH * (1 - v / ymax)
    s = head(
        "End-to-end RTF by lane count",
        "RTF  ·  4763-word article, 201 segments, 28m 31s of audio  ·  f16, M4 / 16 GB  ·  lower is faster",
    )
    for t in (0, 0.25, 0.5, 0.75):
        s.append(f'<line class="grid" x1="{L}" y1="{y(t):.1f}" x2="{L + PW}" y2="{y(t):.1f}"/>')
        s.append(f'<text class="axis" x="{L - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{t:.2f}</text>')

    band = PW / len(E2E)
    bw = 62
    for i, (lanes, rtf, bad) in enumerate(E2E):
        cx = L + band * (i + 0.5)
        h = PH * rtf / ymax
        cls = "bad" if bad else "ok"
        s.append(f'<path class="{cls}" d="{top_rounded(cx - bw / 2, y(rtf), bw, h)}"/>')
        s.append(f'<text class="value" x="{cx:.1f}" y="{y(rtf) - 8:.1f}" text-anchor="middle">{rtf:.3f}</text>')
        if bad:
            s.append(f'<text class="onbar" x="{cx:.1f}" y="{y(rtf) + 18:.1f}" text-anchor="middle">swap</text>')
        s.append(f'<text class="axis" x="{cx:.1f}" y="{T + PH + 18}" text-anchor="middle">{lanes}</text>')

    s.append(f'<line class="base" x1="{L}" y1="{T + PH}" x2="{L + PW}" y2="{T + PH}"/>')
    s.append(f'<text class="axis" x="{L + PW / 2:.1f}" y="{T + PH + 36}" text-anchor="middle">QWEN3TTS_MAX_BATCH (lanes)</text>')
    # The threshold the whole exercise is about.
    s.append(f'<line class="ref" x1="{L}" y1="{y(0.2):.1f}" x2="{L + PW}" y2="{y(0.2):.1f}"/>')
    s.append(f'<text class="note" x="{L + PW}" y="{y(0.2) - 7:.1f}" text-anchor="end">5× realtime = RTF 0.200</text>')
    s += legend()
    s.append("</svg>")
    return "\n".join(s)


def per_lane_chart():
    xs = [b for b, _, _ in PER_LANE]
    lo, hi = math.log2(min(xs)), math.log2(max(xs))
    x = lambda b: L + PW * (math.log2(b) - lo) / (hi - lo)
    ylo, yhi = math.log10(1), math.log10(32)
    y = lambda v: T + PH * (1 - (math.log10(v) - ylo) / (yhi - ylo))

    s = head(
        "Cost per lane, talker trunk alone",
        "ms per lane  ·  span 412 positions, n=5, one batch size per process  ·  log scales",
    )
    for t in (1, 2, 5, 10, 20):
        s.append(f'<line class="grid" x1="{L}" y1="{y(t):.1f}" x2="{L + PW}" y2="{y(t):.1f}"/>')
        s.append(f'<text class="axis" x="{L - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{t}</text>')
    for b in xs:
        s.append(f'<text class="axis" x="{x(b):.1f}" y="{T + PH + 18}" text-anchor="middle">{b}</text>')

    good = [(b, v) for b, v, bad in PER_LANE if not bad]
    bad = [(b, v) for b, v, bad in PER_LANE if bad]
    s.append('<polyline class="okline" points="' + " ".join(f"{x(b):.1f},{y(v):.1f}" for b, v in good) + '"/>')
    # Dashed across the discontinuity: the two regimes are not one trend.
    s.append(
        f'<line class="badline" stroke-dasharray="5 4" x1="{x(good[-1][0]):.1f}" y1="{y(good[-1][1]):.1f}" '
        f'x2="{x(bad[0][0]):.1f}" y2="{y(bad[0][1]):.1f}"/>'
    )
    s.append('<polyline class="badline" points="' + " ".join(f"{x(b):.1f},{y(v):.1f}" for b, v in bad) + '"/>')
    for b, v, is_bad in PER_LANE:
        s.append(f'<circle class="ring {"bad" if is_bad else "ok"}" cx="{x(b):.1f}" cy="{y(v):.1f}" r="5"/>')
    # Selective labels: the ends of the useful range and the cliff.
    for b, v, anchor, dy in ((8, 5.466, "start", -12), (64, 1.470, "start", 18), (128, 25.496, "end", -12)):
        s.append(f'<text class="value" x="{x(b):.1f}" y="{y(v) + dy:.1f}" text-anchor="{anchor}">{v:.2f}</text>')

    s.append(f'<line class="base" x1="{L}" y1="{T + PH}" x2="{L + PW}" y2="{T + PH}"/>')
    s.append(f'<text class="axis" x="{L + PW / 2:.1f}" y="{T + PH + 36}" text-anchor="middle">batch (lanes)</text>')
    s.append(f'<text class="note" x="{L}" y="{y(1.05):.1f}">each added lane ≈ 0.4 ms, flat from 16 to 64</text>')
    s.append(f'<text class="note" x="{x(96):.1f}" y="{y(6.2):.1f}" text-anchor="middle">memory cliff</text>')
    s += legend()
    s.append("</svg>")
    return "\n".join(s)


for name, svg in (("qwen3tts-lanes-rtf", rtf_chart()), ("qwen3tts-lanes-perlane", per_lane_chart())):
    (OUT / f"{name}.svg").write_text(svg)
    print(f"docs/{name}.svg")
