#!/usr/bin/env python3
"""docs/engines-chapter.svg: each engine's time for the 1612-word chapter, over the length of
the speech it produced. The faster voice of each, from docs/reference.md, M4 / 16 GB."""
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs"

# engine, note, wall seconds, audio seconds
ROWS = [
    ("kokoro", "built-in voices", 26.9, 12 * 60 + 15),
    ("qwen3tts", "the default, clones", 72, 11 * 60 + 35),
    ("audio8", "clones, 44.1 kHz", 5 * 60 + 47, 10 * 60 + 59),
    ("cosyvoice", "clones, most languages", 8 * 60 + 15, 11 * 60 + 44),
]

W, H = 620, 252
LEFT, RIGHT, TOP, ROW = 150, 28, 70, 40
SPAN = 13 * 60


def clock(s):
    m, s = divmod(round(s), 60)
    return f"{m}m {s:02d}s" if m else f"{s} s"


def speed(r):
    return f"{r:.0f}" if r >= 10 else f"{r:.1f}"


def chart():
    x = lambda s: LEFT + s / SPAN * (W - LEFT - RIGHT)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        'role="img" aria-label="Time to narrate a 1612-word chapter, per engine, against the length of the speech">',
        """<style>
  text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  .surface { fill: #fcfcfb; }
  .title   { fill: #0b0b0b; font-size: 15px; font-weight: 600; }
  .sub     { fill: #52514e; font-size: 11.5px; }
  .name    { fill: #0b0b0b; font-size: 12.5px; font-weight: 600; }
  .note    { fill: #52514e; font-size: 11px; }
  .axis    { fill: #898781; font-size: 11px; }
  .value   { fill: #0b0b0b; font-size: 11.5px; font-weight: 600; }
  .grid    { stroke: #e1e0d9; stroke-width: 1; }
  .speech  { fill: #e1e0d9; }
  .work    { fill: #2a78d6; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; }
    .title, .name, .value { fill: #ffffff; }
    .sub, .note { fill: #c3c2b7; }
    .grid    { stroke: #2c2c2a; }
    .speech  { fill: #383835; }
    .work    { fill: #3987e5; }
  }
</style>""",
        f'<rect class="surface" width="{W}" height="{H}" rx="6"/>',
        f'<text class="title" x="20" y="26">Narrating a 1612-word chapter</text>',
        f'<text class="sub" x="20" y="44">Blue is the time it took; grey is the speech it produced. M4 / 16 GB.</text>',
    ]
    bottom = TOP + ROW * len(ROWS)
    for m in range(0, 14, 2):
        out.append(f'<line class="grid" x1="{x(m * 60):.1f}" y1="{TOP - 6}" x2="{x(m * 60):.1f}" y2="{bottom}"/>')
        out.append(f'<text class="axis" x="{x(m * 60):.1f}" y="{bottom + 16}" text-anchor="middle">{m} min</text>')
    for i, (name, note, wall, audio) in enumerate(ROWS):
        y = TOP + i * ROW
        out.append(f'<text class="name" x="20" y="{y + 13}">{name}</text>')
        out.append(f'<text class="note" x="20" y="{y + 27}">{note}</text>')
        out.append(f'<rect class="speech" x="{LEFT}" y="{y + 2}" width="{x(audio) - LEFT:.1f}" height="24" rx="3"/>')
        out.append(f'<rect class="work" x="{LEFT}" y="{y + 2}" width="{max(3, x(wall) - LEFT):.1f}" height="24" rx="3"/>')
        out.append(f'<text class="value" x="{x(wall) + 8:.1f}" y="{y + 18}">{clock(wall)}  ·  {speed(audio / wall)}× realtime</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


(OUT / "engines-chapter.svg").write_text(chart())
