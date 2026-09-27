"""fetch.py — the config's YouTube searches, kept to its channel, title prefix and length, downloaded to raw/.

yt-dlp goes stale against YouTube (403s): `yt-dlp -U` first.
"""
import os, subprocess
from common import CFG, D

YT = os.environ.get("YTDLP", "yt-dlp")
os.makedirs(f"{D}/raw", exist_ok=True)
ids = []
for q in CFG["queries"]:
    out = subprocess.run([YT, "--flat-playlist", "--print", "%(id)s\t%(duration)s\t%(channel)s\t%(title)s", f"ytsearch60:{q}"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        vid, dur, chan, title = (line.split("\t") + ["", "", ""])[:4]
        if (chan == CFG.get("channel", chan) and title.upper().startswith(CFG.get("title_prefix", "").upper())
                and dur not in ("NA", "") and float(dur) >= CFG.get("min_seconds", 0) and vid not in ids):
            ids.append(vid)
open(f"{D}/ids.txt", "w").write("\n".join(ids) + "\n")
print(len(ids), "videos")
subprocess.run([YT, "-f", "bestaudio", "-x", "--audio-format", "m4a", "-o", f"{D}/raw/%(id)s.%(ext)s",
                "--download-archive", f"{D}/archive.txt", "-a", f"{D}/ids.txt"])
