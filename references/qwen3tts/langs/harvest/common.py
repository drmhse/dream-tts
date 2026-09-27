"""Shared by the harvest stages: HARVEST=langs/<language>-harvest.json, HARVEST_DIR=the data directory."""
import os, json, subprocess, unicodedata
import numpy as np

CFG = json.load(open(os.environ["HARVEST"]))
D = os.environ["HARVEST_DIR"]
os.makedirs(f"{D}/chunks", exist_ok=True)


def load(path, sr):
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(pcm, np.float32).copy()


def romanize(text):
    """MMS_FA's alphabet is a-z and '; diacritics (Kikuyu ĩ ũ, tone marks) are dropped."""
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if not unicodedata.combining(c))
