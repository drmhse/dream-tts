---
name: dream-tts
description: Set up and drive dream-tts, the offline text-to-speech engine at ~/Desktop/projects/AI/tts/tts-rs — narrate a page, a chapter or a whole book in a cloned voice, on Apple silicon with no Python at runtime. Use when the user asks to narrate, synthesise, voice or read aloud a document, to build or use a voice clone, to resume or control a narration job, or to explain how the engine works. Do not use for editing prose, for other TTS tools, or for the ck blog's publishing workflow.
---

# dream-tts

Offline TTS in Rust with Metal kernels. Four engines behind one CLI; `qwen3tts` is the default
and the one to use unless something specific rules it out. `kokoro` is the exception worth
knowing: 23x realtime in 1.3 GB, English only, and it cannot clone a voice.

Repository: `~/Desktop/projects/AI/tts/tts-rs`. Read `AGENTS.md` in the repo root first; it
carries the rules that hold regardless of the task. `README.md` is the canonical guide and
`docs/reference.md` is the deep one — read the relevant section before making claims about
numbers, because both carry measurements that this skill only summarises.

## Which reference

- **Installing, weights, voices, disk and memory:** `references/setup.md`.
- **Narrating anything, one page to one book:** `references/narrating.md`.
- **Architecture, and where the time goes:** `references/how-it-works.md`.

## The three things people actually ask for

**A sentence or a page.** One command, and the text is imported and segmented for you:

```sh
./dream-tts speak --text-file page.md --out page.wav
./dream-tts speak --text "Hello." --out hello.wav
```

**A book.** Look at the split before spending hours on it, then submit:

```sh
./dream-tts import book.epub --out chapters --dry-run   # check this first
./dream-tts book book.epub --out narration              # watch; Ctrl-C is safe
./dream-tts book book.epub --out narration              # the same command resumes
```

**A voice.** Ten seconds of clean audio and its exact transcript, once per voice, and this is
the only step that wants Python:

```sh
references/qwen3tts/.venv/bin/python references/qwen3tts/export_voice.py \
    --model references/qwen3tts/weights --audio clip.wav \
    --text "the exact words spoken in the clip" \
    --name my-voice --out voices/my-voice
```

## Things to get right

- **`--text-file` narrates; `--raw` does not.** Any document it can import is imported —
  markdown headings, tables and code blocks are stripped or spoken according to the narration
  rules. A `.txt` file is spoken literally. Pass `--raw` only for text that is already narration
  output. When in doubt run `./dream-tts narrate --stats <file>` first and read what it will say.
- **Judge speed on length, never on a paragraph.** `qwen3tts` batches across segments, so it is
  RTF 0.397 on 132 words and 0.148 on a chapter. A slow-looking short render is the engine
  working correctly.
- **It wants 16 GB.** Peak is 12.3–13.0 GB and almost all of it is the codec's activations, so
  no weight format moves it. On a smaller machine use `--engine cosyvoice` (5.0 GB). Do not
  reach for `--quant q8_0` to save memory — it saves 0.58 GB and costs 62% of the speed.
- **One synthesis at a time per install.** An advisory lock in `data_dir` refuses a second, by
  design: two resident engines swap rather than fail, which reads as slowness rather than as a
  mistake.
- **Resume is the same command, and the job id is a hash of the text and settings.** Editing a
  chapter changes the id on purpose; resuming onto audio of the old wording would be silent
  damage.

## Before reporting a number

Every RTF in this repository was measured on one M4 with 16 GB. If asked to verify or improve
performance, read `docs/reference.md#performance` first, and know that this machine drifts
thermally by more than most changes are worth — the harness canary moved 59 to 65 ms across one
afternoon. Interleave A/B runs in one process rather than comparing across sessions, and note
that `cargo build -p qwen3tts` does not relink `dream-tts`, which lives in `tts-cli`.
