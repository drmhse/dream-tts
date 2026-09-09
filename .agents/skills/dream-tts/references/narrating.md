# Narrating, from a sentence to a book

## One page

```sh
./dream-tts speak --text "Hello." --out hello.wav
./dream-tts speak --text-file page.md --out page.wav
./dream-tts speak --text-file page.md --out page.wav --voice voices/my-voice --seed 7
```

`--text-file` **imports and narrates** whatever it is given: EPUB, DOCX, ODT, HTML, PDF,
Markdown or plain text. Headings, tables and code blocks are handled by the narration rules
rather than read aloud verbatim. A `.txt` file is spoken literally. Use `--raw` only for text
that is already narration output.

Look at what it will say before committing to a long render:

```sh
./dream-tts narrate --stats page.md              # to stdout
./dream-tts narrate --stats -o page.txt page.md  # to a file, then --raw it
```

`--stats` reports words, characters, paragraphs and estimated duration, and warns about things
that will be spoken badly — a surviving table pipe, letters it will spell out. Fixing those in
the source is cheaper than re-rendering.

Useful options: `--max-chars` (segment budget, default 220), `--seed` for reproducibility,
`--engine`, `--quant`, `--cpu` (correct, roughly 4x slower).

## A book

The pipeline takes its chapter structure from the filesystem — one `chapter-NNN.md` per
chapter — so importing is a stage in front rather than a branch inside.

```sh
./dream-tts import book.epub --out prep/ --dry-run   # look at this before spending hours
./dream-tts import book.epub --out prep/
./dream-tts book prep/ --out narration
```

Or hand `book` the document directly and let it import:

```sh
./dream-tts book book.epub --out narration      # kick off and watch; Ctrl-C is safe
./dream-tts book book.epub --out narration      # the same command resumes
```

**Always `--dry-run` the import first.** The work is not extracting text, it is splitting one
document into chapters, and that is where it can be wrong:

| format | where the chapters come from |
|---|---|
| EPUB | the OPF spine, in reading order |
| DOCX / ODT | `w:pStyle` / `text:outline-level` |
| HTML, Markdown | heading levels |
| PDF | the outline it declares, else one chapter |

A 400-page PDF with no outline becomes one chapter, which is a wrong split rather than a
failure — worth catching before hours of synthesis.

### Controlling a run

```sh
./dream-tts jobs                              # every job on this machine, server or not
./dream-tts jobs <id>                          # watch one
./dream-tts jobs <id> --control pause          # stop after the chapter in flight, keep its work
./dream-tts jobs <id> --control pause --now    # stop within seconds, discard that chapter
./dream-tts jobs <id> --control resume
./dream-tts jobs <id> --control cancel
./dream-tts book book.epub --out narration --detach   # submit and exit
```

`book` submits to `dream-tts-serve`, which owns the engine and the queue, and starts one if none
is running. Job records are files under `data_dir/jobs/`, so `jobs` answers with no server up —
which is exactly the moment someone is about to start a second run on top of the first.

**Pause states its cost.** Plain `pause` lands on a chapter boundary, which on a reference book
can be twenty minutes away. `--now` interrupts between segments and discards the chapter in
flight, so it is narrated again from the start on resume.

**Resume needs no flag and no remembered id.** The id is a hash of the text that will be spoken
and the settings it will be spoken with, so the same command finds the run and adopts its
finished chapters. Editing a chapter changes the hash deliberately — resuming onto audio of the
old wording would be silent damage.

**No time estimate until two chapters are done.** The model is `fixed + marginal × words`,
fitted over what has finished, because a flat rate is wrong by about 2.5x in whichever direction
the sample leans: extrapolating from a real book's 27-word title page predicted 2h 09m against
an actual hour.

Everything is observable over HTTP: `GET /v1/jobs`, `/v1/jobs/<id>/events` for the stream, and
the service's own page shows every run live.

## Delivery audio

`dream-tts book` produces WAV masters, a resumable job and live progress. Publishing wants more
than that:

```sh
scripts/narrate-book.sh --document book.epub --out narration
scripts/verify-narration.py narration/*.webm
```

This adds delivery encodes and word-level alignment manifests. **It is the one part that needs
`ffmpeg`**, and it says so before it starts rather than an hour in; `--no-align` drops the only
other outside dependency. The `dream-tts` commands themselves need nothing but the binary.
Resumable per stage — a section with a WAV master is never re-synthesised.

Output per chapter: `chapter-NNN.txt` (the narration text), `.wav`, `.webm`, `.map.json` (the
word map a player consumes) and `.manifest.json`.

## What to expect it to cost

A 16-hour document is about 2.5 hours of synthesis at `qwen3tts`'s 0.158, against roughly 12 at
`cosyvoice`'s 0.716. Alignment adds about an hour either way.

**Speed depends on length, because batching does.** The same voice is RTF 0.397 on a 132-word
passage and 0.148 on a 1612-word chapter: 48 segments decode as one batch, and a short passage
has nothing to batch. Never benchmark this engine on a paragraph.

## When the output is wrong

- **Babble at the end of a segment.** The engine warns: *"segment N runs long — the talker kept
  generating after its text ran out"*. A lane past twice the voice's frames-per-character is now
  cut rather than left running, so this is bounded, but the warning still names the segment.
- **A metallic buzz.** The engine warns about degenerate repetition — few distinct codebook-0
  values across many frames. Usually a bad segment of source text.
- **A wrong-sounding clone.** Check the transcript passed to `export_voice.py` matches the clip
  exactly, and that the voice asset belongs to the engine being used.
- **Everything suddenly slow.** Check for swap. Two resident engines do not fit in 16 GB, and
  `--no-gpu-lock` is the flag people use just before discovering why the lock exists.
