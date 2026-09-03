#!/usr/bin/env bash
#
# Download a model checkpoint with real byte progress, resume, and verification.
#
#   scripts/fetch-weights.sh <dest-dir> <base-url> <relative-path>...
#
# Three properties the plain `curl -fsSL` loop this replaces did not have:
#
# 1. **Byte progress across the whole set.** Every file's size is fetched up front with a
#    HEAD, so the bar knows the total before the first byte arrives. A per-file percentage
#    is useless when file 7 of 9 is 4 GB and the other eight are 2 KB each.
#
# 2. **Resume that is accounted for.** `curl -C -` already resumed, silently, so a resumed
#    4 GB download looked identical to a stalled one. Bytes already on disk count as
#    progress here and the summary says how many were skipped.
#
# 3. **Verification.** Hugging Face returns `x-linked-etag` for LFS-backed files, and it is
#    the sha256 of the content — so the big files are checked, not merely counted. Small
#    files have no such header and are checked by length. A file that fails is deleted
#    rather than left as a plausible-looking truncation.
#
# A manifest of verified digests is written to <dest-dir>/.weights-manifest so a re-run
# skips work without re-hashing gigabytes. Delete it to force full re-verification.
set -euo pipefail

command -v curl >/dev/null   || { echo "curl not found" >&2; exit 1; }
command -v shasum >/dev/null || { echo "shasum not found" >&2; exit 1; }

[ $# -ge 3 ] || {
  echo "usage: $0 <dest-dir> <base-url> <relative-path>..." >&2
  exit 2
}
DEST="$1"; BASE="$2"; shift 2
FILES=("$@")
MANIFEST="$DEST/.weights-manifest"

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

TTY=""; [ -t 2 ] && TTY=1

# --------------------------------------------------------------- formatting

human() {
  awk -v b="$1" 'BEGIN {
    split("B KB MB GB TB", u, " "); i = 1
    while (b >= 1024 && i < 5) { b /= 1024; i++ }
    printf (i == 1 ? "%d %s" : "%.1f %s"), b, u[i]
  }'
}

hms() {
  awk -v s="$1" 'BEGIN {
    if (s < 0 || s != s) { printf "--"; exit }
    s = int(s + 0.5)
    if (s < 60) { printf "%ds", s }
    else if (s < 3600) { printf "%dm %02ds", s / 60, s % 60 }
    else { printf "%dh %02dm", s / 3600, (s % 3600) / 60 }
  }'
}

size_of() { stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null || echo 0; }

# --------------------------------------------------------------- head requests
#
# One HEAD per file, before anything downloads, so the total is known up front. Costs a
# round trip each and buys a progress bar that means something.

say "Checking $(printf '%s' "${#FILES[@]}") file(s) on the server"
declare -a SIZES=() DIGESTS=()
TOTAL=0
for f in "${FILES[@]}"; do
  # -L to follow HF's redirect to its CDN, which is where the real headers live.
  headers="$(curl -fsSLI "$BASE/$f" 2>/dev/null)" || die \
    "cannot reach $BASE/$f — check the URL and your connection"
  # Last occurrence wins: a redirect chain repeats these, and the final hop is the truth.
  size="$(printf '%s' "$headers" | awk 'BEGIN{IGNORECASE=1} /^content-length:/ {gsub(/\r/,""); v=$2} END{print v+0}')"
  etag="$(printf '%s' "$headers" | awk 'BEGIN{IGNORECASE=1} /^x-linked-etag:/ {gsub(/[\r"]/,""); v=$2} END{print v}')"
  # Only a 64-hex value is a sha256. HF omits this header for non-LFS files and can send a
  # multipart etag for others; treating either as a digest would fail every small file.
  case "$etag" in
    ????????????????????????????????????????????????????????????????) ;;
    *) etag="" ;;
  esac
  case "$etag" in
    *[!0-9a-f]*) etag="" ;;
  esac
  SIZES+=("$size")
  DIGESTS+=("$etag")
  TOTAL=$((TOTAL + size))
done
say "Total to fetch: $(human "$TOTAL")"

# --------------------------------------------------------------- progress

RENDER_AT=0
render() {   # render <completed-bytes> <current-file-bytes> <index> <name> <started-epoch>
  local done=$(( $1 + $2 )) idx="$3" name="$4" t0="$5"
  local now pct filled bar rate eta elapsed
  now=$(date +%s)
  elapsed=$(( now - t0 )); [ "$elapsed" -lt 1 ] && elapsed=1
  pct=0; [ "$TOTAL" -gt 0 ] && pct=$(( done * 100 / TOTAL ))
  filled=$(( pct * 24 / 100 ))
  bar="$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' $((24 - filled)) '' | tr ' ' '-')"
  rate=$(( (done - SKIPPED_BYTES) / elapsed ))
  if [ "$rate" -gt 0 ] && [ "$done" -lt "$TOTAL" ]; then
    eta="$(hms $(( (TOTAL - done) / rate )))"
  else
    eta="--"
  fi
  if [ -n "$TTY" ]; then
    printf '\r\033[2K  [%d/%d] %s %3d%%  %s / %s  %s/s  eta %s  %s' \
      "$idx" "${#FILES[@]}" "$bar" "$pct" "$(human "$done")" "$(human "$TOTAL")" \
      "$(human "$rate")" "$eta" "$name" >&2
  else
    # Piped: one line per 5% so a build log stays readable and still shows movement.
    if [ "$pct" -ge "$RENDER_AT" ]; then
      printf '  %3d%%  %s / %s  %s/s  eta %s  (%s)\n' \
        "$pct" "$(human "$done")" "$(human "$TOTAL")" "$(human "$rate")" "$eta" "$name" >&2
      RENDER_AT=$(( pct - pct % 5 + 5 ))
    fi
  fi
}

# --------------------------------------------------------------- download

mkdir -p "$DEST"
COMPLETED=0
SKIPPED_BYTES=0
FETCHED=0
SKIPPED=0
T0="$(date +%s)"

verified_digest() {   # verified_digest <relpath>
  [ -f "$MANIFEST" ] || return 1
  awk -v p="$1" '$2 == p { print $1; found=1 } END { exit !found }' "$MANIFEST"
}

record_digest() {     # record_digest <digest> <relpath>
  mkdir -p "$(dirname "$MANIFEST")"
  if [ -f "$MANIFEST" ]; then
    # awk on the whole field, not `grep -v " path$"`: a path is a regex to grep, and every
    # one of these has a `.` in it.
    awk -v p="$2" '$2 != p' "$MANIFEST" >"$MANIFEST.tmp" && mv "$MANIFEST.tmp" "$MANIFEST"
  fi
  printf '%s %s\n' "$1" "$2" >>"$MANIFEST"
}

i=0
for f in "${FILES[@]}"; do
  i=$((i + 1))
  want_size="${SIZES[$((i - 1))]}"
  want_digest="${DIGESTS[$((i - 1))]}"
  out="$DEST/$f"
  mkdir -p "$(dirname "$out")"

  # Already here, right length, and its digest is in the manifest: nothing to do and
  # nothing to re-hash. This is what makes re-running bootstrap free on 4 GB.
  if [ -f "$out" ] && [ "$(size_of "$out")" = "$want_size" ]; then
    if [ -z "$want_digest" ] || [ "$(verified_digest "$f" || true)" = "$want_digest" ]; then
      COMPLETED=$((COMPLETED + want_size))
      SKIPPED_BYTES=$((SKIPPED_BYTES + want_size))
      SKIPPED=$((SKIPPED + 1))
      render "$COMPLETED" 0 "$i" "$f (present)" "$T0"
      continue
    fi
  fi

  # Resume: bytes already on disk are progress, not work to redo.
  have=0
  [ -f "$out" ] && have="$(size_of "$out")"
  if [ "$have" -gt 0 ] && [ "$have" -lt "$want_size" ]; then
    SKIPPED_BYTES=$((SKIPPED_BYTES + have))
  elif [ "$have" -gt "$want_size" ]; then
    # Longer than the server's copy: not a resumable prefix. Start over.
    rm -f "$out"
    have=0
  fi

  # Right length already but not in the manifest: verify it rather than asking curl to
  # resume a complete file, which answers 416 and whose exit status for that is not worth
  # depending on. Happens on the first run after this script replaced the plain curl loop.
  if [ "$have" = "$want_size" ]; then
    SKIPPED_BYTES=$((SKIPPED_BYTES + have))
  else
    curl -fL -C - --retry 3 --retry-delay 2 -s -o "$out" "$BASE/$f" &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
      render "$COMPLETED" "$(size_of "$out")" "$i" "$f" "$T0"
      sleep 0.3
    done
    wait "$pid" || die "download failed: $f (partial file kept; re-run to resume)"
  fi

  got="$(size_of "$out")"
  if [ "$got" != "$want_size" ]; then
    rm -f "$out"
    die "$f is $got bytes and the server said $want_size — deleted rather than kept as a
   plausible-looking truncation. Re-run to retry."
  fi

  if [ -n "$want_digest" ]; then
    if [ -n "$TTY" ]; then printf '\r\033[2K  verifying %s…' "$f" >&2; fi
    got_digest="$(shasum -a 256 "$out" | cut -d' ' -f1)"
    if [ "$got_digest" != "$want_digest" ]; then
      rm -f "$out"
      die "$f failed its sha256 (server said $want_digest, got $got_digest) — deleted"
    fi
    record_digest "$want_digest" "$f"
  fi

  COMPLETED=$((COMPLETED + want_size))
  FETCHED=$((FETCHED + 1))
  render "$COMPLETED" 0 "$i" "$f" "$T0"
done

[ -n "$TTY" ] && printf '\r\033[2K' >&2
elapsed=$(( $(date +%s) - T0 ))
verified=0
for d in "${DIGESTS[@]}"; do [ -n "$d" ] && verified=$((verified + 1)); done
say "$(human "$TOTAL") in $(hms "$elapsed") — $FETCHED fetched, $SKIPPED already present, $verified sha256-verified"
