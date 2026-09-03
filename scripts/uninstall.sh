#!/usr/bin/env bash
#
# Remove what this install put on disk. Nothing is deleted before you have seen the list.
#
#   scripts/uninstall.sh                    what would be removed, and nothing else
#   scripts/uninstall.sh --weights          the checkpoints (the 4-13 GB)
#   scripts/uninstall.sh --weights qwen3tts just that engine's
#   scripts/uninstall.sh --venvs            the torch virtualenvs
#   scripts/uninstall.sh --fixtures         the gate fixtures (refetchable, ~130 MB)
#   scripts/uninstall.sh --build            target/ and bin/
#   scripts/uninstall.sh --all              every one of the above
#   scripts/uninstall.sh --everything       --all, then this directory itself
#
# Add --yes to skip the confirmation. Without a category flag this only reports, because
# the most likely reason someone runs an uninstaller is to find out what it would do.
#
# Deliberately a script and not a `dream-tts` subcommand: the thing that deletes an hour of
# downloading should be something you can read first, and should keep working when the
# binary is the part that is broken.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

ENGINES="qwen3tts audio8 cosyvoice"
YES=""
WHAT=""
ONLY_ENGINE=""

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --weights)
      WHAT="$WHAT weights"
      # An engine id may follow, as in `--weights qwen3tts`.
      case "${2:-}" in
        qwen3tts|audio8|cosyvoice) ONLY_ENGINE="$2"; shift ;;
      esac
      shift ;;
    --venvs)      WHAT="$WHAT venvs";    shift ;;
    --fixtures)   WHAT="$WHAT fixtures"; shift ;;
    --build)      WHAT="$WHAT build";    shift ;;
    --all)        WHAT="weights venvs fixtures build"; shift ;;
    --everything) WHAT="weights venvs fixtures build self"; shift ;;
    --yes|-y)     YES=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

# Where the checkpoints actually are. `dream-tts config` is the authority, since a config
# file can point data_dir somewhere else entirely — an external disk being the whole reason
# that setting exists. Falling back to the install root reproduces the default.
DATA="$ROOT"
if BIN="$("$ROOT/scripts/run-bin.sh" --which dream-tts 2>/dev/null)" && [ -x "$BIN" ]; then
  # No `exit` in the awk: leaving early closes the pipe, the binary dies quietly on
  # SIGPIPE as intended, and `pipefail` would then fail this line. Reading to EOF costs
  # nothing here and keeps the two behaviours from fighting.
  resolved="$("$BIN" config 2>/dev/null | awk '$1 == "data" && !seen { print $2; seen = 1 }' || true)"
  [ -n "$resolved" ] && [ -d "$resolved" ] && DATA="$resolved"
fi
[ "$DATA" = "$ROOT" ] || say "Checkpoints live outside this directory: $DATA"

size_of() {
  [ -e "$1" ] || { echo 0; return; }
  du -sk "$1" 2>/dev/null | awk '{print $1 * 1024}'
}
human() {
  awk -v b="$1" 'BEGIN {
    split("B KB MB GB TB", u, " "); i = 1
    while (b >= 1024 && i < 5) { b /= 1024; i++ }
    printf (i == 1 ? "%d %s" : "%.1f %s"), b, u[i]
  }'
}

wants() { case " $WHAT " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# Build the list of paths, each with the category that selected it.
TARGETS=()
add() { [ -e "$1" ] && TARGETS+=("$1"); }

for e in $ENGINES; do
  [ -z "$ONLY_ENGINE" ] || [ "$ONLY_ENGINE" = "$e" ] || continue
  wants weights  && { add "$DATA/references/$e/weights"; add "$DATA/references/$e/download"; }
  wants venvs    && add "$ROOT/references/$e/.venv"
  wants fixtures && add "$DATA/fixtures/$e"
done
if wants build; then
  add "$ROOT/target"
  add "$ROOT/bin"
fi

if [ -z "$WHAT" ]; then
  # Report mode: everything, with what would remove each part.
  say "Nothing will be deleted. This is what is here:"
  printf '\n'
  total=0
  report() {
    [ -e "$2" ] || return 0
    s="$(size_of "$2")"
    total=$((total + s))
    printf '  %-10s %10s  %s\n' "$1" "$(human "$s")" "${2#"$ROOT"/}"
  }
  for e in $ENGINES; do
    report --weights  "$DATA/references/$e/weights"
    report --weights  "$DATA/references/$e/download"
    report --venvs    "$ROOT/references/$e/.venv"
    report --fixtures "$DATA/fixtures/$e"
  done
  report --build "$ROOT/target"
  report --build "$ROOT/bin"
  printf '\n  %-10s %10s\n' "total" "$(human "$total")"
  printf '\nPass one of those flags to remove that category, --all for every one, or\n'
  printf -- '--everything to also remove this directory. Add --yes to skip the prompt.\n'
  exit 0
fi

if [ "${#TARGETS[@]}" -eq 0 ] && ! wants self; then
  say "Nothing matching those categories is present."
  exit 0
fi

total=0
say "These will be deleted:"
printf '\n'
for t in "${TARGETS[@]}"; do
  s="$(size_of "$t")"
  total=$((total + s))
  printf '  %10s  %s\n' "$(human "$s")" "$t"
done
if wants self; then
  printf '  %10s  %s\n' "" "$ROOT  (the install directory itself)"
fi
printf '\n  %10s  to reclaim\n\n' "$(human "$total")"

if wants weights; then
  warn "Checkpoints are a multi-GB download. ./scripts/bootstrap.sh refetches them."
fi

if [ -z "$YES" ]; then
  # No TTY means no informed consent, so a piped invocation refuses rather than assuming.
  [ -t 0 ] || die "not a terminal — pass --yes if you really mean it"
  printf 'Type the word delete to continue: '
  read -r answer
  [ "$answer" = delete ] || die "not confirmed; nothing was removed"
fi

for t in "${TARGETS[@]}"; do
  # Refuse anything that is not under one of the two roots. Cheap insurance against a
  # config file, an env var or a future edit turning this into `rm -rf /`.
  case "$t" in
    "$ROOT"/*|"$DATA"/*) ;;
    *) die "refusing to delete $t: outside $ROOT and $DATA" ;;
  esac
  rm -rf "$t"
  printf '  removed %s\n' "$t"
done

if wants self; then
  say "Removing $ROOT"
  # From the parent, so the shell is not sitting in a directory being unlinked.
  parent="$(dirname "$ROOT")"
  base="$(basename "$ROOT")"
  [ -n "$base" ] && [ "$ROOT" != "/" ] || die "refusing to remove $ROOT"
  ( cd "$parent" && rm -rf "./$base" )
  say "Gone. Nothing of dream-tts remains except ~/.config/dream-tts if you wrote one."
  exit 0
fi

say "Done. ./scripts/bootstrap.sh restores anything you want back."
