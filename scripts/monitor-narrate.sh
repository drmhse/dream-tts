#!/usr/bin/env bash
#
# Status of a running (or finished) narrate-swe.sh, one snapshot or repeating.
#
#   scripts/monitor-narrate.sh                          # status once
#   scripts/monitor-narrate.sh -w 30                    # refresh every 30 s
#   scripts/monitor-narrate.sh --log narrate-pm.log \
#     --dir narration-project-management-mastery        # non-default paths
#
# It reads nothing but the run's own log and working directory, so it never disturbs the
# render: chapter count and ETA come from the log's `[N/50] ... eta ...` lines, alignment
# quality from the last per-chapter stats, done count from the manifests on disk (a
# manifest is written only when a chapter has finished its whole cycle). The server is not
# polled, and the script has no state of its own.
#
# Exit status is 0 while the run looks alive (or finished cleanly), 1 if it died.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG=""
DIR=""
WATCH=0
INTERVAL=10

while [ $# -gt 0 ]; do
  case "$1" in
    --log) LOG="$2"; shift 2 ;;
    --dir) DIR="$2"; shift 2 ;;
    -w|--watch) WATCH=1; INTERVAL="${2:-10}"; [ $# -ge 2 ] && shift 2 || shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) printf 'unknown argument %s\n' "$1" >&2; exit 2 ;;
  esac
done

# Fall back to the newest narrate-*.log and narration-* working dir in the repo.
if [ -z "$LOG" ]; then
  LOG=$(ls -t narrate-*.log 2>/dev/null | head -1)
fi
if [ -z "$DIR" ]; then
  DIR=$(ls -dt narration-* 2>/dev/null | head -1)
fi
[ -n "$LOG" ] && [ -f "$LOG" ] || { echo "no run log found (--log narrate-pm.log)"; exit 2; }
[ -n "$DIR" ] && [ -d "$DIR" ] || { echo "no working dir found (--dir narration-<slug>)"; exit 2; }

snapshot() {
  local total done_man last_prog eta cur bytes clean
  clean=$(sed 's/\x1b\[[0-9;]*m//g' "$LOG")
  total=$(grep -oE '\[[0-9]+/[0-9]+\]' <<<"$clean" | tail -1 | tr -d '[]')
  total=${total:-?}
  done_man=$(ls "$DIR"/*.manifest.json 2>/dev/null | wc -l | tr -d ' ')
  last_prog=$(grep -oE '\[[0-9]+/[0-9]+\]' <<<"$clean" | tail -1)
  eta=$(grep -oE 'eta [0-9]+:[0-9]{2}:[0-9]{2}' <<<"$clean" | tail -1 | sed 's/eta //')
  # The current chapter is the one named on the last progress line.
  cur=$(grep -oE '^==> \[[0-9]+/[0-9]+\] chapter-[0-9]+' <<<"$clean" | tail -1 \
    | grep -oE 'chapter-[0-9]+')
  bytes=$(du -sh "$DIR" 2>/dev/null | cut -f1)

  local state="running"
  grep -q '^==> Done:' <<<"$clean" && state="finished"
  if ! pgrep -f 'narrate-(swe|book).sh' >/dev/null 2>&1 && [ "$state" != "finished" ]; then
    state="STOPPED (no process)"
  fi

  printf '%-11s %s\n'   "status"   "$state"
  printf '%-11s %s\n'   "chapters" "${last_prog:-0/0} — $done_man manifest(s)"
  printf '%-11s %s\n'   "eta"      "${eta:-unknown}"
  [ -n "$cur" ] && printf '%-11s %s\n' "current" "$cur"
  printf '%-11s %s\n'   "audio"    "$bytes in $DIR"
  printf '%-11s %s\n'   "quality"  "$(grep -E 'cues \(median' <<<"$clean" | tail -1 | sed 's/.*: //')"
  if [ "$state" = "finished" ]; then
    grep '^==> Done:' <<<"$clean" | tail -1
  fi
}

snapshot
if [ "$WATCH" = 1 ]; then
  while true; do sleep "$INTERVAL"; echo; snapshot; done
fi
