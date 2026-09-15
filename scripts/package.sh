#!/usr/bin/env bash
#
# Build the release archive and prove it runs with no toolchain.
#
#   scripts/package.sh [outdir]        default outdir: dist/
#
# A script rather than inline workflow steps for one reason: the release workflow runs only
# on a tag, so a packaging mistake would first be discovered as a broken release. CI calls
# this on every push instead, and it is runnable by hand.
#
# Apple silicon only. Default features are Metal and the `--no-default-features` CPU path is
# a portability guarantee rather than a deployment target — audio8 measures RTF 2.151 there
# against 0.554 on Metal. Publishing a CPU build would invite people to use it.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="${1:-$ROOT/dist}"

# The product, plus the four fixture gates: without them a cargo-less install cannot run
# ./scripts/gates.sh, and qwen3tts's shape-audit tier is the one check that works
# mid-download.
BINS="dream-tts dream-tts-serve audio8-validate cosyvoice-validate kokoro-validate qwen3tts-validate"

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] \
  || die "releases are aarch64-apple-darwin only; this is $(uname -s)/$(uname -m)"

VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)"
[ -n "$VERSION" ] || die "no workspace version in Cargo.toml"
NAME="dream-tts-v$VERSION-aarch64-apple-darwin"

# Built on Sonoma in CI, so without this the binaries would silently require it. Ventura is
# the oldest release Apple silicon and candle's Metal path both still work on.
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"

say "Building $NAME"
# One invocation per package: a virtual workspace cannot select bins across packages in a
# single --bin list. Cargo shares the dependency build between these.
cargo build --release -p tts-cli   --bin dream-tts
cargo build --release -p tts-serve --bin dream-tts-serve
cargo build --release -p audio8    --bin audio8-validate
cargo build --release -p cosyvoice --bin cosyvoice-validate
cargo build --release -p qwen3tts  --bin qwen3tts-validate

# A Homebrew dylib linked in on a build machine would break every machine but that one.
# candle's Metal path needs system frameworks and nothing else.
say "Auditing dynamic dependencies"
for b in $BINS; do
  # Existence first: `grep -v` over the empty output of a failed otool is also empty, so a
  # missing binary would otherwise pass this as clean.
  [ -f "target/release/$b" ] || die "target/release/$b was not built"
  libs="$(otool -L "target/release/$b" | tail -n +2 | awk '{print $1}' \
            | grep -Ev '^(/usr/lib/|/System/)' || true)"
  [ -z "$libs" ] || die "$b links non-system libraries:
$libs"
done
echo "  system frameworks only"

say "Assembling"
rm -rf "$OUT/$NAME"
mkdir -p "$OUT/$NAME/bin" "$OUT/$NAME/examples"
for b in $BINS; do cp "target/release/$b" "$OUT/$NAME/bin/$b"; done
# A working checkout minus the sources. The shims and run-bin.sh find bin/ when there is no
# cargo; gates.sh reports its source-only tiers as skipped there.
cp -R docs scripts voices "$OUT/$NAME/"
cp dream-tts dream-tts-serve dream-tts.example.json README.md LICENSE NOTICE "$OUT/$NAME/"
# The two benchmark fixtures, not the 55 MB of rendered wavs beside them. And no
# references/ — the weights land there and bootstrap creates it.
cp examples/chapter.txt examples/senior.txt "$OUT/$NAME/examples/"
# The real chapter, so a binary install has a document to try `import` and `narrate` on.
mkdir -p "$OUT/$NAME/examples/book"
cp prep-handbook/*.md "$OUT/$NAME/examples/book/" 2>/dev/null || true
printf 'v%s\n' "$VERSION" >"$OUT/$NAME/VERSION"

tar -C "$OUT" -czf "$OUT/$NAME.tar.gz" "$NAME"
( cd "$OUT" && shasum -a 256 "$NAME.tar.gz" >SHA256SUMS )
say "$(cat "$OUT/SHA256SUMS")"

# The claim this whole thing exists to make: a machine with neither cargo nor the sources
# can still run it. Proved by unpacking elsewhere and hiding cargo from PATH.
say "Verifying the archive runs without a toolchain"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
tar -C "$work" -xzf "$OUT/$NAME.tar.gz"
cd "$work/$NAME"
[ ! -f Cargo.toml ] || die "the archive carries sources; it should not"
BARE="PATH=/usr/bin:/bin:/usr/sbin:/sbin"

# Captured, then matched — never `| grep -q`.
#
# `grep -q` leaves on its first match and closes the pipe, and these binaries now let
# SIGPIPE kill them quietly the way every Unix tool does (see tts-cli's main). Under
# `pipefail` that death fails the pipeline even though the match succeeded. Capturing the
# output first tests what is meant instead of testing the pipe.
engines="$(env $BARE ./dream-tts engines)" || die "dream-tts engines failed to run"
case "$engines" in
  *"marks the default engine"*) ;;
  *) die "dream-tts engines printed no default marker" ;;
esac
case "$engines" in
  *"* qwen3tts"*) ;;
  *) die "qwen3tts is not the default engine in the packaged build" ;;
esac
env $BARE ./dream-tts config >/dev/null   || die "dream-tts config failed"

# Narration and import need no weights, no toolchain and no python, so they are the two
# things a binary install can be proved to do before anything is downloaded.
printf '# Title\n\nProse with a 3.2 GB/s rate and `PaymentCaptured(id)`.\n' >"$work/one.md"
narrated="$(env $BARE ./dream-tts narrate "$work/one.md")" || die "dream-tts narrate failed"
case "$narrated" in
  *"3.2 gigabytes per second"*) ;;
  *) die "narrate did not verbalise the rate: $narrated" ;;
esac
env $BARE ./dream-tts import "$work/one.md" --dry-run >/dev/null \
  || die "dream-tts import failed"
# Job discovery must work with no service and no weights: it is what a user hits first, and
# it reads the store rather than the network.
env $BARE ./dream-tts jobs >/dev/null || die "dream-tts jobs failed"
env $BARE ./dream-tts storage >/dev/null  || die "dream-tts storage failed"
env $BARE ./dream-tts --version >/dev/null || die "dream-tts --version failed"

# The shipped example must load, or it is an example that does not work.
cp dream-tts.example.json dream-tts.json
conf="$(env $BARE ./dream-tts config)" || die "dream-tts config failed with a config file"
case "$conf" in
  *dream-tts.json*) ;;
  *) die "the shipped dream-tts.example.json does not load" ;;
esac
rm dream-tts.json
# No weights here, so this must fail cleanly and point at the fix rather than crash.
if env $BARE ./dream-tts speak --text hi --out /dev/null 2>"$work/err"; then
  die "speak succeeded with no weights installed"
fi
# A file, not a pipe, so `grep -q` has nothing to close early.
grep -q 'bootstrap.sh' "$work/err" \
  || die "the no-checkpoint error does not name the command that fixes it"
env $BARE ./scripts/gates.sh >/dev/null || die "gates.sh failed in the packaged archive"

say "$NAME.tar.gz is $(du -h "$OUT/$NAME.tar.gz" | cut -f1) and runs with no cargo on PATH"
