#!/usr/bin/env bash
#
# Download this checkout's release binaries into bin/, so a machine with no Rust toolchain
# can still run everything. curl and shasum only.
#
#   scripts/fetch-prebuilt.sh            the release matching this checkout's version
#   scripts/fetch-prebuilt.sh --force    refetch even if bin/ is already populated
#   scripts/fetch-prebuilt.sh --check    validate the preconditions, download nothing
#
# The version comes from the *checkout*, never from "latest release". A gate that verifies
# fixtures with a binary built from different source is a gate that lies, so this refuses
# to run in a git checkout whose HEAD is not exactly that tag or whose tree is dirty —
# build from source there, which is what a toolchain is for.
set -euo pipefail

cd "$(dirname "$0")/.."

REPO="${DREAM_TTS_RELEASE_REPO:-drmhse/dream-tts}"
BINS="dream-tts dream-tts-serve audio8-validate cosyvoice-validate kokoro-validate qwen3tts-validate"
MODE="${1:-}"
FORCE=""
CHECK=""
case "$MODE" in
  --force) FORCE=1 ;;
  --check) CHECK=1 ;;
  "") ;;
  *) printf 'usage: %s [--force|--check]\n' "$0" >&2; exit 2 ;;
esac

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

command -v curl >/dev/null   || die "curl not found"
command -v shasum >/dev/null || die "shasum not found"
command -v tar >/dev/null    || die "tar not found"

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || die \
  "prebuilt binaries are published for arm64 macOS only, and this is $(uname -s)/$(uname -m).
   Elsewhere, build from source: cargo build --release --no-default-features"

# The workspace manifest in a source checkout; the VERSION file in an unpacked archive.
if [ -f Cargo.toml ]; then
  VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)"
elif [ -f VERSION ]; then
  VERSION="$(tr -d 'v\n' <VERSION)"
else
  die "no Cargo.toml and no VERSION file — cannot tell which release this checkout wants"
fi
[ -n "$VERSION" ] || die "could not read a version out of Cargo.toml"
TAG="v$VERSION"
NAME="dream-tts-$TAG-aarch64-apple-darwin"

# Only meaningful in a git checkout: an unpacked archive has no sources to disagree with.
if [ -d .git ] && command -v git >/dev/null; then
  head_tag="$(git describe --exact-match --tags HEAD 2>/dev/null || true)"
  dirty="$(git status --porcelain 2>/dev/null || true)"
  if [ "$head_tag" != "$TAG" ] || [ -n "$dirty" ]; then
    die "this checkout is not exactly $TAG$([ -n "$dirty" ] && printf ' and has uncommitted changes').
   A prebuilt binary here would not match the source next to it, which makes every
   fixture gate a statement about code you are not reading. Build instead:
       ./scripts/bootstrap.sh --build"
  fi
fi

# Everything above is a precondition. `--check` exists so bootstrap can fail in its first
# second rather than after minutes of HEAD requests and asset fetching.
if [ -n "$CHECK" ]; then
  say "Preconditions for $TAG are satisfied"
  exit 0
fi

if [ -z "$FORCE" ]; then
  have=1
  for b in $BINS; do [ -x "bin/$b" ] || have=""; done
  if [ -n "$have" ]; then
    say "bin/ already holds every binary — skipping (pass --force to refetch)"
    exit 0
  fi
fi

BASE="https://github.com/$REPO/releases/download/$TAG"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

say "Fetching $NAME.tar.gz from $REPO $TAG"
if [ -t 2 ]; then PROGRESS=(-#); else PROGRESS=(-sS); fi
curl -fL "${PROGRESS[@]}" -o "$TMP/$NAME.tar.gz" "$BASE/$NAME.tar.gz" || die \
  "no release asset $NAME.tar.gz under $REPO $TAG.
   Either the tag was never released, or this checkout's version is ahead of it.
   Build from source instead: ./scripts/bootstrap.sh --build"
curl -fsSL -o "$TMP/SHA256SUMS" "$BASE/SHA256SUMS" \
  || die "release $TAG has no SHA256SUMS; refusing to install an unverified binary"

say "Verifying the checksum"
( cd "$TMP" && shasum -a 256 -c --ignore-missing SHA256SUMS ) \
  || die "checksum mismatch on $NAME.tar.gz — deleted, nothing installed"

say "Unpacking into bin/"
mkdir -p bin
tar -xzf "$TMP/$NAME.tar.gz" -C bin --strip-components=2 "$NAME/bin"
for b in $BINS; do
  [ -x "bin/$b" ] || die "the archive did not contain bin/$b"
done
# curl sets no quarantine attribute, so this is belt-and-braces for an archive that came
# from a browser instead. Harmless when there is nothing to remove.
xattr -dr com.apple.quarantine bin 2>/dev/null || true

say "Installed $TAG into bin/ — no toolchain needed from here"
