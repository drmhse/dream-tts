#!/bin/sh
#
# Install dream-tts on Apple silicon with nothing but curl.
#
#   curl -fsSL https://raw.githubusercontent.com/drmhse/dream-tts/main/install.sh | sh
#
# This unpacks a release into ./dream-tts and stops. It does not download 4.3 GB of model
# weights behind your back — it prints the one command that does. Set DREAM_TTS_SETUP=1 to
# have it run that command for you.
#
#   DREAM_TTS_DIR      where to install            (default: ./dream-tts)
#   DREAM_TTS_VERSION  which release               (default: the latest)
#   DREAM_TTS_SETUP    run bootstrap.sh afterwards (default: no; prints the command)
#
# POSIX sh on purpose: this is piped to `sh`, and macOS /bin/sh is bash 3.2 in sh mode.
#
# It duplicates the download-and-verify logic in scripts/fetch-prebuilt.sh, which is
# unavoidable — this runs before there is a checkout to source anything from.
set -eu

REPO="${DREAM_TTS_REPO:-drmhse/dream-tts}"
DIR="${DREAM_TTS_DIR:-$PWD/dream-tts}"

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

for tool in curl tar shasum uname mktemp; do
  command -v "$tool" >/dev/null || die "$tool not found"
done

os="$(uname -s)"; arch="$(uname -m)"
[ "$os" = Darwin ] && [ "$arch" = arm64 ] || die \
  "dream-tts ships prebuilt binaries for arm64 macOS only, and this is $os/$arch.
   It builds and runs elsewhere on CPU fallbacks, roughly 4x slower, from source:
     git clone https://github.com/$REPO && cd dream-tts && ./scripts/bootstrap.sh --build"

# Resolve the latest tag by following the /releases/latest redirect. Cheaper than the JSON
# API, not rate-limited the same way, and no JSON parsing in sh.
TAG="${DREAM_TTS_VERSION:-}"
if [ -z "$TAG" ]; then
  say "Resolving the latest release of $REPO"
  # Not `-f`: a 404 here means "no release yet", which deserves its own message rather than
  # curl's exit 22. The code and the final URL come back together so one request answers
  # both "did it work" and "which tag".
  probe="$(curl -sSLI -o /dev/null -w '%{http_code} %{url_effective}' \
             "https://github.com/$REPO/releases/latest" 2>/dev/null)" \
    || die "could not reach github.com. Are you online?"
  code="${probe%% *}"
  url="${probe#* }"
  case "$code" in
    200) ;;
    404) die "$REPO has no published release yet.
   Either the repository name is wrong, or no v* tag has been built. From source instead:
     git clone https://github.com/$REPO && cd dream-tts && ./scripts/bootstrap.sh --build" ;;
    *) die "github.com answered $code looking for the latest release of $REPO" ;;
  esac
  TAG="${url##*/}"
  case "$TAG" in
    v*) ;;
    *) die "expected a v-prefixed tag, got '$TAG'.
   Pass one explicitly:  DREAM_TTS_VERSION=v0.2.0 ... | sh" ;;
  esac
fi
NAME="dream-tts-$TAG-aarch64-apple-darwin"

if [ -e "$DIR" ] && [ -n "$(ls -A "$DIR" 2>/dev/null || true)" ]; then
  die "$DIR already exists and is not empty.
   Remove it, or choose somewhere else:  DREAM_TTS_DIR=/path/to/install ... | sh"
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT INT TERM

say "Downloading $NAME.tar.gz"
BASE="https://github.com/$REPO/releases/download/$TAG"
curl -fL --progress-bar -o "$TMP/$NAME.tar.gz" "$BASE/$NAME.tar.gz" || die \
  "no asset $NAME.tar.gz in $REPO $TAG.
   Releases are published for arm64 macOS only; check
   https://github.com/$REPO/releases/tag/$TAG"
curl -fsSL -o "$TMP/SHA256SUMS" "$BASE/SHA256SUMS" \
  || die "$TAG has no SHA256SUMS; refusing to install an unverified binary"

say "Verifying the checksum"
( cd "$TMP" && shasum -a 256 -c --ignore-missing SHA256SUMS >/dev/null ) \
  || die "checksum mismatch — nothing installed"

say "Unpacking into $DIR"
mkdir -p "$DIR"
tar -xzf "$TMP/$NAME.tar.gz" -C "$DIR" --strip-components=1
# curl attaches no quarantine attribute; a browser download does. Cheap insurance.
xattr -dr com.apple.quarantine "$DIR" 2>/dev/null || true

[ -x "$DIR/bin/dream-tts" ] || die "the archive did not contain bin/dream-tts"

if [ -n "${DREAM_TTS_SETUP:-}" ]; then
  say "Running bootstrap (downloads ~4.3 GB)"
  cd "$DIR" && exec ./scripts/bootstrap.sh
fi

# A PATH hint rather than a symlink: writing outside the directory the user named is not
# something a piped installer should do unasked, and the shims resolve their own symlinks
# precisely so this line works.
hint=""
case ":$PATH:" in
  *":$HOME/.local/bin:"*) hint="$HOME/.local/bin" ;;
  *":$HOME/bin:"*)        hint="$HOME/bin" ;;
esac

cat <<MSG

$TAG is installed in $DIR

Next, download the model. One engine, ~4.3 GB, curl only, no toolchain and no Python.
It resumes if interrupted, verifies what it downloads, and shows byte progress:

    cd $DIR
    ./scripts/bootstrap.sh

Then:

    ./dream-tts speak --text "Hello from a fresh install." --out hello.wav
    ./dream-tts config          # settings, and where each came from
    ./dream-tts storage         # what is on disk, and what removes it
MSG

if [ -n "$hint" ]; then
  cat <<MSG

To run it from anywhere:

    ln -s "$DIR/dream-tts" "$hint/dream-tts"
MSG
fi

cat <<MSG

To remove it later:

    $DIR/scripts/uninstall.sh          # reports; deletes nothing without a flag

MSG
