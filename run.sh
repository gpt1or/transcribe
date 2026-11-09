#!/usr/bin/env bash
# ============================================================================
#  run.sh - transcribe with one command. Installs what is missing on the
#  first run (see setup.sh), then hands everything to transcribe.py.
#
#    bash run.sh ~/Recordings                     every audio file in a folder
#    bash run.sh call.m4a --diarize --speakers 2  one file, with speaker labels
#    bash run.sh --help                           all options
# ============================================================================
set -eo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$DIR/.venv/bin/python"
CERTS="$DIR/ca-bundle.pem"

if [ $# -eq 0 ]; then
  echo "Usage: bash run.sh <audio files or folders> [options]"
  echo "       bash run.sh --help   for all options"
  exit 2
fi

DIARIZE=0
WANTS_HELP=0
for arg in "$@"; do
  case "$arg" in
    --diarize)           DIARIZE=1 ;;
    -h|--help|--version) WANTS_HELP=1 ;;
  esac
done

# Help works before anything is installed.
if [ "$WANTS_HELP" = 1 ] && [ ! -x "$PY" ]; then
  exec python3 "$DIR/transcribe.py" "$@"
fi

has() {
  [ -x "$PY" ] && "$PY" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$1') else 1)" >/dev/null 2>&1
}

if ! has faster_whisper; then
  echo "First run: installing (needs internet once; afterwards it works offline)."
  if [ "$DIARIZE" = 1 ]; then
    bash "$DIR/setup.sh" --diarize
  else
    bash "$DIR/setup.sh"
  fi
elif [ "$DIARIZE" = 1 ] && ! has pyannote.audio; then
  echo "Speaker labels requested: installing the extra packages first."
  bash "$DIR/setup.sh" --diarize
fi

if [ -s "$CERTS" ]; then
  export SSL_CERT_FILE="$CERTS" REQUESTS_CA_BUNDLE="$CERTS" CURL_CA_BUNDLE="$CERTS"
fi

exec "$PY" "$DIR/transcribe.py" "$@"
