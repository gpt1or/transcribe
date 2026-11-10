#!/usr/bin/env bash
# ============================================================================
#  run.sh - for a clone of the repository: transcribe with one command.
#  Installs the package into .venv on the first run (see setup.sh), then runs
#  it with the arguments given here.
#
#    bash run.sh ~/Recordings                     every audio file in a folder
#    bash run.sh call.m4a --diarize --speakers 2  one file, with speaker labels
#    bash run.sh --help                           all options
#
#  Installed the package with pip or pipx? Then just run:  transcribe ...
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

has() {
  [ -x "$PY" ] && "$PY" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$1') else 1)" >/dev/null 2>&1
}

DIARIZE=0
WANTS_HELP=0
for arg in "$@"; do
  case "$arg" in
    --diarize)           DIARIZE=1 ;;
    -h|--help|--version) WANTS_HELP=1 ;;
  esac
done

# Help works before anything is installed.
if [ "$WANTS_HELP" = 1 ] && ! has hebrew_transcribe; then
  PYTHONPATH="$DIR/src${PYTHONPATH:+:$PYTHONPATH}" exec python3 -m hebrew_transcribe "$@"
fi

if ! has hebrew_transcribe || ! has faster_whisper; then
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

# A token saved by version 1.0 next to the scripts is still honoured.
if [ -z "${HF_TOKEN:-}" ] && [ -s "$DIR/.hf_token" ]; then
  HF_TOKEN="$(cat "$DIR/.hf_token")"
  export HF_TOKEN
fi

if [ -s "$CERTS" ]; then
  export SSL_CERT_FILE="$CERTS" REQUESTS_CA_BUNDLE="$CERTS" CURL_CA_BUNDLE="$CERTS"
fi

exec "$PY" -m hebrew_transcribe "$@"
