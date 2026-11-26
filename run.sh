#!/usr/bin/env bash
# ============================================================================
#  run.sh - for a clone of the repository: transcribe with one command.
#  Installs the package into .venv on the first run (see setup.sh), then runs
#  it with the arguments given here.
#
#    bash run.sh ~/Recordings                     every audio file in a folder
#    bash run.sh call.m4a --diarize --speakers 2  one file, with speaker labels
#    bash run.sh ~/Recordings --language ru       Russian recordings
#    bash run.sh ~/Recordings --language auto     Hebrew or Russian, per file
#    bash run.sh ~/Recordings --ui-language ru    messages in Russian
#    bash run.sh --help                           all options
#
#  Installed the package with pip or pipx? Then just run:  transcribe ...
# ============================================================================
set -eo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$DIR/.venv/bin/python"
CERTS="$DIR/ca-bundle.pem"
UI="${TRANSCRIBE_UI_LANGUAGE:-en}"
LANGUAGE="he"
MODEL="turbo"
DIARIZE=0
WANTS_HELP=0

prev=""
for arg in "$@"; do
  case "$prev" in
    -l|--language) LANGUAGE="$arg" ;;
    -m|--model)    MODEL="$arg" ;;
    --ui-language) UI="$arg" ;;
  esac
  case "$arg" in
    --diarize)           DIARIZE=1 ;;
    -h|--help|--version) WANTS_HELP=1 ;;
    --language=*)        LANGUAGE="${arg#--language=}" ;;
    --model=*)           MODEL="${arg#--model=}" ;;
    --ui-language=*)     UI="${arg#--ui-language=}" ;;
  esac
  prev="$arg"
done
case "$UI" in ru) ;; *) UI="en" ;; esac

# say "English" "Русский"
say() { if [ "$UI" = "ru" ]; then printf '%s\n' "$2"; else printf '%s\n' "$1"; fi; }

if [ $# -eq 0 ]; then
  say "Usage: bash run.sh <audio files or folders> [options]" \
      "Использование: bash run.sh <аудиофайлы или папки> [параметры]"
  say "       bash run.sh --help   for all options" \
      "       bash run.sh --help   — все параметры"
  exit 2
fi

has() {
  [ -x "$PY" ] && "$PY" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$1') else 1)" >/dev/null 2>&1
}

# Help works before anything is installed.
if [ "$WANTS_HELP" = 1 ] && ! has hebrew_transcribe; then
  PYTHONPATH="$DIR/src${PYTHONPATH:+:$PYTHONPATH}" exec python3 -m hebrew_transcribe "$@"
fi

if ! has hebrew_transcribe || ! has faster_whisper; then
  say "First run: installing (needs internet once; afterwards it works offline)." \
      "Первый запуск: установка (интернет нужен один раз, дальше всё работает офлайн)."
  if [ "$DIARIZE" = 1 ]; then
    bash "$DIR/setup.sh" --diarize --model "$MODEL" --language "$LANGUAGE" --ui-language "$UI"
  else
    bash "$DIR/setup.sh" --model "$MODEL" --language "$LANGUAGE" --ui-language "$UI"
  fi
elif [ "$DIARIZE" = 1 ] && ! has pyannote.audio; then
  say "Speaker labels requested: installing the extra packages first." \
      "Запрошено разделение по говорящим: сначала устанавливаются дополнительные пакеты."
  bash "$DIR/setup.sh" --diarize --model "$MODEL" --language "$LANGUAGE" --ui-language "$UI"
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
