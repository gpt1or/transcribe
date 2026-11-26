#!/usr/bin/env bash
# ============================================================================
#  setup.sh - for a clone of the repository: installs hebrew-transcribe into a
#  private virtual environment (.venv) inside this folder, in editable mode, so
#  `git pull` updates it. Nothing is installed system-wide. Safe to run again.
#  run.sh calls it automatically when needed.
#
#  Not using a clone? Install the package instead (see README):
#    pipx install "git+https://github.com/gpt1or/transcribe.git"
# ============================================================================
set -eo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$DIR/.venv"
PY="$VENV/bin/python"
CERTS="$DIR/ca-bundle.pem"
DIARIZE=0
CORP_CERTS=0
MODEL="turbo"
LANGUAGE="he"
UI="${TRANSCRIBE_UI_LANGUAGE:-en}"

# The message language comes first, so that even the usage text is in it.
prev=""
for arg in "$@"; do
  if [ "$prev" = "--ui-language" ]; then UI="$arg"; fi
  case "$arg" in --ui-language=*) UI="${arg#--ui-language=}" ;; esac
  prev="$arg"
done
case "$UI" in ru) ;; *) UI="en" ;; esac

# say "English" "Русский"
say() { if [ "$UI" = "ru" ]; then printf '%s\n' "$2"; else printf '%s\n' "$1"; fi; }

usage() {
  if [ "$UI" = "ru" ]; then
    cat <<'EOF'
Использование: bash setup.sh [--diarize] [--model turbo|full] [--language he|ru|auto]
                             [--corp-certs] [--ui-language en|ru]

  --diarize          установить и разделение по говорящим (pyannote.audio + PyTorch, ~2 ГБ)
  --model full       заранее скачать большую модель «full» вместо «turbo»
  --language ru      заранее скачать модель для русского (auto — для иврита, русского
                     и определения языка); по умолчанию he
  --corp-certs       macOS за корпоративным прокси с проверкой TLS: собрать пакет
                     сертификатов из связок ключей macOS
  --ui-language ru   сообщения на русском
EOF
  else
    cat <<'EOF'
Usage: bash setup.sh [--diarize] [--model turbo|full] [--language he|ru|auto]
                     [--corp-certs] [--ui-language en|ru]

  --diarize          also install speaker labels (pyannote.audio + PyTorch, ~2 GB)
  --model full       pre-download the larger "full" model instead of "turbo"
  --language ru      pre-download the model for Russian (auto: for Hebrew, Russian
                     and language detection); default: he
  --corp-certs       macOS behind a TLS-inspecting corporate proxy: build a
                     certificate bundle from the macOS keychains
  --ui-language ru   messages in Russian
EOF
  fi
}

while [ $# -gt 0 ]; do
  case "$1" in
    --diarize)       DIARIZE=1 ;;
    --corp-certs)    CORP_CERTS=1 ;;
    --model)         MODEL="${2:?--model needs a value}"; shift ;;
    --model=*)       MODEL="${1#--model=}" ;;
    -l|--language)   LANGUAGE="${2:?--language needs a value}"; shift ;;
    --language=*)    LANGUAGE="${1#--language=}" ;;
    --ui-language)   shift ;;
    --ui-language=*) ;;
    -h|--help)       usage; exit 0 ;;
    *)               say "Unknown option: $1" "Неизвестный параметр: $1"; usage; exit 2 ;;
  esac
  shift
done

echo ""
say "== 1/5  Python ==" "== 1/5  Python =="
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  say "x python3 was not found." "x python3 не найден."
  say "  macOS: xcode-select --install   (or: brew install python)" \
      "  macOS: xcode-select --install   (или: brew install python)"
  say "  Linux: install python3 and python3-venv with your package manager" \
      "  Linux: установите python3 и python3-venv через менеджер пакетов"
  exit 1
fi
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  say "x Python 3.10 or newer is required (found: $("$PYTHON" -V 2>&1))." \
      "x Нужен Python 3.10 или новее (найден: $("$PYTHON" -V 2>&1))."
  exit 1
fi
"$PYTHON" -V

echo ""
say "== 2/5  Private environment (.venv) ==" "== 2/5  Отдельное окружение (.venv) =="
if [ ! -x "$PY" ]; then
  "$PYTHON" -m venv "$VENV"
fi
"$PY" -m pip install --upgrade pip -q
say "  ok" "  готово"

echo ""
say "== 3/5  Certificates ==" "== 3/5  Сертификаты =="
if [ "$CORP_CERTS" = 1 ]; then
  if [ "$(uname)" != "Darwin" ]; then
    say "  --corp-certs is for macOS. On Linux, add the proxy's CA certificate to" \
        "  --corp-certs — только для macOS. В Linux добавьте сертификат центра"
    say "  the system store (e.g. update-ca-certificates); it is then used automatically." \
        "  сертификации прокси в системное хранилище (например, update-ca-certificates)."
  else
    # Public CAs from pip's bundle + every certificate in the macOS keychains,
    # which include the corporate proxy's. Certificates only, never private keys.
    "$PY" -c 'import sys; from pip._vendor import certifi; open(sys.argv[1], "w").write(open(certifi.where()).read())' "$CERTS"
    for kc in /System/Library/Keychains/SystemRootCertificates.keychain \
              /Library/Keychains/System.keychain \
              "$HOME/Library/Keychains/login.keychain-db"; do
      if [ -e "$kc" ]; then
        security find-certificate -a -p "$kc" >> "$CERTS" 2>/dev/null || true
      fi
    done
    chmod 600 "$CERTS"
    COUNT="$(grep -c 'BEGIN CERTIFICATE' "$CERTS")"
    say "  ok: ca-bundle.pem, $COUNT certificates (used by run.sh)" \
        "  готово: ca-bundle.pem, сертификатов: $COUNT (использует run.sh)"
  fi
else
  say "  using the system certificate store. If downloads fail behind a" \
      "  используется системное хранилище сертификатов. Если за корпоративным"
  say "  corporate proxy on macOS, run:  bash setup.sh --corp-certs" \
      "  прокси на macOS скачивание не проходит, выполните:  bash setup.sh --corp-certs"
fi
if [ -s "$CERTS" ]; then
  export SSL_CERT_FILE="$CERTS" REQUESTS_CA_BUNDLE="$CERTS" CURL_CA_BUNDLE="$CERTS" PIP_CERT="$CERTS"
fi

echo ""
say "== 4/5  Package ==" "== 4/5  Пакет =="
if [ "$DIARIZE" = 1 ]; then
  say "  with speaker labels (pyannote.audio and PyTorch; this is the large part)" \
      "  с разделением по говорящим (pyannote.audio и PyTorch — это самая большая часть)"
  "$PY" -m pip install -q -e "$DIR[diarize]"
else
  "$PY" -m pip install -q -e "$DIR"
fi
VERSION="$("$PY" -m hebrew_transcribe --version)"
say "  ok: $VERSION" "  готово: $VERSION"

echo ""
say "== 5/5  Speech models (one-time download, ~1.6 GB each) ==" \
    "== 5/5  Модели распознавания (скачиваются один раз, ~1,6 ГБ каждая) =="
"$PY" -m hebrew_transcribe --download-only --model "$MODEL" --language "$LANGUAGE" --ui-language "$UI"

echo ""
echo "=================================================================="
say " Installed. Transcribe with:   bash run.sh <audio or video files, or folders>" \
    " Установлено. Запуск:   bash run.sh <аудио- или видеофайлы, или папки>"
say " Read the Disclaimer in README.md before use." \
    " Перед использованием прочтите отказ от ответственности в README.md."
echo "=================================================================="
