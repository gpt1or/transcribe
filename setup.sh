#!/usr/bin/env bash
# ============================================================================
#  setup.sh - installs transcribe into a private virtual environment (.venv)
#  inside this folder. Nothing is installed system-wide. Safe to run again:
#  it picks up where it stopped. run.sh calls it automatically when needed.
# ============================================================================
set -eo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$DIR/.venv"
PY="$VENV/bin/python"
CERTS="$DIR/ca-bundle.pem"
DIARIZE=0
CORP_CERTS=0
MODEL="turbo"

usage() {
  cat <<'EOF'
Usage: bash setup.sh [--diarize] [--model turbo|full] [--corp-certs]

  --diarize      also install speaker labels (pyannote.audio + PyTorch, ~2 GB)
  --model full   pre-download the slower "full" model instead of "turbo"
  --corp-certs   macOS behind a TLS-inspecting corporate proxy: build a
                 certificate bundle from the macOS keychains
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --diarize)    DIARIZE=1 ;;
    --corp-certs) CORP_CERTS=1 ;;
    --model)      MODEL="${2:?--model needs a value}"; shift ;;
    -h|--help)    usage; exit 0 ;;
    *)            echo "Unknown option: $1"; usage; exit 2 ;;
  esac
  shift
done

echo ""
echo "== 1/5  Python =="
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "x python3 was not found."
  echo "  macOS: xcode-select --install   (or: brew install python)"
  echo "  Linux: install python3 and python3-venv with your package manager"
  exit 1
fi
"$PYTHON" -c 'import sys; v = sys.version_info[:2]; sys.exit(0 if v >= (3, 10) else "x Python 3.10 or newer is required (found %d.%d)" % v)'
"$PYTHON" -V

echo ""
echo "== 2/5  Private environment (.venv) =="
if [ ! -x "$PY" ]; then
  "$PYTHON" -m venv "$VENV"
fi
"$PY" -m pip install --upgrade pip -q
echo "  ok"

echo ""
echo "== 3/5  Packages =="
"$PY" -m pip install -q -r "$DIR/requirements.txt"
if [ "$DIARIZE" = 1 ]; then
  echo "  + speaker labels (pyannote.audio and PyTorch; this is the large part)"
  "$PY" -m pip install -q -r "$DIR/requirements-diarize.txt"
fi
echo "  ok"

echo ""
echo "== 4/5  Certificates =="
if [ "$CORP_CERTS" = 1 ]; then
  if [ "$(uname)" != "Darwin" ]; then
    echo "  --corp-certs is for macOS. On Linux, add the proxy's CA certificate to"
    echo "  the system store (e.g. update-ca-certificates); it is then used automatically."
  else
    # Public CAs from certifi + every certificate in the macOS keychains, which
    # include the corporate proxy's. Certificates only, never private keys.
    "$PY" -c 'import certifi, shutil, sys; shutil.copyfile(certifi.where(), sys.argv[1])' "$CERTS"
    for kc in /System/Library/Keychains/SystemRootCertificates.keychain \
              /Library/Keychains/System.keychain \
              "$HOME/Library/Keychains/login.keychain-db"; do
      if [ -e "$kc" ]; then
        security find-certificate -a -p "$kc" >> "$CERTS" 2>/dev/null || true
      fi
    done
    chmod 600 "$CERTS"
    echo "  ok: ca-bundle.pem, $(grep -c 'BEGIN CERTIFICATE' "$CERTS") certificates (used by run.sh)"
  fi
else
  echo "  using the system certificate store. If downloads fail behind a"
  echo "  corporate proxy on macOS, run:  bash setup.sh --corp-certs"
fi
if [ -s "$CERTS" ]; then
  export SSL_CERT_FILE="$CERTS" REQUESTS_CA_BUNDLE="$CERTS" CURL_CA_BUNDLE="$CERTS"
fi

echo ""
echo "== 5/5  Speech model (one-time download: ~1.6 GB turbo, ~3 GB full) =="
"$PY" "$DIR/transcribe.py" --download-only --model "$MODEL"

echo ""
echo "=================================================================="
echo " Installed. Transcribe with:   bash run.sh <audio files or folders>"
echo " Read the Disclaimer in README.md before use."
echo "=================================================================="
