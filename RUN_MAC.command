#!/bin/bash
# Double-click this file (first time: right-click -> Open). Installs everything, trains (first time only), starts the app.
cd "$(dirname "$0")" || exit 1
echo "============================================================"
echo " SWE015 Chat Moderation - automatic setup and run"
echo "============================================================"

PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)"; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "[PROBLEM] Python 3.10, 3.11 or 3.12 was not found."
  echo "Install Python 3.12 from https://www.python.org/downloads/ and double-click this file again."
  read -r -p "Press Enter to close"; exit 1
fi
echo "[1/4] Using: $PY ($($PY --version))"

fail() { echo; echo "[ERROR] Something failed. Take a screenshot of this window and send it to Claude."; read -r -p "Press Enter to close"; exit 1; }

[ -x .venv/bin/python ] || { echo "[2/4] Creating virtual environment..."; "$PY" -m venv .venv || fail; }
VPY=.venv/bin/python
echo "[3/4] Installing libraries (first time takes several minutes)..."
"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install -r requirements.txt || fail

if [ ! -f models/bert_int8/model_int8.pt ]; then
  echo "[4/4] Training all models and evaluating - BERT can take about 1 hour. Do not close this window."
  "$VPY" train_all.py || fail
else
  echo "[4/4] Models already trained - skipping training."
fi

echo; echo "Starting the app... open http://127.0.0.1:7860 in your browser."
echo "Your results table is in results/results.md"
"$VPY" app.py
