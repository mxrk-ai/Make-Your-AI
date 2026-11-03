#!/usr/bin/env bash
# One-step setup: makes a private Python environment in .venv, installs everything, checks your computer.
#   ./install.sh            everything (language AI + design-first mode)
#   ./install.sh --light    only design-first mode (numpy, no PyTorch, installs in seconds)
set -euo pipefail
cd "$(dirname "$0")"

# Use the newest Python 3.10+ that can make a working environment (some installs have a broken venv).
PY=""
if [ -x .venv/bin/python ]; then
  PY=".venv/bin/python"
else
  for c in python3.13 python3.12 python3.11 python3.10 python3; do
    command -v "$c" >/dev/null || continue
    "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null || continue
    if "$c" -m venv .venv >/dev/null 2>&1; then PY="$c"; break; fi
    rm -rf .venv
  done
fi
if [ -z "$PY" ]; then
  echo "Python 3.10 or newer is needed. Get it from https://www.python.org/downloads/ and run this again."
  exit 1
fi
echo "==> Using $(.venv/bin/python --version)"
.venv/bin/python -m pip install -q --upgrade pip

EXTRAS="llm,dev"
[ "${1:-}" = "--light" ] && EXTRAS="dev"
# On Linux without an NVIDIA GPU, the CPU build of PyTorch is ~10x smaller to download.
if [ "$EXTRAS" != "dev" ] && [ "$(uname)" = "Linux" ] && ! command -v nvidia-smi >/dev/null; then
  .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
fi
echo "==> Installing (the first time takes a few minutes)"
.venv/bin/pip install -q -e ".[$EXTRAS]"

echo
.venv/bin/myai doctor || true
echo
echo "Done. Activate it with:   source .venv/bin/activate"
echo "Then try:                 myai status examples/support-bot"
