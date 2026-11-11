#!/usr/bin/env bash
# Runs a job script on a Colab GPU and always releases the runtime afterwards.
#   scripts/colab_launch.sh <session> <gpu: T4|L4|A100> <script.py> "<key=value args>"
# Needs: `colab` CLI logged in, and HF_TOKEN in .env (checkpoints go to a private Hugging Face repo).
set -uo pipefail
cd "$(dirname "$0")/.."
S="$1"; GPU="$2"; SCRIPT="$3"; ARGS="${4:-}"
PY=.venv/bin/python
rm -rf dist && $PY -m pip wheel . --no-deps -q -w dist || exit 1
echo "== balance before"; colab usage | head -1
colab new -s "$S" --gpu "$GPU" || exit 1
trap 'echo "== stopping runtime"; colab stop -s "$S"; echo "== balance after"; colab usage | head -1' EXIT
colab upload -s "$S" dist/*.whl "/content/$(basename dist/*.whl)" || exit 1
colab upload -s "$S" .env /content/.env || exit 1
colab exec -s "$S" -f "$SCRIPT" --timeout 0 --env "ZOO_ARGS=$ARGS"
