#!/usr/bin/env bash
# Runs a job script on a Colab GPU, detached, and always releases the runtime afterwards.
#   scripts/colab_launch.sh <session> <gpu: T4|L4|A100> <script.py> "<key=value args>"
# The job keeps running on the VM if this terminal loses its connection; we just re-attach.
# Needs: `colab` CLI logged in, and HF_TOKEN in .env (checkpoints go to a private Hugging Face repo).
set -uo pipefail
cd "$(dirname "$0")/.."
S="$1"; GPU="$2"; SCRIPT="$3"; ARGS="${4:-}"
PY=.venv/bin/python
rm -rf dist && $PY -m pip wheel . --no-deps -q -w dist || exit 1
echo "== balance before"; colab usage | head -1
colab sessions 2>/dev/null | grep -q "^\[$S\]" || colab new -s "$S" --gpu "$GPU" || exit 1
stop() { echo "== stopping runtime"; colab stop -s "$S"; echo "== balance after"; colab usage | head -1; }
trap stop EXIT
colab upload -s "$S" dist/*.whl "/content/$(basename dist/*.whl)" || exit 1
colab upload -s "$S" .env /content/.env || exit 1
colab upload -s "$S" "$SCRIPT" /content/job.py || exit 1

RUN="run$(date +%s)"   # each launch has its own log and completion marker
START="import os, subprocess, sys
subprocess.run('pkill -f /content/job.py; pkill -f train_zoo', shell=True)  # nothing left over from an earlier launch
subprocess.Popen(['bash', '-c', f'{sys.executable} -u /content/job.py > /content/$RUN.log 2>&1; touch /content/$RUN.done'],
                 env={**os.environ, 'ZOO_ARGS': os.environ.get('ZOO_ARGS', '')}, start_new_session=True)
print('JOB_STARTED')"
FOLLOW="import os, time
pos = int(open('/content/$RUN.pos').read()) if os.path.exists('/content/$RUN.pos') else 0
t0 = time.time()
while time.time() - t0 < 240:
    with open('/content/$RUN.log', errors='replace') as f:
        f.seek(pos); chunk = f.read(); pos = f.tell()
    if chunk: print(chunk, end='', flush=True)
    open('/content/$RUN.pos', 'w').write(str(pos))
    if os.path.exists('/content/$RUN.done'):
        print('JOB_FINISHED'); break
    time.sleep(10)"

for try in 1 2 3 4 5; do
  echo "$START" | colab exec -s "$S" --timeout 120 --env "ZOO_ARGS=$ARGS" 2>&1 | grep -q JOB_STARTED && break
  echo "== start attempt $try failed, retrying"; sleep 10
  [ "$try" = 5 ] && exit 1
done
fails=0
while [ "$fails" -lt 8 ]; do
  out=$(echo "$FOLLOW" | colab exec -s "$S" --timeout 300 2>&1); rc=$?
  if echo "$out" | grep -q "Connection was lost\|Traceback (most recent call last) ─"; then
    fails=$((fails + 1)); echo "== connection problem ($fails), re-attaching"; sleep 15; continue
  fi
  [ -n "$out" ] && echo "$out"
  echo "$out" | grep -q JOB_FINISHED && exit 0
  [ $rc -eq 0 ] && fails=0 || { fails=$((fails + 1)); sleep 15; }
done
echo "== gave up re-attaching"; exit 1
