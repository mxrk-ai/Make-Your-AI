"""Answers agent requests from a Colab job with a provider on this machine (default: agy).

  python scripts/agent_relay.py <colab session> [provider]

The GPU job (provider "relay") writes /content/agent/req_NNNN.json; this script downloads each one, asks the
agent, and uploads resp_NNNN.txt. It only makes network calls here: no model runs on this machine.
Stops when the job writes /content/agent/done or the session disappears.
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from myai.agent import complete

session, provider = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "agy")
tmp, answered, misses = Path(tempfile.mkdtemp()), set(), 0
run = lambda *a: subprocess.run(["colab", *a], capture_output=True, text=True)
while misses < 60:
    ls = run("ls", "-s", session, "/content/agent")
    if ls.returncode != 0 or "not found" in ls.stdout.lower():
        misses += 1; time.sleep(15); continue
    misses = 0
    names = ls.stdout.split()
    if "done" in names:
        break
    for req in sorted(n for n in names if n.startswith("req_")):
        resp = req.replace("req_", "resp_").replace(".json", ".txt")
        if resp in names or req in answered:
            continue
        run("download", "-s", session, f"/content/agent/{req}", str(tmp / req))
        try:
            q = json.loads((tmp / req).read_text())
            text = complete(q["prompt"], q["system"], provider)
        except Exception as e:
            text = f"ERROR: {e}"
        (tmp / resp).write_text(text)
        run("upload", "-s", session, str(tmp / resp), f"/content/agent/{resp}")
        answered.add(req)
        print(f"answered {req} ({len(text)} chars)", flush=True)
    time.sleep(8)
print("relay finished;", len(answered), "requests answered")
