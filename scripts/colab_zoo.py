"""Runs ON the Colab VM: trains the zoo, checkpointing to a private Hugging Face repo.

Launch from your machine with scripts/colab_launch.sh. Safe to re-run: finished adapters are
downloaded from the checkpoint repo first and skipped.
"""
import glob
import os
import subprocess
import sys
import time

ARGS = dict(a.split("=", 1) for a in os.environ.get("ZOO_ARGS", "").split() if "=" in a)
REPO = ARGS.get("repo", "make-your-ai-zoo")
OUT = "/content/zoo"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "safetensors", "huggingface_hub", "pyyaml",
                *glob.glob("/content/make_your_ai-*.whl"), "--force-reinstall", "--no-deps"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"])  # Colab ships one too old for peft

for line in open("/content/.env"):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1)
        if v:
            os.environ[k] = v

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = f"{api.whoami()['name']}/{REPO}"
api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
try:
    snapshot_download(repo_id, repo_type="dataset", local_dir=OUT, token=os.environ["HF_TOKEN"])
except Exception as e:  # empty repo on the first run
    print("no checkpoint yet:", type(e).__name__)

import threading  # noqa: E402

WORKERS = int(ARGS.get("workers", 3))
KW = dict(steps=int(ARGS.get("steps", 120)), lr=float(ARGS.get("lr", 2e-3)), seeds_seen=int(ARGS.get("seeds", 3)),
          limit=int(ARGS["limit"]) if "limit" in ARGS else None, n_eval=int(ARGS.get("n_eval", 24)))
last = [time.time()]


def push(force=False):
    if force or time.time() - last[0] > 600:  # at most 10 minutes of work can be lost
        last[0] = time.time()
        api.upload_folder(folder_path=OUT, repo_id=repo_id, repo_type="dataset", commit_message="checkpoint")
        print("  checkpoint uploaded", flush=True)


# Step 1 (one process): the files every worker shares.
PREP = {k: KW[k] for k in ("limit", "n_eval")}
subprocess.run([sys.executable, "-c", f"from myai.zoo.trainer import prepare_zoo; prepare_zoo({OUT!r}, **{PREP!r})"], check=True)

# Step 2: several workers share the GPU, each training its own slice of the jobs.
procs = [subprocess.Popen([sys.executable, "-u", "-c",
                           f"from myai.zoo.trainer import train_zoo; train_zoo({OUT!r}, **{KW!r}, shard=({i}, {WORKERS}))"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for i in range(WORKERS)]


def pump(i, p):
    for line in p.stdout:
        if "score=" in line or "Error" in line or "Traceback" in line or "device=" in line:
            print(f"w{i} {line.rstrip()}", flush=True)


threads = [threading.Thread(target=pump, args=(i, p), daemon=True) for i, p in enumerate(procs)]
[t.start() for t in threads]
try:
    while any(p.poll() is None for p in procs):
        time.sleep(20)
        push()
finally:
    [t.join(timeout=5) for t in threads]
    # one results file, so later steps do not care how many workers there were
    lines = []
    for f in sorted(glob.glob(f"{OUT}/results*.jsonl")):
        lines += open(f).read().splitlines()
        os.remove(f)
    open(f"{OUT}/results.jsonl", "w").write("\n".join(dict.fromkeys(lines)) + "\n")
    push(force=True)
print("exit codes", [p.returncode for p in procs], "adapters", len(glob.glob(f"{OUT}/adapters/*.safetensors")))
print("ZOO DONE", flush=True)
