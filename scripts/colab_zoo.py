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

SHARDS = [int(x) for x in ARGS.get("shards", "0,1").split(",")]  # which slices of the job list this VM trains
OF = int(ARGS.get("of", len(SHARDS)))
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
def start(i):
    return subprocess.Popen([sys.executable, "-u", "-c",
                             f"from myai.zoo.trainer import train_zoo; train_zoo({OUT!r}, **{KW!r}, shard=({i}, {OF}))"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})


procs = {i: start(i) for i in SHARDS}
restarts = {i: 0 for i in SHARDS}


def pump(i, p):
    for line in p.stdout:
        if "score=" in line or "Error" in line or "Traceback" in line or "device=" in line:
            print(f"w{i} {line.rstrip()}", flush=True)


def watch(i):
    threading.Thread(target=pump, args=(i, procs[i]), daemon=True).start()


[watch(i) for i in SHARDS]
try:
    while True:
        time.sleep(20)
        push()
        alive = False
        for i in SHARDS:
            rc = procs[i].poll()
            if rc is None:
                alive = True
            elif rc != 0 and restarts[i] < 5:  # crashed (for example out of memory): start it again, it resumes
                restarts[i] += 1
                print(f"w{i} exited with {rc}; restart {restarts[i]}", flush=True)
                procs[i] = start(i); watch(i); alive = True
        if not alive:
            break
finally:
    push(force=True)
print("exit codes", {i: p.returncode for i, p in procs.items()}, "adapters", len(glob.glob(f"{OUT}/adapters/*.safetensors")))
print("ZOO DONE", flush=True)
