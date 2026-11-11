"""Runs ON the Colab VM: trains the generator on the zoo and scores it against every baseline.

Reads the zoo from its private Hugging Face repo; writes the generator and report.json to another private repo.
Safe to re-run: the report resumes task by task.
"""
import glob
import json
import os
import subprocess
import sys
import time

ARGS = dict(a.split("=", 1) for a in os.environ.get("ZOO_ARGS", "").split() if "=" in a)
ZOO, GEN = "/content/zoo", "/content/gen"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "safetensors", "huggingface_hub", "pyyaml"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps",
                *glob.glob("/content/make_your_ai-*.whl")], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"])
for line in open("/content/.env"):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1)
        if v:
            os.environ[k] = v

import torch  # noqa: E402
from huggingface_hub import HfApi, snapshot_download  # noqa: E402

api = HfApi(token=os.environ["HF_TOKEN"])
user = api.whoami()["name"]
zoo_repo, gen_repo = f"{user}/{ARGS.get('zoo', 'make-your-ai-zoo')}", f"{user}/{ARGS.get('gen', 'make-your-ai-generator')}"
snapshot_download(zoo_repo, repo_type="dataset", local_dir=ZOO, token=os.environ["HF_TOKEN"])
api.create_repo(gen_repo, private=True, exist_ok=True)
try:
    snapshot_download(gen_repo, local_dir=GEN, token=os.environ["HF_TOKEN"])
except Exception as e:
    print("no generator checkpoint yet:", type(e).__name__)

from myai.generator.core import Generator, load_zoo, train_generator  # noqa: E402
from myai.generator.report import evaluate_generator, markdown  # noqa: E402

if not os.path.exists(f"{GEN}/generator.json") or ARGS.get("retrain"):
    info = train_generator(ZOO, GEN, steps=int(ARGS.get("steps", 12000)), latent_dim=int(ARGS.get("latent", 64)),
                           d=int(ARGS.get("d", 384)), layers=int(ARGS.get("layers", 6)),
                           min_score=float(ARGS.get("min_score", 0.5)), log=lambda m: print(m, flush=True))
    # Pick the guidance strength by how close generated weights are to trained ones on SEEN tasks (never held-out).
    gen = Generator(GEN)
    seen = [(w, m) for w, m in load_zoo(ZOO, split="seen") if m["seed"] == 0][::4]
    flat = lambda w: torch.cat([w[k].flatten() for k in sorted(w)])
    best = (None, -1)
    for g in (1.0, 1.5, 2.0, 3.0):
        cos = sum(float(torch.nn.functional.cosine_similarity(flat(gen.generate(m["description"], guidance=g)), flat(w), dim=0))
                  for w, m in seen) / len(seen)
        print(f"guidance {g}: mean cosine to trained adapters (seen tasks) {cos:.3f}", flush=True)
        if cos > best[1]:
            best = (g, cos)
    info = json.load(open(f"{GEN}/generator.json"))
    info["guidance"], info["seen_cosine"] = best[0], round(best[1], 3)
    json.dump(info, open(f"{GEN}/generator.json", "w"), indent=1)
    api.upload_folder(folder_path=GEN, repo_id=gen_repo, commit_message="generator")
    print("generator uploaded", flush=True)

guidance = json.load(open(f"{GEN}/generator.json")).get("guidance", 2.0)
last = [time.time()]


def log(m):
    print(m, flush=True)
    if time.time() - last[0] > 600 and os.path.exists(f"{GEN}/report.json"):
        last[0] = time.time()
        api.upload_file(path_or_fileobj=f"{GEN}/report.json", path_in_repo="report.json", repo_id=gen_repo)


try:
    res = evaluate_generator(ZOO, GEN, f"{GEN}/report.json", n_seen=int(ARGS.get("seen", 20)), guidance=guidance, log=log)
    print(markdown(res), flush=True)
finally:
    if os.path.exists(f"{GEN}/report.json"):
        api.upload_file(path_or_fileobj=f"{GEN}/report.json", path_in_repo="report.json", repo_id=gen_repo)
print("GENERATOR DONE", flush=True)
