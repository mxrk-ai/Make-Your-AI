"""Runs ON the Colab VM: compares adapter configurations on a few hard tasks before the zoo is trained."""
import glob
import os
import subprocess
import sys
import time

subprocess.run("pkill -f train_zoo; rm -rf /content/zoo", shell=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "safetensors", "pyyaml"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps",
                *glob.glob("/content/make_your_ai-*.whl")], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"])

import torch  # noqa: E402
from pathlib import Path  # noqa: E402

from myai.zoo import behaviors as B, trainer as T  # noqa: E402
from myai.zoo.prompts import eval_prompts, train_prompts  # noqa: E402
from myai.zoo.spec import LORA_SPEC  # noqa: E402

CONFIGS = {
    "qv r8 200": (dict(target_modules=["q_proj", "v_proj"]), dict(steps=200, lr=2e-3)),
    "qv r8 400 lr4e-3": (dict(target_modules=["q_proj", "v_proj"]), dict(steps=400, lr=4e-3)),
    "qv+down r8 200": (dict(target_modules=["q_proj", "v_proj", "down_proj"]), dict(steps=200, lr=2e-3)),
    "qv+down r8 300": (dict(target_modules=["q_proj", "v_proj", "down_proj"]), dict(steps=300, lr=2e-3)),
    "qkvo r8 200": (dict(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]), dict(steps=200, lr=2e-3)),
    "down r8 200": (dict(target_modules=["down_proj"]), dict(steps=200, lr=2e-3)),
}
TASKS = ["caps", "json", "pirate", "bullets+emoji+hope", "one+caps"]
evalq = eval_prompts(24)
plain = None
for name, (spec_over, kw) in CONFIGS.items():
    spec = {**LORA_SPEC, **spec_over}
    model, tok, dev = T.load_base(spec=spec)
    if plain is None:
        plain = T.make_plain(model, tok, dev, train_prompts(320), Path("/content/bench_plain.jsonl"))
    n = sum(p.numel() for p in T.b_params(model).values())
    row = []
    for tid in TASKS:
        task = B.parse_task_id(tid)
        t0 = time.time()
        info = T.train_adapter(model, tok, dev, T.task_pairs(plain, task), 0, **kw)
        ev = T.eval_adapter(model, tok, dev, task, evalq)
        row.append(f"{tid}={ev['score']:.2f} (loss {info['final_loss']:.2f}, {info['seconds']:.0f}s+{time.time() - t0 - info['seconds']:.0f}s, limit {ev['hit_limit_rate']:.2f})")
    print(f"BENCH {name:18s} params={n:,}  mem={torch.cuda.max_memory_allocated() / 1e9:.1f}GB  " + "  ".join(row), flush=True)
    del model
    torch.cuda.empty_cache()
print("BENCH DONE", flush=True)
