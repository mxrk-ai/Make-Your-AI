"""Diagnostic: is there learnable structure across combinations?

Fits one weight vector per behaviour by least squares on the SEEN adapters (adapter ~ mean + sum of its behaviours'
vectors), then predicts adapters for HELD-OUT combinations and tests them. This is the simplest possible
"generator" and it is told the behaviours (an oracle), so it bounds what a text-conditioned generator can hope for.

  python scripts/diag_additive.py <zoo> [n_tasks=10]
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from myai.generator.core import load_zoo
from myai.zoo import behaviors as B
from myai.zoo import trainer as T
from myai.zoo.prompts import eval_prompts

zoo = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
values = [v for a in B.AXES for v in B.DESCRIBE[a]]
seen = load_zoo(zoo, split="seen", min_score=0.5)
held = [(w, m) for w, m in load_zoo(zoo, split="heldout")]
keys = sorted(seen[0][0])
flat = lambda w: torch.cat([w[k].flatten() for k in keys])
X = torch.stack([flat(w) for w, _ in seen])
D = torch.tensor([[1.0] + [float(v in m["task"].values()) for v in values] for _, m in seen])
E = torch.linalg.lstsq(D, X).solution  # [1 + 15, n_params]
fit = 1 - ((D @ E - X) ** 2).sum() / ((X - X.mean(0)) ** 2).sum()
print(f"seen adapters {len(seen)}; additive model explains {fit:.1%} of their variance")


def unflat(v):
    out, i = {}, 0
    for k in keys:
        shape = seen[0][0][k].shape
        out[k] = v[i:i + shape.numel()].reshape(shape); i += shape.numel()
    return out


model, tok, dev = T.load_base()
T.set_a(model, load_file(str(zoo / "lora_A.safetensors")))
q = eval_prompts(24)
rows = []
for w, m in held[:n]:
    d = torch.tensor([1.0] + [float(v in m["task"].values()) for v in values])
    pred = d @ E
    cos = float(torch.nn.functional.cosine_similarity(pred, flat(w), dim=0))
    s = T.eval_adapter(model, tok, dev, m["task"], q, weights=unflat(pred))
    rows.append((m["task_id"], m["score"], s["score"], cos))
    print(f"{m['task_id']:28s} trained={m['score']:.2f} additive={s['score']:.2f} {s['per_axis']} cos={cos:.2f}", flush=True)
print("mean trained %.2f  additive %.2f  cos %.2f" % tuple(sum(r[i] for r in rows) / len(rows) for i in (1, 2, 3)))
