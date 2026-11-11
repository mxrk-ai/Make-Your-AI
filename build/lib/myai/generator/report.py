"""The honest scoreboard: generated adapters against every baseline, on seen and held-out tasks.

Methods (same eval prompts and same checks for all):
  base          the base model, no adapter, no instruction
  prompt        the base model with the description as its system prompt (no weights changed)
  trained       a LoRA really trained on this task (the upper bound)
  retrieved     the training adapter whose description is closest (the memorisation baseline)
  sum_singles   adding up the trained single-behaviour adapters (needs to know the behaviours; an oracle)
  generated     the adapter drawn by the generator from the description
  gen+prompt    generated adapter plus the description as system prompt
  reconstructed the trained adapter passed through the autoencoder (what the latent space can represent)
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

METHODS = ["base", "prompt", "trained", "retrieved", "sum_singles", "generated", "gen+prompt", "reconstructed"]


def evaluate_generator(zoo_dir: str | Path, gen_dir: str | Path, out_file: str | Path, n_seen: int = 20,
                       n_prompts: int = 24, seed: int = 0, guidance: float = 2.0, log=print) -> dict:
    from ..zoo import behaviors as B
    from ..zoo import trainer as T
    from ..zoo.prompts import eval_prompts
    from .core import Generator

    zoo_dir, out_file = Path(zoo_dir), Path(out_file)
    gen = Generator(gen_dir)
    rng = random.Random(seed)
    tasks = json.loads((zoo_dir / "tasks.json").read_text())
    have = {f.stem for f in (zoo_dir / "adapters").glob("*.safetensors")}
    tasks = [t for t in tasks if f"{t['id']}__s0" in have]
    held = [t for t in tasks if t["split"] == "heldout"]
    seen = [t for t in tasks if t["split"] == "seen"]
    test = held + rng.sample(seen, min(n_seen, len(seen)))
    singles = {t["id"]: T.load_adapter(zoo_dir / "adapters" / f"{t['id']}__s0.safetensors")[0]
               for t in seen if t["size"] == 1}

    model, tok, dev = T.load_base(gen.info["base_model"], gen.info["spec"])
    prompts = eval_prompts(n_prompts)
    base_rows = T.eval_adapter(model, tok, dev, {}, prompts, weights=None)["rows"]
    done = {}
    if out_file.exists():
        done = {r["id"]: r for r in json.loads(out_file.read_text())["tasks"]}
    rows, t0 = [], time.time()
    for i, t in enumerate(test):
        if t["id"] in done:
            rows.append(done[t["id"]]); continue
        task = t["task"]
        desc = B.describe(task, rng)
        trained_w, meta = T.load_adapter(zoo_dir / "adapters" / f"{t['id']}__s0.safetensors")
        _, hit = gen.retrieve(desc)
        retrieved_w = T.load_adapter(zoo_dir / "adapters" / f"{hit['adapter']}.safetensors")[0]
        gen_w = gen.generate(desc, seed=seed, guidance=guidance)
        ev = lambda w, system=None: T.eval_adapter(model, tok, dev, task, prompts, weights=w, system=system)
        s = {"base": T.score_answers([dict(r) for r in base_rows], task)["score"],
             "prompt": ev(None, desc)["score"],
             "trained": meta["score"],
             "retrieved": ev(retrieved_w)["score"]}
        parts = [singles[v] for v in t["id"].split("+") if v in singles]
        if len(parts) == len(task):
            s["sum_singles"] = ev({k: sum(p[k] for p in parts) for k in parts[0]})["score"]
        g = ev(gen_w)
        s["generated"] = g["score"]
        s["gen+prompt"] = ev(gen_w, desc)["score"]
        s["reconstructed"] = ev(gen.codec.decode(gen.codec.encode(trained_w)))["score"]
        rows.append({"id": t["id"], "split": t["split"], "size": t["size"], "description": desc,
                     "retrieved_adapter": hit["adapter"], "scores": s, "generated_per_axis": g["per_axis"],
                     "generated_sample": g["rows"][0]["answer"][:240]})
        log(f"[{i + 1}/{len(test)}] {t['split']:7s} {t['id']:28s} " + "  ".join(f"{k}={v:.2f}" for k, v in s.items())
            + f"  ({(time.time() - t0) / 60:.1f} min)")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps({"tasks": rows, "summary": summarize(rows)}, indent=1, ensure_ascii=False))
    res = {"tasks": rows, "summary": summarize(rows), "n_prompts": n_prompts, "guidance": guidance,
           "generator": {k: gen.info[k] for k in ("trained_on", "steps", "final_loss")}}
    out_file.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


def summarize(rows: list[dict]) -> dict:
    out = {}
    for split in ("seen", "heldout"):
        sub = [r for r in rows if r["split"] == split]
        out[split] = {"tasks": len(sub)}
        for m in METHODS:
            vals = [r["scores"][m] for r in sub if m in r["scores"]]
            if vals:
                out[split][m] = round(sum(vals) / len(vals), 3)
    return out


def markdown(res: dict) -> str:
    s = res["summary"]
    names = {"base": "Base model", "prompt": "Base model + description as system prompt",
             "trained": "Really trained LoRA (upper bound)", "retrieved": "Nearest training adapter (retrieval)",
             "sum_singles": "Sum of single-behaviour adapters (oracle)", "generated": "**Generated adapter**",
             "gen+prompt": "**Generated adapter + system prompt**",
             "reconstructed": "Trained LoRA through the autoencoder"}
    pct = lambda v: "—" if v is None else f"{v * 100:.0f}%"
    lines = [f"| Method | Seen tasks ({s['seen']['tasks']}) | Held-out tasks ({s['heldout']['tasks']}) |", "|---|---|---|"]
    for m in METHODS:
        if m in s["seen"] or m in s["heldout"]:
            lines.append(f"| {names[m]} | {pct(s['seen'].get(m))} | {pct(s['heldout'].get(m))} |")
    return "\n".join(lines)
