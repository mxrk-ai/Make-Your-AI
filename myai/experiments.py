"""Experiments behind the numbers in the README. Each writes a JSON file; none is needed to use the tool.

Run them on a GPU machine (see scripts/colab_pipeline.py); they do a lot of model inference.
"""

from __future__ import annotations

import itertools
import json
import random
import shutil
from pathlib import Path


def patch_experiment(gen_dir: str, report_file: str, out_file: str, work: str = "runs/patch", n: int = 6,
                     provider: str = "agy", log=print) -> dict:
    """Can the agent repair generated adapters that fail their checks? With and without harmonising.

    Takes held-out tasks where the generated adapter scored below 0.9, generates it again, and runs the repair
    loop twice from the same start: plain, and with harmonize 0.3. Scores are on questions the agent never saw.
    """
    from . import generated as G
    from .agent.llm import CALL_LOG
    from .agent.repair import repair
    from .zoo import behaviors as B

    rows = [r for r in json.loads(Path(report_file).read_text())["tasks"]
            if r["split"] == "heldout" and r["scores"]["generated"] < 0.9]
    rows.sort(key=lambda r: r["scores"]["generated"])
    picked = rows[:: max(1, len(rows) // n)][:n]  # spread from bad to nearly good
    results, calls0 = [], len(CALL_LOG)
    for r in picked:
        base = Path(work) / r["id"].replace("+", "_")
        shutil.rmtree(base, ignore_errors=True)
        G.create(r["description"], base, gen_dir)
        worst = min(r["generated_per_axis"], key=r["generated_per_axis"].get)
        task = B.parse_task_id(r["id"])
        what = f"it should always follow this rule: {B.DESCRIBE[worst][task[worst]][0]}"
        entry = {"id": r["id"], "description": r["description"], "what": what, "report_score": r["scores"]["generated"]}
        for label, h in (("plain", 0.0), ("harmonized", 0.3)):
            copy = base.with_name(base.name + "_" + label)
            shutil.rmtree(copy, ignore_errors=True)
            shutil.copytree(base, copy)
            res = repair(copy, what, provider=provider, rounds=3, harmonize=h, generator=gen_dir, log=log)
            entry[label] = {k: res.get(k) for k in ("dev_before", "dev_after", "harmonized_dev", "test_before",
                                                    "test_after", "saved")}
            entry[label]["edits_tried"] = len(res["rounds"])
            entry[label]["rounds"] = res["rounds"]
        results.append(entry)
        mean = lambda k, l: round(sum(e[l][k] for e in results) / len(results), 3)
        out = {"provider": provider, "agent_calls": len(CALL_LOG) - calls0, "n": len(results),
               "plain": {"test_before": mean("test_before", "plain"), "test_after": mean("test_after", "plain"),
                         "saved": sum(e["plain"]["saved"] for e in results)},
               "harmonized": {"test_before": mean("test_before", "harmonized"), "test_after": mean("test_after", "harmonized"),
                              "saved": sum(e["harmonized"]["saved"] for e in results)},
               "tasks": results}
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        Path(out_file).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def flywheel_experiment(zoo_dir: str, gen_dir: str, out_file: str, n_new: int = 40, work: str = "runs/gen_flywheel",
                        log=print) -> dict:
    """Do generated adapters that pass their checks make the generator better?

    1. Generate adapters for NEW combinations (in neither the seen nor the held-out list).
    2. Keep the ones that pass at least 90% of practice questions. No LoRA is trained for them.
    3. Retrain the generator on the zoo plus the keepers.
    4. Compare both generators on the original held-out tasks, on different questions.
    """
    from safetensors.torch import load_file

    from .generator.core import Generator, train_generator
    from .zoo import behaviors as B
    from .zoo import trainer as T
    from .zoo.prompts import eval_prompts

    zoo = Path(zoo_dir)
    tasks = json.loads((zoo / "tasks.json").read_text())
    known = {t["id"] for t in tasks}
    held = [t for t in tasks if t["split"] == "heldout"]
    values = [(a, v) for a in B.AXES for v in B.DESCRIBE[a]]
    new = [dict(c) for c in itertools.combinations(values, 3)
           if len({a for a, _ in c}) == 3 and B.task_id(dict(c)) not in known]
    rng = random.Random(1)
    new = rng.sample(new, n_new)

    gen = Generator(gen_dir)
    model, tok, dev = T.load_base(gen.info["base_model"], gen.info["spec"])
    T.set_a(model, load_file(str(zoo / "lora_A.safetensors")))
    practice, test = eval_prompts(16, seed=5), eval_prompts(24)
    guidance = gen.info.get("guidance", 2.0)
    keep = []
    for i, task in enumerate(new):
        w = gen.generate(B.describe(task, rng), guidance=guidance)
        s = T.eval_adapter(model, tok, dev, task, practice, weights=w)["score"]
        log(f"flywheel [{i + 1}/{n_new}] {B.task_id(task):32s} {s:.2f}")
        if s >= 0.9:
            keep.append((w, {"adapter": f"fly-{B.task_id(task)}", "task": task, "description": B.describe(task),
                             "base_model": gen.info["base_model"], "spec": gen.info["spec"], "score": s,
                             "split": "seen", "seed": 0}))

    def heldout(g):
        scores = [T.eval_adapter(model, tok, dev, t["task"], test,
                                 weights=g.generate(B.describe(t["task"]), guidance=guidance))["score"] for t in held]
        return sum(scores) / len(scores), scores

    before, b_rows = heldout(gen)
    res = {"new_combinations_tried": n_new, "kept": len(keep), "heldout_before": round(before, 3)}
    if keep:
        d = gen.info["denoiser"]
        train_generator(zoo, work, steps=gen.info["steps"], latent_dim=gen.codec.dim, d=d["d"], layers=d["layers"],
                        extra=keep, log=lambda m: None)
        after, a_rows = heldout(Generator(work))
        res["heldout_after"] = round(after, 3)
        res["per_task"] = [{"id": t["id"], "before": b, "after": a} for t, b, a in zip(held, b_rows, a_rows)]
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(out_file).write_text(json.dumps(res, indent=1))
    return res
