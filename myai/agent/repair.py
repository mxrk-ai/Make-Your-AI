"""`myai patch fix`: an LLM agent repairs a generated adapter by editing its weights. No gradient descent.

The loop:
  1. Test the adapter on half of the test questions and collect what goes wrong.
  2. Measure the weights: size per layer, how concentrated each matrix is, what breaks when a part is switched off.
  3. Show all of that to the agent. It answers with a hypothesis and up to three candidate edits.
  4. Try each edit, keep the best one only if the score goes up, tell the agent what happened. Repeat.
  5. Report on the other half of the questions, which the agent never saw. Save only if that half is not worse.

The referee is automatic checks where they exist (format, length, repeating, never stopping) and the
agent LLM as a judge for goals that have no automatic check.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

GROUPS = [(0, 7), (8, 15), (16, 23)]

OPS_HELP = """Available edits (JSON). "where" selects modules: {"layers": [first, last], "module": "<module name from the table>" | "both"} ("both" = all modules).
  {"op": "scale", "where": {...}, "factor": 1.4}     multiply those B matrices (factor < 1 weakens, > 1 strengthens)
  {"op": "zero", "where": {...}}                      switch those modules off
  {"op": "keep_rank", "where": {...}, "k": 1}         keep only the k strongest directions of each matrix (1-3)
  {"op": "blend", "description": "...", "weight": 0.5}   mix in a freshly generated adapter for that description (0-1)
  {"op": "regenerate", "description": "...", "seed": 3}  replace the adapter with a newly generated one"""

SYSTEM = ("You repair small neural network adapters by editing their weights directly. You reason from measurements, "
          "propose small targeted edits, and you answer with JSON only.")


def read_b(adapter_dir: Path) -> dict:
    from safetensors.torch import load_file

    out = {}
    for k, v in load_file(str(Path(adapter_dir) / "adapter_model.safetensors")).items():
        m = re.search(r"layers\.(\d+)\.\w+\.(\w+)\.lora_B", k)
        if m:
            out[f"{m.group(1)}.{m.group(2)}"] = v.float()
    return out


def select(weights: dict, where: dict | None) -> list[str]:
    where = where or {}
    lo, hi = (where.get("layers") or [0, 999])[:2]
    mod = where.get("module", "both")
    return [k for k in weights if lo <= int(k.split(".")[0]) <= hi and (mod == "both" or k.endswith(mod))]


def apply_ops(weights: dict, ops: list[dict], gen=None) -> dict:
    """Returns edited copies; never changes `weights`."""
    import torch

    w = {k: v.clone() for k, v in weights.items()}
    for op in ops:
        kind = op.get("op")
        if kind == "scale":
            f = max(0.0, min(float(op.get("factor", 1.0)), 3.0))
            for k in select(w, op.get("where")):
                w[k] = w[k] * f
        elif kind == "zero":
            for k in select(w, op.get("where")):
                w[k] = torch.zeros_like(w[k])
        elif kind == "keep_rank":
            n = max(1, int(op.get("k", 1)))
            for k in select(w, op.get("where")):
                u, s, vt = torch.linalg.svd(w[k], full_matrices=False)
                s[n:] = 0
                w[k] = (u * s) @ vt
        elif kind in ("blend", "regenerate"):
            if gen is None:
                raise ValueError(f"'{kind}' needs the generator (pass --generator or install the published one)")
            new = gen.generate(str(op["description"]), seed=int(op.get("seed", 1)))
            a = 1.0 if kind == "regenerate" else max(0.0, min(float(op.get("weight", 0.5)), 1.0))
            w = {k: (1 - a) * w[k] + a * new[k] for k in w}
        else:
            raise ValueError(f"unknown op '{kind}'")
    return w


def modules_of(weights: dict) -> list[str]:
    return sorted({k.split(".")[1] for k in weights})


def measure(weights: dict) -> str:
    """A compact table of the weights for the agent: size and concentration per layer and module."""
    import torch

    mods = modules_of(weights)
    lines = ["layer  " + "  ".join(f"|{m}|  top-direction share" for m in mods)]
    for layer in sorted({int(k.split(".")[0]) for k in weights}):
        cells = []
        for m in mods:
            b = weights[f"{layer}.{m}"]
            s = torch.linalg.svdvals(b)
            cells.append(f"{b.norm():>8.3f}  {(s[0] ** 2 / (s ** 2).sum().clamp_min(1e-12)):>19.2f}")
        lines.append(f"{layer:>5}  " + "  ".join(cells))
    return "\n".join(lines)


class Referee:
    """Scores answers. Automatic checks first; the agent LLM judges goals that have no check."""

    def __init__(self, task: dict, what: str, judge_goal: str | None, provider: str | None):
        self.task, self.what, self.judge_goal, self.provider = task, what, judge_goal, provider
        self.cache: dict[str, bool] = {}

    def judge(self, rows: list[dict]) -> None:
        from .llm import complete, extract_json

        todo = [r for r in rows if r["answer"] not in self.cache]
        if todo:
            listing = "\n".join(f"{i}. Q: {r['q']}\n   A: {r['answer'][:500]}" for i, r in enumerate(todo))
            reply = complete(f'Goal for every answer: "{self.judge_goal}".\nFor each answer below say whether it meets '
                             f"the goal. Reply with a JSON list of true/false, one per answer, in order.\n\n{listing}",
                             SYSTEM, self.provider)
            verdicts = extract_json(reply)
            for r, v in zip(todo, verdicts):
                self.cache[r["answer"]] = bool(v)
        for r in rows:
            r["judge"] = self.cache.get(r["answer"], False)

    def score(self, rows: list[dict]) -> dict:
        from ..zoo import trainer as T

        res = T.score_answers(rows, self.task)
        if self.judge_goal:
            self.judge(rows)
        ok = 0
        for r in rows:
            r["ok"] = r["pass"] and not r["hit_limit"] and r.get("judge", True)
            ok += r["ok"]
        res["score"] = round(ok / max(len(rows), 1), 3)
        if self.judge_goal:
            res["per_axis"]["judge"] = round(sum(r["judge"] for r in rows) / len(rows), 3)
        res["loop_rate"] = round(sum(not r["pass"] and all(r["checks"].values()) for r in rows) / len(rows), 3)
        return res


def repair(folder: str | Path, what: str, provider: str | None = None, rounds: int = 4, harmonize: float = 0.0,
           generator: str | None = None, n: int = 12, log=print) -> dict:
    import os

    from .. import generated as G
    from ..generator.core import Generator, find_generator, write_adapter
    from ..zoo import trainer as T
    from ..zoo.prompts import eval_prompts
    from .llm import CALL_LOG, complete, extract_json, load_env

    load_env()
    provider = provider or os.environ.get("MYAI_AGENT_PROVIDER")
    if not provider:
        raise ValueError("Choose the agent LLM: --provider agy (or gemini, anthropic, openai, ollama).")
    folder = Path(folder)
    if not G.is_generated(folder):
        raise ValueError(f"{folder} was not made by `myai generate`. For projects built from documents use `myai patch auto`.")
    meta = G.info(folder)
    try:
        gen = Generator(find_generator(generator))
    except FileNotFoundError:
        gen = None
    spec = gen.info["spec"] if gen else None
    from ..zoo.spec import LORA_SPEC

    model, tok, dev = T.load_base(meta["base_model"], spec or LORA_SPEC)
    original = read_b(folder / "adapter")

    # What to check: the AI's own rules, plus whatever part of the fix request we can check automatically.
    extra, unknown = G.understand(what)
    task = {**meta.get("understood", {}), **extra}
    habit = bool(re.search(r"repeat|loop|ramble|never stop|too long|cut off", what, re.I))
    judge_goal = "; ".join(unknown) if unknown and not habit else None
    ref = Referee(task, what, judge_goal, provider)
    prompts = eval_prompts(2 * n, seed=11)
    dev_q, test_q = prompts[:n], prompts[n:]

    def run(w, qs):
        rows = [{"q": q, **g} for q, g in zip(qs, (T.set_b(model, w), T.generate_answers(model, tok, dev, qs, 300))[1])]
        return {**ref.score(rows), "rows": rows}

    log(f"Goal: {what}\nReferee: " + (", ".join(f"{a}={v}" for a, v in task.items()) or "no format rules")
        + ", no repeating, must stop" + (f", agent judges: {judge_goal}" if judge_goal else ""))
    best_w, best = original, run(original, dev_q)
    log(f"Before: {best['score']:.0%} of {n} practice questions pass")

    # What breaks when a part is switched off? (cheap causal evidence for the agent)
    ablation = []
    for lo, hi in GROUPS:
        for mod in modules_of(original):
            s = run(apply_ops(original, [{"op": "zero", "where": {"layers": [lo, hi], "module": mod}}]), dev_q)
            ablation.append(f"zero layers {lo}-{hi} {mod}: score {s['score']:.2f}, checks {s['per_axis']}, never-stops {s['hit_limit_rate']}")
    history = []
    for rnd in range(1, rounds + 1):
        if best["score"] >= 1.0:
            break
        fails = [r for r in best["rows"] if not r["ok"]][:4]
        prompt = (
            f'The adapter was generated for: "{meta["description"]}".\nThe user wants this fixed: "{what}".\n\n'
            f"Current score {best['score']:.2f} on {n} questions. Per check: {best['per_axis']}. "
            f"Never stops: {best['hit_limit_rate']}. Repeats itself: {best['loop_rate']}.\n\n"
            "Failing answers:\n" + "\n".join(f"- Q: {r['q']}\n  A: {r['answer'][:300]!r}\n  failed: "
                                             f"{[a for a, ok in r['checks'].items() if not ok] or 'repeats / does not stop / judge'}" for r in fails)
            + f"\n\nWeights (LoRA B matrices, rank {best_w[next(iter(best_w))].shape[1]}, 24 layers):\n{measure(best_w)}\n\n"
            "Ablations on the original adapter:\n" + "\n".join(ablation)
            + "\n\nEdits tried so far:\n" + ("\n".join(history) or "none")
            + f"\n\n{OPS_HELP}\n" + ("" if gen else "(blend and regenerate are NOT available in this session.)\n")
            + '\nReply with JSON: {"hypothesis": "one sentence on the cause", "candidates": [[op, ...], [op, ...]]} '
              "with 1 to 3 candidates, each a short list of edits applied together to the current adapter.")
        try:
            plan = extract_json(complete(prompt, SYSTEM, provider))
            cands = [c if isinstance(c, list) else [c] for c in plan.get("candidates", [])][:3]
        except Exception as e:  # a bad reply costs one round, not the whole run
            log(f"Round {rnd}: the agent's reply could not be used ({e})")
            history.append(f"round {rnd}: reply was not valid JSON")
            continue
        log(f"Round {rnd}: {plan.get('hypothesis', '')}")
        round_best = None
        for c in cands:
            try:
                w = apply_ops(best_w, c, gen)
            except Exception as e:
                history.append(f"round {rnd}: {json.dumps(c)} -> invalid ({e})")
                continue
            s = run(w, dev_q)
            history.append(f"round {rnd}: {json.dumps(c)} -> score {s['score']:.2f} (was {best['score']:.2f}), "
                           f"checks {s['per_axis']}, never-stops {s['hit_limit_rate']}")
            log(f"   tried {json.dumps(c)[:150]} -> {s['score']:.0%}")
            if round_best is None or s["score"] > round_best[1]["score"]:
                round_best = (w, s, c)
        if round_best and round_best[1]["score"] > best["score"]:
            best_w, best = round_best[0], round_best[1]
            log(f"   kept: {best['score']:.0%}")
        else:
            log("   nothing helped this round; undone")

    result = {"what": what, "provider": provider, "rounds": history, "dev_before": run(original, dev_q)["score"],
              "dev_after": best["score"], "agent_calls": len(CALL_LOG)}
    if harmonize > 0 and gen is not None and best_w is not original:
        hw = gen.harmonize(best_w, meta["description"], strength=harmonize)
        hs = run(hw, dev_q)
        result["harmonized_dev"] = hs["score"]
        log(f"Harmonised (strength {harmonize}): {hs['score']:.0%} vs {best['score']:.0%} before harmonising")
        if hs["score"] >= best["score"]:
            best_w, best = hw, hs
    before_t, after_t = run(original, test_q), run(best_w, test_q)
    result.update(test_before=before_t["score"], test_after=after_t["score"])
    log(f"\nFresh questions the agent never saw: {before_t['score']:.0%} -> {after_t['score']:.0%}")
    if best_w is not original and after_t["score"] >= before_t["score"] and best["score"] > result["dev_before"]:
        if not (folder / "adapter.orig").exists():
            shutil.copytree(folder / "adapter", folder / "adapter.orig")
        lora_a = gen.lora_a if gen else T.get_a(model)
        write_adapter(folder / "adapter", best_w, lora_a, meta["base_model"], spec or LORA_SPEC)
        result["saved"] = True
        log(f"Saved the repaired adapter (original kept in {folder}/adapter.orig).")
    else:
        result["saved"] = False
        log("No edit held up on fresh questions, so nothing was changed.")
    with (folder / "patch_log.jsonl").open("a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result
