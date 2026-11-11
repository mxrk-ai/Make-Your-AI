"""A generated AI: a folder made by `myai generate`.

  my-ai/
    myai.json     what was asked for, what the generator understood, base model
    adapter/      a standard LoRA adapter (PEFT format)
    REPORT.md     written by `myai eval my-ai`
    model/        written by `myai export my-ai` (merged weights, optional GGUF + Ollama Modelfile)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

UNDERSTAND_MIN = 0.55  # how close a clause must be to a known behaviour to count as understood


def is_generated(folder: str | Path) -> bool:
    f = Path(folder) / "myai.json"
    return f.exists() and json.loads(f.read_text()).get("kind") == "generated"


def info(folder: str | Path) -> dict:
    return json.loads((Path(folder) / "myai.json").read_text())


def understand(description: str) -> tuple[dict, list[str]]:
    """Maps each clause of a description to the closest behaviour the zoo knows.
    Returns (task, clauses that matched nothing). The task is what `myai eval` can check automatically."""
    from .generator import text
    from .zoo.behaviors import DESCRIBE

    known = [(a, v, p) for a in DESCRIBE for v in DESCRIBE[a] for p in DESCRIBE[a][v]]
    bank = text.embed([p for _, _, p in known])
    task, best, unknown = {}, {}, []
    for c in text.clauses(description):
        sims = bank @ text.embed([c])[0]
        i = int(sims.argmax())
        a, v, _ = known[i]
        if float(sims[i]) < UNDERSTAND_MIN:
            unknown.append(c)
        elif float(sims[i]) > best.get(a, 0):
            task[a], best[a] = v, float(sims[i])
    return task, unknown


def slug(description: str) -> str:
    words = re.findall(r"[a-z0-9]+", description.lower())
    return "-".join(words[:5])[:40] or "my-ai"


def create(description: str, out: str | Path | None = None, generator: str | None = None, mode: str = "generate",
           seed: int = 0, guidance: float | None = None) -> Path:
    from .generator.core import Generator, find_generator, write_adapter

    gen = Generator(find_generator(generator))
    out = Path(out or slug(description))
    task, unknown = understand(description)
    extra = {}
    if mode == "retrieve":
        weights, hit = gen.retrieve(description)
        extra["retrieved"] = hit
    else:
        weights = gen.generate(description, seed=seed, guidance=guidance or gen.info.get("guidance", 2.0))
    write_adapter(out / "adapter", weights, gen.lora_a, gen.info["base_model"], gen.info["spec"])
    (out / "myai.json").write_text(json.dumps({
        "kind": "generated", "name": out.name, "description": description, "mode": mode, "seed": seed,
        "base_model": gen.info["base_model"], "understood": task, "not_understood": unknown,
        "system_prompt": description if gen.info.get("use_system_prompt") else None, **extra}, indent=2))
    return out


def load(folder: str | Path, adapter: bool = True):
    """(model, tok, device, system prompt or None) for a generated AI. adapter=False gives the plain base model."""
    from .models import load_model

    folder = Path(folder)
    meta = info(folder)
    model, tok, dev = load_model(meta["base_model"], adapter=folder / "adapter" if adapter else None)
    return model, tok, dev, meta.get("system_prompt")


def check(folder: str | Path, n: int = 20, log=print) -> dict:
    """Before/after test of a generated AI on fresh questions, using the behaviours that were understood."""
    from .zoo import behaviors as B
    from .zoo import trainer as T
    from .zoo.prompts import eval_prompts

    folder = Path(folder)
    meta = info(folder)
    task = meta["understood"]
    if not task:
        raise ValueError("None of the description was matched to a behaviour that can be checked automatically. "
                         f"Try it by hand: myai chat {folder}")
    prompts = eval_prompts(n, seed=7)
    res = {}
    for label, use in (("before", False), ("after", True)):
        model, tok, dev, system = load(folder, adapter=use)
        rows = [{"q": q, **g} for q, g in zip(prompts, T.generate_answers(model, tok, dev, prompts, 300,
                                                                          system=system if use else None))]
        res[label] = {**T.score_answers(rows, task), "rows": rows}
        del model
    lines = [f"# Report card: {meta['name']}", "", f"> {meta['description']}", "",
             f"Checked automatically: **{B.describe(task)}**"]
    if meta["not_understood"]:
        lines.append(f"Not checked (not understood): {'; '.join(meta['not_understood'])}")
    lines += ["", f"| | Base model | Your AI |", "|---|---|---|",
              f"| Answers that follow every rule ({n} fresh questions) | {res['before']['score']:.0%} | {res['after']['score']:.0%} |"]
    for a in task:
        lines.append(f"| {B.DESCRIBE[a][task[a]][0]} | {res['before']['per_axis'].get(a, 0):.0%} | {res['after']['per_axis'].get(a, 0):.0%} |")
    fails = [r for r in res["after"]["rows"] if not r["pass"]]
    lines += ["", f"## Example answer", "", f"**Q:** {res['after']['rows'][0]['q']}  ", f"**A:** {res['after']['rows'][0]['answer'][:400]}",
              "", f"## Answers that broke a rule ({len(fails)} of {n})", ""]
    for r in fails[:10]:
        broken = ", ".join(a for a, ok in r["checks"].items() if not ok) or "repeats itself"
        ans = r["answer"][:300].replace("\n", " ")
        lines.append(f"- **Q:** {r['q']}  \n  **A:** {ans}  \n  _broke: {broken}_")
    if not fails:
        lines.append("None.")
    (folder / "REPORT.md").write_text("\n".join(lines) + "\n")
    (folder / "eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res
