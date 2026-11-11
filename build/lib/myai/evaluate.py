"""Step 4: test the model and write a report card.

eval.jsonl has one test per line:
  {"q": "How do I reset it?", "expect_any": ["hold", "10 seconds"], "kind": "fact"}
  {"q": "What's the stock price?", "kind": "trick"}   # should answer "I don't know"

Optional fields: "expect_all": [...], "forbid": [...]

Besides right/wrong, it measures two bad habits small models have:
  * looping   - repeating the same phrase over and over
  * rambling  - never stopping, hitting the length limit
"""

from __future__ import annotations

import json
import re
from pathlib import Path

IDK_PATTERNS = [r"\bi don'?t know\b", r"\bnot sure\b", r"\bdon'?t have (that|this|enough) information\b",
                r"\bnot (covered|mentioned|in the (docs|documentation|manual))\b", r"\bcan'?t (find|answer)\b"]


def looks_like_idk(text: str) -> bool:
    t = text.lower().replace("’", "'")
    return any(re.search(p, t) for p in IDK_PATTERNS)


def loop_score(text: str, n: int = 4) -> float:
    """Share of word n-grams that are repeats. ~0 for normal text, high when looping."""
    words = text.lower().split()
    if len(words) < n * 3:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1 - len(set(grams)) / len(grams)


def has(answer: str, phrase: str) -> bool:
    """Whole-word match, so "no" doesn't match "know" or "not"."""
    return re.search(r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)", answer.lower()) is not None


def grade(test: dict, answer: str) -> tuple[bool, str]:
    if test.get("kind") == "trick":
        ok = looks_like_idk(answer)
        return ok, "" if ok else "should have said it doesn't know"
    if any(has(answer, f) for f in test.get("forbid", [])):
        return False, "contains a forbidden phrase"
    if test.get("expect_all") and not all(has(answer, e) for e in test["expect_all"]):
        return False, "missing required info"
    if test.get("expect_any") and not any(has(answer, e) for e in test["expect_any"]):
        return False, "missing expected info"
    if looks_like_idk(answer) and not test.get("expect_any") and not test.get("expect_all"):
        return False, "said it doesn't know"
    return True, ""


def generate(model, tok, device, system: str, questions: list[str], max_new_tokens: int, batch_size: int = 8):
    import torch

    from .models import chat_text

    tok.padding_side = "left"
    results = []
    for i in range(0, len(questions), batch_size):
        prompts = [chat_text(tok, system, q) for q in questions[i : i + batch_size]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for row in out[:, enc["input_ids"].shape[1] :]:
            ids = row.tolist()
            stopped = tok.eos_token_id in ids or tok.pad_token_id in ids
            text = tok.decode(ids, skip_special_tokens=True).strip()
            results.append({"answer": text, "hit_limit": not stopped})
    return results


def load_tests(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def evaluate_model(model, tok, device, system: str, tests: list[dict], max_new_tokens: int = 256) -> dict:
    gens = generate(model, tok, device, system, [t["q"] for t in tests], max_new_tokens)
    rows = []
    for t, g in zip(tests, gens):
        ok, why = grade(t, g["answer"])
        rows.append({**t, **g, "pass": ok, "why": why, "loop": round(loop_score(g["answer"]), 3)})
    n = len(rows)
    facts = [r for r in rows if r.get("kind") != "trick"]
    tricks = [r for r in rows if r.get("kind") == "trick"]
    summary = {
        "tests": n,
        "score": round(sum(r["pass"] for r in rows) / n, 3) if n else 0,
        "fact_score": round(sum(r["pass"] for r in facts) / len(facts), 3) if facts else None,
        "honesty_score": round(sum(r["pass"] for r in tricks) / len(tricks), 3) if tricks else None,
        "looping_rate": round(sum(r["loop"] > 0.3 for r in rows) / n, 3) if n else 0,
        "hit_limit_rate": round(sum(r["hit_limit"] for r in rows) / n, 3) if n else 0,
    }
    return {"summary": summary, "rows": rows}


def run_eval(project, label: str, adapter: bool = True, model_path: str | None = None) -> dict:
    from .models import load_model

    if model_path:
        model, tok, device = load_model(model_path)
    else:
        model, tok, device = load_model(project.cfg["base_model"],
                                        adapter=project.adapter_dir if adapter else None)
    tests = load_tests(project.path("eval"))
    res = evaluate_model(model, tok, device, project.system_prompt(), tests,
                         project.cfg["eval_settings"]["max_new_tokens"])
    (project.reports_dir / f"eval_{label}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res


# ---------- report card ----------

LABELS = {
    "score": "Overall score",
    "fact_score": "Knows the documents",
    "honesty_score": "Admits when it doesn't know",
    "looping_rate": "Repeats itself (lower is better)",
    "hit_limit_rate": "Never stops talking (lower is better)",
}


def _fmt(v):
    return "—" if v is None else f"{v*100:.0f}%"


def write_report(project, results: dict[str, dict]) -> Path:
    """results: {"before": eval_result, "after": eval_result, ...} in column order."""
    cols = list(results)
    lines = [f"# Report card: {project.name}", "", f"> {project.goal.splitlines()[0]}", "",
             "| | " + " | ".join(c.capitalize() for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for key, label in LABELS.items():
        lines.append(f"| {label} | " + " | ".join(_fmt(results[c]["summary"].get(key)) for c in cols) + " |")
    last = results[cols[-1]]
    fails = [r for r in last["rows"] if not r["pass"]]
    lines += ["", f"## Still wrong in '{cols[-1]}' ({len(fails)} of {len(last['rows'])})", ""]
    for r in fails[:25]:
        ans = r["answer"].replace("\n", " ")
        lines.append(f"- **Q:** {r['q']}  \n  **A:** {ans[:300]}{'…' if len(ans) > 300 else ''}  \n  _{r['why']}_")
    if not fails:
        lines.append("Nothing. Every test passed.")
    out = project.reports_dir / "REPORT.md"
    out.write_text("\n".join(lines) + "\n")
    return out
