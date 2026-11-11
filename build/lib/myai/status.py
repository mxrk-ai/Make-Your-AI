"""`myai status`: what's done in a project, what's missing, and the one command to run next."""

from __future__ import annotations

import json
import os

from .ingest import TEXT_EXT

DOC_EXT = TEXT_EXT | {".pdf", ".docx"}
PLACEHOLDER = "Replace me"


def doc_files(project) -> list:
    d = project.path("docs")
    return sorted(f for f in d.rglob("*") if f.suffix.lower() in DOC_EXT) if d.exists() else []


def eval_problems(project) -> tuple[list[dict], list[str]]:
    """The test questions, plus anything wrong with them in plain words."""
    f = project.path("eval")
    if not f.exists():
        return [], [f"{f.name} is missing"]
    tests, problems = [], []
    for i, line in enumerate(f.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            t = json.loads(line)
            assert t["q"].strip()
        except Exception:
            problems.append(f"{f.name} line {i} is not valid (needs {{\"q\": ...}})")
            continue
        if PLACEHOLDER in t["q"]:
            problems.append(f"{f.name} line {i} is still the example question. Write your own")
        tests.append(t)
    if len(tests) < 5:
        problems.append(f"only {len(tests)} test questions; write about 20 so the score means something")
    return tests, problems


def _short(path) -> str:
    """The path as the user would type it: relative when it's inside the current folder."""
    rel = os.path.relpath(path)
    return str(path) if rel.startswith("..") else rel


def steps(project) -> list[dict]:
    """Every step of the pipeline: name, done?, detail, and the command that does it."""
    p, d = project, _short(project.root)
    agent = p.cfg["synth"]["backend"] == "agent"
    docs = doc_files(p)
    tests, problems = eval_problems(p)
    tricks = sum(t.get("kind") == "trick" for t in tests)
    n_pairs = len(p.synthetic_file.read_text().splitlines()) if p.synthetic_file.exists() else 0
    n_chunks = len(p.chunks_file.read_text().splitlines()) if p.chunks_file.exists() else 0
    rep = p.reports_dir

    def score(label):
        f = rep / f"eval_{label}.json"
        return json.loads(f.read_text())["summary"]["score"] if f.exists() else None

    before, after = score("before"), score("after")
    return [
        {"name": "Describe your AI (goal.yaml)", "done": "<your topic>" not in p.goal,
         "detail": p.goal.splitlines()[0][:70], "do": f"edit {d}/goal.yaml: say what your AI should do"},
        {"name": "Add your documents (docs/)", "done": bool(docs),
         "detail": f"{len(docs)} file(s)", "do": f"put your .pdf .docx .md .txt files in {p.path('docs')}"},
        {"name": "Write test questions (eval.jsonl)", "done": bool(tests) and not problems,
         "detail": f"{len(tests)} questions, {tricks} trick" + (f"  ({problems[0]})" if problems else ""),
         "do": f"write ~20 questions in {p.path('eval')} (see examples/support-bot/eval.jsonl)"},
        {"name": "Read the documents", "done": n_chunks > 0, "detail": f"{n_chunks} chunks",
         "do": f"myai ingest {d}"},
        {"name": "Write practice questions", "done": n_pairs > 0, "detail": f"{n_pairs} Q&A pairs",
         "do": (f"ask your coding agent: \"write the practice questions for {d}\"" if agent
                else f"myai synth {d}")},
        {"name": "Test the untrained model", "done": before is not None,
         "detail": f"score {before:.0%}" if before is not None else "", "do": f"myai eval {d} --base"},
        {"name": "Train (LoRA)", "done": p.adapter_dir.exists(), "detail": "", "do": f"myai train {d}"},
        {"name": "Test the trained model", "done": after is not None,
         "detail": f"score {after:.0%}" if after is not None else "", "do": f"myai eval {d}"},
        {"name": "Report card", "done": (rep / "REPORT.md").exists(), "detail": "", "do": f"myai report {d}"},
        {"name": "Export your AI", "done": (p.model_dir / "myai.json").exists(), "detail": "",
         "do": f"myai export {d}"},
    ]


def show(project) -> None:
    all_steps = steps(project)
    print(f"Project: {project.name}  ({project.root})\n")
    nxt = None
    for i, s in enumerate(all_steps, 1):
        mark = "✅" if s["done"] else "⬜"
        print(f"  {mark} {i:>2}. {s['name']:34s} {s['detail']}")
        if not s["done"] and nxt is None:
            nxt = s
    if nxt:
        print(f"\nNext: {nxt['do']}")
        if all_steps.index(nxt) >= 3:
            print(f"   or run everything that's left: myai build {_short(project.root)}")
    else:
        print(f"\nDone! Talk to it: myai chat {_short(project.root)}")
        print(f"Report card: {project.reports_dir / 'REPORT.md'}")
