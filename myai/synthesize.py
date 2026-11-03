"""Step 2: turn document chunks into practice questions and answers.

Two ways to do it:
  * backend "agent"  - the AI agent running this repo (Claude Code, Codex, ...)
                       writes the Q&A itself. No API key needed. `prepare_agent_tasks`
                       writes a work list, `validate` checks what the agent wrote.
  * backend "openai" / "anthropic" - the script calls an LLM API for every chunk.
                       "openai" works with any OpenAI-compatible server
                       (OpenAI, OpenRouter, Ollama, LM Studio, vLLM...).

Every line of synthetic.jsonl looks like:
  {"q": "...", "a": "...", "chunk": "manual-3", "kind": "fact" | "trick"}
"""

from __future__ import annotations

import json
import os
import random
import urllib.request
from pathlib import Path

IDK = "I don't know"

PROMPT = """You are writing training data for a small assistant.

The assistant's job:
{goal}

Here is one piece of the source documents:
<<<
{chunk}
>>>

Write {n} question/answer pairs a real user might ask that THIS text answers.
Rules:
- Answers must be fully supported by the text. Never add outside facts.
- Answers follow the assistant's style: {style}
- Vary the phrasing: casual, formal, short, with typos, etc.
- Also write {n_trick} "trick" questions that sound related but the text does NOT
  answer. Their answer must start with "{idk}" and then say briefly what the
  assistant can help with instead.

Reply with ONLY a JSON list, no prose:
[{{"q": "...", "a": "...", "kind": "fact"}}, {{"q": "...", "a": "...", "kind": "trick"}}]
"""


def build_prompt(goal: str, style: list[str], chunk: str, n: int, trick_share: float) -> str:
    n_trick = max(1, round(n * trick_share)) if trick_share > 0 else 0
    return PROMPT.format(
        goal=goal,
        chunk=chunk,
        n=n,
        n_trick=n_trick,
        style="; ".join(style) or "clear and short",
        idk=IDK,
    )


# ---------- API backends (stdlib only, no SDK needed) ----------

def _post(url: str, headers: dict, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def call_llm(backend: str, model: str | None, prompt: str) -> str:
    if backend == "openai":
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        key = os.environ.get("OPENAI_API_KEY", "none")
        out = _post(
            f"{base}/chat/completions",
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            {"model": model or "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]},
        )
        return out["choices"][0]["message"]["content"]
    if backend == "anthropic":
        key = os.environ["ANTHROPIC_API_KEY"]
        out = _post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            {
                "model": model or "claude-haiku-5-5",
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        return "".join(b.get("text", "") for b in out["content"])
    raise ValueError(f"Unknown synth backend: {backend}")


def parse_pairs(text: str) -> list[dict]:
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < 0:
        return []
    try:
        items = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    good = []
    for it in items:
        if isinstance(it, dict) and str(it.get("q", "")).strip() and str(it.get("a", "")).strip():
            kind = "trick" if it.get("kind") == "trick" else "fact"
            good.append({"q": it["q"].strip(), "a": it["a"].strip(), "kind": kind})
    return good


def synthesize_api(project) -> int:
    s = project.cfg["synth"]
    chunks = [json.loads(l) for l in project.chunks_file.read_text().splitlines() if l.strip()]
    total = 0
    with project.synthetic_file.open("w") as f:
        for i, ch in enumerate(chunks, 1):
            prompt = build_prompt(project.goal, project.cfg["style"], ch["text"], s["per_chunk"], s["trick_questions"])
            pairs = parse_pairs(call_llm(s["backend"], s["model"], prompt))
            for p in pairs:
                p["chunk"] = ch["id"]
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
            total += len(pairs)
            print(f"  [{i}/{len(chunks)}] {ch['id']}: {len(pairs)} pairs")
    return total


# ---------- agent backend ----------

def prepare_agent_tasks(project) -> Path:
    """Write a work list the agent follows to create synthetic.jsonl itself."""
    s = project.cfg["synth"]
    chunks = [json.loads(l) for l in project.chunks_file.read_text().splitlines() if l.strip()]
    out = project.run_dir / "SYNTH_TASKS.md"
    lines = [
        "# Synthetic data work list (for the AI agent)",
        "",
        f"Write your results to `{project.synthetic_file.relative_to(project.root)}`,",
        "one JSON object per line: {\"q\": ..., \"a\": ..., \"chunk\": <chunk id>, \"kind\": \"fact\"|\"trick\"}",
        "Then run `myai synth --check <project>` to validate.",
        "",
        "## Instructions (same for every chunk)",
        "",
        build_prompt(project.goal, project.cfg["style"], "<the chunk below>", s["per_chunk"], s["trick_questions"]),
        "",
        "## Chunks",
        "",
    ]
    for ch in chunks:
        lines += [f"### chunk `{ch['id']}` (from {ch['source']})", "", ch["text"], ""]
    out.write_text("\n".join(lines))
    return out


def validate(project) -> dict:
    f = project.synthetic_file
    if not f.exists():
        raise FileNotFoundError(f"{f} does not exist yet")
    chunk_ids = {json.loads(l)["id"] for l in project.chunks_file.read_text().splitlines() if l.strip()}
    stats = {"pairs": 0, "fact": 0, "trick": 0, "bad_lines": 0, "chunks_covered": 0, "chunks_total": len(chunk_ids)}
    covered = set()
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            assert rec["q"].strip() and rec["a"].strip()
        except Exception:
            stats["bad_lines"] += 1
            continue
        stats["pairs"] += 1
        stats["trick" if rec.get("kind") == "trick" else "fact"] += 1
        if rec.get("chunk") in chunk_ids:
            covered.add(rec["chunk"])
    stats["chunks_covered"] = len(covered)
    return stats


def load_pairs(project, seed: int = 42) -> list[dict]:
    pairs = [json.loads(l) for l in project.synthetic_file.read_text().splitlines() if l.strip()]
    random.Random(seed).shuffle(pairs)
    return pairs
