"""The captioner: the agent looks at what an adapter does and writes its description.

Evidence shown to the agent: answers to a few probe questions with and without the adapter, and where in the
network the adapter is strongest. It is scored against the true description two ways: how many of the true
behaviours can be recognised in the caption, and how close the two texts are as sentence embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path

PROBES = ["What is a volcano?", "Why is sleep important?", "How does a bicycle work?", "Tell me about bees."]


def caption_adapter(model, tok, dev, weights: dict, base_rows: list[dict], provider: str | None = None) -> str:
    from ..zoo import trainer as T
    from .llm import complete
    from .repair import SYSTEM, measure

    T.set_b(model, weights)
    rows = T.generate_answers(model, tok, dev, PROBES, 160)
    shown = "\n\n".join(f"Q: {q}\nWithout adapter: {b['answer'][:350]!r}\nWith adapter:    {a['answer'][:350]!r}"
                        for q, b, a in zip(PROBES, base_rows, rows))
    reply = complete(
        "A small adapter (LoRA) was added to a language model. Compare the answers with and without it and describe, "
        "as instructions, the behaviour the adapter adds. Mention every rule you can see (length, format, letter case, "
        "tone, fixed opening or closing phrases). Use short imperative sentences and nothing else.\n\n"
        f"{shown}\n\nAdapter weight sizes per layer:\n{measure(weights)}", SYSTEM.replace("JSON only", "plain text only"), provider)
    return reply.strip().strip('"')


def caption_zoo(zoo_dir: str | Path, out_file: str | Path, n: int = 20, provider: str | None = None, seed: int = 0,
                log=print) -> dict:
    import random

    from ..generated import understand
    from ..generator import text
    from ..generator.core import load_zoo
    from ..zoo import trainer as T
    from .llm import CALL_LOG

    zoo = [(w, m) for w, m in load_zoo(zoo_dir) if m["seed"] == 0 and m["score"] >= 0.7]
    sample = random.Random(seed).sample(zoo, min(n, len(zoo)))
    model, tok, dev = T.load_base(sample[0][1]["base_model"], sample[0][1]["spec"])
    from safetensors.torch import load_file

    T.set_a(model, load_file(str(Path(zoo_dir) / "lora_A.safetensors")))
    T.set_b(model, None)
    base_rows = T.generate_answers(model, tok, dev, PROBES, 160)
    rows = []
    for w, m in sample:
        cap = caption_adapter(model, tok, dev, w, base_rows, provider)
        got, _ = understand(cap)
        truth = m["task"]
        hit = sum(got.get(a) == v for a, v in truth.items())
        cos = float(text.embed_whole([cap])[0] @ text.embed_whole([m["description"]])[0])
        rows.append({"adapter": m["adapter"], "truth": m["description"], "caption": cap, "behaviours_found": hit,
                     "behaviours_total": len(truth), "extra": len(set(got) - set(truth)), "exact": got == truth,
                     "cosine": round(cos, 3)})
        log(f"{m['adapter']:30s} found {hit}/{len(truth)}  cos={cos:.2f}  | {cap[:90]}")
    total = sum(r["behaviours_total"] for r in rows)
    res = {"n": len(rows), "behaviours_recovered": round(sum(r["behaviours_found"] for r in rows) / total, 3),
           "exact_match": round(sum(r["exact"] for r in rows) / len(rows), 3),
           "mean_cosine": round(sum(r["cosine"] for r in rows) / len(rows), 3),
           "agent_calls": len(CALL_LOG), "provider": provider, "rows": rows}
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(out_file).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res
