"""Turns a description into what the generator is conditioned on: one embedding per clause.

"always answer in formal English, max 3 sentences" -> 2 clauses -> 2 vectors.
Clause-level vectors keep each instruction separate instead of blurring them into one.
"""

from __future__ import annotations

import re

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
MAX_CLAUSES = 6
_cache: dict = {}


def clauses(description: str) -> list[str]:
    text = re.sub(r'"[^"]*"', lambda m: m.group(0).replace(".", "․").replace(",", "‚").replace("!", "ǃ"), description)
    parts = re.split(r"[.;!\n]+|,\s*|\s+and\s+(?=[a-z]+\s)", text)
    parts = [p.replace("․", ".").replace("‚", ",").replace("ǃ", "!").strip(" -") for p in parts]
    parts = [p for p in parts if re.search(r"\w", p)]
    return parts[:MAX_CLAUSES] or [description.strip() or "answer normally"]


def embed(texts: list[str]):
    """Sentence embeddings, unit length. Runs on CPU; the model is 90 MB and loads once."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if "m" not in _cache:
        _cache["t"] = AutoTokenizer.from_pretrained(ENCODER)
        _cache["m"] = AutoModel.from_pretrained(ENCODER).eval()
    new = [t for t in dict.fromkeys(texts) if t not in _cache]
    for i in range(0, len(new), 64):
        enc = _cache["t"](new[i:i + 64], padding=True, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            h = _cache["m"](**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        v = torch.nn.functional.normalize((h * mask).sum(1) / mask.sum(1), dim=-1)
        for t, e in zip(new[i:i + 64], v):
            _cache[t] = e
    return torch.stack([_cache[t] for t in texts])


def condition(descriptions: list[str]):
    """-> (cond [batch, MAX_CLAUSES, 384], mask [batch, MAX_CLAUSES] True where a clause exists)."""
    import torch

    cond = torch.zeros(len(descriptions), MAX_CLAUSES, EMBED_DIM)
    mask = torch.zeros(len(descriptions), MAX_CLAUSES, dtype=torch.bool)
    for i, d in enumerate(descriptions):
        e = embed(clauses(d))
        cond[i, :len(e)], mask[i, :len(e)] = e, True
    return cond, mask


def embed_whole(descriptions: list[str]):
    """One vector per description (used by the nearest-neighbour baseline)."""
    return embed(descriptions)
