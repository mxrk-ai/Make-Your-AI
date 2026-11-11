"""The fixed adapter config and the task file format for the zoo.

Every adapter in the zoo uses LORA_SPEC unchanged. That is what lets the generator
treat adapters as points in one shared space (see docs/DESIGN.md, "Shared coordinates").

A task is one YAML file:

    id: formal-short
    description: Always answer in formal English, in at most 3 sentences.
    split: seen            # seen = generator may train on it, heldout = never
    train: formal-short.train.jsonl   # {"q": ..., "a": ...} per line
    eval: formal-short.eval.jsonl     # test prompts with checks, never used for training
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Rank 4 learned length rules (80-93%) but only 67% on ALL CAPS in the pilot, so the zoo uses rank 8.
LORA_SPEC = {
    "r": 8,
    "alpha": 16,
    "dropout": 0.0,
    "target_modules": ["q_proj", "v_proj"],
    "layers": "all",
    "init_seed": 0,        # the same A matrices for every adapter
    "train_b_only": True,  # A stays frozen (LoRA-FA style); the generator only has to produce B
}

# Output sizes of the targeted modules in BASE_MODEL (24 blocks, hidden 896, 2 key/value heads of 64).
_BASE_DIMS = {"layers": 24, "q_proj": 896, "k_proj": 128, "v_proj": 128, "o_proj": 896}

SPLITS = ("seen", "heldout")


def expected_b_params(spec: dict = LORA_SPEC, dims: dict = _BASE_DIMS) -> int:
    """How many numbers the generator has to produce per adapter (the B matrices only)."""
    per_layer = sum(dims[m] * spec["r"] for m in spec["target_modules"])
    return per_layer * dims["layers"]


@dataclass
class Task:
    id: str
    description: str
    split: str
    train: Path
    eval: Path


def load_tasks(folder: str | Path) -> tuple[list[Task], list[str]]:
    """Reads every *.yaml in `folder`. Returns (tasks, problems); problems is empty when all is well."""
    import yaml

    folder = Path(folder)
    tasks, problems, seen_ids = [], [], set()
    for f in sorted(folder.glob("*.yaml")):
        raw = yaml.safe_load(f.read_text()) or {}
        missing = [k for k in ("id", "description", "split", "train", "eval") if not raw.get(k)]
        if missing:
            problems.append(f"{f.name}: missing {', '.join(missing)}")
            continue
        if raw["split"] not in SPLITS:
            problems.append(f"{f.name}: split must be one of {', '.join(SPLITS)}")
        if raw["id"] in seen_ids:
            problems.append(f"{f.name}: duplicate id '{raw['id']}'")
        seen_ids.add(raw["id"])
        t = Task(raw["id"], raw["description"].strip(), raw["split"], folder / raw["train"], folder / raw["eval"])
        for kind, path in (("train", t.train), ("eval", t.eval)):
            if not path.exists():
                problems.append(f"{f.name}: {kind} file not found: {path.name}")
        tasks.append(t)
    if not tasks and not problems:
        problems.append(f"no task files (*.yaml) in {folder}")
    return tasks, problems
