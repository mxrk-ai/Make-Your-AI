"""Loads a project's goal.yaml and fills in defaults."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
    "docs": "docs",
    "eval": "eval.jsonl",
    "style": [],
    "synth": {
        "backend": "agent",  # agent | openai | anthropic
        "model": None,
        "per_chunk": 8,
        "trick_questions": 0.15,  # share of "not in the docs" questions
    },
    "chunking": {"max_chars": 1800, "overlap": 200},
    "train": {
        "epochs": 3,
        "lr": 2e-4,
        "batch_size": 4,
        "max_len": 512,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "seed": 42,
    },
    "eval_settings": {"max_new_tokens": 256},
    "patches": [],
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Project:
    root: Path
    cfg: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.cfg["name"]

    @property
    def goal(self) -> str:
        return self.cfg["goal"].strip()

    @property
    def run_dir(self) -> Path:
        d = self.root / "runs" / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def path(self, key: str) -> Path:
        return (self.root / self.cfg[key]).resolve()

    # Files every step reads or writes, in one place.
    @property
    def chunks_file(self) -> Path:
        return self.run_dir / "chunks.jsonl"

    @property
    def synthetic_file(self) -> Path:
        return self.run_dir / "synthetic.jsonl"

    @property
    def adapter_dir(self) -> Path:
        return self.run_dir / "adapter"

    @property
    def model_dir(self) -> Path:
        return self.run_dir / "model"

    @property
    def reports_dir(self) -> Path:
        d = self.run_dir / "reports"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def system_prompt(self) -> str:
        lines = [self.goal]
        if self.cfg["style"]:
            lines.append("Rules:")
            lines += [f"- {s}" for s in self.cfg["style"]]
        return "\n".join(lines)


def load_project(path: str | Path) -> Project:
    """`path` may be a goal.yaml file or the folder that contains it."""
    p = Path(path)
    if p.is_dir():
        p = p / "goal.yaml"
    if not p.exists():
        raise FileNotFoundError(f"No goal.yaml found at {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    for required in ("name", "goal"):
        if required not in raw:
            raise ValueError(f"goal.yaml is missing '{required}'")
    return Project(root=p.parent.resolve(), cfg=_merge(DEFAULTS, raw))
