"""Talk to your finished AI in the terminal:  myai chat <project>   (or python -m myai.chat <model folder>)"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .export import QWEN_SYSTEM


def resolve(path: str | Path) -> Path:
    """Accepts a model folder or a project folder (then uses its exported model)."""
    p = Path(path)
    if (p / "goal.yaml").exists():
        from .config import load_project

        proj = load_project(p)
        if not (proj.model_dir / "myai.json").exists():
            raise FileNotFoundError(f"This project has no exported model yet. Run: myai export {p}")
        return proj.model_dir
    if not (p / "config.json").exists():
        raise FileNotFoundError(f"{p} is not a model folder or a project folder")
    return p


def chat(path: str | Path, questions: list[str] | None = None) -> None:
    from .models import load_model
    from .patch.tools import generate

    from . import generated as G

    if G.is_generated(path):  # made by `myai generate`: base model + adapter, no export needed
        info = G.info(path)
        model, tok, _, system = G.load(path)
        system = system or QWEN_SYSTEM
    else:
        path = resolve(path)
        info = json.loads((path / "myai.json").read_text()) if (path / "myai.json").exists() else {}
        system = info.get("system_prompt", "You are a helpful assistant.")
        model, tok, _ = load_model(path)
    if questions:
        for q in questions:
            print(f"you > {q}\nai  > {generate(model, tok, q, max_new_tokens=300, system=system).strip()}\n")
        return
    print(f"Chatting with {info.get('name', Path(path).name)}. Ctrl+C to quit.\n")
    while True:
        try:
            q = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q:
            print("ai  >", generate(model, tok, q, max_new_tokens=300, system=system).strip(), "\n")


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python -m myai.chat <model folder or project folder>")
    chat(sys.argv[1])


if __name__ == "__main__":
    main()
