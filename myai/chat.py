"""Talk to your finished AI in the terminal:  python -m myai.chat runs/<name>/model"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python -m myai.chat <model folder>")
    from .patch.tools import generate
    from .models import load_model

    path = Path(sys.argv[1])
    info = json.loads((path / "myai.json").read_text()) if (path / "myai.json").exists() else {}
    system = info.get("system_prompt", "You are a helpful assistant.")
    model, tok, _ = load_model(path)
    print(f"Chatting with {info.get('name', path.name)}. Ctrl+C to quit.\n")
    while True:
        try:
            q = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q:
            print("ai  >", generate(model, tok, q, max_new_tokens=300, system=system).strip(), "\n")


if __name__ == "__main__":
    main()
