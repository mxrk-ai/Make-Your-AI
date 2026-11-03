"""`myai doctor`: checks this computer is ready, and says how to fix what isn't."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

OK, WARN, BAD = "✅", "⚠️ ", "❌"


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def checks() -> list[tuple[str, str, str]]:
    """(mark, what, how to fix) for everything the pipeline needs."""
    out = []
    v = sys.version_info
    out.append((OK, f"Python {v.major}.{v.minor}", "") if v >= (3, 10) else
               (BAD, f"Python {v.major}.{v.minor} is too old", "Install Python 3.10 or newer"))

    missing = [m for m in ("torch", "transformers", "peft", "safetensors") if not _has(m)]
    if missing:
        out.append((WARN, f"Language-AI libraries not installed: {', '.join(missing)}",
                    "Needed to train language AIs: run ./install.sh (design-first mode works without them)"))
    else:
        import torch

        out.append((OK, f"PyTorch {torch.__version__}", ""))
        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            out.append((OK, f"NVIDIA GPU: {torch.cuda.get_device_name(0)} ({gb:.0f} GB). Training takes minutes", ""))
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            out.append((OK, "Apple Silicon GPU. Training takes about a minute for the example", ""))
        else:
            out.append((WARN, "No GPU found, so training runs on the CPU and is slow (expect 30+ minutes)",
                        "It still works. Or train on a free Google Colab GPU."))

    free = shutil.disk_usage(Path.home()).free / 1e9
    out.append((OK, f"{free:.0f} GB free disk", "") if free > 5 else
               (WARN, f"Only {free:.1f} GB free disk", "A model needs about 1 GB to download and 2 GB to export"))

    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    if (cache / "models--Qwen--Qwen2.5-0.5B-Instruct").exists():
        out.append((OK, "Default base model (Qwen2.5-0.5B) already downloaded", ""))
    else:
        out.append((WARN, "Default base model not downloaded yet",
                    "Nothing to do: it downloads (about 1 GB) the first time you run eval or train"))

    keys = [k for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if os.environ.get(k)]
    out.append((OK, f"API key found: {', '.join(keys)}", "") if keys else
               (OK, "No API key, which is fine: your coding agent writes the practice questions", ""))
    return out


def run() -> bool:
    print("Checking your computer...\n")
    ok = True
    for mark, what, fix in checks():
        print(f"  {mark} {what}")
        if fix:
            print(f"       -> {fix}")
        ok &= mark != BAD
    print("\nAll set. Try: myai status examples/support-bot" if ok else "\nFix the ❌ items above, then run myai doctor again.")
    return ok
