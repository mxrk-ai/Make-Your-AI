"""Step 6: bake everything into one model folder you can run anywhere."""

from __future__ import annotations

import json
from pathlib import Path


def export(project, out_dir: Path | None = None, extra_patches: list[str] | None = None) -> Path:
    import torch

    from .models import load_model
    from .patch import Patch, apply_patch, overlap_report

    out_dir = Path(out_dir or project.model_dir)
    model, tok, _ = load_model(project.cfg["base_model"],
                               adapter=project.adapter_dir if project.adapter_dir.exists() else None,
                               device="cpu", dtype=torch.float32)
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()  # fold the LoRA add-on into the weights

    patch_files = list(project.cfg.get("patches") or []) + list(extra_patches or [])
    patches = [Patch.load(project.root / p if not Path(p).is_absolute() else p) for p in patch_files]
    for w in overlap_report(patches):
        print(f"  warning: {w}")
    for p in patches:
        apply_patch(model, p)
        print(f"  applied patch: {p.name}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tok.save_pretrained(str(out_dir))
    (out_dir / "myai.json").write_text(json.dumps({
        "name": project.name,
        "base_model": project.cfg["base_model"],
        "lora": project.adapter_dir.exists(),
        "patches": [p.name for p in patches],
        "system_prompt": project.system_prompt(),
    }, indent=2))
    write_modelfile(project, out_dir)
    return out_dir


def write_modelfile(project, out_dir: Path) -> None:
    """Ollama recipe. Convert to GGUF first (see README), then `ollama create`."""
    sys_prompt = project.system_prompt().replace('"""', "'''")
    (out_dir / "Modelfile").write_text(
        f"FROM ./{project.name}.gguf\n"
        f'SYSTEM """{sys_prompt}"""\n'
        "PARAMETER temperature 0.3\n"
        "PARAMETER repeat_penalty 1.1\n"
    )


# ---------- generated AIs and GGUF ----------

CHATML_TEMPLATE = '''{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
'''
QWEN_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
LLAMA_CPP_TAG = "b4585"  # a fixed llama.cpp release, so conversion is reproducible


def export_generated(folder, out_dir=None, gguf: bool = False, ollama: bool = False) -> Path:
    """Folds a generated adapter into the base model: one normal model folder, optionally a GGUF file."""
    import torch

    from . import generated as G
    from .models import load_model

    folder = Path(folder)
    meta = G.info(folder)
    out_dir = Path(out_dir or folder / "model")
    model, tok, _ = load_model(meta["base_model"], adapter=folder / "adapter", device="cpu", dtype=torch.float32)
    model = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tok.save_pretrained(str(out_dir))
    system = meta.get("system_prompt") or QWEN_SYSTEM  # the adapter was trained under the base model's default prompt
    (out_dir / "myai.json").write_text(json.dumps({"name": meta["name"], "base_model": meta["base_model"],
                                                   "description": meta["description"], "system_prompt": system}, indent=2))
    (out_dir / "Modelfile").write_text(
        f"FROM ./{meta['name']}.gguf\n" f'TEMPLATE """{CHATML_TEMPLATE}"""\n'
        f'SYSTEM """{system.replace(chr(34) * 3, chr(39) * 3)}"""\n'
        'PARAMETER stop "<|im_end|>"\nPARAMETER temperature 0.3\n')
    if gguf:
        to_gguf(out_dir, meta["name"], ollama=ollama)
    return out_dir


def _llama_cpp() -> Path:
    """Finds llama.cpp's converter, fetching the pinned release once if needed."""
    import os
    import subprocess

    for cand in (os.environ.get("LLAMA_CPP_DIR"), Path.home() / ".cache" / "myai" / "llama.cpp"):
        if cand and (Path(cand) / "convert_hf_to_gguf.py").exists():
            return Path(cand)
    dest = Path.home() / ".cache" / "myai" / "llama.cpp"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching the GGUF converter (llama.cpp {LLAMA_CPP_TAG}) once -> {dest}")
    subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--branch", LLAMA_CPP_TAG,
                    "https://github.com/ggerganov/llama.cpp", str(dest)], check=True)
    return dest


def to_gguf(model_dir: Path, name: str, ollama: bool = False) -> Path:
    """model folder -> <name>.gguf next to it (16-bit floats, about 1 GB for a 0.5B model)."""
    import importlib.util
    import shutil
    import subprocess
    import sys

    model_dir = Path(model_dir)
    if importlib.util.find_spec("gguf") is None:
        raise ModuleNotFoundError("gguf", name="gguf")
    lc = _llama_cpp()
    out = model_dir / f"{name}.gguf"
    env = {**__import__("os").environ, "PYTHONPATH": str(lc / "gguf-py")}
    r = subprocess.run([sys.executable, str(lc / "convert_hf_to_gguf.py"), str(model_dir), "--outfile", str(out),
                        "--outtype", "f16"], capture_output=True, text=True, env=env)
    if r.returncode != 0 or not out.exists():
        raise ValueError("GGUF conversion failed:\n" + (r.stderr or r.stdout)[-1500:])
    print(f"  GGUF file: {out}  ({out.stat().st_size / 1e6:.0f} MB)\n  LM Studio: import that file.")
    if ollama:
        if not shutil.which("ollama"):
            raise ValueError("Ollama is not installed. Get it from https://ollama.com, then run this again.")
        subprocess.run(["ollama", "create", name, "-f", "Modelfile"], cwd=str(model_dir), check=True)
        print(f"  Ollama:    ollama run {name}")
    else:
        print(f"  Ollama:    cd {model_dir} && ollama create {name} -f Modelfile && ollama run {name}")
    return out
