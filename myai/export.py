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
