"""Train the generator on a zoo, and use it: description -> adapter.

A generator is a folder:
  codec.safetensors      the autoencoder (adapter <-> latent)
  denoiser.safetensors   the diffusion model
  lora_A.safetensors     the shared frozen A matrices of the zoo
  generator.json         config, base model, adapter spec, what it was trained on
  index.safetensors      latents + description vectors of the training adapters (for retrieval)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HF_REPO = "make-your-ai-generator"  # under the publisher's Hugging Face account, see generator.json / README


def load_zoo(zoo_dir: str | Path, split: str | None = None, min_score: float | None = None) -> list[tuple[dict, dict]]:
    from ..zoo.trainer import load_adapter

    out = []
    for f in sorted((Path(zoo_dir) / "adapters").glob("*.safetensors")):
        w, meta = load_adapter(f)
        if split and meta.get("split") != split:
            continue
        if min_score is not None and meta.get("score", 0) < min_score:
            continue
        out.append((w, meta))
    return out


def train_generator(zoo_dir: str | Path, out_dir: str | Path, latent_dim: int = 64, steps: int = 4000,
                    min_score: float = 0.5, d: int = 256, layers: int = 4, seed: int = 0, device: str | None = None,
                    extra: list[tuple[dict, dict]] | None = None, log=print) -> dict:
    """Fits the autoencoder and trains the diffusion model on the zoo's *seen* adapters only."""
    import shutil

    import torch
    from safetensors.torch import save_file

    from ..models import pick_device
    from ..zoo import behaviors as B
    from . import text
    from .codec import Codec
    from .diffusion import Denoiser, train_denoiser

    zoo_dir, out = Path(zoo_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = load_zoo(zoo_dir, split="seen", min_score=min_score) + list(extra or [])
    if len(data) < 8:
        raise ValueError(f"Only {len(data)} usable adapters in {zoo_dir} (need a score of {min_score}+). Train the zoo first.")
    weights, metas = [w for w, _ in data], [m for _, m in data]
    codec = Codec.fit(weights, latent_dim)
    latents = torch.stack([codec.encode(w) for w in weights])
    log(f"adapters={len(data)}  tokens={codec.n_tokens}  latent={codec.dim}  latent std={latents.std():.3f}")

    rng = random.Random(seed)

    def cond_fn(idx):
        descs = []
        for i in idx:
            m = metas[i]
            # A fresh phrasing each time when the task is built from known behaviours; else the stored text.
            descs.append(B.describe(m["task"], rng) if m.get("task") else m["description"])
        return text.condition(descs)

    model = Denoiser(codec.n_tokens, codec.dim, d=d, layers=layers)
    device = device or pick_device()
    losses = train_denoiser(model, latents, cond_fn, steps=steps, device=device, seed=seed, log=log)
    model.cpu()

    codec.save(out / "codec.safetensors")
    save_file({k: v.half() for k, v in model.state_dict().items()}, str(out / "denoiser.safetensors"))
    shutil.copy2(zoo_dir / "lora_A.safetensors", out / "lora_A.safetensors")
    save_file({"latents": latents.half(), "desc": text.embed_whole([m["description"] for m in metas]).half()},
              str(out / "index.safetensors"))
    info = {"base_model": metas[0]["base_model"], "spec": metas[0]["spec"], "denoiser": model.cfg,
            "text_encoder": text.ENCODER, "trained_on": len(data), "steps": steps, "seed": seed,
            "final_loss": round(sum(losses[-100:]) / len(losses[-100:]), 5),
            "index": [{"adapter": m["adapter"], "description": m["description"]} for m in metas]}
    (out / "generator.json").write_text(json.dumps(info, indent=1))
    return info


class Generator:
    def __init__(self, folder: str | Path):
        import torch
        from safetensors.torch import load_file

        from .codec import Codec
        from .diffusion import Denoiser

        self.folder = Path(folder)
        self.info = json.loads((self.folder / "generator.json").read_text())
        self.codec = Codec.load(self.folder / "codec.safetensors")
        self.model = Denoiser(**self.info["denoiser"])
        self.model.load_state_dict({k: v.float() for k, v in load_file(str(self.folder / "denoiser.safetensors")).items()})
        self.model.eval()
        self.lora_a = load_file(str(self.folder / "lora_A.safetensors"))
        idx = load_file(str(self.folder / "index.safetensors"))
        self.index_latents, self.index_desc = idx["latents"].float(), idx["desc"].float()
        torch.set_grad_enabled(False)

    def generate(self, description: str, seed: int = 0, guidance: float = 2.0, steps: int = 50) -> dict:
        """description -> B matrices, by denoising from pure noise."""
        from . import text
        from .diffusion import sample

        cond, mask = text.condition([description])
        return self.codec.decode(sample(self.model, cond, mask, steps=steps, guidance=guidance, seed=seed)[0])

    def harmonize(self, weights: dict, description: str, strength: float = 0.3, seed: int = 0,
                  guidance: float = 2.0, steps: int = 30) -> dict:
        """Partly re-noise an (edited) adapter and denoise it again, pulling it back towards realistic adapters.
        strength 0 = unchanged (up to autoencoder loss), 1 = generate from scratch."""
        from . import text
        from .diffusion import sample

        z = self.codec.encode(weights).unsqueeze(0)
        if strength <= 0:
            return self.codec.decode(z[0])
        cond, mask = text.condition([description])
        return self.codec.decode(sample(self.model, cond, mask, steps=steps, guidance=guidance, seed=seed,
                                        start=z, start_t=strength)[0])

    def retrieve(self, description: str) -> tuple[dict, dict]:
        """Nearest training adapter by description (the memorisation baseline). Returns (weights, index entry)."""
        from . import text

        sims = self.index_desc @ text.embed_whole([description])[0]
        i = int(sims.argmax())
        return self.codec.decode(self.index_latents[i]), {**self.info["index"][i], "similarity": round(float(sims[i]), 3)}


def write_adapter(folder: str | Path, weights: dict, lora_a: dict, base_model: str, spec: dict) -> Path:
    """Writes a standard PEFT LoRA folder, loadable with PeftModel.from_pretrained or `myai chat`."""
    from peft import LoraConfig
    from safetensors.torch import save_file

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cfg = LoraConfig(r=spec["r"], lora_alpha=spec["alpha"], lora_dropout=0.0,
                     target_modules=list(spec["target_modules"]), task_type="CAUSAL_LM",
                     base_model_name_or_path=base_model)
    cfg.save_pretrained(str(folder))
    tensors = {}
    for key, b in weights.items():
        layer, module = key.split(".")
        prefix = f"base_model.model.model.layers.{layer}.self_attn.{module}"
        tensors[f"{prefix}.lora_A.weight"] = lora_a[key].float().contiguous()
        tensors[f"{prefix}.lora_B.weight"] = b.float().contiguous()
    save_file(tensors, str(folder / "adapter_model.safetensors"))
    return folder


def find_generator(path: str | None = None) -> Path:
    """A local folder if given or present, else download the published generator once."""
    import os

    for cand in (path, os.environ.get("MYAI_GENERATOR"), Path.home() / ".cache" / "myai" / "generator"):
        if cand and (Path(cand) / "generator.json").exists():
            return Path(cand)
    repo = os.environ.get("MYAI_GENERATOR_REPO")
    if not repo:
        raise FileNotFoundError("No generator found. Pass --generator <folder>, or set MYAI_GENERATOR_REPO "
                                "to a Hugging Face repo that holds one.")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, local_dir=str(Path.home() / ".cache" / "myai" / "generator")))
