"""Weight patches: small, shareable files that change specific weights of a model.

Think of them like game mods. A patch is made once, by an AI agent (or a person)
editing weights directly, and then anyone with the same base model can apply it
in a second. Patches can be stacked and removed again.

A patch is one .safetensors file. For each weight it changes it stores one of:
  * sparse  - which positions change and by how much (idx, val)
  * lowrank - the change as two thin matrices, delta = u @ v.T   (u, v)
  * dense   - the full change (fallback for big edits)

plus a checksum of the original weight, so applying a patch to the wrong model
fails loudly instead of silently breaking it.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

SPARSE_MAX_FRACTION = 0.10  # above this share of changed values, store as dense


def checksum(t: torch.Tensor) -> list[float]:
    t = t.detach().to("cpu", torch.float64)
    flat = t.flatten()
    pos = torch.arange(1, flat.numel() + 1, dtype=torch.float64) / flat.numel()
    return [float(flat.sum()), float(flat.abs().sum()), float((flat * pos).sum())]


def checksums_match(a: list[float], b: list[float], rtol: float = 2e-3) -> bool:
    return all(abs(x - y) <= rtol * max(1.0, abs(x), abs(y)) for x, y in zip(a, b))


def _as_lowrank(delta: torch.Tensor, max_rank: int = 16, rtol: float = 1e-4):
    """If a big change is secretly low-rank (most smart edits are), store it as u @ v.T."""
    if delta.dim() != 2 or min(delta.shape) <= 2 * max_rank:
        return None
    U, S, Vh = torch.linalg.svd(delta.double(), full_matrices=False)
    total = float(S.square().sum())
    for k in range(1, max_rank + 1):
        if float(S[k:].square().sum()) <= (rtol ** 2) * total:
            return (U[:, :k] * S[:k]).float(), Vh[:k].T.float()
    return None


@dataclass
class Patch:
    name: str
    description: str = ""
    base_model: str = ""
    author: str = ""
    created: str = field(default_factory=lambda: _dt.date.today().isoformat())
    evals: dict = field(default_factory=dict)  # e.g. {"before": {...}, "after": {...}}
    entries: dict = field(default_factory=dict)  # param name -> entry dict

    # ---------- building ----------

    def add_sparse(self, param: str, original: torch.Tensor, idx: torch.Tensor, val: torch.Tensor):
        self.entries[param] = {"type": "sparse", "shape": list(original.shape), "checksum": checksum(original),
                               "idx": idx.to(torch.int64).cpu(), "val": val.to(torch.float32).cpu()}

    def add_lowrank(self, param: str, original: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
        """delta = u @ v.T ; u: [out, k], v: [in, k]"""
        if u.dim() == 1:
            u, v = u[:, None], v[:, None]
        if param in self.entries and self.entries[param]["type"] == "lowrank":
            e = self.entries[param]  # extend rank instead of overwriting
            e["u"] = torch.cat([e["u"], u.float().cpu()], 1)
            e["v"] = torch.cat([e["v"], v.float().cpu()], 1)
            return
        self.entries[param] = {"type": "lowrank", "shape": list(original.shape), "checksum": checksum(original),
                               "u": u.to(torch.float32).cpu(), "v": v.to(torch.float32).cpu()}

    def add_dense(self, param: str, original: torch.Tensor, delta: torch.Tensor):
        self.entries[param] = {"type": "dense", "shape": list(original.shape), "checksum": checksum(original),
                               "delta": delta.to(torch.float32).cpu()}

    @classmethod
    def from_diff(cls, name: str, before: dict[str, torch.Tensor], after: dict[str, torch.Tensor],
                  atol: float = 0.0, **meta) -> "Patch":
        """Compare two state dicts and keep only what changed."""
        p = cls(name=name, **meta)
        for k, b in before.items():
            a = after[k]
            delta = (a.detach().float().cpu() - b.detach().float().cpu())
            changed = delta.abs() > atol
            n = int(changed.sum())
            if n == 0:
                continue
            if n / delta.numel() <= SPARSE_MAX_FRACTION:
                idx = changed.flatten().nonzero().squeeze(1)
                p.add_sparse(k, b, idx, delta.flatten()[idx])
            elif (uv := _as_lowrank(delta)) is not None:
                p.add_lowrank(k, b, *uv)
            else:
                p.add_dense(k, b, delta)
        return p

    # ---------- info ----------

    def changed_values(self) -> int:
        n = 0
        for e in self.entries.values():
            if e["type"] == "sparse":
                n += e["idx"].numel()
            else:
                rows, cols = e["shape"][0], (e["shape"][1] if len(e["shape"]) > 1 else 1)
                n += rows * cols
        return n

    def stored_values(self) -> int:
        n = 0
        for e in self.entries.values():
            n += sum(e[k].numel() for k in ("idx", "val", "u", "v", "delta") if k in e)
        return n

    def summary(self, model=None) -> str:
        lines = [f"patch: {self.name}", f"  {self.description}" if self.description else "",
                 f"  base model: {self.base_model or '?'}", f"  weights touched: {len(self.entries)} tensors"]
        if model is not None:
            total = sum(p.numel() for p in model.parameters())
            lines.append(f"  values changed: {self.changed_values():,} ({100*self.changed_values()/total:.4f}% of model)")
        else:
            lines.append(f"  values changed: {self.changed_values():,}")
        lines.append(f"  stored numbers: {self.stored_values():,}")
        for k, e in self.entries.items():
            extra = f"{e['idx'].numel():,} values" if e["type"] == "sparse" else (
                f"rank {e['u'].shape[1]}" if e["type"] == "lowrank" else "full")
            lines.append(f"    - {k}  [{e['type']}, {extra}]")
        return "\n".join(l for l in lines if l)

    # ---------- save / load ----------

    def save(self, path: str | Path) -> Path:
        from safetensors.torch import save_file

        path = Path(path)
        if path.suffix != ".safetensors":
            path = path.with_suffix(".safetensors")
        tensors, entry_meta = {}, {}
        for k, e in self.entries.items():
            entry_meta[k] = {"type": e["type"], "shape": e["shape"], "checksum": e["checksum"]}
            for field_ in ("idx", "val", "u", "v", "delta"):
                if field_ in e:
                    tensors[f"{k}::{field_}"] = e[field_].contiguous()
        meta = {"name": self.name, "description": self.description, "base_model": self.base_model,
                "author": self.author, "created": self.created, "evals": self.evals,
                "entries": entry_meta, "format": "myai-patch/1"}
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(path), metadata={"myai_patch": json.dumps(meta)})
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Patch":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            meta = json.loads(f.metadata()["myai_patch"])
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        p = cls(name=meta["name"], description=meta.get("description", ""), base_model=meta.get("base_model", ""),
                author=meta.get("author", ""), created=meta.get("created", ""), evals=meta.get("evals", {}))
        for k, em in meta["entries"].items():
            e = dict(em)
            for field_ in ("idx", "val", "u", "v", "delta"):
                if f"{k}::{field_}" in tensors:
                    e[field_] = tensors[f"{k}::{field_}"]
            p.entries[k] = e
        return p


# ---------- applying ----------

def _delta_for(e: dict, like: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if e["type"] == "sparse":
        return e["idx"].to(like.device), e["val"].to(like.device, like.dtype), None
    if e["type"] == "lowrank":
        return None, None, (e["u"] @ e["v"].T).to(like.device, like.dtype)
    return None, None, e["delta"].to(like.device, like.dtype)


def apply_patch(model, patch: Patch, scale: float = 1.0, check: bool = True) -> None:
    """Adds the patch's changes to the model's weights, in place. scale=-1 removes it."""
    params = dict(model.named_parameters())
    applied = getattr(model, "_myai_applied", {})  # param -> list of patch names already on it
    # Check everything first, so a bad patch changes nothing.
    for k, e in patch.entries.items():
        if k not in params:
            raise KeyError(f"Patch '{patch.name}' changes '{k}', which this model doesn't have. "
                           f"Made for: {patch.base_model or 'unknown model'}")
        if list(params[k].shape) != e["shape"]:
            raise ValueError(f"Shape mismatch on '{k}': model {list(params[k].shape)} vs patch {e['shape']}")
        if check and scale > 0 and not applied.get(k) and not checksums_match(checksum(params[k]), e["checksum"]):
            raise ValueError(f"Patch '{patch.name}' was made for different weights on '{k}'. "
                             f"Is this the right base model ({patch.base_model})? Use check=False to force.")
    with torch.no_grad():
        for k, e in patch.entries.items():
            w = params[k]
            idx, val, dense = _delta_for(e, w)
            if idx is not None:
                w.view(-1).index_add_(0, idx, val * scale)
            else:
                w.add_(dense * scale)
            if scale > 0:
                applied.setdefault(k, []).append(patch.name)
            elif patch.name in applied.get(k, []):
                applied[k].remove(patch.name)
    model._myai_applied = applied


def remove_patch(model, patch: Patch) -> None:
    apply_patch(model, patch, scale=-1.0, check=False)


def overlap_report(patches: list[Patch]) -> list[str]:
    """Warns where two patches change the same weights (they may fight each other)."""
    warnings = []
    for i, a in enumerate(patches):
        for b in patches[i + 1 :]:
            for k in set(a.entries) & set(b.entries):
                ea, eb = a.entries[k], b.entries[k]
                if ea["type"] == eb["type"] == "sparse":
                    shared = len(set(ea["idx"].tolist()) & set(eb["idx"].tolist()))
                    if shared:
                        warnings.append(f"'{a.name}' and '{b.name}' both change {shared:,} values in {k}")
                else:
                    warnings.append(f"'{a.name}' and '{b.name}' both change {k} (whole tensor)")
    return warnings
