"""Tools an AI agent uses to look inside a model and edit its weights.

This is the "weight-space programming" workbench. Typical session:

    from myai.patch import tools as T
    model, tok, dev = T.load("Qwen/Qwen2.5-0.5B-Instruct")
    print(T.overview(model))                       # what's in there
    bad  = T.capture(model, tok, bad_texts)        # activations while doing the bad habit
    good = T.capture(model, tok, good_texts)       # activations while behaving well
    sep  = T.separation(bad, good)                 # which layers tell them apart
    rec  = T.Recorder(model, "no-rambling", base_model="Qwen/Qwen2.5-0.5B-Instruct")
    for layer in top_layers:
        rec.ablate_direction(layer, T.direction(bad, good)[layer])
    # ... test it, keep or undo ...
    rec.to_patch(description="...").save("patches/no-rambling.safetensors")

Everything here is plain PyTorch, so the agent can also write its own code.
"""

from __future__ import annotations

import re

import torch

from ..models import load_model  # re-export
from .format import Patch, checksum

__all__ = ["load", "overview", "layers", "capture", "direction", "separation", "Recorder", "generate"]

load = load_model


def layers(model) -> list:
    """The list of transformer blocks (works for Qwen, Llama, Gemma, Mistral, GPT-2 style models)."""
    m = getattr(model, "base_model", model)
    m = getattr(m, "model", m)
    for path in ("layers", "model.layers", "h", "transformer.h", "decoder.layers"):
        obj = m
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return list(obj)
        except AttributeError:
            continue
    raise AttributeError("Couldn't find the transformer blocks of this model")


def overview(model, max_rows: int = 40) -> str:
    """A readable map of the model: blocks, weight names, shapes."""
    total = sum(p.numel() for p in model.parameters())
    lines = [f"{type(model).__name__}: {total:,} weights, {len(layers(model))} blocks"]
    shown = 0
    for name, p in model.named_parameters():
        m = re.search(r"layers\.(\d+)\.", name)
        if m and int(m.group(1)) > 0:
            continue  # only print block 0 as an example of every block
        lines.append(f"  {name:70s} {tuple(p.shape)}")
        shown += 1
        if shown >= max_rows:
            lines.append("  ...")
            break
    lines.append("(every block has the same weights as block 0)")
    return "\n".join(lines)


def param_name(model, layer: int, contains: str) -> str:
    """Full name of a weight in a block, e.g. param_name(m, 15, 'o_proj') -> 'model.layers.15.self_attn.o_proj.weight'."""
    for name, _ in model.named_parameters():
        if f"layers.{layer}." in name and contains in name and name.endswith("weight"):
            return name
    raise KeyError(f"No weight containing '{contains}' in block {layer}")


@torch.no_grad()
def capture(model, tok, texts: list[str], positions: str = "mean") -> torch.Tensor:
    """What each block writes into the model's 'stream of thought' for these texts.

    Returns a tensor [n_texts, n_layers, hidden]. positions: "mean" over all tokens or "last" token.
    """
    blocks = layers(model)
    device = next(model.parameters()).device
    out = torch.zeros(len(texts), len(blocks), model.config.hidden_size)
    store: dict[int, torch.Tensor] = {}

    def hook(i):
        def fn(_mod, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            store[i] = h.detach()
        return fn

    handles = [b.register_forward_hook(hook(i)) for i, b in enumerate(blocks)]
    try:
        for t_i, text in enumerate(texts):
            enc = tok(text, return_tensors="pt").to(device)
            model(**enc)
            for i in range(len(blocks)):
                h = store[i][0].float()
                out[t_i, i] = h.mean(0).cpu() if positions == "mean" else h[-1].cpu()
    finally:
        for h in handles:
            h.remove()
    return out


def direction(acts_a: torch.Tensor, acts_b: torch.Tensor) -> torch.Tensor:
    """Per-layer unit vector pointing from behaviour B towards behaviour A. Shape [n_layers, hidden]."""
    d = acts_a.mean(0) - acts_b.mean(0)
    return d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def separation(acts_a: torch.Tensor, acts_b: torch.Tensor) -> list[tuple[int, float]]:
    """How clearly each layer tells A from B (higher = clearer). Sorted, best first."""
    d = direction(acts_a, acts_b)
    scores = []
    for layer in range(d.shape[0]):
        pa = acts_a[:, layer] @ d[layer]
        pb = acts_b[:, layer] @ d[layer]
        spread = torch.sqrt((pa.var(unbiased=False) + pb.var(unbiased=False)) / 2).clamp_min(1e-6)
        scores.append((layer, float((pa.mean() - pb.mean()) / spread)))
    return sorted(scores, key=lambda x: -x[1])


@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens: int = 200, system: str | None = None) -> str:
    from ..models import chat_text

    device = next(model.parameters()).device
    text = chat_text(tok, system or "You are a helpful assistant.", prompt)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


class Recorder:
    """Makes edits on a live model and remembers them, so they can be undone or saved as a patch."""

    def __init__(self, model, name: str, base_model: str = "", author: str = ""):
        self.model = model
        self.params = dict(model.named_parameters())
        self.name, self.base_model, self.author = name, base_model, author
        self.originals: dict[str, torch.Tensor] = {}
        self.lowrank: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self.free_edits: set[str] = set()
        self.log: list[str] = []

    def _remember(self, param: str):
        if param not in self.originals:
            self.originals[param] = self.params[param].detach().clone().cpu()

    # --- edit primitives ---

    @torch.no_grad()
    def add_lowrank(self, param: str, u: torch.Tensor, v: torch.Tensor, note: str = ""):
        """W += u @ v.T  (u: [out] or [out,k], v: [in] or [in,k])"""
        self._remember(param)
        w = self.params[param]
        if u.dim() == 1:
            u, v = u[:, None], v[:, None]
        w.add_((u @ v.T).to(w.device, w.dtype))
        self.lowrank.setdefault(param, []).append((u.float().cpu(), v.float().cpu()))
        self.log.append(f"lowrank {param} rank={u.shape[1]} {note}")

    @torch.no_grad()
    def set_values(self, param: str, idx: torch.Tensor, values: torch.Tensor, note: str = ""):
        """Set individual weights (flat indices) to new values."""
        self._remember(param)
        w = self.params[param]
        w.view(-1)[idx.to(w.device)] = values.to(w.device, w.dtype)
        self.free_edits.add(param)
        self.log.append(f"set {idx.numel()} values in {param} {note}")

    @torch.no_grad()
    def edit(self, param: str, fn, note: str = ""):
        """Any edit you like: fn(weight_tensor) modifies it in place."""
        self._remember(param)
        fn(self.params[param])
        self.free_edits.add(param)
        self.log.append(f"edit {param} {note}")

    @torch.no_grad()
    def ablate_direction(self, layer: int, d: torch.Tensor, strength: float = 1.0,
                         targets: tuple[str, ...] = ("o_proj", "down_proj")):
        """Stop block `layer` from writing direction d into the stream (W -= s * d dᵀ W).

        A rank-1 edit per matrix, so the patch stays tiny. strength 1.0 removes the
        direction completely, 0.5 halves it, negative values amplify it.
        """
        for t in targets:
            p = param_name(self.model, layer, t)
            w = self.params[p]
            dd = d.to(w.device, torch.float32)
            dd = dd / dd.norm()
            v = (w.float().T @ dd)  # [in]
            self.add_lowrank(p, -strength * dd, v, note=f"ablate layer {layer} s={strength}")

    # --- undo / save ---

    @torch.no_grad()
    def undo_all(self):
        for p, orig in self.originals.items():
            self.params[p].copy_(orig.to(self.params[p].device, self.params[p].dtype))
        self.originals.clear(); self.lowrank.clear(); self.free_edits.clear()
        self.log.append("undo all")

    def to_patch(self, description: str = "", evals: dict | None = None) -> Patch:
        p = Patch(name=self.name, description=description, base_model=self.base_model, author=self.author,
                  evals=evals or {})
        for param, orig in self.originals.items():
            if param in self.free_edits:  # arbitrary edits: store the exact difference
                diff = Patch.from_diff(self.name, {param: orig}, {param: self.params[param].detach().cpu()})
                if param in diff.entries:
                    p.entries[param] = diff.entries[param]
            else:
                u = torch.cat([a for a, _ in self.lowrank[param]], 1)
                v = torch.cat([b for _, b in self.lowrank[param]], 1)
                p.add_lowrank(param, orig, u, v)
                p.entries[param]["checksum"] = checksum(orig)
        return p
