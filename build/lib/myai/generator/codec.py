"""The autoencoder: adapter weights <-> a small latent, one token per targeted module.

It is a linear autoencoder fitted in closed form (PCA per module). With a few hundred adapters
there is nothing a deeper autoencoder could learn that this does not already capture, and it
cannot overfit in a hidden way: what it loses is measured by `reconstruction check` in the report.
"""

from __future__ import annotations

import json
from pathlib import Path


def token_keys(weights: dict) -> list[str]:
    """Fixed token order: by block number, then module name ('0.q_proj', '0.v_proj', '1.q_proj', ...)."""
    return sorted(weights, key=lambda k: (int(k.split(".")[0]), k.split(".")[1]))


class Codec:
    def __init__(self, keys, shapes, mean, basis, scale):
        self.keys, self.shapes, self.mean, self.basis, self.scale = keys, shapes, mean, basis, scale

    @property
    def n_tokens(self) -> int:
        return len(self.keys)

    @property
    def dim(self) -> int:
        return next(iter(self.basis.values())).shape[0]

    @classmethod
    def fit(cls, adapters: list[dict], dim: int = 64) -> "Codec":
        import torch

        keys = token_keys(adapters[0])
        dim = min(dim, len(adapters) - 1)
        shapes, mean, basis, scale = {}, {}, {}, {}
        for k in keys:
            x = torch.stack([a[k].flatten() for a in adapters]).double()
            shapes[k] = list(adapters[0][k].shape)
            mean[k] = x.mean(0)
            _, _, vt = torch.linalg.svd(x - mean[k], full_matrices=False)
            basis[k] = vt[:dim]
            scale[k] = ((x - mean[k]) @ basis[k].T).std().clamp_min(1e-8)
        f = lambda d: {k: v.float() for k, v in d.items()}
        return cls(keys, shapes, f(mean), f(basis), f(scale))

    def encode(self, weights: dict):
        import torch

        return torch.stack([((weights[k].flatten().float() - self.mean[k]) @ self.basis[k].T) / self.scale[k]
                            for k in self.keys])

    def decode(self, z) -> dict:
        z = z.detach().float().cpu()
        return {k: (z[i] * self.scale[k] @ self.basis[k] + self.mean[k]).reshape(self.shapes[k])
                for i, k in enumerate(self.keys)}

    def save(self, path: str | Path) -> None:
        from safetensors.torch import save_file

        t = {}
        for k in self.keys:
            t[f"mean.{k}"], t[f"basis.{k}"], t[f"scale.{k}"] = self.mean[k], self.basis[k].half(), self.scale[k].reshape(1)
        save_file(t, str(path), metadata={"keys": json.dumps(self.keys), "shapes": json.dumps(self.shapes)})

    @classmethod
    def load(cls, path: str | Path) -> "Codec":
        from safetensors import safe_open

        with safe_open(str(path), "pt") as f:
            keys, shapes = json.loads(f.metadata()["keys"]), json.loads(f.metadata()["shapes"])
            return cls(keys, shapes, {k: f.get_tensor(f"mean.{k}").float() for k in keys},
                       {k: f.get_tensor(f"basis.{k}").float() for k in keys},
                       {k: f.get_tensor(f"scale.{k}").float()[0] for k in keys})
