"""A tiny neural network stored as plain JSON, so an AI can write it by hand.

Format (network.json):
{
  "task": "catch",
  "notes": "how the network works, in plain words",
  "layers": [
    {"W": [[...], [...]], "b": [...], "act": "relu"},   # W is [outputs][inputs]
    {"W": [[...]],        "b": [...], "act": "none"}
  ]
}
act can be: relu, none, sigmoid, tanh, step
No training is involved. Only numpy is needed to run it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ACTS = {
    "relu": lambda x: np.maximum(x, 0),
    "none": lambda x: x,
    "linear": lambda x: x,
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "tanh": np.tanh,
    "step": lambda x: (x > 0).astype(float),
}


class Network:
    def __init__(self, spec: dict):
        self.spec = spec
        self.layers = []
        for i, L in enumerate(spec["layers"]):
            W = np.asarray(L["W"], dtype=float)
            b = np.asarray(L.get("b", np.zeros(W.shape[0])), dtype=float)
            act = L.get("act", "relu")
            if W.ndim != 2 or b.shape != (W.shape[0],):
                raise ValueError(f"layer {i}: W must be [out][in] and b must have length out")
            if act not in ACTS:
                raise ValueError(f"layer {i}: unknown activation '{act}'")
            if self.layers and self.layers[-1][0].shape[0] != W.shape[1]:
                raise ValueError(f"layer {i}: expects {W.shape[1]} inputs but previous layer gives "
                                 f"{self.layers[-1][0].shape[0]}")
            self.layers.append((W, b, act))

    @classmethod
    def load(cls, path: str | Path) -> "Network":
        return cls(json.loads(Path(path).read_text()))

    @property
    def n_in(self) -> int:
        return self.layers[0][0].shape[1]

    @property
    def n_out(self) -> int:
        return self.layers[-1][0].shape[0]

    @property
    def n_weights(self) -> int:
        return sum(W.size + b.size for W, b, _ in self.layers)

    def forward(self, x) -> np.ndarray:
        h = np.asarray(x, dtype=float)
        for W, b, act in self.layers:
            h = ACTS[act](h @ W.T + b)
        return h

    def act(self, x) -> int:
        return int(np.argmax(self.forward(x)))
