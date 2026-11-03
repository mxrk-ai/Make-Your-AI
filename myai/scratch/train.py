"""Design first, then train.

The AI writes a network by hand from its own assumptions (network.json). Training
then starts from *those* weights instead of random ones, and polishes them on
examples. To show whether the design really helped, the same training runs from a
random start with the same shape, data and steps.

Training keeps the best version seen on a separate check set, starting with the
hand-written one, so it never hands back something worse than the design.
Only numpy is needed.
"""

from __future__ import annotations

import copy

import numpy as np

from .network import ACTS, Network
from .tasks import Table, Task


def _act_grad(act: str, z: np.ndarray, h: np.ndarray) -> np.ndarray:
    if act == "relu":
        return (z > 0).astype(float)
    if act in ("none", "linear"):
        return np.ones_like(z)
    if act == "sigmoid":
        return h * (1 - h)
    if act == "tanh":
        return 1 - h**2
    if act == "step":
        return (np.abs(z) < 1).astype(float)  # step has no slope; pass the gradient through near 0
    raise ValueError(act)


def _forward(layers, X):
    hs, zs = [X], []
    for W, b, act in layers:
        z = hs[-1] @ W.T + b
        zs.append(z)
        hs.append(ACTS[act](z))
    return hs, zs


def _grads(layers, X, y):
    """Cross-entropy loss on the outputs (highest output = answer) and its gradient for every weight."""
    hs, zs = _forward(layers, X)
    out = hs[-1] - hs[-1].max(axis=1, keepdims=True)
    p = np.exp(out)
    p /= p.sum(axis=1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    dh = p
    dh[np.arange(len(y)), y] -= 1
    dh /= len(y)
    grads = []
    for i in reversed(range(len(layers))):
        W, _, act = layers[i]
        dz = dh * _act_grad(act, zs[i], hs[i + 1])
        grads.append((dz.T @ hs[i], dz.sum(axis=0)))
        dh = dz @ W
    return loss, grads[::-1]


def _accuracy(layers, X, y) -> float:
    return float((np.argmax(_forward(layers, X)[0][-1], axis=1) == y).mean())


def _to_network(spec: dict, layers) -> Network:
    spec = copy.deepcopy(spec)
    for L, (W, b, act) in zip(spec["layers"], layers):
        L["W"], L["b"], L["act"] = W.tolist(), b.tolist(), act
    return Network(spec)


def score(task: Task, net: Network) -> float:
    """The task's score. For a CSV table it only counts rows training never saw."""
    if isinstance(task, Table):
        return task.evaluate(net, rows="test")["score"]
    return task.evaluate(net)["score"]


def random_like(net: Network, seed: int = 0) -> Network:
    """Same shape and activations as `net`, but random weights (the usual way training starts)."""
    rng = np.random.default_rng(seed)
    spec = {"task": net.spec.get("task"), "layers": [
        {"W": rng.normal(0, np.sqrt(2 / W.shape[1]), W.shape).tolist(), "b": [0.0] * W.shape[0], "act": act}
        for W, _, act in net.layers]}
    return Network(spec)


def train_network(net: Network, task: Task, steps: int = 2000, lr: float = 0.05, batch: int = 64,
                  examples: int = 2000, seed: int = 1, log_every: int = 50) -> tuple[Network, list[dict]]:
    """Polish `net` with gradient descent (Adam). Returns the best network and a history of scores."""
    task.check_shape(net)
    rng = np.random.default_rng(seed)
    X, y = task.labeled(examples, seed)
    Xc, yc = task.labeled(max(examples // 4, 100), seed + 1000)  # check set, decides which version to keep
    layers = [(W.copy(), b.copy(), act) for W, b, act in net.layers]
    m = [(np.zeros_like(W), np.zeros_like(b)) for W, b, _ in layers]
    v = [(np.zeros_like(W), np.zeros_like(b)) for W, b, _ in layers]
    best, best_acc = [(W.copy(), b.copy(), a) for W, b, a in layers], _accuracy(layers, Xc, yc)
    history = [{"step": 0, "score": score(task, net), "check": best_acc}]
    b1, b2 = 0.9, 0.999
    for t in range(1, steps + 1):
        idx = rng.integers(len(y), size=batch)
        loss, grads = _grads(layers, X[idx], y[idx])
        for i, ((W, b, _), (gW, gb)) in enumerate(zip(layers, grads)):
            for j, (w, g) in enumerate(((W, gW), (b, gb))):
                m[i][j][...] = b1 * m[i][j] + (1 - b1) * g
                v[i][j][...] = b2 * v[i][j] + (1 - b2) * g**2
                w -= lr * (m[i][j] / (1 - b1**t)) / (np.sqrt(v[i][j] / (1 - b2**t)) + 1e-8)
        acc = _accuracy(layers, Xc, yc)
        if acc > best_acc:
            best, best_acc = [(W.copy(), b.copy(), a) for W, b, a in layers], acc
        if t % log_every == 0 or t == steps:
            history.append({"step": t, "score": score(task, _to_network(net.spec, best)),
                            "check": best_acc, "loss": float(loss)})
    return _to_network(net.spec, best), history


def compare(net: Network, task: Task, seeds: int = 3, **kw) -> dict:
    """Train from the design and from `seeds` random starts. Same shape, data and steps for all."""
    trained, designed = train_network(net, task, **kw)
    randoms = [train_network(random_like(net, seed=s), task, **kw)[1] for s in range(seeds)]
    random_avg = [{"step": h["step"], "score": float(np.mean([r[i]["score"] for r in randoms]))}
                  for i, h in enumerate(randoms[0])]
    return {"trained": trained, "designed": designed, "random": random_avg}
