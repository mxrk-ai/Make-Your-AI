"""Tasks for write-from-scratch mode.

Each task explains itself in plain words (`describe`), so an AI agent knows
exactly what the inputs and outputs mean, and scores any network (`evaluate`).

Add your own: subclass Task, or use `table` with a CSV of examples.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .network import Network


class Task:
    name = ""
    n_in = 0
    n_out = 0

    def describe(self) -> str:
        raise NotImplementedError

    def evaluate(self, net: Network, episodes: int = 2000, seed: int = 0) -> dict:
        raise NotImplementedError

    def labeled(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """n example inputs with the correct output for each, used to train a network."""
        raise NotImplementedError

    def check_shape(self, net: Network):
        if net.n_in != self.n_in or net.n_out != self.n_out:
            raise ValueError(f"Task '{self.name}' needs {self.n_in} inputs -> {self.n_out} outputs, "
                             f"network has {net.n_in} -> {net.n_out}")


class Catch(Task):
    """Mini arcade game: a ball falls, move the paddle to catch it."""

    name = "catch"

    def __init__(self, width: int = 7, height: int = 7):
        self.w, self.h = width, height
        self.n_in, self.n_out = 2 * width, 3

    def describe(self) -> str:
        w, h = self.w, self.h
        return f"""TASK: catch
A ball falls straight down a grid {w} columns wide and {h} rows tall, one row per turn.
A paddle sits on the bottom row. Each turn the network picks a move for the paddle.
When the ball reaches the bottom row: +1 if the paddle is under it, otherwise -1.
The ball and paddle start in random columns. There is always enough time to reach the ball.

INPUTS ({self.n_in} numbers):
  inputs[0..{w-1}]   ball column, one-hot (a 1 in the ball's column, 0 elsewhere)
  inputs[{w}..{2*w-1}]  paddle column, one-hot
OUTPUTS ({self.n_out} numbers): the highest one is the move
  0 = move left, 1 = stay, 2 = move right   (moves past the edge do nothing)
SCORE: share of balls caught (1.0 = perfect). A random network catches roughly 10-30%."""

    def _obs(self, ball, paddle):
        o = np.zeros(self.n_in)
        o[ball] = 1
        o[self.w + paddle] = 1
        return o

    def labeled(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        # the right move is always "towards the ball"
        rng = np.random.default_rng(seed)
        ball, paddle = rng.integers(self.w, size=n), rng.integers(self.w, size=n)
        X = np.array([self._obs(b, p) for b, p in zip(ball, paddle)])
        return X, np.sign(ball - paddle) + 1

    def evaluate(self, net: Network, episodes: int = 2000, seed: int = 0) -> dict:
        self.check_shape(net)
        rng = np.random.default_rng(seed)
        caught = 0
        for _ in range(episodes):
            ball, paddle = rng.integers(self.w), rng.integers(self.w)
            for _row in range(self.h - 1):
                a = net.act(self._obs(ball, paddle))
                paddle = int(np.clip(paddle + (a - 1), 0, self.w - 1))
            caught += ball == paddle
        return {"score": caught / episodes, "caught": int(caught), "episodes": episodes}


class Thermostat(Task):
    """A rule-following controller: heat, cool, or switch off."""

    name = "thermostat"
    n_in, n_out = 3, 3

    def describe(self) -> str:
        return """TASK: thermostat
Decide what a home thermostat should do.

INPUTS (3 numbers):
  inputs[0] room temperature in °C   (somewhere between 5 and 35)
  inputs[1] target temperature in °C (somewhere between 16 and 26)
  inputs[2] window open?             (1 = open, 0 = closed)
OUTPUTS (3 numbers): the highest one is the decision
  0 = off, 1 = heat, 2 = cool
RULES:
  - window open -> off (never waste energy with the window open)
  - room more than 1 °C below target -> heat
  - room more than 1 °C above target -> cool
  - otherwise -> off
SCORE: share of random situations decided correctly (1.0 = perfect)."""

    @staticmethod
    def rule(temp, target, window):
        if window > 0.5:
            return 0
        if temp < target - 1:
            return 1
        if temp > target + 1:
            return 2
        return 0

    def labeled(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        X = np.column_stack([rng.uniform(5, 35, n), rng.uniform(16, 26, n), rng.integers(0, 2, n)])
        return X, np.array([self.rule(*x) for x in X])

    def evaluate(self, net: Network, episodes: int = 5000, seed: int = 0) -> dict:
        self.check_shape(net)
        X, y = self.labeled(episodes, seed)
        pred = np.argmax(net.forward(X), axis=1)
        wrong = [{"inputs": X[i].round(2).tolist(), "expected": int(y[i]), "got": int(pred[i])}
                 for i in np.nonzero(pred != y)[0][:10]]
        return {"score": float((pred == y).mean()), "examples": episodes, "first_mistakes": wrong}


class Table(Task):
    """Any task given as a CSV: input columns, then one 'label' column with class numbers 0,1,2..."""

    name = "table"

    def __init__(self, csv_path: str | Path, label: str = "label"):
        rows = list(csv.DictReader(Path(csv_path).open()))
        if not rows:
            raise ValueError("CSV is empty")
        self.cols = [c for c in rows[0] if c != label]
        self.X = np.array([[float(r[c]) for c in self.cols] for r in rows])
        self.y = np.array([int(r[label]) for r in rows])
        self.n_in, self.n_out = len(self.cols), int(self.y.max()) + 1
        self.path = csv_path
        # fixed split: training only sees the first 70% (shuffled); "test" is the other 30%
        order = np.random.default_rng(0).permutation(len(self.y))
        cut = int(len(order) * 0.7)
        self.train_rows, self.test_rows = order[:cut], order[cut:]

    def describe(self) -> str:
        cls = ", ".join(f"{k}: {int((self.y == k).sum())} rows" for k in range(self.n_out))
        preview = "\n".join(f"  {dict(zip(self.cols, x.tolist()))} -> {y}" for x, y in zip(self.X[:8], self.y[:8]))
        return f"""TASK: table ({self.path})
INPUTS ({self.n_in} numbers, in this order): {', '.join(self.cols)}
OUTPUTS ({self.n_out} numbers): highest one = predicted class. Classes: {cls}
First rows:
{preview}
SCORE: share of rows predicted correctly. Look at the data, find the rule, write it as weights."""

    def labeled(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rows = np.random.default_rng(seed).choice(self.train_rows, size=n)
        return self.X[rows], self.y[rows]

    def evaluate(self, net: Network, episodes: int = 0, seed: int = 0, rows: str = "all") -> dict:
        """rows: "all", or "test" for only the 30% of rows that training never sees."""
        self.check_shape(net)
        idx = self.test_rows if rows == "test" else np.arange(len(self.y))
        pred = np.argmax(net.forward(self.X[idx]), axis=1)
        return {"score": float((pred == self.y[idx]).mean()), "rows": len(idx)}


TASKS = {"catch": Catch, "thermostat": Thermostat}


def get_task(name: str, csv_path: str | None = None) -> Task:
    if name == "table":
        if not csv_path:
            raise ValueError("task 'table' needs --csv")
        return Table(csv_path)
    if name not in TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {', '.join(TASKS)}, table")
    return TASKS[name]()


def random_network(task: Task, hidden: int = 16, seed: int = 0) -> Network:
    """Baseline: a random untrained network, to compare against."""
    rng = np.random.default_rng(seed)
    return Network({"layers": [
        {"W": rng.normal(0, 1, (hidden, task.n_in)).tolist(), "b": [0.0] * hidden, "act": "relu"},
        {"W": rng.normal(0, 1, (task.n_out, hidden)).tolist(), "b": [0.0] * task.n_out, "act": "none"},
    ]})
