"""Ready-made recipe: remove a bad habit by editing weights.

Idea (same spirit as the MIT "weight-space programming" Qwen fix):
  1. Collect text where the model shows the habit (looping, rambling, making
     things up...) and text where it behaves well.
  2. Look at which blocks clearly "know" the difference.
  3. Remove that difference from those blocks with tiny rank-1 edits.
  4. Keep the edit only if the test score really goes up.

The agent can run this as-is, or use it as a starting point for its own ideas.
"""

from __future__ import annotations

from typing import Callable

from . import tools as T


def fix_habit(model, tok, bad_texts: list[str], good_texts: list[str], score_fn: Callable[[], float],
              name: str, base_model: str = "", max_layers: int = 4,
              strengths: tuple[float, ...] = (0.25, 0.5, 1.0), log=print):
    """Greedy search over (layer, strength). Returns (Recorder, best_score, baseline_score)."""
    bad = T.capture(model, tok, bad_texts)
    good = T.capture(model, tok, good_texts)
    dirs = T.direction(bad, good)
    ranked = T.separation(bad, good)
    log("layers that best tell the habit apart: " + ", ".join(f"{l}({s:.1f})" for l, s in ranked[:8]))

    rec = T.Recorder(model, name, base_model=base_model)
    best = baseline = score_fn()
    log(f"baseline score: {baseline:.3f}")
    for layer, _sep in ranked[:max_layers]:
        kept = None
        for s in strengths:
            trial = T.Recorder(model, "trial")
            trial.ablate_direction(layer, dirs[layer], strength=s)
            sc = score_fn()
            log(f"  layer {layer} strength {s}: {sc:.3f}")
            trial.undo_all()
            if sc > best + 1e-9:
                best, kept = sc, s
        if kept is not None:
            rec.ablate_direction(layer, dirs[layer], strength=kept)
            log(f"  -> keep layer {layer} at strength {kept} (score {best:.3f})")
    return rec, best, baseline
