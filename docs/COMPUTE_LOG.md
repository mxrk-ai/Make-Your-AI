# Compute log

Budget: 200 Colab compute units per month. GPU defaults: L4 (about 1.7 units/hr), T4 (about 1.2) for small
evals, A100 (about 5.4) only with approval. Rates are the published approximate ones; the "units" column
below is always the measured difference in `colab usage` before and after a job.

## Balance checks

| Date | Balance (units) | Note |
|---|---|---|
| 2026-11-11 | 199.97 | First check after CLI login. One CPU runtime was already running (0.08 units/hr), not started by this project. |

## GPU jobs

| Date | Phase | Job | GPU | Hours | Units | Result / checkpoint |
|---|---|---|---|---|---|---|
| 2026-11-11 | 1 | Pilot attempt 1 (crashed at start: library conflict on the VM) | T4 | 0.03 | 0.03 | Nothing. Fixed in `scripts/colab_zoo.py`. |
| 2026-11-11 | 1 | Pilot: 3 adapters, rank 4 (one 93%, three 80%, caps 67%), then stopped by hand | T4 | 0.2 | 0.20 | Led to rank 8 and 16-bit training. |

## Agent LLM calls

| Date | Phase | Provider / model | Calls | Approx. cost | Note |
|---|---|---|---|---|---|
| 2026-11-11 | 5 | agy (subscription, default model) | 4 | $0 | Functional test of `patch fix` on a fake adapter. |

## Per phase

| Phase | Estimated units | Used units | API cost |
|---|---|---|---|
| 0 Setup | 0 | 0 | $0 |
| 1 Zoo | 30–40 | | |
| 2 Captioner | 3–5 | | |
| 3 Generator | 20–30 | | |
| 4 Generate + baselines | 10–15 | | |
| 5 Patch loop | 10 | | |
| 6 Export | 0 | | |
| 8 Research preview | 30–40 | | |
