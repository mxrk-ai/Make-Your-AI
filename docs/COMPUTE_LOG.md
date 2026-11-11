# Compute log

Budget: 200 Colab compute units per month. GPU defaults: L4 (about 1.7 units/hr), T4 (about 1.2) for small
evals, A100 (about 5.4) only with approval. Rates are the published approximate ones; the "units" column
below is always the measured difference in `colab usage` before and after a job.

## Balance checks

| Date | Balance (units) | Note |
|---|---|---|
| 2026-11-11 | 183.60 | After all jobs. No active sessions. Total used: 16.37 units. |
| 2026-11-11 | 199.97 | First check after CLI login. One CPU runtime was already running (0.08 units/hr), not started by this project. |

## GPU jobs

| Date | Phase | Job | GPU | Hours | Units | Result / checkpoint |
|---|---|---|---|---|---|---|
| 2026-11-11 | 1 | Pilot attempt 1 (crashed at start: library conflict on the VM) | T4 | 0.03 | 0.03 | Nothing. Fixed in `scripts/colab_zoo.py`. |
| 2026-11-11 | 1 | Pilot: 3 adapters, rank 4 (one 93%, three 80%, caps 67%), then stopped by hand | T4 | 0.2 | 0.20 | Led to rank 8 and 16-bit training. |
| 2026-11-11 | 1 | Config benchmark (rank, steps, learning rate) | L4 | | about 1 | Kept rank 8, 200 steps, lr 2e-3. |
| 2026-11-11 | 1 | Zoo: 333 adapters on two L4 sessions, 4 shards | L4 x2 | about 1.7 each | about 6.7 | Private dataset repo `make-your-ai-zoo-v1`. Mean score 0.88. |
| 2026-11-11 | 3-8 | Pipeline `stage=main`: generator, guidance search, scoreboard, demo, best-of-4, flywheel | L4 | about 3.3 | about 5.5 | `results/report.json`, `demo.json`, `bestof.json`, `flywheel.json`. The session ended on its own after the last upload. |
| 2026-11-11 | 2, 5 | Pipeline `stage=agent`: captioner (20), agent repair (5 adapters, with and without harmonize) | L4 | about 1.8 | about 3 | `results/caption.json`, `results/patch.json`. |

Per-job units after the pilot are estimates (jobs overlapped, so `colab usage` differences mix them). The total,
16.37 units, is measured.

## Agent LLM calls

| Date | Phase | Provider / model | Calls | Approx. cost | Note |
|---|---|---|---|---|---|
| 2026-11-11 | 5 | agy (subscription, default model) | 4 | $0 | Functional test of `patch fix` on a fake adapter. |
| 2026-11-11 | 2 | agy through `scripts/agent_relay.py` | 20 | $0 | Captioner experiment. |
| 2026-11-11 | 5 | agy through `scripts/agent_relay.py` | 29 | $0 | Repair experiment. |

## Per phase

| Phase | Estimated units | Used units | API cost |
|---|---|---|---|
| 0 Setup | 0 | 0 | $0 |
| 1 Zoo | 30–40 | about 7.9 | $0 |
| 2 Captioner | 3–5 | about 1 | $0 |
| 3 Generator | 20–30 | under 1 | $0 |
| 4 Generate + baselines | 10–15 | about 3.5 | $0 |
| 5 Patch loop | 10 | about 2 | $0 |
| 6 Export | 0 | 0 | $0 |
| 8 Research preview | 30–40 | about 1.5 | $0 |
