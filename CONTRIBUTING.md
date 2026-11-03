# Contributing

Thanks for helping! Everything here should stay usable by someone who knows nothing about machine learning.

## Set up

```bash
./install.sh && source .venv/bin/activate
pytest -q        # offline, a few seconds: builds a tiny model, downloads nothing
```

## What's most useful

- 🧩 **Patches** for popular small models in `patches/<base-model>/`: the `.safetensors` file plus a short `.md`
  with what it fixes, held-out before/after scores, and what it might break. See [patches/README.md](patches/README.md).
- ✍️ **Design-first tasks**: a new `Task` in `myai/scratch/tasks.py` (with `describe`, `labeled` and `evaluate`),
  plus a hand-written `network.json` whose `notes` explain every weight.
- 📚 **Example projects** in `examples/`: `goal.yaml`, `docs/`, and an `eval.jsonl` with ~20 questions (about 1 in 4 trick).

## Rules

- **Report real numbers.** Never tune `eval.jsonl` to make a score look better, and include the bad results too.
- **Plain words** in every message a user sees. Say what to do next, not just what went wrong.
- Add a test for new behaviour. Tests must run offline.
- Keep PRs small and focused.
