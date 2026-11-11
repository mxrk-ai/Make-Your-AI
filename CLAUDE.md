# Instructions for AI agents (Claude Code, Codex, Cursor, ...)

You are running **Make Your AI**. A user wants their own small AI. Your job is to build it
from their project folder, test it honestly, and fix its bad habits by editing weights.

The user may not know anything about machine learning. Talk to them in plain words,
show progress, and never claim results you didn't measure.

## Fast path: the user wants a behaviour, not knowledge

If the user asks for a way of answering ("always reply in JSON", "talk like a pirate, max 3 sentences") and has
no documents to teach, do not train anything:

```bash
myai behaviors                                   # what the generator can make
myai generate "<their description>" --out <dir>  # seconds; prints what it understood and what it did not
myai eval <dir>                                  # before/after report card on fresh questions
myai chat <dir> --ask "a question"
myai export <dir> --gguf --ollama                # optional
```

- Show the user the "understood / not understood" lines. If a part was not understood, say so plainly and offer
  the closest behaviour from `myai behaviors`, or the documents path below for knowledge.
- If `myai eval` shows failures, try `myai patch fix <dir> "<what is wrong>" --provider <agy|gemini|anthropic|openai|ollama>`.
  It keeps a change only if it holds on questions it never practised on. Report the before/after numbers it prints.
- `myai status <dir>` shows the next step.

Everything below is the other path: teaching a model the user's own documents.

## 0. Setup

```bash
./install.sh && source .venv/bin/activate   # or: pip install -e ".[llm,dev]"
myai doctor          # checks Python, PyTorch, GPU, disk. Fix any ❌ first
pytest -q            # should pass offline in a few seconds
```

A project is a folder with `goal.yaml`, `docs/` and `eval.jsonl`. If the user has none yet,
run `myai init <dir> --goal "..." --docs <their files>` and help them fill it in: ask what the AI should do,
get their files, and write ~20 test questions with them (about 1 in 4 should be "trick" questions the docs don't answer).
All outputs go to `<project>/runs/<name>/`.

**Always start with `myai status <project>`.** It shows which steps are done and the next command, so you
never redo finished work. Run it again after each step and show it to the user: it's how they follow progress.

## 1. Read the documents

```bash
myai ingest <project>
```

## 2. Write the practice questions (you do this yourself)

When `synth.backend` is `agent` (the default), YOU write the training data. No API key needed.

```bash
myai synth <project>     # writes runs/<name>/SYNTH_TASKS.md: the instructions + every chunk
```

Read `SYNTH_TASKS.md`, then write `runs/<name>/synthetic.jsonl`, one line per pair:
`{"q": "...", "a": "...", "chunk": "<chunk id>", "kind": "fact"}` (or `"kind": "trick"`).

Quality rules (these decide how good the final AI is):
- Every answer must be supported by its chunk. Add no outside facts.
- Follow the `style` rules in goal.yaml exactly. The small model copies your style.
- Vary the questions: casual, formal, typos, short ones, "my X is broken" ones.
- About 15–20% should be trick questions. Their answers start with "I don't know" and say what you *can* help with.
- **Never copy questions from `eval.jsonl`.** That's the exam, and the AI must not see it before the test.
- Aim for about `per_chunk` pairs per chunk. For big projects work chunk by chunk and append.

```bash
myai synth --check <project>   # must show 0 bad_lines and cover every chunk
```

## 3. Test the base model, train, test again

```bash
myai eval <project> --base     # "before" scores
myai train <project>           # LoRA, a few minutes on GPU / Apple Silicon
myai eval <project>            # "after" scores
myai report <project>          # writes reports/REPORT.md
```

Read REPORT.md and the failed answers. If facts are weak, the usual fix is better or more
practice data (step 2), not more epochs. If it rambles, loops or makes things up, go to step 4.

## 4. Fix bad habits by editing weights (weight-space programming)

Fast path, automatic:
```bash
myai patch auto <project> --habit looping      # or rambling, dishonest
```
It compares the model's internal activity on bad vs good answers, finds the blocks that
separate them, removes that direction with rank-1 edits, and keeps an edit only if the score
on half of the tests goes up. Then it reports on the *other* half.

Manual path, where you do the reasoning: write a short Python script using the workbench.
```python
from myai.patch import tools as T
model, tok, dev = T.load(base_model, adapter="<project>/runs/<name>/adapter")
model = model.merge_and_unload()
print(T.overview(model))
bad  = T.capture(model, tok, texts_showing_the_habit)
good = T.capture(model, tok, texts_without_it)
print(T.separation(bad, good)[:8])
rec = T.Recorder(model, "<patch-name>", base_model="<base> + <name> LoRA")
rec.ablate_direction(layer, T.direction(bad, good)[layer], strength=0.5)
# test with myai.evaluate.evaluate_model(...) ; rec.undo_all() if worse
rec.to_patch(description="...", evals={...}).save("<project>/runs/<name>/patches/<patch-name>.safetensors")
```
You can also use `rec.set_values(...)` or `rec.edit(param, fn)` for any edit you can reason about.

Rules for weight edits:
- Form a hypothesis first ("layers 15–19 push towards repeating"), test it, and write down what you found.
- Change as little as possible. Prefer rank-1 edits on a few blocks.
- Keep an edit only if held-out scores improve. Undo everything else.
- Weight edits fix **habits**. Don't try to add new knowledge this way; use step 2 for that.

Add the patch to `patches:` in goal.yaml, then evaluate with label `patched`:
`myai export <project> --out /tmp/m && myai eval <project> --model /tmp/m --label patched && myai report <project>`

## 5. Export

```bash
myai export <project>          # runs/<name>/model  (LoRA + patches baked in)
myai chat <project>                 # or: myai chat <project> --ask "a question"
```
For Ollama: convert to GGUF with llama.cpp (`python convert_hf_to_gguf.py runs/<name>/model --outfile runs/<name>/model/<name>.gguf`)
then `ollama create <name> -f runs/<name>/model/Modelfile`.

## Design-first mode: write the weights from nothing, then train

For tiny rule-like tasks, you write the network's weights yourself first, then training polishes them.
```bash
myai scratch list
myai scratch describe catch             # inputs, outputs, rules, scoring
myai scratch describe table --csv data.csv   # the user's own data
```
1. **Design.** Write `network.json` (format in `myai/scratch/network.py`) by reasoning about what each
   neuron should compute, and put the explanation in `"notes"`. Say which parts are a guess.
   Don't search randomly. The point is that the starting weights are *designed*.
2. **Test the design:** `myai scratch test network.json` (add `--csv data.csv` for a table).
3. **Train from the design:** `myai scratch train network.json [--csv data.csv]`.
   It trains your weights and a random start side by side (same shape, data, steps) and saves
   `network.trained.json`. Report both rows honestly, including when random catches up.
   If the design is already 100%, say so and skip training.

Shape tips: keep inputs in their real units and give each hidden unit a meaning; training can only
adjust units that are active (a ReLU that is always 0 never learns), so don't leave a unit at all zeros.

Examples: `examples/scratch-catch/network.json`, `examples/scratch-thermostat/network.json` (perfect by
design), `examples/scratch-plant/` (a first guess that training improves).

## Always

- Report real numbers from the report card, including the bad ones.
- Never edit `eval.jsonl` to make scores look better.
- Tell the user at the end: where the model is, how to chat with it, what it's still bad at.

---

# Working on this repo itself (contributors and their agents)

Everything above is for building a user's AI. This part is for changing the code.
The plan and the reasons behind it are in [docs/DESIGN.md](docs/DESIGN.md). Read it first.

- **Tests:** `make test` must pass offline in seconds. New code gets a test that needs no downloads (`tests/tiny_model.py`).
- **Unbuilt commands** call `_not_yet(...)` in `myai/cli.py`. Never make a command look like it works before it does.
- **Small steps:** make each phase work end to end on about 5 tasks before scaling up. Commit often.
- **Compute:** CPU work runs locally. GPU jobs run on Colab through the `colab` CLI: L4 by default, T4 for small
  evals, A100 only after asking. Every job checkpoints, can resume, and stops its runtime when it ends or fails.
  Write units used per job in [docs/COMPUTE_LOG.md](docs/COMPUTE_LOG.md) (`colab usage` before and after).
- **Honest results:** every number in the README comes from an eval in the repo. Always report the baselines:
  base model, real trained LoRA, nearest-neighbour retrieval. Held-out tasks are reported separately.
- **Reproducible:** fixed seeds, configs in files, one command per experiment.
- **Zoo adapters** all use `myai.zoo.LORA_SPEC` unchanged. Changing it means retraining the whole zoo.
- **Secrets** live in `.env` (see `.env.example`). Never print them, commit them or ask for them in chat.
- **Neutral repo:** examples are generic and in English. No personal, regional or nationality-specific references.
- **Licensing:** check any new model, dataset or dependency and add it to [LICENSES.md](LICENSES.md).
