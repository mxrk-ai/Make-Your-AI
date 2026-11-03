<div align="center">

# 🧠 Make Your AI

**Give it your documents and a goal. A coding agent builds you your own small AI,<br>tests it honestly, and fixes its bad habits by editing its weights directly.**

Runs on your laptop · Private and offline · No ML knowledge needed

[![tests](https://github.com/mxrk-ai/Make-Your-AI/actions/workflows/tests.yml/badge.svg)](https://github.com/mxrk-ai/Make-Your-AI/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

</div>

---

```
   your files            your goal                        your own AI
 ┌────────────┐    ┌──────────────────────┐         ┌──────────────────┐
 │ manual.pdf │    │ "Answer customer     │         │ small model      │
 │ faq.md     │ +  │  questions. Say 'I   │  agent  │ + report card    │
 │ notes.txt  │    │  don't know' if..."  │ ──────▶ │ + fixed habits   │
 └────────────┘    └──────────────────────┘         └──────────────────┘
```

## Start in 3 steps

**1. Install** (makes a private environment in `.venv` and checks your computer):

```bash
git clone https://github.com/mxrk-ai/Make-Your-AI && cd Make-Your-AI
./install.sh
source .venv/bin/activate
```

**2. Try the example.** A support bot for a made-up robot vacuum. This downloads a 1 GB base model once.

```bash
myai build examples/support-bot
myai chat examples/support-bot
```

**3. Make your own.** `init` asks what your AI should do and where your files are:

```bash
myai init my-ai
```

Then open this folder in **Claude Code** (or Codex, Cursor, any coding agent) and say:

> **"Build my AI from `my-ai`."**

The agent follows [`CLAUDE.md`](CLAUDE.md): it writes the practice questions, trains, tests, fixes habits, and tells you
honestly what your AI is still bad at. You can check progress at any time:

```bash
$ myai status my-ai
  ✅  1. Describe your AI (goal.yaml)       Answer questions about our cafe.
  ✅  2. Add your documents (docs/)         3 file(s)
  ⬜  3. Write test questions (eval.jsonl)  2 questions, 1 trick  (eval.jsonl line 1 is still the example question. Write your own)
  ...
Next: write ~20 questions in my-ai/eval.jsonl
```

Something not working? `myai doctor` checks Python, PyTorch, your GPU and disk space, and says how to fix what's wrong.

## Real result: the example on a MacBook (M4 Pro)

| | Untrained | After 40 s of training |
|---|---|---|
| Overall score (24 test questions it never saw) | 8% | **54%** |
| Knows the manual | 11% | **56%** |
| Says "I don't know" instead of making things up | 0% | **50%** |
| Repeats itself / never stops talking | 0% / 0% | 0% / 0% |

That's a big jump, but 54% is not perfect. It still mixes up numbers that sit close together in the manual, and on
half the trick questions it still makes something up. The [report card](examples/support-bot/runs/orbit-support/reports/REPORT.md)
lists every wrong answer. Every project gets one like it, so you know what to trust before you use your AI.

## How it works

1. **Reads your documents:** PDF, Word, Markdown, text, HTML.
2. **Writes practice questions** from them: real ones plus *trick* questions the documents don't answer, so your AI
   learns to say "I don't know". Your coding agent writes these, so no API key is needed.
3. **Tests the untrained model** on your own test questions, which training never sees.
4. **Teaches a small model** (Qwen 0.5B by default) with LoRA on a laptop GPU, a Mac, or a free Colab.
5. **Tests it again** and writes a report card: *knows the documents*, *admits when it doesn't know*,
   *repeats itself*, *never stops talking*.
6. **Fixes bad habits by editing weights** (below), then **exports** one model folder that runs offline.

## Plug-and-play weight patches 🧩

When the model loops, rambles, or invents answers, the agent doesn't retrain it. It looks inside, finds where the habit
lives, and removes it with a tiny edit. Every fix is saved as a **patch**, a small file like a game mod:

```bash
myai patch auto   examples/support-bot --habit dishonest        # find & fix a habit automatically
myai patch info   my-patch.safetensors                          # what does it change?
myai patch apply  Qwen/Qwen2.5-0.5B-Instruct p1.safetensors p2.safetensors --out my-model
```

- **Tiny.** A typical fix is a few rank-1 edits: thousands of numbers, not millions.
- **Safe.** Each patch carries a checksum of the weights it was made for. Applying it to the wrong model fails instead
  of silently breaking it.
- **Stackable and removable.** Stack several patches and you're warned when two touch the same weights. Remove one and you get the exact original back.
- **Honest.** `patch auto` searches on half your tests and reports on the other half, so the score isn't memorized.

Habits: `looping`, `rambling`, `dishonest`. Add a patch to `patches:` in `goal.yaml` and `myai export` bakes it in.
Made a useful one? Share it in [`patches/`](patches/).

## Design it from nothing, then train ✍️ → 🏋️

For small, rule-like tasks such as games, sensor readings, or simple decisions. Normally a network starts as random
numbers and training has to discover everything. Here the agent **writes the weights by hand first**, from its own
reasoning, and explains each one. Then training starts from *those* weights and polishes what the agent couldn't work out.

```bash
$ myai scratch train examples/scratch-plant/network.json --csv examples/scratch-plant/data.csv
designed start: 88.3% -> 98.3% after training
random start:   52.2% -> 85.3% after training (same shape, same data, same steps)
```

It always trains a random start next to the design, so you can see whether the design actually helped. Training never
hands back something worse than the design, and scores on your own CSV only count rows training never saw.
No PyTorch needed (`./install.sh --light`).

Some designs are perfect with no training at all:

```bash
$ myai scratch test examples/scratch-catch/network.json
score:   100.0%   (random network: 11.4%)
```

Open [`examples/scratch-catch/network.json`](examples/scratch-catch/network.json). Every weight is explained in plain words.

## All commands

| Command | What it does |
|---|---|
| `myai doctor` | check this computer is ready |
| `myai init <dir>` | start a project (asks what it should do and copies your files in) |
| `myai status <project>` | what's done, what's missing, the next command to run |
| `myai build <project>` | run every step that's left |
| `myai chat <project>` | talk to your AI (`--ask "..."` for one question) |
| `myai ingest / synth / eval / train / report / export` | the single steps, if you want control |
| `myai synth --check <project>` | validate the practice questions |
| `myai patch auto / info / apply / diff` | weight patches |
| `myai scratch list / describe / test / train` | design-first mode |

## More ways to run it

- **No coding agent?** Set `synth.backend: openai` (OpenAI, OpenRouter, Ollama, LM Studio…) or `anthropic` in
  `goal.yaml`, set the API key, and run `myai build my-ai`.
- **Ollama:** `myai export` writes a `Modelfile`. Convert to GGUF with llama.cpp
  (`python convert_hf_to_gguf.py <project>/runs/<name>/model --outfile <project>/runs/<name>/model/<name>.gguf`),
  then `ollama create <name> -f <project>/runs/<name>/model/Modelfile`.
- **Another base model:** change `base_model` in `goal.yaml` to any Hugging Face chat model.

## For AI agents: the workbench

`myai.patch.tools` gives an agent what it needs to do weight-space programming on a real model:

```python
from myai.patch import tools as T
model, tok, dev = T.load("Qwen/Qwen2.5-0.5B-Instruct")
print(T.overview(model))                    # map of every weight
bad, good = T.capture(model, tok, looping_texts), T.capture(model, tok, normal_texts)
print(T.separation(bad, good)[:5])          # which blocks "know" the difference
rec = T.Recorder(model, "no-looping")
rec.ablate_direction(15, T.direction(bad, good)[15])   # rank-1 surgical edit
rec.to_patch(description="stops loops").save("patches/no-looping.safetensors")
```

## What this can and can't do (honestly)

| ✅ Works | ❌ Doesn't (yet) |
|---|---|
| Teaching a small model *your* documents with LoRA | Making a 0.5B model as smart as a frontier model |
| Fixing specific bad habits (loops, rambling, inventing answers) with weight edits | Adding whole new knowledge *only* by editing weights |
| Designing tiny networks (games, rules, simple data) from nothing, then training them | Writing a language model from zero weights |

Small models still make mistakes. Read the report card before you trust your AI with anything important.

## Where the idea comes from

Weight patches and design-first mode are inspired by the blog post **"Sparks of Weight Space Programming: LLMs can
understand and write neural network weights"**. It showed a frontier LLM reading networks' weights, editing a small
language model's weights to stop it looping, and writing working game networks from scratch. Every experiment's full
logs: [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs). The directional-ablation recipe in
`patch auto` builds on earlier interpretability work on finding and removing behaviour directions inside models.

## Contributing

PRs welcome, especially 🧩 patches for popular small models, ✍️ new design-first tasks, and 📚 example projects.
See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
