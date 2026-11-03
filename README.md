<div align="center">

# 🧠 Make Your AI

**Give it your documents and a goal. An AI agent builds you your own small AI,<br>then fixes its bad habits by editing its weights directly.**

Runs on your laptop · Private and offline · No ML knowledge needed

</div>

---

```
   your files            your goal                       your own AI
 ┌────────────┐    ┌──────────────────────┐        ┌──────────────────┐
 │ manual.pdf │    │ "Answer customer     │        │ small model      │
 │ faq.md     │ +  │  questions. Say 'I   │ agent  │ + report card    │
 │ notes.txt  │    │  don't know' if..."  │ ─────▶ │ + fixed habits   │
 └────────────┘    └──────────────────────┘        └──────────────────┘
```

Open this repo with **Claude Code** (or Codex, Cursor, any coding agent) and say:

> **"Build my AI from `examples/support-bot`."**

The agent reads [`CLAUDE.md`](CLAUDE.md) and does every step by itself.

## What happens

1. **Reads your documents.** PDF, Word, Markdown, text, HTML.
2. **Writes practice questions** from them: real questions plus *trick* questions the documents don't answer, so your AI learns to say "I don't know" instead of making things up.
3. **Teaches a small model** (Qwen 0.5B by default) using LoRA. It trains on a laptop GPU, a Mac, or a free Colab.
4. **Tests it** before and after, and writes a report card.
5. **Fixes bad habits by editing the weights directly.** If the model loops, rambles, or invents answers, the agent looks inside it, finds where the habit lives, and removes it with a tiny surgical edit. No retraining.
6. **Exports** one model folder that runs offline (plus a recipe for Ollama).

You get a **report card** (`runs/<name>/reports/REPORT.md`) that compares before, after and patched on: *knows the documents*, *admits when it doesn't know*, *repeats itself*, *never stops talking*. It also lists every question your AI still gets wrong.

## Two modes

### 1. Make an AI from your documents
For anything with language: support bots, study helpers, internal-docs assistants, product experts. It starts from a small pretrained model and specializes it.

### 2. Design it from nothing, then train ✍️ → 🏋️
For small, rule-like tasks such as games, sensor readings, or simple decisions.

Normally a network starts as random numbers and training has to discover everything. Here the
agent **writes the weights by hand first**, from its own reasoning about the task. Then training
starts from *those* weights and polishes what the agent couldn't work out.

```
 the agent reasons            training polishes            you get
 "water below 30%"   ───▶    learns the temperature  ───▶  a network that is
  (writes weights)            effect it couldn't see        explained AND accurate
```

```bash
$ myai scratch train examples/scratch-plant/network.json --csv examples/scratch-plant/data.csv
designed start: 88.3% -> 98.3% after training
random start:   52.2% -> 85.3% after training (same shape, same data, same steps)
```

The command always trains a random start next to the design, with the same shape, data and steps,
so you can see whether the design actually helped. Training keeps the best version it sees, starting
with the hand-written one, so it never hands back something worse. Scores on your own CSV only count
rows that training never saw.

If the agent's design is already perfect, you can skip training. The repo ships two networks written
this way that score 100% with no training at all:

```bash
$ myai scratch test examples/scratch-catch/network.json
network: examples/scratch-catch/network.json  (39 weights)
score:   100.0%   (random network: 11.4%)
```

Open [`examples/scratch-catch/network.json`](examples/scratch-catch/network.json). Every weight is explained in plain words.

## Plug-and-play weight patches 🧩

Every fix the agent makes is saved as a **patch**: a small file that changes specific weights, like a game mod.

```bash
myai patch info   patches/no-rambling.safetensors          # what does it change?
myai patch apply  Qwen/Qwen2.5-0.5B-Instruct p1.safetensors p2.safetensors --out my-model
myai patch auto   examples/support-bot --habit looping     # find & fix a habit automatically
```

- **Tiny.** A typical fix is a few rank-1 edits, so it stores thousands of numbers, not millions.
- **Safe.** Each patch carries a checksum of the weights it was made for. Applying it to the wrong model fails instead of silently breaking it.
- **Stackable and removable.** Apply several patches and it warns you when two of them touch the same weights. Remove one and you get the exact original back.
- **Honest.** `patch auto` searches on half of your tests and reports on the other half, so the score isn't just memorized.

Made a useful patch? Share it in [`patches/`](patches/).

## Quick start

```bash
git clone https://github.com/mxrk-ai/Make-Your-AI && cd Make-Your-AI
pip install -e ".[llm]"

# try the example (a support bot for a made-up robot vacuum)
myai build examples/support-bot
python -m myai.chat examples/support-bot/runs/orbit-support/model
```

Make your own:

```bash
myai init my-ai              # creates my-ai/goal.yaml, my-ai/docs/, my-ai/eval.jsonl
# 1. describe your AI in goal.yaml
# 2. drop your files in my-ai/docs/
# 3. write ~20 test questions in my-ai/eval.jsonl
# 4. open the repo in Claude Code and say "build my AI from my-ai"
```

No agent? Set `synth.backend: openai` (works with OpenAI, OpenRouter, Ollama, LM Studio…) or `anthropic` in `goal.yaml`, then run `myai build my-ai`.

## Commands

| Command | What it does |
|---|---|
| `myai init <dir>` | start a new project |
| `myai ingest <project>` | read documents into chunks |
| `myai synth <project>` | write practice Q&A (or a work list for the agent) |
| `myai synth --check <project>` | validate the Q&A file |
| `myai eval <project> [--base]` | test the model (`--base` = before training) |
| `myai train <project>` | teach the model with LoRA |
| `myai report <project>` | before/after report card |
| `myai export <project>` | bake LoRA + patches into one model folder |
| `myai build <project>` | all of the above |
| `myai patch info / apply / diff / auto` | weight patches |
| `myai scratch list / describe / test` | design a tiny network from nothing |
| `myai scratch train <network.json>` | train starting from the designed weights, compared to a random start |

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

Small models still make mistakes. The report card shows you exactly where, so read it before you trust your AI with anything important.

## Where the idea comes from

Weight patches and design-first mode are inspired by the blog post **"Sparks of Weight Space Programming: LLMs can understand and write neural network weights"**. It showed a frontier LLM reading networks' weights, editing a small language model's weights to stop it looping, and writing working game networks from scratch. Every experiment's full logs: [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs).

This repo turns that idea into a tool anyone can use. The directional-ablation recipe in `patch auto` builds on earlier interpretability work on finding and removing behaviour directions inside models.

## Contributing

PRs welcome, especially:
- 🧩 **patches** for popular small models (with before/after evals)
- ✍️ **scratch tasks** (new games and puzzles for agents to solve in weights)
- 📚 **example projects**

Run `pytest` before sending a PR. The tests build a tiny offline model, so no downloads are needed.

## License

MIT
