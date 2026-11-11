<div align="center">

# 🧠 Make Your AI

**Describe the AI you want in one sentence. Get a working custom model in seconds, with no training.**

Like an image generator, but instead of pictures it makes small AIs.

[![tests](https://github.com/mxrk-ai/Make-Your-AI/actions/workflows/tests.yml/badge.svg)](https://github.com/mxrk-ai/Make-Your-AI/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-FSL--1.1--ALv2-blue)

</div>

---

```bash
$ myai generate "talk like a pirate, max 3 sentences" --out pirate
Made your AI in a few seconds, with no training: pirate/
  understood:      Answer in at most three sentences. Talk like a pirate.

$ myai chat pirate --ask "What is a volcano?"
ai  > {{DEMO_ANSWER}}
```

A **generator** (a small diffusion model) has learned what trained LoRA adapters look like. You give it a
description, it starts from noise and draws the adapter's weights. The adapter plugs into a small open model
(Qwen2.5-0.5B, 1 GB) that runs on a laptop, offline. Then:

- **`myai eval`** tests your AI on fresh questions and writes a before/after report card.
- **`myai patch fix`** lets an LLM agent look inside the weights and repair what is still wrong, with no retraining.
- **`myai export --gguf`** gives you one file for Ollama or LM Studio.

> "The generator is the intuition, the agent is the reasoning."

## Start in 3 steps

**1. Install** (makes a private environment in `.venv` and checks your computer):

```bash
git clone https://github.com/mxrk-ai/Make-Your-AI && cd Make-Your-AI
./install.sh
source .venv/bin/activate
```

**2. Make an AI.** The first run downloads the generator ({{GEN_MB}} MB) and the base model (1 GB), once.

```bash
myai generate "reply only in JSON, in one sentence" --out json-bot
myai chat json-bot
```

**3. Test it, fix it, take it with you.**

```bash
myai eval json-bot                                   # before / after report card
myai patch fix json-bot "never add text outside the JSON" --provider agy    # optional: agent repair
myai export json-bot --gguf --ollama                 # then: ollama run json-bot
```

`myai status json-bot` shows what is done and the next command. `myai doctor` checks your computer.

## What it can make today

The generator knows **15 behaviours in 5 groups**. Ask for one, or combine up to one from each group, in your own
words. `myai behaviors` prints this list.

| Group | Behaviours |
|---|---|
| Length | exactly one sentence · at most three sentences |
| Format | ALL CAPS · all lowercase · JSON only · bullet points · numbered list |
| Style | pirate speak · an emoji after every sentence · very excited · formal English |
| Opener | starts with "Sure thing!" · starts with "Certainly." |
| Closer | ends with "Hope that helps!" · ends with "That is all." |

Ask for something else ("be kind", "answer in French") and `myai generate` tells you which part it did **not
understand** instead of pretending. For knowledge from your own documents, use [the other mode](#the-other-mode-teach-it-your-documents).

## Does it work? The honest scoreboard

Every task has automatic checks (is it valid JSON? one sentence? any lowercase letter?). An answer passes only if
**every** rule holds. Scores are the share of {{N_PROMPTS}} fresh questions that pass, averaged over tasks. Questions and
topics used for testing never appear in training.

- **Seen tasks:** combinations the generator trained on.
- **Held-out tasks:** {{N_HELD}} combinations it never saw (each single behaviour appears elsewhere, the mix is new).

{{RESULTS_TABLE}}

{{RESULTS_READING}}

Reproduce: `myai generator-report <zoo> <generator> --out report.json`. Raw numbers: [`results/`](results/).

## How it was built

1. **A zoo of {{N_ADAPTERS}} adapters.** {{N_TASKS}} tasks (every single behaviour, every pair, a sample of triples), up to 3
   adapters each. Training data is the base model's own plain answers rewritten by rules into each behaviour, so an
   adapter learns only the *change in behaviour*. All adapters share one frozen half (LoRA "A"), so they live in one
   coordinate system and the generator only has to produce the other half: 196,608 numbers.
2. **An autoencoder** compresses each adapter to 48 small tokens, one per network module.
3. **A diffusion transformer** learns to denoise those tokens, conditioned on a sentence embedding of each clause
   of the description.
4. **Baselines and held-out tasks** keep it honest (table above).

Full design and what changed along the way: [docs/DESIGN.md](docs/DESIGN.md). Compute used: [docs/COMPUTE_LOG.md](docs/COMPUTE_LOG.md)
({{UNITS}} Colab compute units in total).

## Agent repair: `myai patch fix`

When the report card shows failures, an LLM agent can repair the adapter **by editing weights, with no gradient
descent**. It sees the failing answers, the size and shape of every weight matrix, and what breaks when parts are
switched off. It proposes small edits (scale or switch off some layers, keep only the strongest direction, blend in
a freshly generated adapter). Each edit is kept only if the score rises on practice questions, and the final result
is saved only if it also holds on questions the agent never saw.

```bash
myai patch fix my-ai "stops repeating itself" --provider agy
```

Providers: `agy` (a command-line agent on a subscription), `gemini`, `anthropic`, `openai`, `ollama`. Only `agy`
has been exercised in this repo's experiments.

{{PATCH_RESULTS}}

## Train your own generator

```bash
myai zoo train my-zoo                       # trains every adapter (resumable; needs a GPU, see scripts/colab_launch.sh)
myai train-generator my-zoo --out my-gen    # minutes on a laptop
myai generator-report my-zoo my-gen --out report.json
myai generate "..." --generator my-gen
```

To teach it new behaviours, add them to [`myai/zoo/behaviors.py`](myai/zoo/behaviors.py): a description, a rewrite
rule and a check.

## The other mode: teach it your documents

`generate` changes how a model behaves. To make it **know your content**, give it your files and a goal; a coding
agent writes practice questions, trains a LoRA on your laptop, tests it honestly and fixes bad habits by editing weights.

```bash
myai init my-ai          # asks what it should do and where your files are
myai build my-ai         # or open the repo in a coding agent and say: "Build my AI from my-ai."
```

The bundled example (a support bot for a made-up robot vacuum) goes from 8% to 54% on 24 questions it never saw,
after 40 seconds of training on a MacBook; the [report card](examples/support-bot/runs/orbit-support/reports/REPORT.md)
lists every wrong answer. Details for agents are in [CLAUDE.md](CLAUDE.md). There is also a design-first mode for
tiny rule-like networks (`myai scratch`).

## All commands

| Command | What it does |
|---|---|
| `myai generate "<description>"` | make an AI from a description, no training |
| `myai behaviors` | what the generator knows how to make |
| `myai chat <folder>` | talk to your AI (`--ask "..."` for one question) |
| `myai eval <folder>` | before/after report card |
| `myai patch fix <folder> "<what to fix>"` | agent repairs the weights |
| `myai export <folder> --gguf [--ollama]` | one model folder, a GGUF file, optionally registered in Ollama |
| `myai status <folder>` · `myai doctor` | what's next · is this computer ready |
| `myai zoo train` · `train-generator` · `generator-report` · `caption` | build and score your own generator |
| `myai init / build / ingest / synth / train / report` | the from-your-documents mode |
| `myai patch auto / info / apply / diff` | weight patches for trained projects |
| `myai scratch list / describe / test / train` | design-first mode |

`mya` works as a short alias for `myai`.

## Limitations (read this)

{{LIMITATIONS}}

## Where the ideas come from

- **Generating weights:** work on diffusion and hypernetworks over neural network weights (for example D2NWG, RPG /
  ORAL, Text-to-LoRA, Drag-and-Drop LLMs).
- **Editing weights with an agent:** the blog post "Sparks of Weight Space Programming: LLMs can understand and write
  neural network weights" (logs: [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)), and
  earlier interpretability work on finding and removing behaviour directions.
- **This repo's part** is wiring them together with evals as the referee, plus re-noising an agent-edited adapter
  through the generator (`--harmonize`, research preview).

## Contributing

PRs welcome, especially new behaviours with checks, support for other base models, and repairs that hold up on
held-out questions. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Code: [FSL-1.1-ALv2](LICENSE.md) (free for any use except offering a competing commercial product; each version
becomes Apache 2.0 two years after its release). Version 0.1.0 was released under MIT and stays MIT
([LICENSE-MIT-v0.1](LICENSE-MIT-v0.1)). Generator weights: non-commercial, commercial license on request.
Upstream licenses: [LICENSES.md](LICENSES.md).
