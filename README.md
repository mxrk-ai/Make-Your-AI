<div align="center">

# 🧠 Make Your AI

### Your own small AI, on your own laptop.

**Teach it your documents, or describe how it should answer.**
No cloud bill, no data leaving your machine, and a report card that tells you the truth.

[![tests](https://github.com/mxrk-ai/Make-Your-AI/actions/workflows/tests.yml/badge.svg)](https://github.com/mxrk-ai/Make-Your-AI/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-FSL--1.1--ALv2-blue)
[![generator](https://img.shields.io/badge/%F0%9F%A4%97%20generator-v1-yellow)](https://huggingface.co/mxrk18/make-your-ai-generator-v1)

</div>

---

> **Status: research preview.** Every number below was measured and the raw files are in [`results/`](results/).
> Some of them are not good. Read [Limitations](#limitations-read-this) before relying on it.

## Why use it

- **Private by default.** Everything runs on your computer, on a small open model (Qwen2.5-0.5B, 1 GB). Your documents never leave it.
- **Honest by design.** Every AI you make gets a before/after report card on questions it never saw, wrong answers included.
- **Fixable without retraining.** Bad habits (rambling, looping, making things up) are repaired by editing weights directly, and an edit is kept only if the score goes up.
- **Yours to take anywhere.** One command exports a single file for Ollama or LM Studio.

## Two ways to make an AI

| | You give it | You get | How long | How well it works today |
|---|---|---|---|---|
| **Teach it your documents** (`myai build`) | your files and ~20 test questions | a model that answers from your content | minutes of training | bundled example: 8% → 54% on 24 unseen questions |
| **Describe a behaviour** (`myai generate`) | one sentence | a model that answers in that style | seconds, no training | 89% on combinations it has seen, 36% on new ones |

After either one, **`myai eval`** writes an honest before/after report card, **weight patches** fix bad habits by
editing weights with no retraining, and **`myai export --gguf`** gives you one file for Ollama or LM Studio.

- For a company or product assistant, go to [Teach it your documents](#the-other-mode-teach-it-your-documents).
- The rest of this page starts with `generate`, the newer and more experimental part.

## Describe a behaviour: `myai generate`

```bash
$ myai generate "talk like a pirate, max 3 sentences" --out pirate
Made your AI in a few seconds, with no training: pirate/
  understood:      Answer in at most three sentences. Talk like a pirate.

$ myai chat pirate --ask "Why is sleep important?"
ai  > Arr, sleep be crucial for several reasons: it helps the body recover and repair after daytime activities.
      It regulates hormones necessary for mood regulation and cognitive function. Sleep also plays a vital role in
      memory consolidation and learning new skills, making it essential for overall health and well-being, matey.
```

A **generator** (a small diffusion model) has learned what trained LoRA adapters look like. You give it a
description, it starts from noise and draws the adapter's weights, like an image generator draws a picture. The
adapter plugs into the base model. An LLM agent can then repair what is still wrong by editing the weights
(`myai patch fix`).

## Start in 3 steps

**1. Install** (makes a private environment in `.venv` and checks your computer):

```bash
git clone https://github.com/mxrk-ai/Make-Your-AI && cd Make-Your-AI
./install.sh
source .venv/bin/activate
```

**2. Make an AI.** The first run downloads the generator (57 MB) and the base model (1 GB), once.

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
**every** rule holds. Scores are the share of 24 fresh questions that pass, averaged over tasks. Questions and
topics used for testing never appear in training.

- **Seen tasks:** combinations the generator trained on.
- **Held-out tasks:** 30 combinations it never saw (each single behaviour appears elsewhere, the mix is new).

| Method | Seen tasks (15) | Held-out tasks (30) |
|---|---|---|
| Base model | 0% | 0% |
| Base model + description as system prompt | 5% | 5% |
| Really trained LoRA (upper bound) | 87% | 84% |
| Nearest training adapter (retrieval; on seen tasks this is the trained LoRA itself) | — | 25% |
| Sum of single-behaviour adapters (oracle) | — | 8% |
| **Generated adapter** | **89%** | **36%** |
| Generated adapter + system prompt | 38% | 34% |
| Generated, best of 4 tries (picked on 12 other questions) | — | 38% |
| Trained LoRA through the autoencoder | — | 45% |

**How to read it.**

- **On combinations it trained on, generation works:** 89%, the same as a really trained adapter (87%), in seconds.
- **On new combinations it mostly does not work yet:** 36%, against 84% for a trained adapter. It beats copying the
  nearest known adapter (25%; better on 16 of the 30 tasks, worse on 8), but that is a small win. 9 of the 30
  held-out tasks reach 70% or more and 10 are at 10% or less. Pairs do better (40%) than triples (29%).
- **Where it is lost:** the compression step alone drops a trained held-out adapter from 84% to 45%, so the
  generator cannot do much better than that with this autoencoder. Fixing the autoencoder is the next step.
- **Trying four times barely helps** (36% to 38%; even always picking the luckiest try gives 42%).
- **The flywheel did nothing:** 40 new combinations were generated, none scored 90% or more, so none was added and
  the generator was not retrained.
- A 0.5B model ignores a system prompt for these rules (5%), which is why an adapter is needed at all.

Reproduce: `myai generator-report <zoo> <generator> --out report.json`. Raw numbers: [`results/`](results/).

## How it was built

1. **A zoo of 333 adapters.** 131 tasks (every single behaviour, every pair, a sample of triples), up to 3
   adapters each. Training data is the base model's own plain answers rewritten by rules into each behaviour, so an
   adapter learns only the *change in behaviour*. All adapters share one frozen half (LoRA "A"), so they live in one
   coordinate system and the generator only has to produce the other half: 196,608 numbers.
2. **An autoencoder** compresses each adapter to 48 small tokens, one per network module.
3. **A diffusion transformer** learns to denoise those tokens, conditioned on a sentence embedding of each clause
   of the description.
4. **Baselines and held-out tasks** keep it honest (table above).

Full design and what changed along the way: [docs/DESIGN.md](docs/DESIGN.md). Compute used: [docs/COMPUTE_LOG.md](docs/COMPUTE_LOG.md)
(16.4 Colab compute units in total).

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

**Measured** on 5 generated held-out adapters that failed their report card (12 practice questions for the agent,
12 other questions for the final number, `agy` as the agent, 29 agent calls):

| | Before | After repair | Adapters improved |
|---|---|---|---|
| Agent edits only | 18% | 45% | 3 of 5 |
| Agent edits, then re-noised through the generator (`--harmonize`) | 18% | 50% | 3 of 5 |

The three it fixed went from 8%, 42% and 42% to 83-92%, 50-75% and 83-92%. The two at 0% stayed at 0% after 9
edits each: the agent can strengthen a behaviour that is partly there, it cannot create a missing one. Five
adapters is a small sample, so the difference between the two rows is not a finding. Raw runs:
[`results/patch.json`](results/patch.json).

The agent can also **describe an adapter** from its weights and sample answers (`myai caption`). On 20 trained
adapters it named 51% of the behaviours and got the full set right for 2 of 20 ([`results/caption.json`](results/caption.json)).

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

- **Only the 15 listed behaviours.** It changes how a model answers, not what it knows. No new facts, no new skills.
- **New combinations are unreliable** (36% on held-out tasks). Run `myai eval` on every AI you make; it tells you
  the real score for yours. Combinations the generator trained on are fine (89%).
- **ALL CAPS is weak everywhere**, including in trained adapters; "ALL CAPS and very excited" scored 25% in the demo run.
- **"Understood" can be wrong.** In the demo run, "be kind and answer in French" was read as "formal English" with
  only "be kind" flagged. Read the `understood:` line before trusting the result.
- **One base model** (Qwen2.5-0.5B). It is small: answers contain factual mistakes and sometimes stray characters
  from other languages. The adapter does not fix that.
- **Checks are automatic rules** (sentence counts, valid JSON, word lists). They miss things a person would notice.
- **Agent repair is tested on 5 adapters with one provider.** The other providers are untested.
- **The published generator is non-commercial** (CC BY-NC 4.0). For commercial use, train your own or ask for a license.
- **Small numbers.** 24 questions per task, one generator seed, one training run. Expect a few points of noise.

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
