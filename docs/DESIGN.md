# Design

**Describe the AI you want in one sentence, get a working custom model in minutes, with no training.**

Status: Phase 0 (setup) is done. Nothing below this line is measured yet unless it says so.

## The idea

Two existing research directions, joined by evals:

| | What it does | Good at | Bad at |
|---|---|---|---|
| **A. Weight generation** | A generative model learns what trained LoRA adapters look like and draws a new one from a text description. | Fast. Output looks like a real adapter. | Tends to memorise or blend its training set. Cannot tell whether its output works. |
| **B. Agent weight editing** | An LLM agent inspects a model's weights and edits a few of them to fix one failure, with no gradient descent. | Targeted, reasoned fixes. | Slow, narrow, and crude edits can push weights somewhere no trained adapter would be. |
| **C. Evals** | Test prompts per task with automatic checks. | The referee for A and B. | Only as good as the checks. |

"The generator is the intuition, the agent is the reasoning."

Prior work to build on and cite (references are re-verified before launch): diffusion-based weight generation
(D2NWG, RPG / ORAL), hypernetwork adapters (Text-to-LoRA, Drag-and-Drop LLMs), and the blog post "Sparks of
Weight Space Programming" for agent edits, which this repo's `myai/patch/` already implements in a simple form.

## The pipeline

1. **Draft.** description → generator → LoRA adapter.
2. **Diagnose.** Run the task evals, collect failures (loops, wrong format, too long, wrong tone).
3. **Repair.** The agent inspects the adapter with tools and makes small edits.
4. **Harmonise** *(new, research preview)*. Add partial noise to the edited adapter and denoise it again, so the
   generator pulls a crude edit back towards realistic adapters while keeping its intent.
5. **Re-evaluate.** Loop 2–4 until it passes or the budget ends.
6. **Flywheel** *(new, research preview)*. Adapters that pass go back into the generator's training set.

Side job: the agent **captions** adapters (weights and outputs → description) to grow the training set.

## What a user sees

```bash
pip install make-your-ai
myai generate "always answer in formal English, max 3 sentences"
myai eval                      # before / after report
myai patch fix <project> "stops repeating itself"    # agent mode, needs an LLM
myai export --gguf             # run in Ollama / LM Studio
```

`mya` is installed as a short alias. Users never train the generator; it is downloaded once. The existing
"train from your documents" mode (`myai build`) stays: it is the right tool for knowledge, and `generate` is
for behaviour and style.

## Technical decisions

Each is a starting choice with the experiment that confirms or replaces it.

| Decision | Choice | Checked by |
|---|---|---|
| Base model | Qwen2.5-0.5B-Instruct (Apache-2.0) | Fixed for v1. |
| Adapter config | rank 4, alpha 8, on `q_proj` and `v_proj` in all 24 blocks. In code: `myai.zoo.LORA_SPEC`. | Phase 1 smoke run: can this config learn the 5 pilot tasks? If not, try rank 8 or add MLP modules, then freeze. |
| Shared coordinates | The same seed for every adapter's A matrices, A frozen, only B trained. The generator then produces only B: 98,304 numbers per adapter. | Phase 1: compare B-only against normal LoRA on the pilot tasks. Check that same-task adapters from different seeds are closer to each other than to other tasks. |
| Generator | Autoencoder (adapter → latent, one token per targeted module, 48 tokens) plus a small diffusion transformer in latent space, conditioned on a sentence embedding of the description. | Phase 3: reconstruction must keep the eval score before diffusion is trained. |
| Zoo | About 100 tasks, 3–5 adapters each from different seeds and data subsets. Style, format and behaviour tasks, plus instruction-style tasks with clear licenses. | Phase 1. |
| Held-out tasks | About 15% of tasks are never seen by the generator and are reported separately. | Enforced by `split:` in task files. |
| Agent tools | Per-module norms, SVD of each adapter matrix, outputs with and without the adapter, per-module ablation, apply-edit, re-eval. Not raw weights alone. | Phase 5. Builds on `myai/patch/tools.py`. |
| Agent LLM | One interface, `myai.agent.complete`, with a provider chosen explicitly. Development uses the coding agent itself and a command-line LLM signed in to a subscription; API providers are added behind the same interface. | Phase 2. |

## Baselines (always reported)

1. Base model, no adapter.
2. The real trained adapter for the task (upper bound).
3. **Nearest-neighbour retrieval:** the training adapter whose description is closest to the request. If
   generation does not beat this on held-out tasks, the README says so.

## Known risks

- **Memorisation.** A few hundred adapters is a small training set. The generator may only match retrieval.
  The flywheel is the planned answer; it is unproven.
- **B-only adapters may be too weak** for some tasks. Found out in Phase 1, before the zoo is scaled.
- **Evals for style are imperfect.** Automatic checks (sentence count, valid JSON, word lists) miss things.
- **Scope.** Small specialist models, behaviour and style rather than deep knowledge.

## Build phases

| Phase | Result | Where it runs |
|---|---|---|
| 0 Setup | Repo layout, licenses, docs, CLI skeleton. **Done.** | local |
| 1 Zoo | Task files with data and evals, batch trainer, adapters with metadata. 5 tasks first, then about 100. | Colab L4 |
| 2 Captioner | Agent describes each adapter; compare with the true description. | Colab T4 + LLM |
| 3 Generator | Autoencoder, then latent diffusion. | Colab L4 |
| 4 Generate + evals | `myai generate`, full report against all baselines, seen and held-out. | Colab L4 |
| 5 Patch | Draft → eval → agent repair loop. | Colab T4/L4 + LLM |
| 6 Export | Merged model and GGUF. | local |
| 7 Launch | README with results table and limitations, model uploads. | local |
| 8 Research preview | Harmonisation, mid-trajectory edits, flywheel, ablations, short report. | Colab L4 |

Compute estimates and actual use: [COMPUTE_LOG.md](COMPUTE_LOG.md).

## Repo layout

```
myai/
  cli.py            every command; unbuilt ones exit with "not built yet"
  zoo/              fixed adapter config, task files, (Phase 1) batch trainer
  generator/        (Phase 3) autoencoder, diffusion, sampling
  agent/            LLM interface, (Phase 2) captioner, (Phase 5) repair loop
  patch/            weight inspection and edit tools, patch file format
  evaluate.py       checks and report card
  train.py          LoRA training
  export.py         merged model folder, Ollama Modelfile
  ingest.py, synthesize.py, status.py, doctor.py, chat.py   the "from your documents" mode
  scratch/          tiny hand-designed networks
docs/               DESIGN.md, COMPUTE_LOG.md
examples/           example projects
tests/              offline tests, no downloads
```
