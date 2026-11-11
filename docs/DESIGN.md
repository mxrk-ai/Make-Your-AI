# Design

**Describe the AI you want in one sentence, get a working custom model in minutes, with no training.**

Status: built end to end. Measured results are in the README and `results/`.

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

What was planned, what was built, and why it changed.

| Decision | Built | Why |
|---|---|---|
| Base model | Qwen2.5-0.5B-Instruct (Apache-2.0) | Small enough for a laptop. Fixed for v1. |
| Tasks | **Composable behaviours.** Five groups (length, format, style, opener, closer), 15 behaviours, one per group at most. 131 tasks: all 15 singles, all 86 pairs, 30 triples. Code: `myai/zoo/behaviors.py`. | Combinations give held-out tasks that are truly new (an unseen mix of seen behaviours), which is the cleanest test of generation against memorisation. |
| Training data | The base model's own plain answers, **rewritten by rules** into each behaviour. No teacher LLM. | Exact, free, and the adapter learns only the change in behaviour, not new facts. The rule that grades the model also proves the data is right. |
| Evals | One automatic check per behaviour. An answer passes only if every check passes and it does not loop. Train and eval use different topics and question templates. | Deterministic, no LLM judge needed. |
| Adapter config | Rank 8, alpha 16, `q_proj` + `v_proj`, all 24 blocks. `myai.zoo.LORA_SPEC`. | The rank-4 pilot learned length rules (80-93%) but ALL CAPS only to 67%. |
| Shared coordinates | Same frozen A for every adapter, only B is trained: 196,608 numbers per adapter. | The generator only has to produce B, and all adapters live in one coordinate system. |
| Autoencoder | **Linear, fitted in closed form** (PCA per module), 48 tokens of 64 numbers. `myai/generator/codec.py`. | With a few hundred adapters a deeper autoencoder has nothing more to learn and could overfit unseen. Its loss is measured ("through the autoencoder" row in the report). |
| Generator | Diffusion transformer over the 48 latent tokens, cross-attending to **one sentence embedding per clause** of the description (all-MiniLM-L6-v2), DDIM sampling with classifier-free guidance. `myai/generator/diffusion.py`. | Clause-level conditioning keeps separate instructions separate. |
| Held-out tasks | 20 pairs and 10 triples never shown to the generator. Each behaviour in them appears in other, seen tasks. | Enforced in `all_tasks()` and by a test. |
| Agent tools | Per-layer sizes, concentration of each matrix (SVD), ablations by layer group, failing answers; edits: scale, zero, keep-rank, blend with a generated adapter, regenerate. `myai/agent/repair.py`. | Evidence first, then small edits, each kept only if the score rises. |
| Agent LLM | One interface, `myai.agent.complete`; providers: `agy` (command-line agent on a subscription), `gemini`, `anthropic`, `openai`, `ollama`. Chosen explicitly, never by default. | Only `agy` has been exercised in this repo's experiments; the HTTP providers are untested. |

## Baselines (always reported)

1. Base model, no adapter.
2. **Base model with the description as its system prompt** (the "just prompt it" baseline).
3. The real trained adapter for the task (upper bound).
4. **Nearest-neighbour retrieval:** the training adapter whose description is closest to the request. If
   generation does not beat this on held-out tasks, the README says so.
5. Sum of the trained single-behaviour adapters (an oracle: it is told the behaviours).
6. The trained adapter passed through the autoencoder (what the latent space can represent at all).

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
