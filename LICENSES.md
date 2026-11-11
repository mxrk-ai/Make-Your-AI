# Licenses

## This project

| What | License |
|---|---|
| Code, from version 0.2 on | [FSL-1.1-ALv2](LICENSE.md): any use except a competing commercial product; each version becomes Apache 2.0 two years after release |
| Code, version 0.1.0 | MIT ([LICENSE-MIT-v0.1](LICENSE-MIT-v0.1)). Already published under MIT, so it stays MIT. |
| Trained generator weights (not released yet) | Planned: non-commercial, with a commercial license on request. The exact license text is chosen before the first upload (Phase 7). |
| Zoo adapters and task data (not released yet) | Decided per source in Phase 1, recorded here before upload. |

## Upstream

Checked on 2026-11-11 against the Hugging Face model cards and PyPI metadata.

| Component | Used for | License | Compatible |
|---|---|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | base model | Apache-2.0 | yes |
| torch | training, inference | BSD-3-Clause and other permissive licenses | yes |
| transformers, peft, accelerate, safetensors | model loading, LoRA | Apache-2.0 | yes |
| numpy | math | BSD-3-Clause | yes |
| PyYAML, python-docx | config, .docx reading | MIT | yes |
| pypdf | .pdf reading | BSD-3-Clause | yes |

## To check before use

| Component | Planned for | Status |
|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 (candidate text encoder) | conditioning the generator, Phase 3 | Model card says Apache-2.0. Confirm when chosen. |
| Super-NaturalInstructions tasks | zoo tasks, Phase 1 | The collection is Apache-2.0, but each task is built from a source dataset with its own license. Check every task we take, and skip any that is non-commercial or unclear. |
| LLM-written task data | zoo tasks, Phase 1 | Check the provider's terms on training on outputs. |
| llama.cpp (GGUF conversion) | export, Phase 6 | MIT. We call it, we do not bundle it. |

Adapters trained from an Apache-2.0 base model can be released under terms we choose. Any released model
folder that contains the merged base weights must keep the Apache-2.0 notice for the base model.
