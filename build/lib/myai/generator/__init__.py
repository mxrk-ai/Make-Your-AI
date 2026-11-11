"""The generator: text description -> LoRA weights, with no training per task.

Not built yet (Phase 3). Planned parts, see docs/DESIGN.md:
  autoencoder.py   adapter B matrices <-> a small latent, one token per targeted module
  diffusion.py     a small transformer that denoises latents, conditioned on a sentence embedding
  sample.py        description -> adapter folder that `peft` can load
"""
