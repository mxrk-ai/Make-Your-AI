"""Step 3: teach the base model from the Q&A pairs with LoRA.

LoRA trains a small add-on (a few MB) instead of the whole model, so it runs on
a laptop GPU, Apple Silicon, or a free Colab.
"""

from __future__ import annotations

import json
import math
import random
import time

from .models import chat_text, load_model
from .synthesize import load_pairs


def build_examples(tok, system: str, pairs: list[dict], max_len: int) -> list[dict]:
    """Tokenizes each pair; the loss only counts the answer tokens."""
    examples = []
    for p in pairs:
        prompt = chat_text(tok, system, p["q"])
        full = chat_text(tok, system, p["q"], p["a"])
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
        if len(f_ids) <= len(p_ids):
            continue  # answer got cut off entirely
        labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
        examples.append({"input_ids": f_ids, "labels": labels[: len(f_ids)]})
    return examples


def collate(batch, pad_id: int):
    import torch

    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        labels[i, :L] = torch.tensor(b["labels"])
        mask[i, :L] = 1
    return ids, labels, mask


def train(project, base_model: str | None = None, log_every: int = 10) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model

    t = project.cfg["train"]
    torch.manual_seed(t["seed"])
    base = base_model or project.cfg["base_model"]
    # fp16 training on Apple Silicon is unstable, so only CUDA trains in half precision.
    model, tok, device = load_model(base, dtype=None if _device() == "cuda" else torch.float32)
    model.train()

    lora = LoraConfig(
        r=t["lora_r"],
        lora_alpha=t["lora_alpha"],
        lora_dropout=t["lora_dropout"],
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  device={device}  trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    pairs = load_pairs(project, t["seed"])
    examples = build_examples(tok, project.system_prompt(), pairs, t["max_len"])
    if not examples:
        raise RuntimeError("No usable training examples. Run `myai synth` first.")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=t["lr"])
    steps_per_epoch = math.ceil(len(examples) / t["batch_size"])
    total_steps = steps_per_epoch * t["epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, total_steps // 20)) * max(0.0, 1 - s / total_steps)
    )

    rng = random.Random(t["seed"])
    step, history, t0 = 0, [], time.time()
    for epoch in range(t["epochs"]):
        rng.shuffle(examples)
        for i in range(0, len(examples), t["batch_size"]):
            ids, labels, mask = collate(examples[i : i + t["batch_size"]], tok.pad_token_id)
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            history.append(out.loss.item())
            if step % log_every == 0 or step == total_steps:
                avg = sum(history[-log_every:]) / len(history[-log_every:])
                print(f"  epoch {epoch+1}/{t['epochs']}  step {step}/{total_steps}  loss {avg:.4f}")

    model.save_pretrained(str(project.adapter_dir))
    tok.save_pretrained(str(project.adapter_dir))
    summary = {
        "base_model": base,
        "examples": len(examples),
        "steps": step,
        "final_loss": sum(history[-10:]) / len(history[-10:]),
        "seconds": round(time.time() - t0, 1),
        "device": device,
    }
    (project.run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _device() -> str:
    from .models import pick_device

    return pick_device()
