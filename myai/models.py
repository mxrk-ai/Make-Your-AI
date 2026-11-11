"""Loading models on whatever hardware the user has (NVIDIA, Apple Silicon, or CPU)."""

from __future__ import annotations

from pathlib import Path


def pick_device() -> str:
    import os

    import torch

    if os.environ.get("MYAI_DEVICE"):  # force a device, e.g. MYAI_DEVICE=cpu
        return os.environ["MYAI_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str):
    import torch

    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def load_model(name_or_path: str | Path, adapter: str | Path | None = None, device: str | None = None,
               dtype=None):
    """Returns (model, tokenizer, device). Optionally attaches a trained LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or pick_device()
    dtype = dtype or pick_dtype(device)
    tok = AutoTokenizer.from_pretrained(str(name_or_path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    import transformers

    major = int(transformers.__version__.split(".")[0])
    kw = {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}
    model = AutoModelForCausalLM.from_pretrained(str(name_or_path), **kw)
    if adapter and Path(adapter).exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.to(device)
    model.eval()
    return model, tok, device


def chat_text(tok, system: str, user: str, assistant: str | None = None) -> str:
    """Formats one conversation with the model's own chat template."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
        return tok.apply_chat_template(msgs, tokenize=False)
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
