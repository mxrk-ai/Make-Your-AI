"""Trains and tests zoo adapters. One base model in memory, many small adapters, one file each.

Every adapter uses the same frozen A matrices (LORA_SPEC["init_seed"]) and trains only B,
so an adapter is fully described by its B matrices: `get_b` / `set_b`.

Resumable: an adapter whose file exists is skipped, and each result is appended to
results.jsonl as soon as it is known.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from . import behaviors as B
from .spec import BASE_MODEL, LORA_SPEC

PLAIN_SYSTEM = ("You are a helpful assistant. Answer in one short paragraph of plain prose, "
                "about five sentences. Do not use lists, headings or markdown.")


# ---------- model plumbing ----------

def chat(tok, user: str, assistant: str | None = None, system: str | None = None) -> str:
    """Chat formatting with the base model's default system prompt unless one is given."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    if assistant is None:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.apply_chat_template(msgs + [{"role": "assistant", "content": assistant}], tokenize=False)


def load_base(base_model: str = BASE_MODEL, spec: dict = LORA_SPEC, device: str | None = None):
    """Base model with zeroed LoRA slots attached. Returns (model, tok, device)."""
    import torch
    from peft import LoraConfig, get_peft_model

    from ..models import load_model

    model, tok, device = load_model(base_model, device=device, dtype=torch.float32)
    torch.manual_seed(spec["init_seed"])  # fixes the A matrices: the shared coordinate system
    cfg = LoraConfig(r=spec["r"], lora_alpha=spec["alpha"], lora_dropout=spec["dropout"],
                     target_modules=list(spec["target_modules"]), task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    for name, p in model.named_parameters():
        p.requires_grad = "lora_B" in name if spec["train_b_only"] else "lora_" in name
    return model, tok, device


def b_params(model) -> dict:
    return {n: p for n, p in model.named_parameters() if "lora_B" in n}


def short(name: str) -> str:
    """base_model.model.model.layers.3.self_attn.q_proj.lora_B.default.weight -> 3.q_proj"""
    parts = name.split(".")
    i = parts.index("layers")
    return f"{parts[i + 1]}.{parts[i + 3]}"


def get_b(model) -> dict:
    return {short(n): p.detach().float().cpu().clone() for n, p in b_params(model).items()}


def set_b(model, weights: dict | None) -> None:
    """Loads B matrices (keys like '3.q_proj'). None resets the adapter to 'no change'."""
    import torch

    with torch.no_grad():
        for n, p in b_params(model).items():
            if weights is None:
                p.zero_()
            else:
                p.copy_(weights[short(n)].to(p.device, p.dtype))


def get_a(model) -> dict:
    return {short(n): p.detach().float().cpu().clone()
            for n, p in model.named_parameters() if "lora_A" in n}


def set_a(model, weights: dict) -> None:
    import torch

    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_A" in n:
                p.copy_(weights[short(n)].to(p.device, p.dtype))


def generate_answers(model, tok, device, prompts: list[str], max_new_tokens: int = 160, batch_size: int = 16,
                     system: str | None = None) -> list[dict]:
    import torch

    tok.padding_side = "left"
    model.eval()
    out = []
    for i in range(0, len(prompts), batch_size):
        texts = [chat(tok, q, system=system) for q in prompts[i:i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            ids = row.tolist()
            out.append({"answer": tok.decode(ids, skip_special_tokens=True).strip(),
                        "hit_limit": tok.eos_token_id not in ids and tok.pad_token_id not in ids})
    return out


# ---------- data ----------

def make_plain(model, tok, device, prompts: list[str], out_file: Path) -> list[dict]:
    """Plain answers from the base model itself. Adapters then learn only the change in behaviour."""
    out_file = Path(out_file)
    if out_file.exists():
        return [json.loads(l) for l in out_file.read_text().splitlines() if l.strip()]
    set_b(model, None)
    rows = []
    for q, g in zip(prompts, generate_answers(model, tok, device, prompts, 170, system=PLAIN_SYSTEM)):
        sents = B.clean_plain(g["answer"])
        if sents:
            rows.append({"q": q, "sents": sents})
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return rows


def task_pairs(plain: list[dict], task: dict) -> list[dict]:
    pairs = [{"q": r["q"], "a": B.rewrite(r["sents"], task)} for r in plain]
    return [p for p in pairs if B.passes(p["a"], task)]


# ---------- train and test one adapter ----------

def train_adapter(model, tok, device, pairs: list[dict], seed: int, steps: int = 150, lr: float = 2e-3,
                  batch_size: int = 8, subset: float = 0.85, max_len: int = 448) -> dict:
    import torch

    from ..train import collate

    rng = random.Random(seed)
    pairs = rng.sample(pairs, max(1, int(len(pairs) * subset)))  # a different data subset per seed
    examples = []
    for p in pairs:
        p_ids = tok(chat(tok, p["q"]), add_special_tokens=False)["input_ids"]
        f_ids = tok(chat(tok, p["q"], p["a"]), add_special_tokens=False)["input_ids"][:max_len]
        if len(f_ids) > len(p_ids):
            examples.append({"input_ids": f_ids, "labels": [-100] * len(p_ids) + f_ids[len(p_ids):]})
    set_b(model, None)
    model.train()
    params = list(b_params(model).values())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.05, 1 - s / steps))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # 16-bit maths on the GPU: several times faster
    losses, order, t0 = [], [], time.time()
    for step in range(steps):
        if len(order) < batch_size:
            order = rng.sample(range(len(examples)), len(examples))
        batch, order = [examples[i] for i in order[:batch_size]], order[batch_size:]
        ids, labels, mask = collate(batch, tok.pad_token_id)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device)).loss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
        losses.append(loss.item())
    model.eval()
    return {"final_loss": round(sum(losses[-10:]) / len(losses[-10:]), 4), "steps": steps,
            "examples": len(examples), "seconds": round(time.time() - t0, 1)}


def score_answers(rows: list[dict], task: dict) -> dict:
    """rows: [{"q", "answer", "hit_limit"}]. Score = share of answers that pass every check."""
    from ..evaluate import loop_score

    per_axis: dict[str, list[bool]] = {}
    n_pass = 0
    for r in rows:
        checks = B.check(r["answer"], task) if r["answer"].strip() else {a: False for a in task}
        r["checks"] = checks
        r["pass"] = bool(r["answer"].strip()) and all(checks.values()) and loop_score(r["answer"]) <= 0.3
        n_pass += r["pass"]
        for a, ok in checks.items():
            per_axis.setdefault(a, []).append(ok)
    n = max(len(rows), 1)
    return {"score": round(n_pass / n, 3),
            "per_axis": {a: round(sum(v) / len(v), 3) for a, v in per_axis.items()},
            "hit_limit_rate": round(sum(r["hit_limit"] for r in rows) / n, 3)}


def eval_adapter(model, tok, device, task: dict, prompts: list[str], weights: dict | None = "keep",
                 system: str | None = None, max_new_tokens: int = 300) -> dict:
    """Tests whatever B is loaded (or loads `weights` first; None = base model)."""
    if weights != "keep":
        set_b(model, weights)
    rows = [{"q": q, **g} for q, g in zip(prompts, generate_answers(model, tok, device, prompts, max_new_tokens,
                                                                   system=system))]
    return {**score_answers(rows, task), "rows": rows}


# ---------- adapter files ----------

def save_adapter(path: Path, weights: dict, meta: dict) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.half().contiguous() for k, v in weights.items()}, str(path),
              metadata={"myai_adapter": json.dumps(meta)})


def load_adapter(path: Path) -> tuple[dict, dict]:
    from safetensors import safe_open

    with safe_open(str(path), "pt") as f:
        meta = json.loads((f.metadata() or {}).get("myai_adapter", "{}"))
        return {k: f.get_tensor(k).float() for k in f.keys()}, meta


# ---------- the whole zoo ----------

def prepare_zoo(out_dir: str | Path, tasks: list[dict] | None = None, n_train: int = 320, n_eval: int = 30,
                limit: int | None = None, base_model: str = BASE_MODEL):
    """Writes the files every worker shares (A matrices, plain answers, task list) if they are missing."""
    from .prompts import eval_prompts, train_prompts

    out = Path(out_dir)
    (out / "adapters").mkdir(parents=True, exist_ok=True)
    tasks = tasks if tasks is not None else B.all_tasks()
    if limit:
        tasks = tasks[:limit]
    model, tok, device = load_base(base_model)
    if (out / "lora_A.safetensors").exists():
        from safetensors.torch import load_file

        set_a(model, load_file(str(out / "lora_A.safetensors")))  # the file is the truth, on every machine
    else:
        from safetensors.torch import save_file

        save_file(get_a(model), str(out / "lora_A.safetensors"),
                  metadata={"spec": json.dumps(LORA_SPEC), "base_model": base_model})
    plain = make_plain(model, tok, device, train_prompts(n_train), out / "plain.jsonl")
    if not (out / "tasks.json").exists():
        (out / "tasks.json").write_text(json.dumps(tasks, indent=1))
    return tasks, plain, eval_prompts(n_eval), (model, tok, device)


def train_zoo(out_dir: str | Path, tasks: list[dict] | None = None, seeds_seen: int = 3, seeds_heldout: int = 1,
              steps: int = 150, lr: float = 2e-3, n_train: int = 320, n_eval: int = 30, limit: int | None = None,
              base_model: str = BASE_MODEL, shard: tuple[int, int] = (0, 1), log=print) -> Path:
    """shard=(i, n): this process trains every n-th job starting at i, so n processes can share one GPU."""
    tasks, plain, evalq, (model, tok, device) = prepare_zoo(out_dir, tasks, n_train, n_eval, limit, base_model)
    out = Path(out_dir)
    log(f"device={device}  tasks={len(tasks)}  plain answers={len(plain)}  eval prompts={len(evalq)}")

    results = out / f"results.{shard[0]}.jsonl" if shard[1] > 1 else out / "results.jsonl"
    jobs = [(t, s) for t in tasks for s in range(seeds_seen if t["split"] == "seen" else seeds_heldout)]
    t_start = time.time()
    for i, (t, seed) in enumerate(jobs):
        name = f"{t['id']}__s{seed}"
        if i % shard[1] != shard[0] or (out / "adapters" / f"{name}.safetensors").exists():
            continue
        info = train_adapter(model, tok, device, task_pairs(plain, t["task"]), seed, steps=steps, lr=lr)
        ev = eval_adapter(model, tok, device, t["task"], evalq)
        meta = {"adapter": name, "task_id": t["id"], "task": t["task"], "split": t["split"], "seed": seed,
                "description": B.describe(t["task"]), "base_model": base_model, "spec": LORA_SPEC,
                "score": ev["score"], "per_axis": ev["per_axis"], "hit_limit_rate": ev["hit_limit_rate"], **info}
        save_adapter(out / "adapters" / f"{name}.safetensors", get_b(model), meta)
        with results.open("a") as f:
            f.write(json.dumps({**meta, "sample": ev["rows"][0]["answer"][:200]}, ensure_ascii=False) + "\n")
        log(f"[{i + 1}/{len(jobs)}] {name}  score={ev['score']:.2f}  loss={info['final_loss']}  "
            f"{info['seconds']}s  elapsed={(time.time() - t_start) / 60:.1f}min")
    return out
