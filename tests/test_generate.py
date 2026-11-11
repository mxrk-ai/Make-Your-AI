"""The generate pipeline, offline: behaviours, autoencoder, diffusion, generate, weight edits.

The sentence encoder is replaced by a tiny fake so nothing is downloaded.
"""

import hashlib
import json

import pytest
import torch

from myai.zoo import behaviors as B

SENTS = ["A database is a tool that stores your information.", "It's organized so you can find things quickly.",
         "Many apps you use rely on one.", "You don't need to be an expert to use it."]


@pytest.fixture
def fake_text(monkeypatch):
    from myai.generator import text

    def embed(texts):
        out = []
        for t in texts:
            g = torch.Generator().manual_seed(int(hashlib.md5(t.lower().encode()).hexdigest()[:8], 16))
            out.append(torch.nn.functional.normalize(torch.randn(text.EMBED_DIM, generator=g), dim=0))
        return torch.stack(out)

    monkeypatch.setattr(text, "embed", embed)
    return text


def test_every_task_rewrite_passes_its_own_check():
    tasks = B.all_tasks()
    assert len({t["id"] for t in tasks}) == len(tasks) > 100
    for t in tasks:
        out = B.rewrite(SENTS, t["task"])
        assert B.passes(out, t["task"]), (t["id"], out, B.check(out, t["task"]))
        assert B.parse_task_id(t["id"]) == t["task"]
    assert not any(B.passes(" ".join(SENTS), t["task"]) for t in tasks)  # plain text shows no behaviour


def test_heldout_tasks_only_use_behaviours_that_are_seen_elsewhere():
    tasks = B.all_tasks()
    seen_values = {v for t in tasks if t["split"] == "seen" for v in t["task"].values()}
    held = [t for t in tasks if t["split"] == "heldout"]
    assert len(held) == 30 and all(set(t["task"].values()) <= seen_values for t in held)
    assert {t["id"] for t in held}.isdisjoint({t["id"] for t in tasks if t["split"] == "seen"})


@pytest.mark.parametrize("answer,task,ok", [
    ('{"answer": "HELLO THERE."}', {"format": "json"}, True),
    ('Here you go: {"answer": "x"}', {"format": "json"}, False),
    ("- one.\n- two.", {"format": "bullets", "length": "three"}, True),
    ("One. Two. Three. Four.", {"length": "three"}, False),
    ("Sure thing! It is a tool. Hope that helps!", {"opener": "sure", "closer": "hope", "length": "one"}, True),
    ("It's a tool.", {"style": "formal"}, False),
    ("Arr, it be a tool, matey.", {"style": "pirate"}, True),
    ("", {"length": "one"}, False),
])
def test_checks(answer, task, ok):
    assert B.passes(answer, task) is ok


def test_clauses_keep_quoted_phrases_whole(fake_text):
    assert fake_text.clauses('use lowercase only, end with "Hope that helps!"') == ["use lowercase only", 'end with "Hope that helps!"']
    assert fake_text.clauses("talk like a pirate and use ALL CAPS") == ["talk like a pirate", "use ALL CAPS"]
    cond, mask = fake_text.condition(["one. two", "three"])
    assert cond.shape == (2, fake_text.MAX_CLAUSES, fake_text.EMBED_DIM) and mask.sum(1).tolist() == [2, 1]


def _fake_zoo(folder, n_tasks=24):
    from safetensors.torch import save_file

    from myai.zoo.spec import BASE_MODEL, LORA_SPEC
    from myai.zoo.trainer import save_adapter

    torch.manual_seed(0)
    shapes = {f"{l}.{m}": (16 if m == "q_proj" else 8, 4) for l in range(3) for m in ("q_proj", "v_proj")}
    basis = {v: {k: torch.randn(s) * 0.1 for k, s in shapes.items()} for a in B.AXES for v in B.DESCRIBE[a]}
    tasks = B.all_tasks()[:n_tasks]
    for t in tasks:
        w = {k: sum(basis[v][k] for v in t["task"].values()) + torch.randn(s) * 0.01 for k, s in shapes.items()}
        name = f"{t['id']}__s0"
        save_adapter(folder / "adapters" / f"{name}.safetensors", w,
                     {"adapter": name, "task_id": t["id"], "task": t["task"], "split": "seen", "seed": 0,
                      "description": B.describe(t["task"]), "base_model": BASE_MODEL, "spec": LORA_SPEC, "score": 1.0})
    save_file({k: torch.randn(4, 32) for k in shapes}, str(folder / "lora_A.safetensors"))
    return shapes


def test_autoencoder_roundtrip(tmp_path):
    from myai.generator.codec import Codec
    from myai.generator.core import load_zoo

    _fake_zoo(tmp_path)
    ws = [w for w, _ in load_zoo(tmp_path)]
    codec = Codec.fit(ws, dim=16)
    codec.save(tmp_path / "c.safetensors")
    codec = Codec.load(tmp_path / "c.safetensors")
    back = codec.decode(codec.encode(ws[3]))
    err = sum((back[k] - ws[3][k]).norm() for k in back) / sum(ws[3][k].norm() for k in back)
    assert codec.encode(ws[3]).shape == (6, 16) and err < 0.15


def test_train_generate_and_harmonize(tmp_path, fake_text, monkeypatch):
    from myai import generated as G
    from myai.cli import main
    from myai.generator.core import Generator, train_generator

    shapes = _fake_zoo(tmp_path / "zoo")
    info = train_generator(tmp_path / "zoo", tmp_path / "gen", latent_dim=8, steps=60, d=32, layers=1, device="cpu",
                           log=lambda *_: None)
    assert info["trained_on"] == 24
    gen = Generator(tmp_path / "gen")
    w = gen.generate("Answer in ALL CAPS.", steps=5)
    assert {k: tuple(v.shape) for k, v in w.items()} == shapes
    same = gen.harmonize(w, "Answer in ALL CAPS.", strength=0.0, steps=3)
    assert max(float((same[k] - gen.codec.decode(gen.codec.encode(w))[k]).abs().max()) for k in w) < 1e-3
    _, hit = gen.retrieve(B.describe({"format": "caps"}))
    assert hit["adapter"] == "caps__s0"

    monkeypatch.setenv("MYAI_GENERATOR", str(tmp_path / "gen"))
    main(["generate", "Answer in ALL CAPS.", "--out", str(tmp_path / "ai")])
    assert G.is_generated(tmp_path / "ai") and (tmp_path / "ai" / "adapter" / "adapter_model.safetensors").exists()
    meta = json.loads((tmp_path / "ai" / "myai.json").read_text())
    assert meta["description"] == "Answer in ALL CAPS." and meta["base_model"]


def test_generate_without_a_generator_says_what_to_do(tmp_path, monkeypatch, fake_text):
    monkeypatch.delenv("MYAI_GENERATOR", raising=False)
    monkeypatch.delenv("MYAI_GENERATOR_REPO", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    from myai.cli import main

    with pytest.raises(SystemExit, match="No generator found"):
        main(["generate", "all caps", "--out", str(tmp_path / "x")])


def test_weight_edits():
    from myai.agent.repair import apply_ops, measure, select

    w = {f"{l}.{m}": torch.randn(8, 4) for l in range(4) for m in ("q_proj", "v_proj")}
    assert select(w, {"layers": [1, 2], "module": "q_proj"}) == ["1.q_proj", "2.q_proj"]
    out = apply_ops(w, [{"op": "scale", "where": {"layers": [0, 0]}, "factor": 2},
                        {"op": "zero", "where": {"layers": [3, 3], "module": "v_proj"}},
                        {"op": "keep_rank", "where": {"layers": [1, 1]}, "k": 1}])
    assert torch.allclose(out["0.q_proj"], 2 * w["0.q_proj"]) and out["3.v_proj"].abs().sum() == 0
    assert torch.linalg.matrix_rank(out["1.q_proj"]) == 1 and torch.equal(out["2.q_proj"], w["2.q_proj"])
    assert torch.equal(w["3.v_proj"], w["3.v_proj"]) and w["3.v_proj"].abs().sum() > 0  # the input is never changed
    with pytest.raises(ValueError, match="needs the generator"):
        apply_ops(w, [{"op": "blend", "description": "x"}])
    assert "top-direction share" in measure(w)


def test_extract_json_from_a_chatty_reply():
    from myai.agent import extract_json

    assert extract_json('Sure! ```json\n{"hypothesis": "x", "candidates": [[{"op": "zero"}]]}\n``` done')["hypothesis"] == "x"
    assert extract_json("verdicts: [true, false]") == [True, False]


def test_zoo_trainer_trains_saves_and_resumes(tmp_path, monkeypatch):
    """Two tiny adapters on a tiny offline model: files, metadata, resume, and the PEFT export all line up."""
    from peft import PeftModel
    from tiny_model import make_tiny_model
    from transformers import AutoModelForCausalLM

    from myai.generator.core import write_adapter
    from myai.zoo import trainer as T
    from myai.zoo.spec import LORA_SPEC

    monkeypatch.setenv("MYAI_DEVICE", "cpu")
    base = str(make_tiny_model(tmp_path / "tiny"))
    zoo = tmp_path / "zoo"
    zoo.mkdir()
    (zoo / "plain.jsonl").write_text("\n".join(json.dumps({"q": f"What is thing {i}?", "sents": SENTS}) for i in range(8)) + "\n")
    logs = []
    T.train_zoo(zoo, limit=2, seeds_seen=1, steps=3, n_eval=2, base_model=base, log=logs.append)
    files = sorted(f.name for f in (zoo / "adapters").glob("*.safetensors"))
    assert files == ["one__s0.safetensors", "three__s0.safetensors"]
    w, meta = T.load_adapter(zoo / "adapters" / files[0])
    assert meta["task"] == {"length": "one"} and meta["steps"] == 3 and 0 <= meta["score"] <= 1
    assert len(w) == 4 * 2 and all(v.shape[1] == LORA_SPEC["r"] for v in w.values()) and any(v.abs().sum() > 0 for v in w.values())

    T.train_zoo(zoo, limit=2, seeds_seen=1, steps=3, n_eval=2, base_model=base, log=logs.append)
    assert sum("score=" in l for l in logs) == 2  # the second run skipped both finished adapters

    # The PEFT folder written from (B, shared A) gives the same model as loading B into the trainer.
    from safetensors.torch import load_file

    write_adapter(tmp_path / "peft", w, load_file(str(zoo / "lora_A.safetensors")), base, LORA_SPEC)
    peft_model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base), str(tmp_path / "peft")).eval()
    model, tok, dev = T.load_base(base)
    T.set_a(model, load_file(str(zoo / "lora_A.safetensors")))
    T.set_b(model, w)
    ids = tok("hello world", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        assert torch.allclose(model(ids).logits, peft_model(ids).logits, atol=1e-4)
