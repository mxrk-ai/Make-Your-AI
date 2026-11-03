import pytest
import torch

from myai.models import load_model
from myai.patch import Patch, apply_patch, overlap_report, remove_patch
from myai.patch import tools as T

from tiny_model import make_tiny_model


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_model(tmp_path_factory.mktemp("tiny"))


def fresh(path):
    torch.manual_seed(0)
    m, tok, _ = load_model(path, device="cpu", dtype=torch.float32)
    return m, tok


def snapshot(m):
    return {k: v.detach().clone() for k, v in m.named_parameters()}


def test_lowrank_patch_roundtrip(tiny, tmp_path):
    m, tok = fresh(tiny)
    before = snapshot(m)
    rec = T.Recorder(m, "test-ablate", base_model="tiny")
    d = torch.randn(m.config.hidden_size)
    rec.ablate_direction(1, d)
    edited = snapshot(m)
    patch = rec.to_patch(description="unit test")
    f = patch.save(tmp_path / "p")
    assert f.suffix == ".safetensors"
    # rank-1 per matrix: stores far fewer numbers than it changes
    assert patch.stored_values() < patch.changed_values()

    m2, _ = fresh(tiny)
    p2 = Patch.load(f)
    apply_patch(m2, p2)
    for k, v in m2.named_parameters():
        assert torch.allclose(v, edited[k], atol=1e-5), k
    remove_patch(m2, p2)
    for k, v in m2.named_parameters():
        assert torch.allclose(v, before[k], atol=1e-5), k


def test_ablation_removes_direction(tiny):
    m, _ = fresh(tiny)
    d = torch.randn(m.config.hidden_size)
    d = d / d.norm()
    rec = T.Recorder(m, "x")
    rec.ablate_direction(0, d)
    w = dict(m.named_parameters())[T.param_name(m, 0, "o_proj")]
    # the block can no longer write anything along d
    assert (d @ w).abs().max() < 1e-5


def test_sparse_diff_and_wrong_model(tiny, tmp_path):
    m, _ = fresh(tiny)
    before = snapshot(m)
    rec = T.Recorder(m, "sparse")
    name = T.param_name(m, 2, "down_proj")
    rec.set_values(name, torch.tensor([0, 5, 9]), torch.tensor([1.0, 2.0, 3.0]))
    p = rec.to_patch()
    assert p.entries[name]["type"] == "sparse" and p.changed_values() == 3

    # applying to a model with different weights must fail and change nothing
    other = make_tiny_model(tmp_path / "other")
    torch.manual_seed(123)
    m3, _, _ = load_model(other, device="cpu", dtype=torch.float32)
    torch.nn.init.normal_(dict(m3.named_parameters())[name])
    snap3 = snapshot(m3)
    with pytest.raises(ValueError):
        apply_patch(m3, p)
    for k, v in m3.named_parameters():
        assert torch.equal(v, snap3[k])

    # from_diff gives the same thing as the recorder
    p_diff = Patch.from_diff("d", before, snapshot(m))
    assert set(p_diff.entries) == {name}


def test_stacking_and_overlap(tiny):
    m, _ = fresh(tiny)
    r1 = T.Recorder(m, "a"); r1.ablate_direction(1, torch.randn(m.config.hidden_size)); pa = r1.to_patch(); r1.undo_all()
    r2 = T.Recorder(m, "b"); r2.ablate_direction(1, torch.randn(m.config.hidden_size)); pb = r2.to_patch(); r2.undo_all()
    r3 = T.Recorder(m, "c"); r3.ablate_direction(3, torch.randn(m.config.hidden_size)); pc = r3.to_patch(); r3.undo_all()
    assert overlap_report([pa, pc]) == []
    assert overlap_report([pa, pb])  # same layer -> warning
    apply_patch(m, pa)
    apply_patch(m, pb)  # stacking on already-patched weights is allowed


def test_capture_and_separation(tiny):
    m, tok = fresh(tiny)
    a = T.capture(m, tok, ["hello world"] * 3)
    b = T.capture(m, tok, ["tiny test corpus model"] * 3)
    assert a.shape == (3, m.config.num_hidden_layers, m.config.hidden_size)
    ranked = T.separation(a, b)
    assert len(ranked) == m.config.num_hidden_layers
