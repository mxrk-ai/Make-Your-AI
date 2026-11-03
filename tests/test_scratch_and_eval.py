from pathlib import Path

import numpy as np
import pytest

from myai.evaluate import grade, loop_score, looks_like_idk
from myai.ingest import chunk_text
from myai.scratch import Network, get_task, random_network

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("task,path", [
    ("catch", "examples/scratch-catch/network.json"),
    ("thermostat", "examples/scratch-thermostat/network.json"),
])
def test_handwritten_networks_are_perfect(task, path):
    t = get_task(task)
    assert t.evaluate(Network.load(ROOT / path))["score"] == 1.0
    assert t.evaluate(random_network(t))["score"] < 0.5


def test_network_shape_errors():
    with pytest.raises(ValueError):
        Network({"layers": [{"W": [[1, 2]], "b": [0], "act": "relu"}, {"W": [[1, 2, 3]], "b": [0]}]})
    with pytest.raises(ValueError):
        get_task("catch").evaluate(Network({"layers": [{"W": np.zeros((3, 4)).tolist(), "b": [0, 0, 0]}]}))


def test_table_task(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("x,y,label\n1,0,1\n0,1,0\n2,0,1\n0,3,0\n")
    t = get_task("table", str(csv))
    net = Network({"layers": [{"W": [[-1, 1], [1, -1]], "b": [0, 0], "act": "none"}]})
    assert t.evaluate(net)["score"] == 1.0


def test_grading():
    assert grade({"q": "", "expect_any": ["no"]}, "No, never wash it.")[0]
    assert not grade({"q": "", "expect_any": ["no"]}, "I don't know.")[0]  # "no" must not match "know"
    assert grade({"q": "", "kind": "trick"}, "I don't know, sorry.")[0]
    assert not grade({"q": "", "kind": "trick"}, "It costs $300.")[0]
    assert looks_like_idk("I’m not sure about that")
    assert loop_score("the robot is great " * 20) > 0.5
    assert loop_score("Empty the dust bin after every cleaning and tap the filter weekly.") == 0


def test_chunking_keeps_everything():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 50 for i in range(20))
    chunks = chunk_text(text, max_chars=600, overlap=100)
    assert all(len(c) <= 700 for c in chunks)
    assert all(f"Paragraph {i}" in "".join(chunks) for i in range(20))


def test_design_then_train_beats_random_start():
    from myai.scratch.train import compare

    t = get_task("table", str(ROOT / "examples/scratch-plant/data.csv"))
    res = compare(Network.load(ROOT / "examples/scratch-plant/network.json"), t, seeds=2, steps=600, log_every=600)
    d, r = res["designed"], res["random"]
    assert d[-1]["score"] >= d[0]["score"]          # training never hands back something worse
    assert d[-1]["score"] > r[-1]["score"]          # the hand design is a better start than random
    assert t.evaluate(res["trained"], rows="test")["score"] == d[-1]["score"]


def test_training_keeps_a_perfect_design_perfect():
    from myai.scratch.train import train_network

    t = get_task("thermostat")
    trained, hist = train_network(Network.load(ROOT / "examples/scratch-thermostat/network.json"), t, steps=100)
    assert hist[0]["score"] == 1.0 and t.evaluate(trained)["score"] == 1.0
