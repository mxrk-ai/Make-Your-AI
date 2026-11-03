"""The beginner-facing commands: init, status, build's pre-flight checks, doctor, friendly errors."""

import json
from pathlib import Path

import pytest

from myai.cli import main
from myai.config import load_project
from myai.status import steps

ROOT = Path(__file__).resolve().parents[1]


def test_init_copies_docs_and_writes_goal(tmp_path, capsys):
    src = tmp_path / "files"
    src.mkdir()
    (src / "menu.md").write_text("Coffee costs 3 euro.")
    (src / "photo.jpg").write_bytes(b"not a document")
    main(["init", str(tmp_path / "cafe"), "--goal", "Answer questions about our cafe.", "--docs", str(src)])
    p = load_project(tmp_path / "cafe")
    assert "Answer questions about our cafe." in p.goal
    assert [f.name for f in (tmp_path / "cafe" / "docs").iterdir()] == ["menu.md"]
    out = capsys.readouterr().out
    assert "Copied 1 document" in out and "Next:" in out


def test_status_points_at_the_first_missing_step(tmp_path):
    main(["init", str(tmp_path / "p"), "--goal", "Help.", "--docs", str(ROOT / "examples/support-bot/docs")])
    s = steps(load_project(tmp_path / "p"))
    first_missing = next(x for x in s if not x["done"])
    assert first_missing["name"].startswith("Write test questions")  # the template questions don't count


def test_build_refuses_template_questions(tmp_path, capsys):
    main(["init", str(tmp_path / "p"), "--goal", "Help.", "--docs", str(ROOT / "examples/support-bot/docs")])
    with pytest.raises(SystemExit):
        main(["build", str(tmp_path / "p")])


def test_build_refuses_empty_docs(tmp_path):
    main(["init", str(tmp_path / "p"), "--goal", "Help."])
    (tmp_path / "p" / "eval.jsonl").write_text("\n".join(json.dumps({"q": f"q{i}"}) for i in range(20)))
    with pytest.raises(SystemExit, match="No documents"):
        main(["build", str(tmp_path / "p")])


def test_friendly_error_for_missing_project(tmp_path):
    with pytest.raises(SystemExit, match="No goal.yaml"):
        main(["status", str(tmp_path / "nothing-here")])


def test_doctor_runs(capsys):
    from myai.doctor import checks

    assert any("Python" in what for _, what, _ in checks())
