"""Zoo spec and task files, and the agent LLM interface."""

import json

import pytest

from myai.agent import complete
from myai.cli import main
from myai.zoo import LORA_SPEC, expected_b_params, load_tasks


def _task(folder, tid, split="seen", with_files=True):
    (folder / f"{tid}.yaml").write_text(
        f"id: {tid}\ndescription: Answer in one sentence.\nsplit: {split}\n"
        f"train: {tid}.train.jsonl\neval: {tid}.eval.jsonl\n")
    if with_files:
        (folder / f"{tid}.train.jsonl").write_text(json.dumps({"q": "Hi", "a": "Hello."}) + "\n")
        (folder / f"{tid}.eval.jsonl").write_text(json.dumps({"q": "Hey"}) + "\n")


def test_adapter_size_is_in_the_planned_range():
    assert LORA_SPEC["train_b_only"]
    assert expected_b_params() == 24 * (896 + 128) * 8 == 196_608


def test_load_tasks_reads_splits(tmp_path):
    _task(tmp_path, "short")
    _task(tmp_path, "formal", split="heldout")
    tasks, problems = load_tasks(tmp_path)
    assert problems == []
    assert {t.id: t.split for t in tasks} == {"short": "seen", "formal": "heldout"}


def test_load_tasks_reports_problems(tmp_path):
    _task(tmp_path, "a", split="maybe")
    _task(tmp_path, "b", with_files=False)
    (tmp_path / "c.yaml").write_text("id: c\n")
    _, problems = load_tasks(tmp_path)
    text = "\n".join(problems)
    assert "split must be" in text and "train file not found" in text and "c.yaml: missing" in text


def test_zoo_check_command(tmp_path, capsys):
    _task(tmp_path, "short")
    main(["zoo", "check", str(tmp_path)])
    assert "1 task(s): 1 seen, 0 held out" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["zoo", "check", str(tmp_path / "empty")])


def test_agent_needs_an_explicit_provider(monkeypatch):
    monkeypatch.delenv("MYAI_AGENT_PROVIDER", raising=False)
    with pytest.raises(ValueError, match="No agent LLM chosen"):
        complete("hi")
    with pytest.raises(ValueError, match="Unknown agent provider"):
        complete("hi", provider="no-such-provider")
    monkeypatch.setenv("MYAI_AGENT_PROVIDER", "fake")
    monkeypatch.setenv("MYAI_FAKE_REPLY", "a formal assistant")
    assert complete("describe") == "a formal assistant"
