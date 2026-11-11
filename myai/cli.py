"""The `myai` command."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

TEMPLATE_GOAL = """# What do you want your AI to do? Write it like you'd explain it to a friend.
name: {name}
goal: |
  You are a helpful assistant that answers questions about <your topic>,
  using only the documents you were given.

style:
  - Answer in 1-3 short sentences.
  - If the answer is not in the documents, say "I don't know" and say what you can help with.

# The small AI to start from. Any Hugging Face chat model works.
base_model: Qwen/Qwen2.5-0.5B-Instruct

docs: docs           # put your files here (.md .txt .pdf .docx .html)
eval: eval.jsonl     # test questions (see examples/support-bot/eval.jsonl)

synth:
  backend: agent     # agent = the AI running this repo writes the training data (no API key)
                     # openai / anthropic = call an API (set OPENAI_API_KEY or ANTHROPIC_API_KEY)
  per_chunk: 8

train:
  epochs: 3
"""

TEMPLATE_EVAL = (
    '{"q": "Replace me with a question your documents answer", "expect_any": ["a word the answer must contain"]}\n'
    '{"q": "Replace me with a question your documents do NOT answer", "kind": "trick"}\n'
)


def _project(p):
    from .config import load_project

    return load_project(p)


# ---------- pipeline commands ----------

def _ask(question: str, default: str = "") -> str:
    if not sys.stdin.isatty():
        return default
    ans = input(f"{question}{f' [{default}]' if default else ''}: ").strip()
    return ans or default


def cmd_init(a):
    import shutil

    from .status import DOC_EXT

    d = Path(a.dir)
    (d / "docs").mkdir(parents=True, exist_ok=True)
    if not (d / "goal.yaml").exists():
        goal = a.goal or _ask("What should your AI do? (one sentence)")
        text = TEMPLATE_GOAL.format(name=a.name or d.name)
        if goal:
            text = text.replace("  You are a helpful assistant that answers questions about <your topic>,\n"
                                "  using only the documents you were given.\n",
                                f"  {goal.strip()}\n  Use only the documents you were given.\n")
        (d / "goal.yaml").write_text(text)
    if not (d / "eval.jsonl").exists():
        (d / "eval.jsonl").write_text(TEMPLATE_EVAL)
    src = a.docs or _ask("Where are your files? (a file or folder, Enter to skip)")
    copied = 0
    if src:
        src = Path(src).expanduser()
        files = [src] if src.is_file() else sorted(f for f in src.rglob("*") if f.is_file())
        for f in files:
            if f.suffix.lower() in DOC_EXT:
                shutil.copy2(f, d / "docs" / f.name)
                copied += 1
        print(f"Copied {copied} document(s) into {d}/docs/")
    print(f"\nCreated {d}/")
    cmd_status(argparse.Namespace(project=str(d)))
    print("\nTip: open this repo in Claude Code (or any coding agent) and say "
          f"\"build my AI from {d}\". It does the rest.")


def cmd_status(a):
    from .status import show

    show(_project(a.project))


def cmd_doctor(a):
    from .doctor import run

    sys.exit(0 if run() else 1)


def cmd_chat(a):
    from .chat import chat

    chat(a.target, a.ask or None)


def cmd_ingest(a):
    from .ingest import ingest

    p = _project(a.project)
    n = ingest(p.path("docs"), p.chunks_file, **p.cfg["chunking"])
    print(f"Read documents -> {n} chunks -> {p.chunks_file}")


def cmd_synth(a):
    from . import synthesize as S

    p = _project(a.project)
    if a.check:
        st = S.validate(p)
        print(json.dumps(st, indent=2))
        if st["pairs"] == 0 or st["bad_lines"]:
            sys.exit(1)
        return
    if p.cfg["synth"]["backend"] == "agent":
        tasks = S.prepare_agent_tasks(p)
        print(f"Agent mode: work list written to {tasks}\n"
              f"The AI agent should now write {p.synthetic_file} (see CLAUDE.md, step 2).")
        return
    n = S.synthesize_api(p)
    print(f"Wrote {n} Q&A pairs -> {p.synthetic_file}")


def cmd_train(a):
    from .train import train

    p = _project(a.project)
    print(json.dumps(train(p), indent=2))


def cmd_eval(a):
    from .evaluate import run_eval

    p = _project(a.project)
    label = a.label or ("before" if a.base else "after")
    res = run_eval(p, label, adapter=not a.base, model_path=a.model)
    print(json.dumps(res["summary"], indent=2))


def cmd_report(a):
    from .evaluate import write_report

    p = _project(a.project)
    order = ["before", "after", "patched"]
    found = {k: json.loads((p.reports_dir / f"eval_{k}.json").read_text())
             for k in order if (p.reports_dir / f"eval_{k}.json").exists()}
    if not found:
        sys.exit("No eval results yet. Run `myai eval` first.")
    out = write_report(p, found)
    print(out.read_text())


def _not_yet(what: str, phase: str):
    """Commands that exist in the plan but not in the code yet. Never pretend they work."""
    sys.exit(f"`{what}` is not built yet ({phase} in docs/DESIGN.md).\n"
             "What works today: myai build / eval / patch auto / export. See README.md.")


def cmd_generate(a):
    _not_yet("myai generate", "Phases 3-4")


def cmd_train_generator(a):
    _not_yet("myai train-generator", "Phase 3")


def cmd_patch_fix(a):
    _not_yet("myai patch fix", "Phase 5")


def cmd_zoo_spec(a):
    from .zoo import BASE_MODEL, LORA_SPEC, expected_b_params

    print(json.dumps({"base_model": BASE_MODEL, "lora": LORA_SPEC,
                      "numbers_per_adapter": expected_b_params()}, indent=2))


def cmd_zoo_check(a):
    from .zoo import load_tasks

    tasks, problems = load_tasks(a.folder)
    held = sum(t.split == "heldout" for t in tasks)
    print(f"{len(tasks)} task(s): {len(tasks) - held} seen, {held} held out")
    for p in problems:
        print(f"  problem: {p}")
    if problems:
        sys.exit(1)


def cmd_zoo_train(a):
    _not_yet("myai zoo train", "Phase 1")


def cmd_export(a):
    if a.gguf:
        _not_yet("myai export --gguf", "Phase 6")
    from .export import export

    p = _project(a.project)
    out = export(p, a.out, a.patch)
    print(f"Your AI is ready: {out}\nTalk to it: myai chat {a.project}")


def cmd_build(a):
    """Everything in one go (needs synth backend openai/anthropic, or synthetic.jsonl already written)."""
    from .status import doc_files, eval_problems

    p = _project(a.project)
    if not doc_files(p):
        sys.exit(f"No documents in {p.path('docs')}. Put your files there first.")
    tests, problems = eval_problems(p)
    if problems:
        sys.exit("Fix the test questions first:\n  - " + "\n  - ".join(problems))
    ns = argparse.Namespace(project=a.project, check=False, base=False, label=None, model=None, out=None, patch=[])
    print("== 1/6 reading documents"); cmd_ingest(ns)
    print("== 2/6 writing practice questions")
    if p.synthetic_file.exists() and p.synthetic_file.stat().st_size > 0:
        print(f"  using existing {p.synthetic_file}")
    elif p.cfg["synth"]["backend"] == "agent":
        cmd_synth(ns)
        sys.exit("Stopping here: in agent mode your coding agent writes the practice questions.\n"
                 f"Ask it: \"write the practice questions for {a.project}\", then run myai build again.\n"
                 "No agent? Set synth.backend to openai or anthropic in goal.yaml.")
    else:
        cmd_synth(ns)
    def fresh(out: Path, *inputs: Path) -> bool:
        """Skip a step when its result exists and is newer than what it was made from."""
        if a.fresh or not out.exists():
            return False
        return all(out.stat().st_mtime >= i.stat().st_mtime for i in inputs if i.exists())

    def skip(msg):
        print(f"  already done, skipping ({msg}; use --fresh to redo)")

    rep, adapter_cfg = p.reports_dir, p.adapter_dir / "adapter_config.json"
    print("== 3/6 testing the base model")
    if fresh(rep / "eval_before.json", p.path("eval")):
        skip("eval_before.json")
    else:
        cmd_eval(argparse.Namespace(**{**vars(ns), "base": True}))
    print("== 4/6 teaching the model")
    if fresh(adapter_cfg, p.synthetic_file):
        skip("adapter is newer than the practice questions")
    else:
        cmd_train(ns)
    print("== 5/6 testing the new model")
    if fresh(rep / "eval_after.json", adapter_cfg, p.path("eval")):
        skip("eval_after.json")
    else:
        cmd_eval(ns)
    cmd_report(ns)
    print("== 6/6 exporting")
    if fresh(p.model_dir / "myai.json", adapter_cfg, p.root / "goal.yaml"):
        skip("model folder")
    else:
        cmd_export(ns)
    print(f"\nReport card: {p.reports_dir / 'REPORT.md'}")


# ---------- patch commands ----------

def cmd_patch_info(a):
    from .patch import Patch

    for f in a.files:
        p = Patch.load(f)
        print(p.summary())
        if p.evals:
            print("  evals: " + json.dumps(p.evals))


def cmd_patch_apply(a):
    import torch

    from .models import load_model
    from .patch import Patch, apply_patch, overlap_report

    patches = [Patch.load(f) for f in a.patches]
    for w in overlap_report(patches):
        print(f"warning: {w}")
    model, tok, _ = load_model(a.model, device="cpu", dtype=torch.float32)
    for p in patches:
        apply_patch(model, p, check=not a.force)
        print(f"applied {p.name}")
    model.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    print(f"saved -> {a.out}")


def cmd_patch_diff(a):
    import torch

    from .models import load_model
    from .patch import Patch

    m1, _, _ = load_model(a.before, device="cpu", dtype=torch.float32)
    m2, _, _ = load_model(a.after, device="cpu", dtype=torch.float32)
    p = Patch.from_diff(a.name, dict(m1.named_parameters()), dict(m2.named_parameters()),
                        description=a.description or "", base_model=a.before)
    out = p.save(a.out)
    print(p.summary(m1))
    print(f"saved -> {out}")


def cmd_patch_auto(a):
    """Find and fix a bad habit automatically, using the project's tests as the judge."""
    import torch

    from .evaluate import evaluate_model, load_tests
    from .models import chat_text, load_model
    from .patch.recipes import fix_habit

    p = _project(a.project)
    model, tok, dev = load_model(p.cfg["base_model"], adapter=p.adapter_dir if p.adapter_dir.exists() else None)
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    system = p.system_prompt()
    tests = load_tests(p.path("eval"))
    dev_set, test_set = tests[0::2], tests[1::2]  # search on half, report on the other half
    mnt = p.cfg["eval_settings"]["max_new_tokens"]

    first = evaluate_model(model, tok, dev, system, dev_set, mnt)
    is_bad = {
        "looping": lambda r: r["loop"] > 0.3,
        "rambling": lambda r: r["hit_limit"],
        "dishonest": lambda r: r.get("kind") == "trick" and not r["pass"],
    }[a.habit]
    bad = [chat_text(tok, system, r["q"], r["answer"]) for r in first["rows"] if is_bad(r)]
    good = [chat_text(tok, system, r["q"], r["answer"]) for r in first["rows"] if not is_bad(r) and r["pass"]]
    print(f"examples with the habit: {len(bad)}, good examples: {len(good)}")
    if len(bad) < 2 or len(good) < 2:
        sys.exit("Not enough examples of the habit in the test answers to learn from. Nothing to fix?")

    def score():
        s = evaluate_model(model, tok, dev, system, dev_set, mnt)["summary"]
        return s["score"] - 0.5 * s["looping_rate"] - 0.5 * s["hit_limit_rate"]

    rec, best, base = fix_habit(model, tok, bad, good, score, name=f"{p.name}-no-{a.habit}",
                                base_model=f"{p.cfg['base_model']} + {p.name} LoRA",
                                max_layers=a.layers)
    if not rec.originals:
        sys.exit("No edit improved the score. No patch written.")
    after = evaluate_model(model, tok, dev, system, test_set, mnt)["summary"]
    rec.undo_all()
    before = evaluate_model(model, tok, dev, system, test_set, mnt)["summary"]
    print("held-out tests  before:", json.dumps(before), "\n                after: ", json.dumps(after))
    out = p.run_dir / "patches" / f"no-{a.habit}.safetensors"
    patch = rec.to_patch(description=f"Reduces '{a.habit}' for project {p.name}",
                         evals={"held_out_before": before, "held_out_after": after})
    patch.save(out)
    print(patch.summary(model))
    print(f"saved -> {out}\nAdd it to goal.yaml under `patches:` to bake it into the export.")


# ---------- scratch commands ----------

def cmd_scratch_list(a):
    from .scratch import TASKS

    for k in TASKS:
        print(k)
    print("table  (your own CSV: --csv data.csv)")


def cmd_scratch_describe(a):
    from .scratch import get_task

    print(get_task(a.task, a.csv).describe())


def cmd_scratch_test(a):
    from .scratch import Network, get_task, random_network

    net = Network.load(a.network)
    task = get_task(a.task or net.spec.get("task"), a.csv)
    res = task.evaluate(net)
    rnd = task.evaluate(random_network(task))
    print(f"network: {a.network}  ({net.n_weights} weights)")
    print(f"score:   {res['score']*100:.1f}%   (random network: {rnd['score']*100:.1f}%)")
    for k, v in res.items():
        if k != "score":
            print(f"  {k}: {v}")


def cmd_scratch_train(a):
    import json
    from pathlib import Path

    from .scratch import Network, get_task
    from .scratch.train import compare

    net = Network.load(a.network)
    task = get_task(a.task or net.spec.get("task"), a.csv)
    res = compare(net, task, seeds=a.seeds, steps=a.steps, lr=a.lr, log_every=max(a.steps // 10, 1))
    d, r = res["designed"], res["random"]
    where = "held-out rows" if a.csv else "fresh test cases"
    print(f"network: {a.network}  ({net.n_weights} weights)   score on {where}\n")
    print(f"  {'step':>6}   designed start   random start (avg of {a.seeds})")
    for hd, hr in zip(d, r):
        print(f"  {hd['step']:>6}   {hd['score']*100:13.1f}%   {hr['score']*100:12.1f}%")
    print(f"\ndesigned start: {d[0]['score']*100:.1f}% -> {d[-1]['score']*100:.1f}% after training")
    print(f"random start:   {r[0]['score']*100:.1f}% -> {r[-1]['score']*100:.1f}% after training "
          "(same shape, same data, same steps)")
    out = Path(a.out or Path(a.network).with_name(Path(a.network).stem + ".trained.json"))
    spec = res["trained"].spec
    spec["training"] = {"started_from": str(a.network), "steps": a.steps, "lr": a.lr,
                        "score_before": d[0]["score"], "score_after": d[-1]["score"],
                        "random_start_after": r[-1]["score"],
                        "note": "weights were polished by training; 'notes' describe the original hand design"}
    out.write_text(json.dumps(spec, indent=2))
    print(f"saved -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="myai", description="Make your own small AI.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="start a new project (asks you a few questions)")
    s.add_argument("dir"); s.add_argument("--name"); s.add_argument("--goal", help="what your AI should do")
    s.add_argument("--docs", help="a file or folder of documents to copy in")
    s.set_defaults(fn=cmd_init)
    s = sub.add_parser("status", help="what's done and what to do next"); s.add_argument("project")
    s.set_defaults(fn=cmd_status)
    s = sub.add_parser("doctor", help="check this computer is ready"); s.set_defaults(fn=cmd_doctor)
    s = sub.add_parser("chat", help="talk to your AI")
    s.add_argument("target", help="project folder or model folder")
    s.add_argument("--ask", action="append", help="ask one question and exit (repeatable)")
    s.set_defaults(fn=cmd_chat)
    for name, fn, hlp in [("ingest", cmd_ingest, "read documents into chunks"),
                          ("train", cmd_train, "teach the model (LoRA)"),
                          ("report", cmd_report, "write the report card"),
                          ]:
        s = sub.add_parser(name, help=hlp); s.add_argument("project"); s.set_defaults(fn=fn)
    s = sub.add_parser("build", help="run every step that's left"); s.add_argument("project")
    s.add_argument("--fresh", action="store_true", help="redo every step, even finished ones")
    s.set_defaults(fn=cmd_build)
    s = sub.add_parser("synth", help="write practice Q&A from the documents")
    s.add_argument("project"); s.add_argument("--check", action="store_true", help="validate synthetic.jsonl")
    s.set_defaults(fn=cmd_synth)
    s = sub.add_parser("eval", help="test a model")
    s.add_argument("project"); s.add_argument("--base", action="store_true", help="test the untouched base model")
    s.add_argument("--label"); s.add_argument("--model", help="test a model folder instead")
    s.set_defaults(fn=cmd_eval)
    s = sub.add_parser("export", help="save the finished model")
    s.add_argument("project"); s.add_argument("--out"); s.add_argument("--patch", action="append", default=[])
    s.add_argument("--gguf", action="store_true", help="(coming) also write a GGUF file for Ollama / LM Studio")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("generate", help="(coming) describe the AI you want, get an adapter with no training")
    s.add_argument("description"); s.add_argument("--base", default="qwen2.5-0.5b"); s.add_argument("--out")
    s.set_defaults(fn=cmd_generate)
    s = sub.add_parser("train-generator", help="(coming) train the generator on your own adapter collection")
    s.add_argument("zoo"); s.set_defaults(fn=cmd_train_generator)
    zz = sub.add_parser("zoo", help="the adapter collection the generator learns from").add_subparsers(dest="zcmd", required=True)
    s = zz.add_parser("spec", help="the fixed adapter config every zoo adapter uses"); s.set_defaults(fn=cmd_zoo_spec)
    s = zz.add_parser("check", help="validate a folder of task files"); s.add_argument("folder")
    s.set_defaults(fn=cmd_zoo_check)
    s = zz.add_parser("train", help="(coming) train every adapter in the zoo"); s.add_argument("folder")
    s.set_defaults(fn=cmd_zoo_train)

    pp = sub.add_parser("patch", help="weight patches (plug-and-play edits)").add_subparsers(dest="pcmd", required=True)
    s = pp.add_parser("info"); s.add_argument("files", nargs="+"); s.set_defaults(fn=cmd_patch_info)
    s = pp.add_parser("apply"); s.add_argument("model"); s.add_argument("patches", nargs="+")
    s.add_argument("--out", required=True); s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_patch_apply)
    s = pp.add_parser("diff", help="make a patch from two model folders")
    s.add_argument("before"); s.add_argument("after"); s.add_argument("--name", required=True)
    s.add_argument("--out", required=True); s.add_argument("--description")
    s.set_defaults(fn=cmd_patch_diff)
    s = pp.add_parser("auto", help="find and fix a bad habit automatically")
    s.add_argument("project"); s.add_argument("--habit", choices=["looping", "rambling", "dishonest"], required=True)
    s.add_argument("--layers", type=int, default=4); s.set_defaults(fn=cmd_patch_auto)
    s = pp.add_parser("fix", help="(coming) an LLM agent inspects the weights and repairs what you describe")
    s.add_argument("project"); s.add_argument("what", help='e.g. "stops repeating itself"')
    s.set_defaults(fn=cmd_patch_fix)

    sc = sub.add_parser("scratch", help="design a tiny network from nothing, then train it").add_subparsers(dest="scmd", required=True)
    s = sc.add_parser("list"); s.set_defaults(fn=cmd_scratch_list)
    s = sc.add_parser("describe"); s.add_argument("task"); s.add_argument("--csv"); s.set_defaults(fn=cmd_scratch_describe)
    s = sc.add_parser("test"); s.add_argument("network"); s.add_argument("--task"); s.add_argument("--csv")
    s.set_defaults(fn=cmd_scratch_test)
    s = sc.add_parser("train", help="start from your hand-written weights, then train")
    s.add_argument("network"); s.add_argument("--task"); s.add_argument("--csv"); s.add_argument("--out")
    s.add_argument("--steps", type=int, default=2000); s.add_argument("--lr", type=float, default=0.05)
    s.add_argument("--seeds", type=int, default=3, help="random starts to compare against")
    s.set_defaults(fn=cmd_scratch_train)

    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except KeyboardInterrupt:
        sys.exit("\nStopped.")
    except ModuleNotFoundError as e:
        if os.environ.get("MYAI_DEBUG"):
            raise
        sys.exit(f"Missing library '{e.name}'. Run ./install.sh (or: pip install -e \".[llm]\"), "
                 "then myai doctor.")
    except (FileNotFoundError, ValueError, KeyError) as e:
        if os.environ.get("MYAI_DEBUG"):
            raise
        sys.exit(f"Error: {e}\n(set MYAI_DEBUG=1 to see the full details)")


if __name__ == "__main__":
    main()
