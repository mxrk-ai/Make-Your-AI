"""Runs ON the Colab VM: everything after the zoo. Reads the zoo from its private Hugging Face repo and writes
the generator and all result files to another private repo. Every stage is skipped if its result exists.

  stage=main    train generator -> scoreboard -> demo answers -> best-of-4 -> flywheel
  stage=agent   wait for the generator -> captioner -> agent repair (needs scripts/agent_relay.py running locally)
"""
import glob
import json
import os
import subprocess
import sys
import time

ARGS = dict(a.split("=", 1) for a in os.environ.get("ZOO_ARGS", "").split() if "=" in a)
STAGE = ARGS.get("stage", "main")
ZOO, GEN = "/content/zoo", "/content/gen"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "safetensors", "huggingface_hub", "pyyaml"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps",
                *glob.glob("/content/make_your_ai-*.whl")], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"])
for line in open("/content/.env"):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1)
        if v:
            os.environ[k] = v

import torch  # noqa: E402
from huggingface_hub import HfApi, snapshot_download  # noqa: E402

api = HfApi(token=os.environ["HF_TOKEN"])
user = api.whoami()["name"]
zoo_repo, gen_repo = f"{user}/{ARGS.get('zoo', 'make-your-ai-zoo-v1')}", f"{user}/{ARGS.get('gen', 'make-your-ai-generator')}"
api.create_repo(gen_repo, private=True, exist_ok=True)
snapshot_download(zoo_repo, repo_type="dataset", local_dir=ZOO, token=os.environ["HF_TOKEN"])
say = lambda m: print(m, flush=True)


def pull():
    try:
        snapshot_download(gen_repo, local_dir=GEN, token=os.environ["HF_TOKEN"])
    except Exception as e:
        say(f"nothing to pull yet: {type(e).__name__}")


def push(name=None):
    if name:
        api.upload_file(path_or_fileobj=f"{GEN}/{name}", path_in_repo=name, repo_id=gen_repo)
    else:
        api.upload_folder(folder_path=GEN, repo_id=gen_repo, commit_message="generator", ignore_patterns=["*.lock", ".cache/*"])
    say(f"uploaded {name or 'generator'}")


have = lambda name: os.path.exists(f"{GEN}/{name}")
pull()

from safetensors.torch import load_file  # noqa: E402

from myai import generated as G  # noqa: E402
from myai.generator.core import Generator, load_zoo, train_generator  # noqa: E402
from myai.generator.report import evaluate_generator, markdown  # noqa: E402
from myai.zoo import behaviors as B  # noqa: E402
from myai.zoo import trainer as T  # noqa: E402
from myai.zoo.prompts import eval_prompts  # noqa: E402

if STAGE == "main":
    if not have("generator.json"):
        zoo_all = load_zoo(ZOO)
        say(f"zoo: {len(zoo_all)} adapters, mean score {sum(m['score'] for _, m in zoo_all) / len(zoo_all):.3f}, "
            f"below 0.5: {sum(m['score'] < 0.5 for _, m in zoo_all)}")
        train_generator(ZOO, GEN, steps=int(ARGS.get("steps", 12000)), latent_dim=int(ARGS.get("latent", 64)),
                        d=int(ARGS.get("d", 384)), layers=int(ARGS.get("layers", 6)), log=say)
        # Guidance strength is chosen by behaviour on SEEN tasks only (held-out tasks are never used for tuning).
        gen = Generator(GEN)
        model, tok, dev = T.load_base(gen.info["base_model"], gen.info["spec"])
        T.set_a(model, gen.lora_a)
        q = eval_prompts(16, seed=5)
        seen = [m for _, m in load_zoo(ZOO, split="seen") if m["seed"] == 0][::8]
        best = (2.0, -1)
        for g in (1.0, 2.0, 3.0, 4.0):
            s = sum(T.eval_adapter(model, tok, dev, m["task"], q, weights=gen.generate(m["description"], guidance=g))["score"]
                    for m in seen) / len(seen)
            say(f"guidance {g}: {s:.3f} on {len(seen)} seen tasks")
            if s > best[1]:
                best = (g, s)
        info = json.load(open(f"{GEN}/generator.json"))
        info["guidance"], info["guidance_search"] = best[0], round(best[1], 3)
        json.dump(info, open(f"{GEN}/generator.json", "w"), indent=1)
        del model
        torch.cuda.empty_cache()
        push()
    gen = Generator(GEN)
    guidance = gen.info["guidance"]

    last = [time.time()]

    def log(m):
        say(m)
        if time.time() - last[0] > 600 and have("report.json"):
            last[0] = time.time(); push("report.json")

    res = json.load(open(f"{GEN}/report.json")) if have("report.json") else {}
    if "generator" not in res:  # unfinished or missing
        res = evaluate_generator(ZOO, GEN, f"{GEN}/report.json", n_seen=int(ARGS.get("seen", 15)), guidance=guidance, log=log)
        push("report.json")
    say(markdown(res))

    model, tok, dev = T.load_base(gen.info["base_model"], gen.info["spec"])
    T.set_a(model, gen.lora_a)
    if not have("demo.json"):
        demos = []
        for desc in ["talk like a pirate, max 3 sentences", "always answer in formal English, max 3 sentences",
                     "reply only in JSON, in one sentence", "answer with bullet points and end with \"Hope that helps!\"",
                     "ALL CAPS and very excited", "be kind and answer in French"]:
            task, unknown = G.understand(desc)
            w = gen.generate(desc, guidance=guidance)
            qs = ["What is a volcano?", "Why is sleep important?"]
            T.set_b(model, w)
            ans = [r["answer"] for r in T.generate_answers(model, tok, dev, qs, 300)]
            score = T.eval_adapter(model, tok, dev, task, eval_prompts(20, seed=7), weights=w)["score"] if task else None
            demos.append({"description": desc, "understood": B.describe(task) if task else "", "not_understood": unknown,
                          "known_combination": B.task_id(task) in {m["task_id"] for _, m in load_zoo(ZOO, split="seen")},
                          "score": score, "qa": list(zip(qs, ans))})
            say(json.dumps(demos[-1], ensure_ascii=False)[:600])
        json.dump(demos, open(f"{GEN}/demo.json", "w"), indent=1, ensure_ascii=False)
        push("demo.json")

    if not have("bestof.json"):
        tasks = [t for t in json.load(open(f"{ZOO}/tasks.json")) if t["split"] == "heldout"]
        practice, test, rows = eval_prompts(12, seed=3), eval_prompts(24), []
        for t in tasks:
            desc = B.describe(t["task"])
            cands = [gen.generate(desc, seed=s, guidance=guidance) for s in range(4)]
            pr = [T.eval_adapter(model, tok, dev, t["task"], practice, weights=w)["score"] for w in cands]
            te = [T.eval_adapter(model, tok, dev, t["task"], test, weights=w)["score"] for w in cands]
            pick = max(range(4), key=lambda i: pr[i])
            rows.append({"id": t["id"], "first": te[0], "best_of_4": te[pick], "oracle_best": max(te)})
            say(f"bestof {t['id']:28s} first={te[0]:.2f} picked={te[pick]:.2f}")
        mean = lambda k: round(sum(r[k] for r in rows) / len(rows), 3)
        json.dump({"heldout_first": mean("first"), "heldout_best_of_4": mean("best_of_4"), "oracle": mean("oracle_best"),
                   "tasks": rows}, open(f"{GEN}/bestof.json", "w"), indent=1)
        push("bestof.json")
    del model
    torch.cuda.empty_cache()

    if not have("flywheel.json"):
        from myai.experiments import flywheel_experiment

        r = flywheel_experiment(ZOO, GEN, f"{GEN}/flywheel.json", n_new=int(ARGS.get("fly", 40)), work="/content/gen_fly", log=say)
        say(json.dumps({k: v for k, v in r.items() if k != "per_task"}))
        push("flywheel.json")

if STAGE == "agent":
    os.environ["MYAI_AGENT_PROVIDER"] = "relay"
    while not have("generator.json"):
        say("waiting for the generator..."); time.sleep(60); pull()
    gen = Generator(GEN)
    if not have("caption.json"):
        from myai.agent.caption import caption_zoo

        r = caption_zoo(ZOO, f"{GEN}/caption.json", n=int(ARGS.get("captions", 20)), provider="relay", log=say)
        say(json.dumps({k: v for k, v in r.items() if k != "rows"}))
        push("caption.json")
    if not have("patch.json"):
        from myai.experiments import patch_experiment

        # A quick look at which held-out tasks the generator gets wrong (the repair experiment starts from these).
        model, tok, dev = T.load_base(gen.info["base_model"], gen.info["spec"])
        T.set_a(model, gen.lora_a)
        rows = []
        for t in [t for t in json.load(open(f"{ZOO}/tasks.json")) if t["split"] == "heldout"][:16]:
            desc = B.describe(t["task"])
            e = T.eval_adapter(model, tok, dev, t["task"], eval_prompts(16, seed=5), weights=gen.generate(desc, guidance=gen.info["guidance"]))
            rows.append({"id": t["id"], "split": "heldout", "description": desc, "scores": {"generated": e["score"]},
                         "generated_per_axis": e["per_axis"]})
            say(f"prescreen {t['id']:28s} {e['score']:.2f}")
        del model
        torch.cuda.empty_cache()
        json.dump({"tasks": rows}, open("/content/prescreen.json", "w"))
        r = patch_experiment(GEN, "/content/prescreen.json", f"{GEN}/patch.json", work="/content/patch",
                             n=int(ARGS.get("patches", 5)), provider="relay", log=say)
        say(json.dumps({k: v for k, v in r.items() if k != "tasks"}))
        push("patch.json")
    open("/content/agent/done", "w").write("done")
say("PIPELINE DONE")
