"""One interface for every LLM the agent can use.

    from myai.agent import complete
    text = complete("Describe this adapter...", system="You are ...", provider="fake")

A provider is a function (prompt, system, model) -> text. Real providers
(anthropic, openai, gemini, gemini-cli, ollama) are added in Phase 2; only "fake" exists
now so the rest of the code can be built and tested offline.

The provider is chosen by, in order: the `provider` argument, the MYAI_AGENT_PROVIDER
environment variable. There is no silent default, so a job never calls a paid API by accident.
"""

from __future__ import annotations

import os
from typing import Callable

Provider = Callable[[str, str, "str | None"], str]
PROVIDERS: dict[str, Provider] = {}
PLANNED = ()


def register(name: str):
    def deco(fn: Provider) -> Provider:
        PROVIDERS[name] = fn
        return fn
    return deco


@register("fake")
def _fake(prompt: str, system: str, model: str | None) -> str:
    """Offline stand-in for tests: returns MYAI_FAKE_REPLY, or echoes the prompt."""
    return os.environ.get("MYAI_FAKE_REPLY", f"[fake] {prompt}")


def complete(prompt: str, system: str = "", provider: str | None = None, model: str | None = None) -> str:
    name = provider or os.environ.get("MYAI_AGENT_PROVIDER")
    if not name:
        raise ValueError("No agent LLM chosen. Pass provider=... or set MYAI_AGENT_PROVIDER "
                         f"(available: {', '.join(PROVIDERS)}).")
    if name not in PROVIDERS:
        hint = " It is planned but not built yet." if name in PLANNED else ""
        raise ValueError(f"Unknown agent provider '{name}'.{hint} Available: {', '.join(PROVIDERS)}.")
    return PROVIDERS[name](prompt, system, model)


# ---------- real providers ----------
# Each one is a few lines over plain HTTP or a command-line tool, so there are no extra dependencies.

def _post(url: str, body: dict, headers: dict) -> dict:
    import json
    import urllib.request

    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def _key(name: str) -> str:
    load_env()
    if not os.environ.get(name):
        raise ValueError(f"{name} is not set. Put it in a .env file (see .env.example).")
    return os.environ[name]


def load_env(path: str = ".env") -> None:
    """Reads KEY=value lines from .env into the environment (existing variables win)."""
    if os.path.exists(path):
        for line in open(path):
            k, sep, v = line.strip().partition("=")
            if sep and v and not k.startswith("#"):
                os.environ.setdefault(k, v)


@register("agy")
def _agy(prompt: str, system: str, model: str | None) -> str:
    """The `agy` command-line agent, signed in with a subscription. No API key."""
    import shutil
    import subprocess
    import tempfile

    if not shutil.which("agy"):
        raise ValueError("The `agy` command is not installed or not on PATH.")
    cmd = ["agy", "--print", f"{system}\n\n{prompt}" if system else prompt, "--print-timeout", "300s"]
    if model or os.environ.get("MYAI_AGENT_MODEL"):
        cmd += ["--model", model or os.environ["MYAI_AGENT_MODEL"]]
    with tempfile.TemporaryDirectory() as empty:  # run in an empty folder: it only has to think, not touch files
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=empty, timeout=400)
    if r.returncode != 0:
        raise ValueError(f"agy failed: {(r.stderr or r.stdout)[-400:]}")
    return r.stdout.strip()


@register("gemini")
def _gemini(prompt: str, system: str, model: str | None) -> str:
    model = model or os.environ.get("MYAI_AGENT_MODEL") or "gemini-2.5-flash"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    r = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", body,
              {"x-goog-api-key": _key("GEMINI_API_KEY")})
    return r["candidates"][0]["content"]["parts"][0]["text"].strip()


@register("anthropic")
def _anthropic(prompt: str, system: str, model: str | None) -> str:
    model = model or os.environ.get("MYAI_AGENT_MODEL") or "claude-sonnet-5-5"
    body = {"model": model, "max_tokens": 4000, "messages": [{"role": "user", "content": prompt}]}
    if system:
        body["system"] = system
    r = _post("https://api.anthropic.com/v1/messages", body,
              {"x-api-key": _key("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"})
    return "".join(b.get("text", "") for b in r["content"]).strip()


@register("openai")
def _openai(prompt: str, system: str, model: str | None) -> str:
    model = model or os.environ.get("MYAI_AGENT_MODEL") or "gpt-4o"
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    r = _post(f"{base}/chat/completions", {"model": model, "messages": msgs},
              {"authorization": f"Bearer {_key('OPENAI_API_KEY')}"})
    return r["choices"][0]["message"]["content"].strip()


@register("ollama")
def _ollama(prompt: str, system: str, model: str | None) -> str:
    """A local model served by Ollama. Free and private; needs a capable model to do useful repairs."""
    model = model or os.environ.get("MYAI_AGENT_MODEL")
    if not model:
        raise ValueError("Set MYAI_AGENT_MODEL to a model you have in Ollama (see `ollama list`).")
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    r = _post(os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat",
              {"model": model, "messages": msgs, "stream": False}, {})
    return r["message"]["content"].strip()


def extract_json(text: str):
    """The first JSON object or array in an LLM reply (they like to wrap it in prose or code fences)."""
    import json
    import re

    for m in re.finditer(r"[\[{]", text):
        try:
            return json.JSONDecoder().raw_decode(text[m.start():])[0]
        except ValueError:
            continue
    raise ValueError(f"No JSON found in the agent's reply: {text[:200]}")


CALL_LOG: list[dict] = []
_raw_complete = complete


def complete(prompt: str, system: str = "", provider: str | None = None, model: str | None = None) -> str:  # noqa: F811
    """Same as above, and counts calls so jobs can report how much agent use they cost."""
    import time

    t0 = time.time()
    out = _raw_complete(prompt, system, provider, model)
    CALL_LOG.append({"provider": provider or os.environ.get("MYAI_AGENT_PROVIDER"), "seconds": round(time.time() - t0, 1),
                     "chars_in": len(prompt) + len(system), "chars_out": len(out)})
    return out


@register("relay")
def _relay(prompt: str, system: str, model: str | None) -> str:
    """For jobs on a remote GPU machine that has no agent of its own: writes the request to a folder and waits
    for a reply file. `scripts/agent_relay.py` on your own machine answers them with a real provider."""
    import json
    import time
    from pathlib import Path

    d = Path(os.environ.get("MYAI_RELAY_DIR", "/content/agent"))
    d.mkdir(parents=True, exist_ok=True)
    n = len(list(d.glob("req_*.json")))
    (d / f"req_{n:04d}.json").write_text(json.dumps({"prompt": prompt, "system": system}))
    reply = d / f"resp_{n:04d}.txt"
    for _ in range(360):
        if reply.exists():
            time.sleep(1)  # let the upload finish
            return reply.read_text().strip()
        time.sleep(5)
    raise ValueError("No reply from the agent relay within 30 minutes. Is scripts/agent_relay.py running?")
