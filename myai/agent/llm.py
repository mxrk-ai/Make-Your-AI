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
PLANNED = ("anthropic", "openai", "gemini", "gemini-cli", "ollama")


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
