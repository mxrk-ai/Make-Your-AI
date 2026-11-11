"""The agent: an LLM that captions adapters and repairs them by editing weights."""

from .llm import PROVIDERS, complete, register

__all__ = ["PROVIDERS", "complete", "register"]
