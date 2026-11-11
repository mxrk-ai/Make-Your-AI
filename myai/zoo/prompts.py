"""The questions every zoo task is trained and tested on.

Train and eval use different topics AND different question templates, so a test never
repeats something the adapter saw. Everything is generated from the lists below with a seed.
"""

from __future__ import annotations

import random

TRAIN_TOPICS = [
    "photosynthesis", "the water cycle", "gravity", "volcanoes", "the immune system", "electricity",
    "magnets", "the moon", "rainbows", "earthquakes", "recycling", "compound interest", "inflation",
    "a budget", "supply and demand", "a database", "an operating system", "encryption", "a web browser",
    "machine learning", "a password manager", "open source software", "cloud storage", "a spreadsheet",
    "a thermostat", "a bicycle", "a compass", "a telescope", "a refrigerator", "solar panels",
    "bread baking", "composting", "a balanced diet", "sleep", "stretching", "hydration", "meditation",
    "a library", "a museum", "a map", "a calendar", "time zones", "an alphabet", "a dictionary",
    "a poem", "a novel", "a symphony", "jazz", "a painting", "photography", "chess", "a marathon",
    "teamwork", "a job interview", "public speaking", "note taking", "a to-do list", "a deadline",
    "bees", "whales", "forests", "deserts", "coral reefs", "migration", "seeds", "weather forecasts",
    "bridges", "trains", "airplanes", "lighthouses", "vaccines", "vitamins", "the heart", "the lungs",
]

EVAL_TOPICS = [
    "tides", "glaciers", "batteries", "a microscope", "a search engine", "a firewall", "savings accounts",
    "taxes", "fermentation", "a greenhouse", "tea", "running shoes", "a backpack", "a podcast",
    "an orchestra", "sculpture", "owls", "rivers", "satellites", "elevators", "a first aid kit",
    "a recipe", "a contract", "a résumé",
]

TRAIN_TEMPLATES = [
    "What is {t}?", "Explain {t} in simple terms.", "Why does {t} matter?", "How does {t} work?",
    "Tell me about {t}.", "Can you describe {t}?", "What should a beginner know about {t}?",
    "Give me a quick overview of {t}.", "What is interesting about {t}?", "what's the point of {t}",
    "I keep hearing about {t}. What is it?", "Help me understand {t}.", "What are common mistakes people make with {t}?",
    "How would you explain {t} to a child?", "What is one useful fact about {t}?", "Why do people care about {t}?",
]

EVAL_TEMPLATES = [
    "Could you explain {t} to me?", "What do I need to know about {t}?", "Summarize {t}.",
    "What is the main idea behind {t}?", "quick question: how do {t} work?", "Describe {t} for a newcomer.",
    "What makes {t} useful?", "Teach me something about {t}.",
]


def _make(topics: list[str], templates: list[str], n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    pairs = [(t, tpl) for t in topics for tpl in templates]
    rng.shuffle(pairs)
    return [tpl.format(t=t) for t, tpl in pairs[:n]]


def train_prompts(n: int = 320, seed: int = 0) -> list[str]:
    return _make(TRAIN_TOPICS, TRAIN_TEMPLATES, n, seed)


def eval_prompts(n: int = 30, seed: int = 1) -> list[str]:
    return _make(EVAL_TOPICS, EVAL_TEMPLATES, n, seed)
