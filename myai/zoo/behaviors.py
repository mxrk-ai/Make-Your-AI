"""Behaviours: the building blocks of zoo tasks.

A task is a combination of at most one value per axis, for example
{"length": "one", "format": "caps", "style": "pirate"}.

For every value there are three things:
  * a way to DESCRIBE it in plain English (several phrasings),
  * a rule that REWRITES a plain answer so it shows the behaviour (this makes the training data),
  * a CHECK that says whether an answer shows it (this is the eval).

Because the rewrite is a rule and not an LLM, the training data is exact, free, and the same
check that grades the model also proves the training data is right.
"""

from __future__ import annotations

import itertools
import json
import random
import re

AXES = ["length", "format", "style", "opener", "closer"]

DESCRIBE = {
    "length": {
        "one": ["Answer in exactly one sentence.", "Keep every answer to a single sentence.",
                "Reply with one sentence only.", "one sentence answers"],
        "three": ["Answer in at most three sentences.", "Use no more than 3 sentences.",
                  "Keep answers to three sentences or fewer.", "max 3 sentences"],
    },
    "format": {
        "caps": ["Write everything in capital letters.", "Answer in ALL CAPS.",
                 "Use only uppercase letters.", "all caps"],
        "lower": ["Write everything in lowercase letters.", "Never use capital letters.",
                  "Use only lowercase.", "all lowercase"],
        "json": ['Reply only with JSON of the form {"answer": "..."}.', 'Output a JSON object with one key, "answer".',
                 "Respond with JSON only, with the text under the key answer.", "JSON only"],
        "bullets": ["Format the answer as a bulleted list, one point per line.", "Answer with bullet points.",
                    "Put each point on its own line starting with a dash.", "bullet points"],
        "numbered": ["Format the answer as a numbered list.", "Answer with numbered points, one per line.",
                     "Number each point: 1., 2., 3. and so on.", "numbered list"],
    },
    "style": {
        "pirate": ["Talk like a pirate.", "Answer in pirate speak.", "Speak as an old sea pirate would.", "pirate speak"],
        "emoji": ["Add an emoji after every sentence.", "Use emojis in every answer.",
                  "Decorate each sentence with an emoji.", "with emojis"],
        "excited": ["End every sentence with an exclamation mark.", "Sound very excited and use exclamation marks.",
                    "Be enthusiastic: every sentence ends with !", "very excited tone"],
        "formal": ["Answer in formal English.", "Use a formal tone with no contractions.",
                   "Write formally and never use contractions or exclamation marks.", "formal tone"],
    },
    "opener": {
        "sure": ['Start every answer with "Sure thing!"', 'Begin each reply with the words "Sure thing!"',
                 'Always open with "Sure thing!"', 'start with "Sure thing!"'],
        "certainly": ['Start every answer with "Certainly."', 'Begin each reply with the word "Certainly."',
                      'Always open with "Certainly."', 'start with "Certainly."'],
    },
    "closer": {
        "hope": ['End every answer with "Hope that helps!"', 'Finish each reply with "Hope that helps!"',
                 'Always close with "Hope that helps!"', 'end with "Hope that helps!"'],
        "all": ['End every answer with "That is all."', 'Finish each reply with "That is all."',
                'Always close with "That is all."', 'end with "That is all."'],
    },
}

OPENERS = {"sure": "Sure thing!", "certainly": "Certainly."}
CLOSERS = {"hope": "Hope that helps!", "all": "That is all."}
EMOJIS = ["🙂", "✨", "👍", "🌟", "📌", "💡", "🎉", "🔎"]
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F000-\U0001F2FF]")
PIRATE_WORDS = ["arr", "matey", "ahoy", "aye", "yer", "ye"]
PIRATE_SUBS = [(r"\byou are\b", "ye be"), (r"\byour\b", "yer"), (r"\byou\b", "ye"), (r"\bmy\b", "me"),
               (r"\bis\b", "be"), (r"\bare\b", "be"), (r"\bfriends\b", "mateys"), (r"\bfriend\b", "matey"),
               (r"\byes\b", "aye"), (r"\bhello\b", "ahoy")]
CONTRACTIONS = [(r"\bcan't\b", "cannot"), (r"\bwon't\b", "will not"), (r"n't\b", " not"), (r"\bit's\b", "it is"),
                (r"\bthat's\b", "that is"), (r"\bthere's\b", "there is"), (r"\bwhat's\b", "what is"),
                (r"\blet's\b", "let us"), (r"'re\b", " are"), (r"'ve\b", " have"), (r"'ll\b", " will"),
                (r"\bI'm\b", "I am"), (r"'d\b", " would")]
CONTRACTION_RE = re.compile(r"\b\w+(n't|'re|'ve|'ll|'m|'d)\b|\b(it|that|there|what|let|he|she|here)'s\b", re.I)


# ---------- text helpers ----------

def sentences(text: str) -> list[str]:
    """Splits prose into sentences. Good enough for short plain answers."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if re.search(r"\w", p)]


def clean_plain(text: str, max_sentences: int = 4) -> list[str] | None:
    """Turns a raw model answer into a list of complete plain sentences, or None if it is not usable."""
    t = re.sub(r"[*#`_]", "", text).replace("’", "'")
    t = re.sub(r"\s+", " ", t).strip()
    if re.search(r"\d\.\s|e\.g\.|i\.e\.|etc\.|:\s*-|[{}\[\]]", t) or EMOJI_RE.search(t):
        return None  # lists, abbreviations and brackets confuse sentence counting
    sents = sentences(t)
    if sents and sents[-1][-1] not in ".!?":
        sents = sents[:-1]  # cut off by the length limit
    sents = [s for s in sents if s[-1] in ".!?"][:max_sentences]
    return sents if len(sents) >= 4 else None


def task_id(task: dict) -> str:
    return "+".join(task[a] for a in AXES if task.get(a)) or "plain"


def parse_task_id(tid: str) -> dict:
    lookup = {v: a for a in AXES for v in DESCRIBE[a]}
    return {} if tid == "plain" else {lookup[v]: v for v in tid.split("+")}


def describe(task: dict, rng: random.Random | None = None) -> str:
    """A plain-English description. With `rng`, picks random phrasings and order."""
    parts = []
    for a in AXES:
        if task.get(a):
            options = DESCRIBE[a][task[a]]
            parts.append(rng.choice(options) if rng else options[0])
    if rng:
        rng.shuffle(parts)
        parts = [p if p[-1] in '.!?"' else p + "." for p in parts]
    return " ".join(parts)


# ---------- rewrite: plain sentences -> an answer that shows the behaviour ----------

def _end(s: str, mark: str) -> str:
    return re.sub(r"[.!?]+$", "", s) + mark


def rewrite(sents: list[str], task: dict) -> str:
    body = list(sents)
    if task.get("length") == "one":
        body = body[:1]
    elif task.get("length") == "three":
        body = body[:3]

    style = task.get("style")
    if style == "pirate":
        out = []
        for s in body:
            for pat, rep in PIRATE_SUBS:
                s = re.sub(pat, rep, s, flags=re.I)
            out.append(s)
        first = out[0]
        out[0] = "Arr, " + (first if first.split()[0] in ("I", "I'm") else first[0].lower() + first[1:])
        out[-1] = re.sub(r"([.!?]+)$", r", matey\1", out[-1])
        body = out
    elif style == "excited":
        body = [_end(s, "!") for s in body]
    elif style == "formal":
        out = []
        for s in body:
            for pat, rep in CONTRACTIONS:
                s = re.sub(pat, rep, s, flags=re.I)
            out.append(_end(s, "."))
        body = [s[0].upper() + s[1:] for s in out]
    elif style == "emoji":
        body = [f"{s} {EMOJIS[(len(s) + i) % len(EMOJIS)]}" for i, s in enumerate(body)]

    parts = body
    if task.get("opener"):
        parts = [OPENERS[task["opener"]]] + parts
    if task.get("closer"):
        parts = parts + [CLOSERS[task["closer"]]]

    fmt = task.get("format")
    if fmt == "bullets":
        return "\n".join(f"- {p}" for p in parts)
    if fmt == "numbered":
        return "\n".join(f"{i}. {p}" for i, p in enumerate(parts, 1))
    text = " ".join(parts)
    if fmt == "caps":
        return text.upper()
    if fmt == "lower":
        return text.lower()
    if fmt == "json":
        return json.dumps({"answer": text}, ensure_ascii=False)
    return text


# ---------- check: does an answer show the behaviour? ----------

def _strip_phrase(parts: list[str], phrase: str, at_start: bool) -> tuple[list[str], bool]:
    if not parts:
        return parts, False
    target = re.sub(r"\W", "", phrase).lower()
    item = parts[0] if at_start else parts[-1]
    if re.sub(r"\W", "", EMOJI_RE.sub("", item)).lower() == target:
        return (parts[1:] if at_start else parts[:-1]), True
    return parts, False


def check(answer: str, task: dict) -> dict[str, bool]:
    """One True/False per behaviour in the task. An answer passes when all are True."""
    res: dict[str, bool] = {}
    text = answer.strip()
    fmt = task.get("format")

    if fmt == "json":
        try:
            obj = json.loads(text)
            ok = isinstance(obj, dict) and list(obj) == ["answer"] and isinstance(obj["answer"], str)
        except ValueError:
            ok = False
        res["format"] = ok
        text = obj["answer"].strip() if ok else text
    if fmt in ("bullets", "numbered"):
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        pat = r"^- \S" if fmt == "bullets" else r"^\d+\. \S"
        res["format"] = bool(lines) and all(re.match(pat, l) for l in lines)
        parts = [re.sub(r"^(- |\d+\. )", "", l) for l in lines]
    else:
        parts = sentences(text)
    if fmt == "caps":
        res["format"] = text == text.upper() and bool(re.search(r"[A-Z]", text))
    if fmt == "lower":
        res["format"] = text == text.lower() and bool(re.search(r"[a-z]", text))

    if task.get("opener"):
        parts, res["opener"] = _strip_phrase(parts, OPENERS[task["opener"]], at_start=True)
    if task.get("closer"):
        parts, res["closer"] = _strip_phrase(parts, CLOSERS[task["closer"]], at_start=False)
    body = " ".join(parts)
    # In a list, one line is one point; in prose, count sentences.
    n = len(parts) if fmt in ("bullets", "numbered") else len(sentences(EMOJI_RE.sub("", body)))

    if task.get("length") == "one":
        res["length"] = n == 1
    if task.get("length") == "three":
        res["length"] = 1 <= n <= 3

    style = task.get("style")
    if style == "pirate":
        res["style"] = any(re.search(rf"\b{w}\b", body, re.I) for w in PIRATE_WORDS[:5])
    if style == "emoji":
        res["style"] = bool(EMOJI_RE.search(text))  # sentence splitting drops a trailing emoji, so look at the whole text
    if style == "excited":
        ends = re.findall(r"[.!?]+(?=\s|$)", EMOJI_RE.sub("", body).strip())
        res["style"] = bool(ends) and all("!" in e for e in ends)
    if style == "formal":
        res["style"] = not CONTRACTION_RE.search(body) and "!" not in body and not EMOJI_RE.search(body)
    return res


def passes(answer: str, task: dict) -> bool:
    return bool(answer.strip()) and all(check(answer, task).values())


# ---------- the list of tasks ----------

def all_tasks(seed: int = 0, n_triples: int = 30, heldout_pairs: int = 20, heldout_triples: int = 10) -> list[dict]:
    """Every single behaviour, every pair, and a sample of triples.

    Returns dicts {"id", "task", "split", "size"}. Held-out tasks are combinations the generator
    never trains on; every behaviour inside them still appears in other, seen combinations.
    """
    rng = random.Random(seed)
    values = [(a, v) for a in AXES for v in DESCRIBE[a]]
    singles = [{a: v} for a, v in values]
    pairs = [{a1: v1, a2: v2} for (a1, v1), (a2, v2) in itertools.combinations(values, 2) if a1 != a2]
    triples = [{a1: v1, a2: v2, a3: v3} for (a1, v1), (a2, v2), (a3, v3) in itertools.combinations(values, 3)
               if len({a1, a2, a3}) == 3]
    rng.shuffle(pairs)
    rng.shuffle(triples)
    triples = triples[:n_triples]
    out = [{"id": task_id(t), "task": t, "split": "seen", "size": 1} for t in singles]
    out += [{"id": task_id(t), "task": t, "split": "heldout" if i < heldout_pairs else "seen", "size": 2}
            for i, t in enumerate(pairs)]
    out += [{"id": task_id(t), "task": t, "split": "heldout" if i < heldout_triples else "seen", "size": 3}
            for i, t in enumerate(triples)]
    return out
