"""Built-in prompt sets.

These are deliberately short, original, and free of any licence encumbrance so
that the pipeline runs end to end with no download. For a submission, swap in a
public benchmark with ``--prompt-file``; anything that provides a JSONL with a
``prompt`` field works.
"""

from __future__ import annotations

import json
from pathlib import Path

OPEN = [
    "Tell me something interesting about the ocean.",
    "What is a good way to spend a rainy afternoon?",
    "Describe a small town you would like to visit.",
    "What makes a good friendship?",
    "Explain why people enjoy music.",
    "Suggest a name for a coffee shop and justify it.",
    "What would you tell a student who is afraid of failing?",
    "Describe the perfect breakfast.",
    "Why do humans tell stories?",
    "What is the most useful skill a person can learn?",
]

STORY = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Write a short story about a robot learning to garden.",
    "Write a short story set in a library that rearranges itself at night.",
    "Write a short story about two strangers who miss the same train.",
    "Write a short story about a town where it rains only on Sundays.",
    "Write a short story about a map that shows places that do not exist yet.",
    "Write a short story about a chef who can taste memories.",
    "Write a short story about the last bookshop on a small island.",
]

EXPLAIN = [
    "Explain how a refrigerator keeps food cold.",
    "Explain why the sky changes colour at sunset.",
    "Explain how vaccines train the immune system.",
    "Explain what a cryptographic hash function does.",
    "Explain why compound interest grows so quickly.",
    "Explain how a language model predicts the next word.",
    "Explain the difference between weather and climate.",
    "Explain how a bicycle stays upright.",
]

ADVICE = [
    "How should I prepare for a job interview?",
    "How can I stay focused while working from home?",
    "What should I do if I disagree with my manager?",
    "How can I learn a new language as an adult?",
    "How do I start running as a beginner?",
    "What is a sensible way to budget a small salary?",
    "How can I be a better listener?",
    "What should I consider before adopting a pet?",
]

LIST = [
    "List five ways to reduce household waste.",
    "List five habits that improve sleep quality.",
    "List five questions worth asking before starting a project.",
    "List five things to check when reviewing a research paper.",
    "List five board games suitable for a rainy day.",
    "List five ways to make meetings shorter.",
]

PROMPT_SETS = {
    "open": OPEN,
    "story": STORY,
    "explain": EXPLAIN,
    "advice": ADVICE,
    "list": LIST,
}


def get_prompts(name: str = "open", limit: int | None = None) -> list[str]:
    if name in PROMPT_SETS:
        prompts = list(PROMPT_SETS[name])
    elif name == "all":
        prompts = [p for group in PROMPT_SETS.values() for p in group]
    else:
        raise KeyError(f"unknown prompt set {name!r}; choose from {sorted(PROMPT_SETS)} or 'all'")
    return prompts[:limit] if limit else prompts


def load_prompt_file(path: str | Path) -> list[str]:
    path = Path(path)
    prompts = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                prompts.append(line)
                continue
            prompts.append(payload["prompt"] if isinstance(payload, dict) else payload)
    return prompts


def resolve_prompts(name: str, path: str | None = None, limit: int | None = None) -> list[str]:
    prompts = load_prompt_file(path) if path else get_prompts(name)
    return prompts[:limit] if limit else prompts
