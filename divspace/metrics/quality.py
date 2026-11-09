"""Surface quality checks.

Diversity is only interesting if it does not come from broken text, so every
diversity comparison must be reported next to these signals. They are deliberately
cheap and model-free so that they can run on the full result set.
"""

from __future__ import annotations

import re
from typing import Sequence

from .lexical import rep_n, words

_TERMINAL = re.compile(r"[.!?\"')\]]\s*$")


def completion_rate(texts: Sequence[str]) -> float:
    if not texts:
        return 0.0
    complete = sum(1 for t in texts if t.strip() and _TERMINAL.search(t.strip()))
    return complete / len(texts)


def mean_length(texts: Sequence[str], unit: str = "words") -> float:
    if not texts:
        return 0.0
    if unit == "chars":
        return sum(len(t) for t in texts) / len(texts)
    return sum(len(words(t)) for t in texts) / len(texts)


def length_std(texts: Sequence[str]) -> float:
    if len(texts) < 2:
        return 0.0
    values = [len(words(t)) for t in texts]
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def degeneracy(texts: Sequence[str]) -> float:
    return rep_n(texts, n=4)


def empty_rate(texts: Sequence[str]) -> float:
    if not texts:
        return 0.0
    return sum(1 for t in texts if not t.strip()) / len(texts)


def quality_report(texts: Sequence[str]) -> dict:
    return {
        "completion_rate": completion_rate(texts),
        "empty_rate": empty_rate(texts),
        "mean_words": mean_length(texts),
        "std_words": length_std(texts),
        "degeneracy_4": degeneracy(texts),
    }
