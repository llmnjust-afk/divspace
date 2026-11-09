"""Lexical diversity metrics: distinct-n, rep-n, self-BLEU."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

_WORD = re.compile(r"[\w']+", re.UNICODE)
_CJK = r"\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af"
_CJK_CHAR = re.compile(f"[{_CJK}]")
_CJK_SPLIT = re.compile(f"[{_CJK}]|[^{_CJK}]+")


def words(text: str) -> list[str]:
    """Whitespace/punctuation-free tokenisation that also works for CJK text.

    Scripts without spaces are emitted character by character; alphabetic runs
    stay whole. Without this split a Chinese sentence would collapse to a single
    token and every distinct-n score would be meaningless.
    """
    tokens: list[str] = []
    for match in _WORD.finditer(text.lower()):
        token = match.group()
        if _CJK_CHAR.search(token):
            tokens.extend(piece for piece in _CJK_SPLIT.findall(token) if piece.strip())
        else:
            tokens.append(token)
    return tokens


def ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def distinct_n(texts: Sequence[str], n: int = 2) -> float:
    """Unique n-grams over total n-grams, pooled across all samples of one prompt."""
    counter: Counter = Counter()
    for text in texts:
        counter.update(ngram_counts(words(text), n))
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return len(counter) / total


def rep_n(texts: Sequence[str], n: int = 3) -> float:
    """Fraction of n-gram slots that are not the first occurrence of that n-gram."""
    counter: Counter = Counter()
    for text in texts:
        counter.update(ngram_counts(words(text), n))
    total = sum(counter.values())
    if total == 0:
        return 0.0
    repeated = sum(count - 1 for count in counter.values() if count > 1)
    return repeated / total


def type_token_ratio(texts: Sequence[str]) -> float:
    tokens = [token for text in texts for token in words(text)]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _bleu_precisions(hypothesis: Sequence[str], reference: Sequence[str], max_n: int) -> tuple[list[float], float]:
    precisions = []
    for n in range(1, max_n + 1):
        hyp_counts = ngram_counts(hypothesis, n)
        ref_counts = ngram_counts(reference, n)
        if not hyp_counts:
            precisions.append(0.0)
            continue
        overlap = sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
        precisions.append(overlap / sum(hyp_counts.values()))
    brevity = 1.0
    if len(hypothesis) < len(reference) and len(hypothesis) > 0:
        brevity = math.exp(1.0 - len(reference) / len(hypothesis))
    elif len(hypothesis) == 0:
        brevity = 0.0
    return precisions, brevity


def sentence_bleu(hypothesis: str, reference: str, max_n: int = 4) -> float:
    hyp, ref = words(hypothesis), words(reference)
    precisions, brevity = _bleu_precisions(hyp, ref, max_n)
    if min(precisions) <= 0:
        return 0.0
    return brevity * math.exp(sum(math.log(p) for p in precisions) / max_n)


def self_bleu(texts: Sequence[str], max_n: int = 4) -> float:
    """Mean pairwise BLEU among the samples of a single prompt; lower is more diverse."""
    texts = [t for t in texts if t.strip()]
    if len(texts) < 2:
        return 0.0
    total, pairs = 0.0, 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            total += sentence_bleu(texts[i], texts[j], max_n)
            pairs += 1
    return total / max(pairs, 1)
