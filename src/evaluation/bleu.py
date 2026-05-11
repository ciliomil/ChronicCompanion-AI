"""Char-level BLEU-1~4 with multi-reference support.

This is a pure-Python implementation tailored to Chinese eval:

- Tokenization: split into individual Unicode characters, dropping whitespace.
  CJK is naturally character-tokenized; ASCII strings degrade to char-level
  too, which is fine for our short reply lengths.
- Multi-reference: clip n-gram counts at the per-n-gram max across references
  (the standard BLEU-multi-ref rule).
- Smoothing: NIST "smoothing method 1" — when a higher-order precision is
  zero, fall back to ``1 / (2 * total_ngrams)`` so the geometric mean stays
  finite. Sentence-level BLEU benefits from this; corpus-level is mostly
  unaffected because total counts are usually large.
- Brevity penalty: classic BLEU brevity penalty against the reference whose
  length is closest to the candidate (ties broken by shorter reference).

Both sentence-level (per-sample) and corpus-level (aggregate) computations
share the same n-gram primitives.
"""

from __future__ import annotations

import math
from collections import Counter


_MAX_N = 4


def tokenize(text: str) -> list[str]:
    """Char-level tokenization with whitespace stripped."""
    if not text:
        return []
    return [c for c in text if not c.isspace()]


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _clipped_counts(
    cand_tokens: list[str],
    ref_token_lists: list[list[str]],
    n: int,
) -> tuple[int, int]:
    """Return ``(clipped_match, candidate_total)`` for n-grams of order ``n``."""
    cand_ngrams = _ngrams(cand_tokens, n)
    if not cand_ngrams:
        return 0, 0
    cand_counts = Counter(cand_ngrams)

    max_ref_counts: Counter[tuple[str, ...]] = Counter()
    for ref in ref_token_lists:
        ref_counts = Counter(_ngrams(ref, n))
        for ng, c in ref_counts.items():
            if c > max_ref_counts[ng]:
                max_ref_counts[ng] = c

    clipped = sum(min(c, max_ref_counts[ng]) for ng, c in cand_counts.items())
    total = sum(cand_counts.values())
    return clipped, total


def _closest_ref_length(cand_len: int, ref_lens: list[int]) -> int:
    """Pick the reference length closest to ``cand_len`` (ties: shorter)."""
    if not ref_lens:
        return 0
    return min(ref_lens, key=lambda r: (abs(r - cand_len), r))


def _brevity_penalty(cand_len: int, ref_len: int) -> float:
    if cand_len == 0:
        return 0.0
    if cand_len > ref_len:
        return 1.0
    return math.exp(1.0 - ref_len / cand_len)


def sentence_bleu(
    candidate: str,
    references: list[str],
    *,
    max_n: int = _MAX_N,
) -> dict[str, float]:
    """Cumulative sentence-level BLEU-1..max_n with NIST method-1 smoothing."""
    cand_tokens = tokenize(candidate)
    ref_token_lists = [tokenize(r) for r in references if r]
    if not cand_tokens or not ref_token_lists:
        return {f"bleu_{n}": 0.0 for n in range(1, max_n + 1)}

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        clip, total = _clipped_counts(cand_tokens, ref_token_lists, n)
        if total == 0:
            # Candidate too short for this n-gram order. BLEU is undefined
            # here; treat as 1.0 so a perfectly matching short answer still
            # scores 1.0 instead of being penalized for lacking long n-grams.
            precisions.append(1.0)
        elif clip == 0:
            precisions.append(1.0 / (2.0 * total))  # NIST method 1
        else:
            precisions.append(clip / total)

    cand_len = len(cand_tokens)
    ref_len = _closest_ref_length(cand_len, [len(r) for r in ref_token_lists])
    bp = _brevity_penalty(cand_len, ref_len)

    out: dict[str, float] = {}
    for n_max in range(1, max_n + 1):
        ps = precisions[:n_max]
        log_avg = sum(math.log(p) for p in ps) / n_max
        out[f"bleu_{n_max}"] = bp * math.exp(log_avg)
    return out


class CorpusBleuAccumulator:
    """Streaming corpus-level BLEU accumulator (multi-ref).

    Aggregates clipped match counts and totals across all (candidate, refs)
    pairs and computes BLEU-1..max_n at the end. This is the standard
    corpus-level computation:

        precision_n = sum_clipped_n / sum_total_n
        bp          = brevity_penalty(sum_cand_len, sum_closest_ref_len)
        bleu_N      = bp * exp((1/N) * sum_{n<=N} log(precision_n))
    """

    def __init__(self, max_n: int = _MAX_N) -> None:
        self.max_n = max_n
        self._clip = [0] * max_n
        self._total = [0] * max_n
        self._cand_len = 0
        self._ref_len = 0
        self._n_samples = 0
        self._n_with_refs = 0

    def add(self, candidate: str, references: list[str]) -> None:
        self._n_samples += 1
        cand_tokens = tokenize(candidate)
        ref_token_lists = [tokenize(r) for r in references if r]
        if not ref_token_lists:
            # No references → cannot score this sample; still count it as a sample.
            return
        self._n_with_refs += 1
        for i in range(self.max_n):
            clip, total = _clipped_counts(cand_tokens, ref_token_lists, i + 1)
            self._clip[i] += clip
            self._total[i] += total
        self._cand_len += len(cand_tokens)
        self._ref_len += _closest_ref_length(
            len(cand_tokens),
            [len(r) for r in ref_token_lists],
        )

    def result(self) -> dict[str, float | int]:
        out: dict[str, float | int] = {
            "n_samples": self._n_samples,
            "n_with_refs": self._n_with_refs,
        }
        if self._n_with_refs == 0:
            for n_max in range(1, self.max_n + 1):
                out[f"bleu_{n_max}"] = 0.0
            return out

        smoothed: list[float] = []
        for clip, total in zip(self._clip, self._total):
            if total == 0:
                # Whole corpus too short for this order — treat as 1.0 (see
                # ``sentence_bleu`` for the same convention).
                smoothed.append(1.0)
            elif clip == 0:
                smoothed.append(1.0 / (2.0 * total))
            else:
                smoothed.append(clip / total)

        bp = _brevity_penalty(self._cand_len, self._ref_len)
        for n_max in range(1, self.max_n + 1):
            ps = smoothed[:n_max]
            log_avg = sum(math.log(p) for p in ps) / n_max
            out[f"bleu_{n_max}"] = bp * math.exp(log_avg)
        return out


def corpus_bleu(
    candidates: list[str],
    references_list: list[list[str]],
    *,
    max_n: int = _MAX_N,
) -> dict[str, float | int]:
    """Convenience wrapper for one-shot corpus-level BLEU."""
    if len(candidates) != len(references_list):
        raise ValueError(
            f"candidates / references_list length mismatch: "
            f"{len(candidates)} vs {len(references_list)}"
        )
    acc = CorpusBleuAccumulator(max_n=max_n)
    for cand, refs in zip(candidates, references_list):
        acc.add(cand, refs)
    return acc.result()
