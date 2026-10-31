"""Deterministic pseudo-relevance feedback helpers for lexical retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}|[^\W\d_]{2,}", re.UNICODE)
STOPWORDS = {
    "about", "after", "again", "also", "been", "being", "between", "could",
    "does", "during", "from", "have", "into", "more", "other", "over", "such",
    "than", "that", "their", "there", "these", "they", "this", "those", "through",
    "under", "were", "which", "while", "with", "would", "what", "when", "where",
    "will", "your", "you", "the", "and", "for", "not", "are", "can", "how", "why",
    "was", "has", "had", "its", "our", "out", "all", "any", "but", "who", "one",
}


def tokenize_terms(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text) if token.lower() not in STOPWORDS]


def document_frequencies(documents: Iterable[str]) -> dict[str, int]:
    frequencies: Counter[str] = Counter()
    for document in documents:
        frequencies.update(set(tokenize_terms(document)))
    return dict(frequencies)


def select_expansion_terms(
    query: str,
    seed_documents: Iterable[str],
    *,
    document_count: int,
    document_frequency: dict[str, int] | None = None,
    max_terms: int = 8,
) -> list[str]:
    """Select high-signal terms from the first-pass documents.

    Terms already present in the query are excluded. Frequency is discounted by
    rank order and weighted by corpus IDF, which prevents generic terms from
    dominating the expansion.
    """

    query_terms = set(tokenize_terms(query))
    frequencies: Counter[str] = Counter()
    for rank, document in enumerate(seed_documents, 1):
        weight = 1.0 / math.sqrt(float(rank))
        for term in set(tokenize_terms(document)):
            if term not in query_terms:
                frequencies[term] += weight
    doc_freq = document_frequency or {}

    def score(term: str) -> float:
        frequency = max(1, doc_freq.get(term, 1))
        idf = math.log((document_count + 1.0) / (frequency + 0.5))
        return frequencies[term] * max(0.1, idf)

    return [term for term, _ in sorted(frequencies.items(), key=lambda item: (-score(item[0]), item[0]))[:max_terms]]


def weighted_rrf_merge(
    ranked_lists: list[tuple[list[str], float]], *, limit: int = 50, rrf_k: int = 60
) -> list[str]:
    scores: dict[str, float] = {}
    first_rank: dict[str, int] = {}
    for ranked, weight in ranked_lists:
        if weight <= 0:
            continue
        for rank, doc_id in enumerate(ranked, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (rrf_k + rank)
            first_rank[doc_id] = min(first_rank.get(doc_id, rank), rank)
    return [
        doc_id
        for doc_id, _ in sorted(
            scores.items(), key=lambda item: (-item[1], first_rank[item[0]], item[0])
        )[:max(1, limit)]
    ]
