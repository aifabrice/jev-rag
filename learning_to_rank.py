"""Small, dependency-free pairwise learning-to-rank utilities.

The benchmark uses this module as a deliberately conservative second-stage
ranker.  It learns only from development qrels and consumes rank-derived
features from already-produced retrieval/reranking runs; it never needs the
test qrels or a new model API call.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable


DEFAULT_RRF_K = 60.0


def reciprocal_rank(rank: int | None, *, rrf_k: float = DEFAULT_RRF_K) -> float:
    """Return a stable reciprocal-rank feature, or zero when absent."""

    if rank is None or rank < 1:
        return 0.0
    return 1.0 / (rrf_k + float(rank))


def normalized_rank(rank: int | None, depth: int) -> float:
    """Return a [0, 1] feature where the first rank is one."""

    if rank is None or rank < 1:
        return 0.0
    if depth <= 1:
        return 1.0
    return max(0.0, 1.0 - (float(rank) - 1.0) / float(depth - 1))


def make_feature_names(source_names: Iterable[str]) -> list[str]:
    names = list(source_names)
    output: list[str] = []
    for source in names:
        output.extend(
            [
                f"{source}.rrf",
                f"{source}.rank_norm",
                f"{source}.present",
            ]
        )
    output.extend(["source_count", "rrf_sum", "rrf_max", "rank_norm_max"])
    return output


def build_feature_vectors(
    source_rankings: dict[str, dict[str, list[str]]],
    candidate_docs: dict[str, list[str]],
    *,
    rrf_k: float = DEFAULT_RRF_K,
) -> dict[str, dict[str, list[float]]]:
    """Build deterministic rank features for every query/document candidate."""

    source_names = list(source_rankings)
    output: dict[str, dict[str, list[float]]] = {}
    for query_id, docs in candidate_docs.items():
        query_features: dict[str, list[float]] = {}
        unique_docs = list(dict.fromkeys(docs))
        per_source: dict[str, dict[str, int]] = {}
        for source in source_names:
            per_source[source] = {
                doc_id: rank
                for rank, doc_id in enumerate(source_rankings[source].get(query_id, []), 1)
            }
        depths = {
            source: max(1, len(source_rankings[source].get(query_id, [])))
            for source in source_names
        }
        for doc_id in unique_docs:
            features: list[float] = []
            rrf_values: list[float] = []
            rank_values: list[float] = []
            for source in source_names:
                rank = per_source[source].get(doc_id)
                rrf = reciprocal_rank(rank, rrf_k=rrf_k)
                rank_norm = normalized_rank(rank, depths[source])
                features.extend([rrf, rank_norm, 1.0 if rank is not None else 0.0])
                rrf_values.append(rrf)
                rank_values.append(rank_norm)
            present = sum(1.0 for value in rrf_values if value > 0.0)
            features.extend(
                [
                    present,
                    sum(rrf_values),
                    max(rrf_values, default=0.0),
                    max(rank_values, default=0.0),
                ]
            )
            query_features[doc_id] = features
        output[query_id] = query_features
    return output


def _sigmoid_neg(margin: float) -> float:
    """Stable sigmoid(-margin), used by the pairwise logistic objective."""

    if margin >= 0.0:
        exp_value = math.exp(-margin)
        return exp_value / (1.0 + exp_value)
    exp_value = math.exp(margin)
    return 1.0 / (1.0 + exp_value)


@dataclass
class PairwiseLinearRanker:
    """A tiny pairwise logistic ranker with feature standardization.

    This is intentionally dependency-free so the local project remains easy to
    install.  It is not intended to replace a full LambdaMART implementation;
    it learns a transparent linear combination of the existing pipeline ranks.
    """

    feature_names: list[str]
    means: list[float]
    scales: list[float]
    weights: list[float]
    bias: float = 0.0

    @classmethod
    def fit(
        cls,
        feature_vectors: dict[str, dict[str, list[float]]],
        qrels: dict[str, dict[str, int]],
        feature_names: list[str],
        *,
        epochs: int = 80,
        learning_rate: float = 0.08,
        l2: float = 0.002,
        max_negatives_per_positive: int = 48,
        seed: int = 17,
    ) -> "PairwiseLinearRanker":
        rows = [
            vector
            for query in feature_vectors.values()
            for vector in query.values()
        ]
        if not rows:
            raise ValueError("cannot train a ranker without feature vectors")
        width = len(feature_names)
        if any(len(row) != width for row in rows):
            raise ValueError("feature vector width does not match feature names")
        means = [sum(row[index] for row in rows) / len(rows) for index in range(width)]
        scales = []
        for index, mean in enumerate(means):
            variance = sum((row[index] - mean) ** 2 for row in rows) / len(rows)
            scales.append(math.sqrt(variance) or 1.0)

        def standardize(vector: list[float]) -> list[float]:
            return [(value - mean) / scale for value, mean, scale in zip(vector, means, scales)]

        standardized = {
            query_id: {
                doc_id: standardize(vector)
                for doc_id, vector in query.items()
            }
            for query_id, query in feature_vectors.items()
        }
        pairs: list[tuple[list[float], list[float], float]] = []
        rng = random.Random(seed)
        for query_id, query in standardized.items():
            relevant = qrels.get(query_id, {})
            positives = [doc for doc in query if relevant.get(doc, 0) > 0]
            negatives = [doc for doc in query if relevant.get(doc, 0) <= 0]
            if not positives or not negatives:
                continue
            for positive in positives:
                ordered_negatives = list(negatives)
                rng.shuffle(ordered_negatives)
                for negative in ordered_negatives[:max_negatives_per_positive]:
                    weight = max(1.0, float(relevant.get(positive, 1)))
                    pairs.append((query[positive], query[negative], weight))
        if not pairs:
            raise ValueError("development candidates contain no positive/negative pairs")

        weights = [0.0] * width
        bias = 0.0
        for _ in range(max(1, epochs)):
            rng.shuffle(pairs)
            for positive, negative, pair_weight in pairs:
                diff = [left - right for left, right in zip(positive, negative)]
                margin = sum(weight * value for weight, value in zip(weights, diff))
                probability = _sigmoid_neg(margin)
                step = learning_rate * pair_weight * probability
                for index, value in enumerate(diff):
                    weights[index] += step * value - learning_rate * l2 * weights[index]
                bias += step
        return cls(feature_names, means, scales, weights, bias)

    def score(self, vector: list[float]) -> float:
        if len(vector) != len(self.weights):
            raise ValueError("feature vector width does not match trained ranker")
        standardized = [
            (value - mean) / scale
            for value, mean, scale in zip(vector, self.means, self.scales)
        ]
        return self.bias + sum(weight * value for weight, value in zip(self.weights, standardized))

    def rank(self, vectors: dict[str, list[float]]) -> list[str]:
        return [
            doc_id
            for doc_id, _ in sorted(
                vectors.items(), key=lambda item: (-self.score(item[1]), item[0])
            )
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "means": self.means,
            "scales": self.scales,
            "weights": self.weights,
            "bias": self.bias,
        }
