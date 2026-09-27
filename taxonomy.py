"""Deterministic hierarchical knowledge classification for Jev RAG.

The taxonomy is learned from corpus embeddings only. Relevance labels and
benchmark queries never participate in tree construction.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TAXONOMY_VERSION = 1
DEFAULT_TOP_BRANCHES = 12
DEFAULT_LEAF_SIZE = 64
DEFAULT_ROUTE_LEAVES = 4
DEFAULT_ROUTE_MIN_ITEMS = 200
DEFAULT_SECONDARY_MARGIN = 0.025
DEFAULT_TAXONOMY_RRF_K = 60
DEFAULT_TAXONOMY_ROUTE_WEIGHT = 2.0
DEFAULT_TAXONOMY_EXTRA_CANDIDATES = 20

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u3400-\u4dbf\u4e00-\u9fff]{2,8}")
_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were",
    "has", "have", "had", "not", "but", "can", "may", "into", "than", "then",
    "their", "there", "these", "those", "using", "used", "use", "between", "about",
    "what", "when", "where", "which", "who", "why", "how", "also", "more", "such",
    "study", "studies", "result", "results", "effect", "effects", "patients", "patient",
    "based", "data", "analysis", "associated", "compared", "however", "during", "after",
    "before", "over", "under", "through", "other", "some", "each", "both", "all",
}


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by optional dependency checks
        raise RuntimeError("taxonomy mode requires numpy") from exc
    return np


def _normalise_rows(values: Any) -> Any:
    np = _numpy()
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("taxonomy vectors must be a two-dimensional matrix")
    if not len(matrix):
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _spherical_kmeans(vectors: Any, cluster_count: int, seed: int) -> tuple[Any, Any]:
    """Small deterministic spherical k-means implementation with no sklearn dependency."""
    np = _numpy()
    vectors = _normalise_rows(vectors)
    count = len(vectors)
    if not count:
        return np.empty(0, dtype=np.int32), np.empty((0, 0), dtype=np.float32)
    cluster_count = max(1, min(int(cluster_count), count))
    if cluster_count == 1:
        center = vectors.mean(axis=0, keepdims=True)
        center = _normalise_rows(center)
        return np.zeros(count, dtype=np.int32), center

    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, count))
    selected = [first]
    closest = vectors @ vectors[first]
    for _ in range(1, cluster_count):
        next_index = int(np.argmin(closest))
        selected.append(next_index)
        closest = np.maximum(closest, vectors @ vectors[next_index])
    centers = vectors[np.asarray(selected)].copy()
    assignments = np.full(count, -1, dtype=np.int32)

    for _ in range(30):
        similarities = vectors @ centers.T
        updated = np.argmax(similarities, axis=1).astype(np.int32)
        if np.array_equal(updated, assignments):
            break
        assignments = updated
        new_centers = []
        best_similarity = similarities.max(axis=1)
        for cluster in range(cluster_count):
            members = vectors[assignments == cluster]
            if len(members):
                center = members.mean(axis=0)
            else:
                replacement = int(np.argmin(best_similarity))
                center = vectors[replacement]
                assignments[replacement] = cluster
            center = center / max(float(np.linalg.norm(center)), 1e-12)
            new_centers.append(center)
        centers = np.asarray(new_centers, dtype=np.float32)
    return assignments, centers


def _representative_terms(texts: Iterable[str], limit: int = 5) -> list[str]:
    documents: list[list[str]] = []
    frequencies: Counter[str] = Counter()
    document_frequency: Counter[str] = Counter()
    for text in texts:
        tokens = [
            token.lower()
            for token in _WORD_RE.findall(text[:2400])
            if token.lower() not in _STOPWORDS and not token.isdigit()
        ]
        if not tokens:
            continue
        documents.append(tokens)
        frequencies.update(tokens)
        document_frequency.update(set(tokens))
    total = max(1, len(documents))
    scored = {
        token: frequencies[token] * math.log((total + 1) / (document_frequency[token] + 1) + 1)
        for token in frequencies
    }
    return sorted(scored, key=lambda token: (-scored[token], token))[:limit]


def _node_label(texts: list[str], fallback: str) -> tuple[str, str]:
    terms = _representative_terms(texts)
    label = " / ".join(terms[:3]) if terms else fallback
    description = ", ".join(terms) if terms else fallback
    return label, description


def build_taxonomy(
    vectors: Any,
    item_ids: list[str],
    texts: list[str],
    *,
    top_branches: int = DEFAULT_TOP_BRANCHES,
    leaf_size: int = DEFAULT_LEAF_SIZE,
    secondary_margin: float = DEFAULT_SECONDARY_MARGIN,
    seed: int = 17,
) -> tuple[dict[str, Any], Any]:
    """Build a two-level category tree and multi-label leaf assignments."""
    np = _numpy()
    vectors = _normalise_rows(vectors)
    if len(item_ids) != len(vectors) or len(texts) != len(vectors):
        raise ValueError("taxonomy item ids, texts, and vectors must have the same length")
    if not item_ids:
        raise ValueError("cannot build a taxonomy for an empty corpus")
    if leaf_size < 2:
        raise ValueError("taxonomy leaf_size must be at least 2")

    top_count = max(1, min(top_branches, math.ceil(len(item_ids) / leaf_size)))
    top_assignments, top_centroids = _spherical_kmeans(vectors, top_count, seed)
    nodes: list[dict[str, Any]] = [
        {
            "id": "root",
            "parent_id": None,
            "depth": 0,
            "label": "All knowledge",
            "description": "Root of the automatically generated knowledge taxonomy",
            "item_count": len(item_ids),
        }
    ]
    leaf_centroids: list[Any] = []
    leaf_parent_ids: list[str] = []
    primary_leaf_indexes = np.full(len(item_ids), -1, dtype=np.int32)

    for top_index in range(top_count):
        member_indexes = np.flatnonzero(top_assignments == top_index)
        top_id = f"topic-{top_index + 1:02d}"
        top_texts = [texts[int(index)] for index in member_indexes]
        label, description = _node_label(top_texts, top_id)
        nodes.append(
            {
                "id": top_id,
                "parent_id": "root",
                "depth": 1,
                "label": label,
                "description": description,
                "item_count": int(len(member_indexes)),
            }
        )
        local_count = max(1, math.ceil(len(member_indexes) / leaf_size))
        local_assignments, local_centroids = _spherical_kmeans(
            vectors[member_indexes], local_count, seed + 1009 * (top_index + 1)
        )
        for local_index in range(local_count):
            local_members = member_indexes[local_assignments == local_index]
            leaf_id = f"{top_id}/leaf-{local_index + 1:02d}"
            leaf_texts = [texts[int(index)] for index in local_members]
            leaf_label, leaf_description = _node_label(leaf_texts, leaf_id)
            nodes.append(
                {
                    "id": leaf_id,
                    "parent_id": top_id,
                    "depth": 2,
                    "label": leaf_label,
                    "description": leaf_description,
                    "item_count": int(len(local_members)),
                }
            )
            global_leaf_index = len(leaf_centroids)
            leaf_centroids.append(local_centroids[local_index])
            leaf_parent_ids.append(top_id)
            primary_leaf_indexes[local_members] = global_leaf_index

    centroid_matrix = _normalise_rows(np.asarray(leaf_centroids, dtype=np.float32))
    similarities = vectors @ centroid_matrix.T
    assignments: dict[str, list[str]] = {}
    leaf_ids = [node["id"] for node in nodes if node["depth"] == 2]
    for item_index, item_id in enumerate(item_ids):
        primary = int(primary_leaf_indexes[item_index])
        primary_score = float(similarities[item_index, primary])
        ordered = np.argsort(-similarities[item_index], kind="stable")
        memberships = [leaf_ids[primary]]
        for candidate in ordered:
            candidate = int(candidate)
            if candidate == primary:
                continue
            if float(similarities[item_index, candidate]) < primary_score - secondary_margin:
                break
            memberships.append(leaf_ids[candidate])
            break
        assignments[str(item_id)] = memberships

    node_by_id = {node["id"]: node for node in nodes}
    for node in nodes:
        if node["depth"] != 2:
            continue
        node["assigned_count"] = sum(
            node["id"] in membership for membership in assignments.values()
        )
    tree = {
        "version": TAXONOMY_VERSION,
        "config": {
            "top_branches": top_branches,
            "leaf_size": leaf_size,
            "secondary_margin": secondary_margin,
            "seed": seed,
        },
        "item_count": len(item_ids),
        "top_node_count": top_count,
        "leaf_node_count": len(leaf_ids),
        "nodes": nodes,
        "assignments": assignments,
        "leaf_ids": leaf_ids,
        "leaf_parent_ids": leaf_parent_ids,
        "labels": {node_id: node_by_id[node_id]["label"] for node_id in node_by_id},
    }
    return tree, centroid_matrix


def save_taxonomy(prefix: Path, tree: dict[str, Any], centroids: Any) -> None:
    np = _numpy()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(tree, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with prefix.with_suffix(".npz").open("wb") as handle:
        np.savez(handle, leaf_centroids=np.asarray(centroids, dtype=np.float32))


def load_taxonomy(prefix: Path) -> tuple[dict[str, Any], Any] | None:
    np = _numpy()
    json_path = prefix.with_suffix(".json")
    vector_path = prefix.with_suffix(".npz")
    if not json_path.exists() or not vector_path.exists():
        return None
    tree = json.loads(json_path.read_text(encoding="utf-8"))
    if int(tree.get("version", -1)) != TAXONOMY_VERSION:
        return None
    with np.load(vector_path, allow_pickle=False) as cached:
        centroids = cached["leaf_centroids"].copy()
    if len(centroids) != int(tree.get("leaf_node_count", -1)):
        return None
    return tree, centroids


def route_taxonomy(
    query_vector: Any,
    tree: dict[str, Any],
    leaf_centroids: Any,
    *,
    route_leaves: int = DEFAULT_ROUTE_LEAVES,
    min_items: int = DEFAULT_ROUTE_MIN_ITEMS,
) -> tuple[set[str], dict[str, Any]]:
    """Select query-relevant leaves, expanding until a safe corpus scope exists."""
    np = _numpy()
    vector = np.asarray(query_vector, dtype=np.float32)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    centroids = _normalise_rows(leaf_centroids)
    similarities = centroids @ vector
    ordered = np.argsort(-similarities, kind="stable").tolist()
    assignments = tree["assignments"]
    leaf_ids = tree["leaf_ids"]
    labels = tree.get("labels") or {}
    selected: list[int] = []
    allowed: set[str] = set()
    for leaf_index in ordered:
        selected.append(int(leaf_index))
        leaf_id = leaf_ids[int(leaf_index)]
        allowed.update(
            item_id for item_id, memberships in assignments.items() if leaf_id in memberships
        )
        if len(selected) >= max(1, route_leaves) and len(allowed) >= max(1, min_items):
            break
    routes = [
        {
            "id": leaf_ids[index],
            "label": labels.get(leaf_ids[index], leaf_ids[index]),
            "parent_id": tree["leaf_parent_ids"][index],
            "score": round(float(similarities[index]), 6),
        }
        for index in selected
    ]
    return allowed, {
        "routes": routes,
        "selected_leaf_count": len(selected),
        "scope_items": len(allowed),
        "tree_items": int(tree["item_count"]),
        "top_node_count": int(tree["top_node_count"]),
        "leaf_node_count": int(tree["leaf_node_count"]),
    }


def taxonomy_rank_fusion(
    routed: list[dict[str, Any]],
    global_results: list[dict[str, Any]],
    *,
    id_key: str = "rowid",
    limit: int = 50,
    rrf_k: int = DEFAULT_TAXONOMY_RRF_K,
    route_weight: float = DEFAULT_TAXONOMY_ROUTE_WEIGHT,
) -> list[dict[str, Any]]:
    """Boost routed evidence while retaining a full-corpus fallback ranking."""
    items: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    for rank, item in enumerate(routed, start=1):
        identity = str(item[id_key])
        merged = items.setdefault(identity, dict(item))
        merged.update(item)
        merged["taxonomy_rank"] = rank
        scores[identity] = scores.get(identity, 0.0) + route_weight / (rrf_k + rank)
    for rank, item in enumerate(global_results, start=1):
        identity = str(item[id_key])
        merged = items.setdefault(identity, dict(item))
        merged.update(item)
        merged["global_rank"] = rank
        scores[identity] = scores.get(identity, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(
        items,
        key=lambda identity: (
            -scores[identity],
            items[identity].get("taxonomy_rank", 10**9),
            items[identity].get("global_rank", 10**9),
            identity,
        ),
    )[: max(1, limit)]
    results = []
    for rank, identity in enumerate(ordered, start=1):
        item = dict(items[identity])
        item["taxonomy_rrf_score"] = round(scores[identity], 8)
        item["retrieval_rank"] = rank
        results.append(item)
    return results


def taxonomy_candidate_expansion(
    routed: list[dict[str, Any]],
    global_results: list[dict[str, Any]],
    *,
    id_key: str = "rowid",
    extra_candidates: int = DEFAULT_TAXONOMY_EXTRA_CANDIDATES,
) -> list[dict[str, Any]]:
    """Keep the global ranking intact and append routed candidates it missed."""
    routed_ranks = {
        str(item[id_key]): rank for rank, item in enumerate(routed, start=1)
    }
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for global_rank, source in enumerate(global_results, start=1):
        item = dict(source)
        identity = str(item[id_key])
        seen.add(identity)
        item["global_rank"] = global_rank
        if identity in routed_ranks:
            item["taxonomy_rank"] = routed_ranks[identity]
            item["taxonomy_routed"] = True
        else:
            item["taxonomy_routed"] = False
        item["retrieval_rank"] = len(results) + 1
        results.append(item)
    added = 0
    for taxonomy_rank, source in enumerate(routed, start=1):
        identity = str(source[id_key])
        if identity in seen:
            continue
        item = dict(source)
        item["taxonomy_rank"] = taxonomy_rank
        item["global_rank"] = None
        item["taxonomy_routed"] = True
        item["retrieval_rank"] = len(results) + 1
        results.append(item)
        seen.add(identity)
        added += 1
        if added >= max(0, extra_candidates):
            break
    return results
