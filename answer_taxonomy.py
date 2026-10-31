"""Answer-supervised taxonomy for offline retrieval experiments.

This module deliberately separates training-time supervision from query-time
routing.  A taxonomy is learned from development queries and their relevant
documents, while test queries only select leaves from the learned centroids.
The structure is a small hierarchy with multi-label document assignments and
soft routing; it is not a hard filter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from taxonomy import _normalise_rows, _spherical_kmeans, _numpy


ANSWER_TAXONOMY_VERSION = 1
DEFAULT_ANSWER_TOP_BRANCHES = 6
DEFAULT_ANSWER_LEAF_SIZE = 24
DEFAULT_ANSWER_ROUTE_LEAVES = 4
DEFAULT_ANSWER_MIN_ITEMS = 200
DEFAULT_ANSWER_SECONDARY_MARGIN = 0.025
DEFAULT_ANSWER_ROUTE_WEIGHT = 0.25


def build_answer_supervised_taxonomy(
    corpus_vectors: Any,
    doc_ids: list[str],
    dev_query_vectors: Any,
    dev_query_ids: list[str],
    dev_qrels: dict[str, dict[str, int]],
    *,
    top_branches: int = DEFAULT_ANSWER_TOP_BRANCHES,
    leaf_size: int = DEFAULT_ANSWER_LEAF_SIZE,
    secondary_margin: float = DEFAULT_ANSWER_SECONDARY_MARGIN,
    seed: int = 41,
) -> tuple[dict[str, Any], Any]:
    """Learn query-intent leaves from development relevance supervision.

    Each development query contributes a prototype made from its embedding and
    the relevance-weighted mean of its relevant document embeddings.  Clusters
    of these prototypes become answer-oriented categories.  The test split is
    not needed to build the tree and must only call :func:`route_answer_taxonomy`.
    """
    np = _numpy()
    corpus = _normalise_rows(corpus_vectors)
    query_matrix = _normalise_rows(dev_query_vectors)
    if len(doc_ids) != len(corpus):
        raise ValueError("answer taxonomy document ids and vectors must have the same length")
    if len(dev_query_ids) != len(query_matrix):
        raise ValueError("answer taxonomy query ids and vectors must have the same length")
    if leaf_size < 2 or not dev_query_ids:
        raise ValueError("answer taxonomy needs queries and leaf_size >= 2")

    doc_index = {str(doc_id): index for index, doc_id in enumerate(doc_ids)}
    records: list[tuple[str, Any, list[str]]] = []
    for query_index, query_id in enumerate(dev_query_ids):
        relevant = [
            (str(doc_id), int(score))
            for doc_id, score in (dev_qrels.get(str(query_id)) or {}).items()
            if str(doc_id) in doc_index and int(score) > 0
        ]
        if not relevant:
            continue
        indexes = np.asarray([doc_index[doc_id] for doc_id, _ in relevant], dtype=np.int64)
        weights = np.asarray([max(1, score) for _, score in relevant], dtype=np.float32)
        doc_center = (corpus[indexes] * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1.0)
        prototype = _normalise_rows(
            np.asarray([0.5 * query_matrix[query_index] + 0.5 * doc_center], dtype=np.float32)
        )[0]
        records.append((str(query_id), prototype, [doc_id for doc_id, _ in relevant]))
    if not records:
        raise ValueError("development qrels did not match any corpus documents")

    prototypes = np.asarray([record[1] for record in records], dtype=np.float32)
    top_count = max(1, min(int(top_branches), len(records)))
    top_assignments, _ = _spherical_kmeans(prototypes, top_count, seed)

    nodes: list[dict[str, Any]] = [
        {
            "id": "root",
            "parent_id": None,
            "depth": 0,
            "label": "Answer-oriented knowledge",
            "description": "Categories learned from development query support evidence",
            "item_count": len(doc_ids),
        }
    ]
    leaf_centroids: list[Any] = []
    leaf_parent_ids: list[str] = []
    leaf_query_indexes: list[list[int]] = []
    for top_index in range(top_count):
        member_indexes = np.flatnonzero(top_assignments == top_index)
        top_id = f"answer-topic-{top_index + 1:02d}"
        local_count = max(1, (len(member_indexes) + leaf_size - 1) // leaf_size)
        local_assignments, _ = _spherical_kmeans(
            prototypes[member_indexes], local_count, seed + 1009 * (top_index + 1)
        )
        nodes.append(
            {
                "id": top_id,
                "parent_id": "root",
                "depth": 1,
                "label": top_id,
                "description": "answer-supported query intent group",
                "item_count": int(len(member_indexes)),
            }
        )
        for local_index in range(local_count):
            local_members = member_indexes[local_assignments == local_index]
            leaf_id = f"{top_id}/leaf-{local_index + 1:02d}"
            center = _normalise_rows(
                np.asarray([prototypes[local_members].mean(axis=0)], dtype=np.float32)
            )[0]
            leaf_centroids.append(center)
            leaf_parent_ids.append(top_id)
            leaf_query_indexes.append([int(index) for index in local_members.tolist()])
            nodes.append(
                {
                    "id": leaf_id,
                    "parent_id": top_id,
                    "depth": 2,
                    "label": leaf_id,
                    "description": "query group with shared relevant evidence",
                    "item_count": int(len(local_members)),
                }
            )

    leaf_ids = [node["id"] for node in nodes if node["depth"] == 2]
    centroids = _normalise_rows(np.asarray(leaf_centroids, dtype=np.float32))
    doc_similarities = corpus @ centroids.T
    assignments: dict[str, list[str]] = {str(doc_id): [] for doc_id in doc_ids}
    for doc_index_value, doc_id in enumerate(doc_ids):
        ordered = np.argsort(-doc_similarities[doc_index_value], kind="stable")
        primary = int(ordered[0])
        memberships = [leaf_ids[primary]]
        primary_score = float(doc_similarities[doc_index_value, primary])
        for candidate in ordered[1:]:
            candidate = int(candidate)
            if float(doc_similarities[doc_index_value, candidate]) < primary_score - secondary_margin:
                break
            memberships.append(leaf_ids[candidate])
            break
        assignments[str(doc_id)] = memberships

    # Ensure every supervised support document is reachable from the leaf that
    # learned from its query, even when its corpus embedding is an outlier.
    for record_index, (_, _, support_doc_ids) in enumerate(records):
        leaf_index = next(
            index for index, members in enumerate(leaf_query_indexes) if record_index in members
        )
        leaf_id = leaf_ids[leaf_index]
        for doc_id in support_doc_ids:
            memberships = assignments.setdefault(str(doc_id), [])
            if leaf_id not in memberships:
                memberships.append(leaf_id)

    tree = {
        "version": ANSWER_TAXONOMY_VERSION,
        "config": {
            "top_branches": top_branches,
            "leaf_size": leaf_size,
            "secondary_margin": secondary_margin,
            "seed": seed,
            "supervision": "dev_query_plus_relevant_documents",
        },
        "item_count": len(doc_ids),
        "supervised_query_count": len(records),
        "top_node_count": top_count,
        "leaf_node_count": len(leaf_ids),
        "nodes": nodes,
        "assignments": assignments,
        "leaf_ids": leaf_ids,
        "leaf_parent_ids": leaf_parent_ids,
        "labels": {node["id"]: node["label"] for node in nodes},
    }
    return tree, centroids


def save_answer_taxonomy(prefix: Path, tree: dict[str, Any], centroids: Any) -> None:
    np = _numpy()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(tree, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with prefix.with_suffix(".npz").open("wb") as handle:
        np.savez(handle, leaf_centroids=np.asarray(centroids, dtype=np.float32))


def load_answer_taxonomy(prefix: Path) -> tuple[dict[str, Any], Any] | None:
    np = _numpy()
    json_path = prefix.with_suffix(".json")
    vector_path = prefix.with_suffix(".npz")
    if not json_path.exists() or not vector_path.exists():
        return None
    tree = json.loads(json_path.read_text(encoding="utf-8"))
    if int(tree.get("version", -1)) != ANSWER_TAXONOMY_VERSION:
        return None
    with np.load(vector_path, allow_pickle=False) as cached:
        centroids = cached["leaf_centroids"].copy()
    if len(centroids) != int(tree.get("leaf_node_count", -1)):
        return None
    return tree, centroids


def route_answer_taxonomy(
    query_vector: Any,
    tree: dict[str, Any],
    leaf_centroids: Any,
    *,
    route_leaves: int = DEFAULT_ANSWER_ROUTE_LEAVES,
    min_items: int = DEFAULT_ANSWER_MIN_ITEMS,
) -> tuple[set[str], dict[str, Any]]:
    """Soft-route a query to answer-oriented leaves with global fallback scope."""
    np = _numpy()
    vector = np.asarray(query_vector, dtype=np.float32)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    centroids = _normalise_rows(leaf_centroids)
    similarities = centroids @ vector
    ordered = np.argsort(-similarities, kind="stable").tolist()
    assignments = tree["assignments"]
    leaf_ids = tree["leaf_ids"]
    selected: list[int] = []
    allowed: set[str] = set()
    for leaf_index in ordered:
        leaf_index = int(leaf_index)
        selected.append(leaf_index)
        leaf_id = leaf_ids[leaf_index]
        allowed.update(
            item_id for item_id, memberships in assignments.items() if leaf_id in memberships
        )
        if len(selected) >= max(1, route_leaves) and len(allowed) >= max(1, min_items):
            break
    routes = [
        {
            "id": leaf_ids[index],
            "label": tree.get("labels", {}).get(leaf_ids[index], leaf_ids[index]),
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
        "supervised_query_count": int(tree.get("supervised_query_count", 0)),
        "top_node_count": int(tree["top_node_count"]),
        "leaf_node_count": int(tree["leaf_node_count"]),
    }


def answer_taxonomy_candidate_expansion(
    routed: list[dict[str, Any]],
    global_results: list[dict[str, Any]],
    *,
    id_key: str = "path",
    extra_candidates: int = 20,
) -> list[dict[str, Any]]:
    """Keep global candidates first and append unseen routed candidates."""
    routed_ranks = {str(item[id_key]): rank for rank, item in enumerate(routed, start=1)}
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for global_rank, source in enumerate(global_results, start=1):
        item = dict(source)
        identity = str(item[id_key])
        seen.add(identity)
        item["global_rank"] = global_rank
        item["answer_taxonomy_rank"] = routed_ranks.get(identity)
        item["answer_taxonomy_routed"] = identity in routed_ranks
        item["retrieval_rank"] = len(results) + 1
        results.append(item)
    added = 0
    for taxonomy_rank, source in enumerate(routed, start=1):
        identity = str(source[id_key])
        if identity in seen:
            continue
        item = dict(source)
        item["answer_taxonomy_rank"] = taxonomy_rank
        item["global_rank"] = None
        item["answer_taxonomy_routed"] = True
        item["retrieval_rank"] = len(results) + 1
        results.append(item)
        seen.add(identity)
        added += 1
        if added >= max(0, extra_candidates):
            break
    return results


def answer_taxonomy_rank_fusion(
    routed: list[dict[str, Any]],
    global_results: list[dict[str, Any]],
    *,
    id_key: str = "path",
    limit: int = 50,
    rrf_k: int = 60,
    route_weight: float = DEFAULT_ANSWER_ROUTE_WEIGHT,
) -> list[dict[str, Any]]:
    """Softly boost routed candidates while preserving a global fallback."""
    if route_weight < 0 or rrf_k < 1:
        raise ValueError("answer taxonomy fusion parameters are invalid")
    items: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    for rank, source in enumerate(routed, start=1):
        identity = str(source[id_key])
        item = items.setdefault(identity, dict(source))
        item.update(source)
        item["answer_taxonomy_rank"] = rank
        scores[identity] = scores.get(identity, 0.0) + route_weight / (rrf_k + rank)
    for rank, source in enumerate(global_results, start=1):
        identity = str(source[id_key])
        item = items.setdefault(identity, dict(source))
        item.update(source)
        item["global_rank"] = rank
        scores[identity] = scores.get(identity, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(
        items,
        key=lambda identity: (
            -scores[identity],
            items[identity].get("global_rank", 10**9),
            items[identity].get("answer_taxonomy_rank", 10**9),
            identity,
        ),
    )[: max(1, limit)]
    results: list[dict[str, Any]] = []
    routed_ids = {str(source[id_key]) for source in routed}
    for position, identity in enumerate(ordered, start=1):
        item = dict(items[identity])
        item["answer_taxonomy_rrf_score"] = round(scores[identity], 8)
        item["answer_taxonomy_routed"] = identity in routed_ids
        item["retrieval_rank"] = position
        results.append(item)
    return results
