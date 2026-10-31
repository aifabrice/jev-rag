#!/usr/bin/env python3
"""Evaluate a dev-trained learning-to-rank layer over cached BEIR runs.

The script intentionally operates on cached result JSON files.  It learns from
only the development qrels and evaluates on the untouched test qrels, so this
experiment is reproducible without additional embedding, planner, or Jev API
calls.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning_to_rank import (  # noqa: E402
    PairwiseLinearRanker,
    build_feature_vectors,
    make_feature_names,
)
from scripts.benchmark_beir import (  # noqa: E402
    aggregate,
    ensure_dataset,
    load_qrels,
    metrics_for_query,
)


DEFAULT_SOURCE_SPECS = {
    "base_retrieval": (
        "nfcorpus-agentic-hybrid-jev-prior-dev.json",
        "nfcorpus-agentic-hybrid-jev-prior-test.json",
        "retrieval_ranked_doc_ids",
    ),
    "base_jev_prior": (
        "nfcorpus-agentic-hybrid-jev-prior-dev.json",
        "nfcorpus-agentic-hybrid-jev-prior-test.json",
        "ranked_doc_ids",
    ),
    "agentic_jev": (
        "nfcorpus-agentic-hybrid-jev-dev.json",
        "nfcorpus-agentic-hybrid-jev-test.json",
        "ranked_doc_ids",
    ),
    "hybrid": (
        "nfcorpus-hybrid-dev.json",
        "nfcorpus-hybrid-text-embedding-3-large-top50.json",
        "ranked_doc_ids",
    ),
}


def load_result_file(path: Path, field: str) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output: dict[str, list[str]] = {}
    for row in payload.get("results", []):
        query_id = str(row["query_id"])
        ranked = row.get(field)
        if not isinstance(ranked, list):
            raise ValueError(f"{path.name}: result {field!r} is not a list")
        output[query_id] = [str(doc_id) for doc_id in ranked]
    if not output:
        raise ValueError(f"{path.name}: no results found")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--cache-root", type=Path, default=ROOT / ".knowledge/public-benchmarks")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--candidate-depth",
        type=int,
        default=None,
        help="maximum candidate depth before LTR; defaults to top-k for base and 4*top-k for union",
    )
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--candidate-pool", choices=("base", "union"), default="base")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.002)
    parser.add_argument("--max-negatives-per-positive", type=int, default=48)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k < 1 or args.epochs < 1 or args.max_negatives_per_positive < 1:
        raise SystemExit("top-k, epochs, and max-negatives-per-positive must be positive")
    results_dir = args.results_dir or args.cache_root / "results"
    dataset_root = ensure_dataset(args.dataset, args.cache_root)
    dev_qrels = load_qrels(dataset_root, "dev")
    test_qrels = load_qrels(dataset_root, "test")

    dev_sources: dict[str, dict[str, list[str]]] = {}
    test_sources: dict[str, dict[str, list[str]]] = {}
    source_files: dict[str, dict[str, str]] = {}
    for source_name, (dev_name, test_name, field) in DEFAULT_SOURCE_SPECS.items():
        dev_path = results_dir / dev_name
        test_path = results_dir / test_name
        if not dev_path.exists() or not test_path.exists():
            raise SystemExit(f"missing cached source runs: {dev_path} or {test_path}")
        dev_sources[source_name] = load_result_file(dev_path, field)
        test_sources[source_name] = load_result_file(test_path, field)
        source_files[source_name] = {"dev": dev_name, "test": test_name, "field": field}

    source_names = list(dev_sources)
    dev_query_ids = sorted(set(dev_qrels) & set(dev_sources["base_retrieval"]))
    test_query_ids = sorted(set(test_qrels) & set(test_sources["base_retrieval"]))
    if not dev_query_ids or not test_query_ids:
        raise SystemExit("cached runs do not overlap qrels")

    def candidate_docs(
        sources: dict[str, dict[str, list[str]]], query_ids: list[str]
    ) -> dict[str, list[str]]:
        output: dict[str, list[str]] = {}
        for query_id in query_ids:
            if args.candidate_pool == "base":
                docs = list(sources["base_retrieval"].get(query_id, []))
            else:
                docs = []
                for source in source_names:
                    docs.extend(sources[source].get(query_id, []))
            depth = args.candidate_depth or (
                args.top_k if args.candidate_pool == "base" else args.top_k * 4
            )
            output[query_id] = list(dict.fromkeys(docs))[:depth]
        return output

    dev_candidates = candidate_docs(dev_sources, dev_query_ids)
    test_candidates = candidate_docs(test_sources, test_query_ids)
    dev_features = build_feature_vectors(dev_sources, dev_candidates, rrf_k=args.rrf_k)
    test_features = build_feature_vectors(test_sources, test_candidates, rrf_k=args.rrf_k)
    feature_names = make_feature_names(source_names)
    ranker = PairwiseLinearRanker.fit(
        dev_features,
        dev_qrels,
        feature_names,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        max_negatives_per_positive=args.max_negatives_per_positive,
    )

    def evaluate(
        features: dict[str, dict[str, list[float]]],
        qrels: dict[str, dict[str, int]],
        query_ids: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for query_id in query_ids:
            started = time.perf_counter()
            ranked = ranker.rank(features[query_id])[: args.top_k]
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            rows.append(
                {
                    "query_id": query_id,
                    "ranked_doc_ids": ranked,
                    "candidate_count": len(features[query_id]),
                    "metrics": metrics_for_query(ranked, qrels[query_id], [1, 3, 5, 10, 30, 50]),
                    "latency_ms": round(elapsed_ms, 4),
                }
            )
        return rows, aggregate(rows, [1, 3, 5, 10, 30, 50])

    dev_rows, dev_summary = evaluate(dev_features, dev_qrels, dev_query_ids)
    test_rows, test_summary = evaluate(test_features, test_qrels, test_query_ids)
    baseline_test_rows = []
    for query_id in test_query_ids:
        ranked = test_sources["base_jev_prior"][query_id][: args.top_k]
        baseline_test_rows.append(
            {
                "query_id": query_id,
                "ranked_doc_ids": ranked,
                "metrics": metrics_for_query(ranked, test_qrels[query_id], [1, 3, 5, 10, 30, 50]),
                "latency_ms": 0.0,
            }
        )
    baseline_summary = aggregate(baseline_test_rows, [1, 3, 5, 10, 30, 50])
    payload = {
        "benchmark": "cached-beir-learning-to-rank",
        "dataset": args.dataset,
        "train_split": "dev",
        "eval_split": "test",
        "top_k": args.top_k,
        "candidate_pool": args.candidate_pool,
        "feature_sources": source_files,
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "max_negatives_per_positive": args.max_negatives_per_positive,
            "qrels_used": "dev only",
        },
        "dev_fit_summary": dev_summary,
        "baseline_test_summary": baseline_summary,
        "ltr_test_summary": test_summary,
        "delta_ndcg@10": round(
            test_summary["metrics"]["ndcg@10"] - baseline_summary["metrics"]["ndcg@10"],
            6,
        ),
        "ranker": ranker.as_dict(),
        "results": test_rows,
    }
    output = args.output or args.cache_root / "results" / "nfcorpus-ltr-test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "baseline_ndcg@10": baseline_summary["metrics"]["ndcg@10"],
        "ltr_ndcg@10": test_summary["metrics"]["ndcg@10"],
        "delta_ndcg@10": payload["delta_ndcg@10"],
        "baseline_recall@50": baseline_summary["metrics"]["recall@50"],
        "ltr_recall@50": test_summary["metrics"]["recall@50"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
