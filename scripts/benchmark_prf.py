#!/usr/bin/env python3
"""No-cost pseudo-relevance-feedback validation over cached Agentic Hybrid runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_kb import KnowledgeBase  # noqa: E402
from pseudo_relevance import (  # noqa: E402
    document_frequencies,
    select_expansion_terms,
    weighted_rrf_merge,
)
from scripts.benchmark_beir import (  # noqa: E402
    aggregate,
    ensure_dataset,
    jsonl,
    load_qrels,
    metrics_for_query,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--cache-root", type=Path, default=ROOT / ".knowledge/public-benchmarks")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--lexical-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--seed-k", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--expansion-terms", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--expansion-weight", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    return parser.parse_args()


def load_cached_run(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["query_id"]): row for row in payload["results"]}


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir or args.cache_root / "results"
    dataset_root = ensure_dataset(args.dataset, args.cache_root)
    queries = {str(row["_id"]): str(row["text"]) for row in jsonl(dataset_root / "queries.jsonl")}
    corpus = {
        str(row["_id"]): " ".join((str(row.get("title") or ""), str(row.get("text") or ""))).strip()
        for row in jsonl(dataset_root / "corpus.jsonl")
    }
    doc_freq = document_frequencies(corpus.values())
    runtime_root = args.cache_root / f"{args.dataset}-jev-rag"
    documents_root = runtime_root / "documents"
    path_to_doc_id = json.loads((runtime_root / "path_to_doc_id.json").read_text(encoding="utf-8"))
    dev_run = load_cached_run(results_dir / "nfcorpus-agentic-hybrid-jev-prior-dev.json")
    test_run = load_cached_run(results_dir / "nfcorpus-agentic-hybrid-jev-prior-test.json")
    dev_qrels = load_qrels(dataset_root, "dev")
    test_qrels = load_qrels(dataset_root, "test")

    def evaluate(split: str, qrels: dict[str, dict[str, int]], cached: dict[str, dict[str, Any]], config: dict[str, Any] | None = None):
        db = KnowledgeBase(runtime_root / f"{args.dataset}-{split}.db", documents_root, [])
        rows: list[dict[str, Any]] = []
        try:
            for query_id in sorted(set(qrels) & set(cached) & set(queries)):
                row = cached[query_id]
                baseline = list(row["retrieval_ranked_doc_ids"])
                if config is None:
                    ranked = baseline[: args.top_k]
                    expansions: list[str] = []
                else:
                    seed_docs = [corpus[doc_id] for doc_id in baseline[: config["seed_k"]] if doc_id in corpus]
                    expansions = select_expansion_terms(
                        queries[query_id],
                        seed_docs,
                        document_count=len(corpus),
                        document_frequency=doc_freq,
                        max_terms=config["expansion_terms"],
                    )
                    expanded_query = " ".join([queries[query_id], *expansions])
                    lexical = db.lexical_search(expanded_query, args.lexical_k)
                    lexical_ids = [path_to_doc_id[item["path"]] for item in lexical if item["path"] in path_to_doc_id]
                    ranked = weighted_rrf_merge(
                        [(baseline, 1.0), (lexical_ids, config["expansion_weight"])],
                        limit=args.top_k,
                        rrf_k=args.rrf_k,
                    )
                rows.append({
                    "query_id": query_id,
                    "ranked_doc_ids": ranked,
                    "expansion_terms": expansions,
                    "metrics": metrics_for_query(ranked, qrels[query_id], [1, 3, 5, 10, 30, 50]),
                    "latency_ms": 0.0,
                })
        finally:
            db.close()
        return rows, aggregate(rows, [1, 3, 5, 10, 30, 50])

    baseline_dev_rows, baseline_dev = evaluate("dev", dev_qrels, dev_run)
    baseline_test_rows, baseline_test = evaluate("test", test_qrels, test_run)
    best_config: dict[str, Any] | None = None
    best_dev = float("-inf")
    for seed_k in args.seed_k:
        for expansion_terms in args.expansion_terms:
            for expansion_weight in args.expansion_weight:
                config = {
                    "seed_k": seed_k,
                    "expansion_terms": expansion_terms,
                    "expansion_weight": expansion_weight,
                }
                _, summary = evaluate("dev", dev_qrels, dev_run, config)
                score = summary["metrics"]["ndcg@10"]
                if score > best_dev:
                    best_dev, best_config = score, config
    if best_config is None:
        raise SystemExit("no PRF configuration evaluated")
    test_rows, test_summary = evaluate("test", test_qrels, test_run, best_config)
    payload = {
        "benchmark": "cached-beir-pseudo-relevance-feedback",
        "dataset": args.dataset,
        "train_split": "dev",
        "eval_split": "test",
        "qrels_used_for_selection": "dev only",
        "candidate_pool": "baseline Agentic Hybrid retrieval + expanded lexical RRF",
        "selected_config": best_config,
        "baseline_dev_summary": baseline_dev,
        "prf_dev_ndcg@10": best_dev,
        "baseline_test_summary": baseline_test,
        "prf_test_summary": test_summary,
        "delta_ndcg@10": round(test_summary["metrics"]["ndcg@10"] - baseline_test["metrics"]["ndcg@10"], 6),
        "results": test_rows,
    }
    output = args.output or args.cache_root / "results" / "nfcorpus-prf-retrieval-test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "selected_config": best_config,
        "baseline_dev_ndcg@10": baseline_dev["metrics"]["ndcg@10"],
        "prf_dev_ndcg@10": best_dev,
        "baseline_test_ndcg@10": baseline_test["metrics"]["ndcg@10"],
        "prf_test_ndcg@10": test_summary["metrics"]["ndcg@10"],
        "delta_ndcg@10": payload["delta_ndcg@10"],
        "baseline_test_recall@50": baseline_test["metrics"]["recall@50"],
        "prf_test_recall@50": test_summary["metrics"]["recall@50"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
