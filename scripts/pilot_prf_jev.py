#!/usr/bin/env python3
"""Run a small Jev pilot on the cached pseudo-relevance-feedback candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jev_test import load_dotenv  # noqa: E402
from local_kb import KnowledgeBase, jev_rerank  # noqa: E402
from scripts.benchmark_beir import ensure_dataset, jsonl, load_qrels, metrics_for_query  # noqa: E402
from scripts.benchmark_prf import load_cached_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / ".knowledge/public-benchmarks")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument(
        "--prf-result",
        type=Path,
        default=ROOT / ".knowledge/experiments/nfcorpus-prf-retrieval-test.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    results_dir = args.results_dir or args.cache_root / "results"
    dataset_root = ensure_dataset("nfcorpus", args.cache_root)
    queries = {str(row["_id"]): str(row["text"]) for row in jsonl(dataset_root / "queries.jsonl")}
    qrels = load_qrels(dataset_root, "test")
    cached = load_cached_run(results_dir / "nfcorpus-agentic-hybrid-jev-prior-test.json")
    baseline_full = cached
    prf_payload = json.loads(args.prf_result.read_text(encoding="utf-8"))
    prf_rows = {str(row["query_id"]): row for row in prf_payload["results"]}
    runtime_root = args.cache_root / "nfcorpus-jev-rag"
    documents_root = runtime_root / "documents"
    db = KnowledgeBase(runtime_root / "nfcorpus-test.db", documents_root, [])
    path_to_doc_id = json.loads((runtime_root / "path_to_doc_id.json").read_text(encoding="utf-8"))
    documents_by_id = {}
    for row in db.connection.execute(
        "SELECT rowid, doc_id, passage_no, path, title, heading, start_line, end_line, body FROM passages ORDER BY rowid"
    ):
        item = dict(row)
        item.update({"rowid": int(item["rowid"]), "passage_no": int(item["passage_no"])})
        if item["path"] in path_to_doc_id:
            documents_by_id[path_to_doc_id[item["path"]]] = item

    pilot_rows = []
    try:
        query_ids = sorted(set(qrels) & set(cached) & set(prf_rows))[: args.limit]
        baseline_metrics = [
            metrics_for_query(baseline_full[query_id]["ranked_doc_ids"], qrels[query_id], [10, 50])
            for query_id in query_ids
        ]
        for query_id in query_ids:
            candidate_ids = list(prf_rows[query_id]["ranked_doc_ids"])
            candidates = []
            for rank, doc_id in enumerate(candidate_ids, 1):
                if doc_id not in documents_by_id:
                    continue
                item = dict(documents_by_id[doc_id])
                item["retrieval_rank"] = rank
                candidates.append(item)
            reranked, meta = jev_rerank(
                db,
                queries[query_id],
                candidates,
                provider=args.provider,
                timeout=args.timeout,
                use_cache=True,
                batch_size=args.batch_size,
                retrieval_prior_weight=0.25,
                rank_fusion_k=60,
            )
            ranked_ids = [path_to_doc_id[item["path"]] for item in reranked]
            pilot_rows.append({
                "query_id": query_id,
                "ranked_doc_ids": ranked_ids,
                "metrics": metrics_for_query(ranked_ids, qrels[query_id], [1, 3, 5, 10, 30, 50]),
                "jev": meta,
            })
            print(f"Evaluated {len(pilot_rows)}/{len(query_ids)} queries", file=sys.stderr)
    finally:
        db.close()
    output = args.output or args.cache_root / "experiments/nfcorpus-prf-jev-pilot10.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "nfcorpus-prf-jev-pilot",
        "queries": len(pilot_rows),
        "selected_config": prf_payload["selected_config"],
        "baseline_full_chain_ndcg@10": round(
            sum(item["ndcg@10"] for item in baseline_metrics) / len(baseline_metrics), 6
        ) if baseline_metrics else 0.0,
        "baseline_full_chain_recall@50": round(
            sum(item["recall@50"] for item in baseline_metrics) / len(baseline_metrics), 6
        ) if baseline_metrics else 0.0,
        "summary": {
            "ndcg@10": round(sum(row["metrics"]["ndcg@10"] for row in pilot_rows) / len(pilot_rows), 6)
            if pilot_rows else 0.0,
            "recall@50": round(sum(row["metrics"]["recall@50"] for row in pilot_rows) / len(pilot_rows), 6)
            if pilot_rows else 0.0,
        },
        "results": pilot_rows,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **payload["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
