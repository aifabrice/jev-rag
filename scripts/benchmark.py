#!/usr/bin/env python3
"""Run a small, reproducible retrieval benchmark against a local folder."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_kb import KnowledgeBase, run_search  # noqa: E402


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        case = json.loads(raw)
        if not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError(f"{path}:{line_number}: query must be a non-empty string")
        expected = case.get("expected_paths")
        if not isinstance(expected, list) or not expected:
            raise ValueError(f"{path}:{line_number}: expected_paths must be a non-empty list")
        cases.append(case)
    if not cases:
        raise ValueError(f"{path}: no benchmark cases found")
    return cases


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Jev RAG retrieval on JSONL cases")
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "benchmarks" / "sample_queries.jsonl",
    )
    parser.add_argument("--documents", type=Path, default=ROOT / "knowledge")
    parser.add_argument("--db", type=Path, default=ROOT / ".knowledge" / "benchmark.db")
    parser.add_argument("--chunking", choices=["auto", "none", "paragraph"], default="auto")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--use-jev", action="store_true")
    parser.add_argument("--provider", choices=["openrouter", "typesafe"], default="openrouter")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cases = load_cases(args.cases)
    kb = KnowledgeBase(args.db, args.documents, [])
    rows: list[dict[str, Any]] = []
    try:
        kb.index(args.chunking)
        for case in cases:
            result = run_search(
                kb,
                case["query"],
                top_k=args.top_k,
                top_n=args.top_n,
                use_jev=args.use_jev,
                provider=args.provider,
                timeout=args.timeout,
                threshold=args.threshold,
            )
            expected = {str(path) for path in case["expected_paths"]}
            first_rank = next(
                (
                    int(item["final_rank"])
                    for item in result["results"]
                    if str(item["path"]) in expected
                ),
                None,
            )
            rows.append(
                {
                    "query": case["query"],
                    "expected_paths": sorted(expected),
                    "first_relevant_rank": first_rank,
                    "hit": first_rank is not None,
                    "reciprocal_rank": 0.0 if first_rank is None else 1.0 / first_rank,
                    "returned_count": result["returned_count"],
                    "latency_ms": float(result["timing"]["total_ms"]),
                }
            )
    finally:
        kb.close()

    latencies = [row["latency_ms"] for row in rows]
    summary = {
        "mode": "bm25+jev" if args.use_jev else "bm25",
        "cases": len(rows),
        "hit_rate": round(sum(row["hit"] for row in rows) / len(rows), 4),
        "mrr": round(statistics.fmean(row["reciprocal_rank"] for row in rows), 4),
        "latency_ms": {
            "median": round(statistics.median(latencies), 2),
            "p95": round(percentile(latencies, 0.95), 2),
        },
        "results": rows,
    }
    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Mode: {summary['mode']} | cases: {summary['cases']}")
        print(f"Hit rate: {summary['hit_rate']:.1%} | MRR: {summary['mrr']:.4f}")
        print(
            "Latency: "
            f"median {summary['latency_ms']['median']:.2f} ms | "
            f"p95 {summary['latency_ms']['p95']:.2f} ms"
        )
        for row in rows:
            rank = row["first_relevant_rank"] if row["first_relevant_rank"] else "miss"
            print(f"- rank={rank} {row['latency_ms']:.2f} ms | {row['query']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
