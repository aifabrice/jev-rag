#!/usr/bin/env python3
"""Measure cold per-query latency for every pipeline shown on the project site.

The corpus index and corpus embedding matrix are treated as one-time setup.
Agentic planning, query embeddings, Jev reranking, Passage Gate, and Line
Search run with result caches disabled so the reported latency represents a
new query rather than a cache lookup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jev_test import load_dotenv  # noqa: E402
from local_kb import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    KnowledgeBase,
    _passage_fingerprint,
    load_or_create_embedding_index,
    run_search,
    sha256_text,
)
from scripts.benchmark_beir import (  # noqa: E402
    ensure_dataset,
    load_qrels,
    load_queries,
    materialize_corpus,
)


PIPELINES: tuple[dict[str, Any], ...] = (
    {
        "id": "bm25",
        "label": "BM25 top 30",
        "kwargs": {"retrieval_mode": "bm25", "top_k": 30, "use_jev": False},
    },
    {
        "id": "jev30",
        "label": "BM25 top 30 + Jev",
        "kwargs": {"retrieval_mode": "bm25", "top_k": 30, "use_jev": True},
    },
    {
        "id": "jev50",
        "label": "BM25 top 50 + Jev",
        "kwargs": {"retrieval_mode": "bm25", "top_k": 50, "use_jev": True},
    },
    {
        "id": "lineSearch",
        "label": "Two-level Jev Line Search",
        "kwargs": {
            "retrieval_mode": "line-search",
            "line_search_top_k": 100,
            "use_jev": True,
        },
    },
    {
        "id": "passageGate",
        "label": "Hybrid + Unified Passage Gate",
        "kwargs": {"retrieval_mode": "hybrid-gate", "use_jev": True},
    },
    {
        "id": "agentic",
        "label": "Agentic lexical top 50",
        "kwargs": {"retrieval_mode": "agentic", "use_jev": False},
    },
    {
        "id": "rrf",
        "label": "BM25 + embedding RRF",
        "kwargs": {"retrieval_mode": "hybrid", "use_jev": False},
    },
    {
        "id": "agenticJev",
        "label": "Agentic lexical top 50 + Jev",
        "kwargs": {"retrieval_mode": "agentic", "use_jev": True},
    },
    {
        "id": "hybrid",
        "label": "Hybrid top 50 + Jev",
        "kwargs": {"retrieval_mode": "hybrid", "use_jev": True},
    },
    {
        "id": "agenticHybrid",
        "label": "Agentic Hybrid + Jev/retrieval fusion",
        "kwargs": {"retrieval_mode": "agentic-hybrid", "use_jev": True},
    },
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def install_compatible_embedding_cache(
    kb: KnowledgeBase,
    model: str,
    benchmark_matrix: Path,
    path_to_doc_id: dict[str, str],
) -> Path:
    """Reuse the existing normalized benchmark matrix for latency-only setup.

    The benchmark and application indexes contain the same one-record-per-file
    NFCorpus documents. Reordering the matrix to SQLite passage order avoids a
    second paid corpus-embedding build. Retrieval quality is not evaluated by
    this latency suite.
    """
    import numpy as np

    passages = kb.embedding_passages()
    fingerprint = _passage_fingerprint(model, passages)
    model_id = sha256_text(model)[:12]
    cache_path = kb.db_path.parent / "embeddings" / (
        f"{kb.db_path.stem}-{model_id}-{fingerprint}.npz"
    )
    if cache_path.exists():
        return cache_path

    with np.load(benchmark_matrix, allow_pickle=False) as source:
        doc_ids = [str(value) for value in source["doc_ids"].tolist()]
        vectors = source["vectors"]
        positions = {doc_id: index for index, doc_id in enumerate(doc_ids)}
        ordered = np.asarray(
            [vectors[positions[path_to_doc_id[item["path"]]]] for item in passages],
            dtype=np.float32,
        )
    rowids = np.asarray([int(item["rowid"]) for item in passages], dtype=np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, rowids=rowids, vectors=ordered)
    return cache_path


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "queries": len(rows),
        "median_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(percentile(latencies, 0.95), 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--provider", choices=("openrouter", "typesafe"), default="openrouter")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--cache-root", type=Path, default=ROOT / ".knowledge" / "public-benchmarks"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / ".knowledge"
            / "public-benchmarks"
            / "results"
            / "nfcorpus-pipeline-latency-live.json"
        ),
    )
    parser.add_argument(
        "--pipelines",
        nargs="*",
        choices=[pipeline["id"] for pipeline in PIPELINES],
        help="Optional subset; defaults to every website pipeline.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.queries < 1:
        raise SystemExit("--queries must be positive")
    load_dotenv(args.env_file)

    dataset_root = ensure_dataset("nfcorpus", args.cache_root)
    runtime_root = args.cache_root / "nfcorpus-jev-rag"
    documents, path_to_doc_id = materialize_corpus(dataset_root, runtime_root)
    queries = load_queries(dataset_root)
    qrels = load_qrels(dataset_root, "test")
    kb = KnowledgeBase(runtime_root / "nfcorpus-test.db", documents, [])
    try:
        kb.index("none")
        query_ids = []
        for query_id in sorted(query_id for query_id in qrels if query_id in queries):
            try:
                has_lexical_candidate = bool(kb.lexical_search(queries[query_id], 30))
            except ValueError:
                has_lexical_candidate = False
            if has_lexical_candidate:
                query_ids.append(query_id)
            if len(query_ids) == args.queries:
                break
        if len(query_ids) < args.queries:
            raise RuntimeError("Not enough non-empty BM25 queries for the requested sample")

        matrix_fingerprint = hashlib.sha256(
            json.dumps(
                [
                    DEFAULT_EMBEDDING_MODEL,
                    "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
                    sorted(path_to_doc_id.values()),
                ]
            ).encode("utf-8")
        ).hexdigest()[:16]
        benchmark_matrix = runtime_root / "embeddings" / (
            f"openai%2Ftext-embedding-3-large-{matrix_fingerprint}.npz"
        )
        if not benchmark_matrix.exists():
            matches = sorted(
                path
                for path in (runtime_root / "embeddings").glob("*.npz")
                if not path.name.startswith("queries-")
                and path.name.startswith("openai%2Ftext-embedding-3-large-")
            )
            if not matches:
                raise RuntimeError("Existing NFCorpus corpus embedding matrix was not found")
            benchmark_matrix = matches[0]
        install_compatible_embedding_cache(
            kb, DEFAULT_EMBEDDING_MODEL, benchmark_matrix, path_to_doc_id
        )
        # Load the one-time corpus index before any per-query timer starts.
        load_or_create_embedding_index(kb, DEFAULT_EMBEDDING_MODEL, args.timeout)

        selected = set(args.pipelines or [pipeline["id"] for pipeline in PIPELINES])
        output: dict[str, Any] = {
            "benchmark": "BEIR NFCorpus pipeline latency",
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provider": args.provider,
            "sample": {
                "split": "test",
                "queries": len(query_ids),
                "query_ids": query_ids,
                "selection": "first lexicographically sorted test queries with non-empty BM25 top 30",
            },
            "measurement": {
                "includes": [
                    "local retrieval",
                    "online query embedding when used",
                    "online Agentic planning when used",
                    "online Jev decision calls when used",
                ],
                "excludes": [
                    "one-time corpus indexing",
                    "one-time corpus embedding build",
                    "final answer generation",
                ],
                "result_cache": "disabled",
                "corpus_embedding_matrix": "prebuilt and memory-resident",
            },
            "pipelines": [],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for pipeline in PIPELINES:
            if pipeline["id"] not in selected:
                continue
            rows = []
            print(f"Measuring {pipeline['label']}...", file=sys.stderr, flush=True)
            for index, query_id in enumerate(query_ids, start=1):
                started = time.perf_counter()
                result = run_search(
                    kb,
                    queries[query_id],
                    top_n=50,
                    provider=args.provider,
                    timeout=args.timeout,
                    use_cache=False,
                    embedding_model=DEFAULT_EMBEDDING_MODEL,
                    agentic_rounds=2,
                    agentic_queries=5,
                    agentic_per_query_k=100,
                    agentic_top_k=50,
                    hybrid_top_k=50,
                    vector_top_k=50,
                    rrf_k=60,
                    **pipeline["kwargs"],
                )
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                rows.append(
                    {
                        "query_id": query_id,
                        "latency_ms": elapsed_ms,
                        "candidate_count": result["candidate_count"],
                        "stages": result["timing"],
                    }
                )
                print(
                    f"  {index}/{len(query_ids)} {query_id}: {elapsed_ms:.2f} ms",
                    file=sys.stderr,
                    flush=True,
                )
            output["pipelines"].append(
                {
                    "id": pipeline["id"],
                    "label": pipeline["label"],
                    "summary": summarize(rows),
                    "results": rows,
                }
            )
            args.output.write_text(
                json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        print(json.dumps(output, ensure_ascii=False, indent=2))
    finally:
        kb.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
