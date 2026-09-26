#!/usr/bin/env python3
"""Run Jev RAG against a public BEIR retrieval benchmark.

The adapter intentionally uses Jev RAG's real local-folder ingestion and
SQLite FTS5/BM25 search path. Downloaded datasets, materialized documents,
indexes, and results live under .knowledge/ and are ignored by Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jev_test import load_dotenv  # noqa: E402
from local_kb import KnowledgeBase, jev_rerank  # noqa: E402


DATASETS = {
    "nfcorpus": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
        "sha256": "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
    }
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, target: Path) -> None:
    target = target.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            destination = (target / member.filename).resolve()
            if target not in destination.parents and destination != target:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        zipped.extractall(target)


def ensure_dataset(name: str, cache_root: Path) -> Path:
    config = DATASETS[name]
    dataset_root = cache_root / name
    corpus_path = dataset_root / "corpus.jsonl"
    if corpus_path.exists():
        return dataset_root

    cache_root.mkdir(parents=True, exist_ok=True)
    archive = cache_root / f"{name}.zip"
    if not archive.exists():
        print(f"Downloading {config['url']}", file=sys.stderr)
        urllib.request.urlretrieve(config["url"], archive)
    actual = file_sha256(archive)
    if actual != config["sha256"]:
        raise RuntimeError(
            f"Checksum mismatch for {archive}: expected {config['sha256']}, got {actual}"
        )
    safe_extract(archive, cache_root)
    if not corpus_path.exists():
        raise RuntimeError(f"Dataset archive did not contain {corpus_path}")
    return dataset_root


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if raw.strip():
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc


def materialize_corpus(dataset_root: Path, runtime_root: Path) -> tuple[Path, dict[str, str]]:
    documents = runtime_root / "documents"
    mapping_path = runtime_root / "path_to_doc_id.json"
    if documents.exists() and mapping_path.exists():
        return documents, json.loads(mapping_path.read_text(encoding="utf-8"))

    documents.mkdir(parents=True, exist_ok=True)
    path_to_doc_id: dict[str, str] = {}
    for record in jsonl(dataset_root / "corpus.jsonl"):
        doc_id = str(record["_id"])
        filename = urllib.parse.quote(doc_id, safe="") + ".txt"
        title = str(record.get("title") or "").strip()
        body = str(record.get("text") or "").strip()
        content = "\n\n".join(part for part in (title, body) if part) + "\n"
        (documents / filename).write_text(content, encoding="utf-8")
        path_to_doc_id[filename] = doc_id
    mapping_path.write_text(
        json.dumps(path_to_doc_id, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return documents, path_to_doc_id


def load_queries(dataset_root: Path) -> dict[str, str]:
    return {str(item["_id"]): str(item["text"]) for item in jsonl(dataset_root / "queries.jsonl")}


def load_qrels(dataset_root: Path, split: str) -> dict[str, dict[str, int]]:
    path = dataset_root / "qrels" / f"{split}.tsv"
    qrels: dict[str, dict[str, int]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for raw in lines[1:]:
        if not raw.strip():
            continue
        query_id, doc_id, score_text = raw.split("\t")[:3]
        score = int(score_text)
        if score > 0:
            qrels.setdefault(query_id, {})[doc_id] = score
    return qrels


def dcg(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float:
    return sum(
        (2 ** relevant.get(doc_id, 0) - 1) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_ids[:k], 1)
    )


def metrics_for_query(
    ranked_ids: list[str], relevant: dict[str, int], cutoffs: list[int]
) -> dict[str, float]:
    output: dict[str, float] = {}
    ideal_ids = [doc_id for doc_id, _ in sorted(relevant.items(), key=lambda item: -item[1])]
    relevant_ids = set(relevant)
    for k in cutoffs:
        retrieved = ranked_ids[:k]
        hits = [rank for rank, doc_id in enumerate(retrieved, 1) if doc_id in relevant_ids]
        ideal = dcg(ideal_ids, relevant, k)
        output[f"ndcg@{k}"] = dcg(retrieved, relevant, k) / ideal if ideal else 0.0
        output[f"recall@{k}"] = len(hits) / len(relevant_ids) if relevant_ids else 0.0
        output[f"precision@{k}"] = len(hits) / k
        output[f"mrr@{k}"] = 1.0 / hits[0] if hits else 0.0
        denominator = min(len(relevant_ids), k)
        output[f"map@{k}"] = (
            sum(index / rank for index, rank in enumerate(hits, 1)) / denominator
            if denominator
            else 0.0
        )
    return output


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def aggregate(rows: list[dict[str, Any]], cutoffs: list[int]) -> dict[str, Any]:
    metric_names = list(rows[0]["metrics"])
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "queries": len(rows),
        "metrics": {
            name: round(statistics.fmean(row["metrics"][name] for row in rows), 6)
            for name in metric_names
        },
        "latency_ms": {
            "median": round(statistics.median(latencies), 2),
            "p95": round(percentile(latencies, 0.95), 2),
        },
        "cutoffs": cutoffs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Jev RAG on a public BEIR dataset")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="nfcorpus")
    parser.add_argument("--split", choices=["dev", "test", "train"], default="test")
    parser.add_argument("--top-k", type=int, default=100, help="BM25 candidate pool size")
    parser.add_argument("--use-jev", action="store_true", help="Rerank the BM25 candidate pool")
    parser.add_argument("--provider", choices=["openrouter", "typesafe"], default="openrouter")
    parser.add_argument("--limit-queries", type=int, help="Deterministic prefix for a pilot run")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--cache-root", type=Path, default=ROOT / ".knowledge" / "public-benchmarks"
    )
    parser.add_argument("--output", type=Path, help="Write the complete machine-readable result")
    parser.add_argument("--no-jev-cache", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.top_k <= 100:
        raise SystemExit("--top-k must be between 1 and 100")
    if args.limit_queries is not None and args.limit_queries < 1:
        raise SystemExit("--limit-queries must be positive")

    load_dotenv(args.env_file)
    dataset_root = ensure_dataset(args.dataset, args.cache_root)
    runtime_root = args.cache_root / f"{args.dataset}-jev-rag"
    documents, path_to_doc_id = materialize_corpus(dataset_root, runtime_root)
    queries = load_queries(dataset_root)
    qrels = load_qrels(dataset_root, args.split)
    query_ids = sorted(query_id for query_id in qrels if query_id in queries)
    if args.limit_queries:
        query_ids = query_ids[: args.limit_queries]

    db_name = f"{args.dataset}-{args.split}.db"
    kb = KnowledgeBase(runtime_root / db_name, documents, [])
    rows: list[dict[str, Any]] = []
    total_usage: dict[str, float] = {}
    try:
        index_stats = kb.index("none")
        for number, query_id in enumerate(query_ids, 1):
            query = queries[query_id]
            started = time.perf_counter()
            try:
                candidates = kb.lexical_search(query, args.top_k)
            except ValueError:
                candidates = []
            jev_meta: dict[str, Any] = {"used": False}
            if args.use_jev:
                candidates, jev_meta = jev_rerank(
                    kb,
                    query,
                    candidates,
                    provider=args.provider,
                    timeout=args.timeout,
                    use_cache=not args.no_jev_cache,
                )
                for key, value in (jev_meta.get("usage") or {}).items():
                    if isinstance(value, (int, float)):
                        total_usage[key] = total_usage.get(key, 0.0) + float(value)
            ranked_ids = [path_to_doc_id[item["path"]] for item in candidates]
            cutoffs = sorted({k for k in (1, 3, 5, 10, 30, 100) if k <= args.top_k})
            rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "ranked_doc_ids": ranked_ids,
                    "candidate_count": len(ranked_ids),
                    "metrics": metrics_for_query(ranked_ids, qrels[query_id], cutoffs),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "jev": jev_meta,
                }
            )
            if number % 25 == 0 or number == len(query_ids):
                print(f"Evaluated {number}/{len(query_ids)} queries", file=sys.stderr)
    finally:
        kb.close()

    summary = aggregate(rows, cutoffs)
    result = {
        "benchmark": "BEIR",
        "dataset": args.dataset,
        "split": args.split,
        "mode": f"bm25+jev:{args.provider}" if args.use_jev else "bm25",
        "top_k": args.top_k,
        "limited": args.limit_queries is not None,
        "dataset_sha256": DATASETS[args.dataset]["sha256"],
        "index": index_stats,
        "summary": summary,
        "usage": total_usage,
        "results": rows,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
