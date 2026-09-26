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
import os
import statistics
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jev_test import load_dotenv  # noqa: E402
from local_kb import KnowledgeBase, jev_rerank, make_snippet  # noqa: E402


DATASETS = {
    "nfcorpus": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
        "sha256": "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
    }
}

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"


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


def request_embeddings(
    texts: list[str],
    model: str,
    timeout: float,
    batch_size: int = 64,
    retries: int = 2,
) -> tuple[list[list[float]], dict[str, float]]:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少环境变量 OPENROUTER_API_KEY")

    vectors: list[list[float]] = []
    total_usage: dict[str, float] = {}
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        body = json.dumps({"model": model, "input": batch}, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "jev-rag-hybrid-benchmark/0.2.0",
            "X-Title": "Jev RAG Hybrid Benchmark",
        }
        payload: dict[str, Any] = {}
        for attempt in range(retries + 1):
            request = urllib.request.Request(
                OPENROUTER_EMBEDDINGS_URL, data=body, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code in {408, 425, 429, 500, 502, 503, 504, 529}
                if retryable and attempt < retries:
                    time.sleep(0.5 * (2**attempt))
                    continue
                raise RuntimeError(f"Embedding HTTP {exc.code}: {detail[:1000]}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < retries:
                    time.sleep(0.5 * (2**attempt))
                    continue
                raise RuntimeError(f"Embedding 网络请求失败: {exc}") from exc

        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(batch):
            raise RuntimeError(f"Embedding 响应格式异常: {json.dumps(payload)[:1000]}")
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        vectors.extend(item["embedding"] for item in ordered)
        for key, value in (payload.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                total_usage[key] = total_usage.get(key, 0.0) + float(value)
    return vectors, total_usage


def load_or_create_corpus_embeddings(
    dataset_root: Path,
    runtime_root: Path,
    model: str,
    timeout: float,
) -> tuple[Any, list[str], dict[str, Any]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("混合检索 benchmark 需要 numpy（pip install numpy）") from exc

    records = list(jsonl(dataset_root / "corpus.jsonl"))
    doc_ids = [str(record["_id"]) for record in records]
    fingerprint = hashlib.sha256(
        json.dumps([model, DATASETS["nfcorpus"]["sha256"], doc_ids]).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = runtime_root / "embeddings"
    cache_path = cache_dir / f"{urllib.parse.quote(model, safe='')}-{fingerprint}.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_ids = [str(value) for value in cached["doc_ids"].tolist()]
        if cached_ids == doc_ids:
            return cached["vectors"], doc_ids, {
                "model": model,
                "cache_hit": True,
                "documents": len(doc_ids),
                "dimensions": int(cached["vectors"].shape[1]),
                "usage": {},
            }

    texts = [
        "\n\n".join(
            part for part in (str(record.get("title") or ""), str(record.get("text") or ""))
            if part.strip()
        )
        for record in records
    ]
    started = time.perf_counter()
    raw_vectors: list[list[float]] = []
    usage: dict[str, float] = {}
    batch_size = 64
    for offset in range(0, len(texts), batch_size):
        batch_vectors, batch_usage = request_embeddings(
            texts[offset : offset + batch_size], model, timeout, batch_size=batch_size
        )
        raw_vectors.extend(batch_vectors)
        for key, value in batch_usage.items():
            usage[key] = usage.get(key, 0.0) + value
        print(
            f"Embedded corpus {min(offset + batch_size, len(texts))}/{len(texts)} documents",
            file=sys.stderr,
        )
    vectors = np.asarray(raw_vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, doc_ids=np.asarray(doc_ids), vectors=vectors)
    return vectors, doc_ids, {
        "model": model,
        "cache_hit": False,
        "documents": len(doc_ids),
        "dimensions": int(vectors.shape[1]),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "usage": usage,
    }


def load_or_create_query_embeddings(
    runtime_root: Path,
    model: str,
    query_ids: list[str],
    queries: dict[str, str],
    timeout: float,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("混合检索 benchmark 需要 numpy（pip install numpy）") from exc

    fingerprint = hashlib.sha256(
        json.dumps(
            [model, [(query_id, queries[query_id]) for query_id in query_ids]],
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = runtime_root / "embeddings"
    cache_path = cache_dir / f"queries-{urllib.parse.quote(model, safe='')}-{fingerprint}.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_ids = [str(value) for value in cached["query_ids"].tolist()]
        if cached_ids == query_ids:
            return dict(zip(query_ids, cached["vectors"].tolist())), {
                "model": model,
                "cache_hit": True,
                "queries": len(query_ids),
                "dimensions": int(cached["vectors"].shape[1]),
                "usage": {},
            }

    started = time.perf_counter()
    vectors, usage = request_embeddings(
        [queries[query_id] for query_id in query_ids],
        model,
        timeout,
        batch_size=64,
    )
    matrix = np.asarray(vectors, dtype=np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, query_ids=np.asarray(query_ids), vectors=matrix)
    return dict(zip(query_ids, vectors)), {
        "model": model,
        "cache_hit": False,
        "queries": len(query_ids),
        "dimensions": int(matrix.shape[1]),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "usage": usage,
    }


def vector_rank(
    query_vector: list[float], corpus_vectors: Any, doc_ids: list[str], limit: int
) -> list[str]:
    import numpy as np

    query = np.asarray(query_vector, dtype=np.float32)
    query /= max(float(np.linalg.norm(query)), 1e-12)
    scores = corpus_vectors @ query
    limit = min(max(1, limit), len(doc_ids))
    indexes = np.argpartition(-scores, limit - 1)[:limit]
    indexes = indexes[np.argsort(-scores[indexes])]
    return [doc_ids[int(index)] for index in indexes]


def reciprocal_rank_fusion(
    lexical: list[dict[str, Any]],
    vector_doc_ids: list[str],
    documents: dict[str, dict[str, Any]],
    path_to_doc_id: dict[str, str],
    limit: int,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    lexical_by_id = {path_to_doc_id[item["path"]]: item for item in lexical}
    lexical_ranks = {doc_id: rank for rank, doc_id in enumerate(lexical_by_id, 1)}
    vector_ranks = {doc_id: rank for rank, doc_id in enumerate(vector_doc_ids, 1)}
    scores: dict[str, float] = {}
    for doc_id, rank in lexical_ranks.items():
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    for doc_id, rank in vector_ranks.items():
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    ranked_ids = sorted(
        scores,
        key=lambda doc_id: (
            -scores[doc_id],
            lexical_ranks.get(doc_id, 10**9),
            vector_ranks.get(doc_id, 10**9),
        ),
    )[:limit]
    results: list[dict[str, Any]] = []
    for position, doc_id in enumerate(ranked_ids, 1):
        item = dict(lexical_by_id.get(doc_id) or documents[doc_id])
        item["original_bm25_rank"] = lexical_ranks.get(doc_id)
        item["vector_rank"] = vector_ranks.get(doc_id)
        item["rrf_score"] = round(scores[doc_id], 8)
        # jev_rerank uses bm25_rank as its deterministic tie breaker. For a
        # hybrid candidate pool, the fused retrieval rank is the fair fallback.
        item["bm25_rank"] = position
        item["retrieval_rank"] = position
        results.append(item)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Jev RAG on a public BEIR dataset")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="nfcorpus")
    parser.add_argument("--split", choices=["dev", "test", "train"], default="test")
    parser.add_argument("--top-k", type=int, default=100, help="BM25 candidate pool size")
    parser.add_argument("--use-jev", action="store_true", help="Rerank the BM25 candidate pool")
    parser.add_argument("--provider", choices=["openrouter", "typesafe"], default="openrouter")
    parser.add_argument(
        "--embedding-model",
        help="Enable BM25 + OpenRouter embedding retrieval with reciprocal-rank fusion",
    )
    parser.add_argument("--vector-top-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
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
    if not 1 <= args.vector_top_k <= 100:
        raise SystemExit("--vector-top-k must be between 1 and 100")
    if args.rrf_k < 1:
        raise SystemExit("--rrf-k must be positive")
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
    cutoffs = sorted({k for k in (1, 3, 5, 10, 30, args.top_k, 100) if k <= args.top_k})

    db_name = f"{args.dataset}-{args.split}.db"
    kb = KnowledgeBase(runtime_root / db_name, documents, [])
    rows: list[dict[str, Any]] = []
    total_usage: dict[str, float] = {}
    embedding_usage: dict[str, float] = {}
    embedding_index: dict[str, Any] | None = None
    embedding_queries: dict[str, Any] | None = None
    try:
        index_stats = kb.index("none")
        corpus_vectors = None
        vector_doc_ids: list[str] = []
        documents_by_id: dict[str, dict[str, Any]] = {}
        query_vectors_by_id: dict[str, list[float]] = {}
        if args.embedding_model:
            corpus_vectors, vector_doc_ids, embedding_index = load_or_create_corpus_embeddings(
                dataset_root, runtime_root, args.embedding_model, args.timeout
            )
            query_vectors_by_id, embedding_queries = load_or_create_query_embeddings(
                runtime_root,
                args.embedding_model,
                query_ids,
                queries,
                args.timeout,
            )
            for key, value in (embedding_queries.get("usage") or {}).items():
                if isinstance(value, (int, float)):
                    embedding_usage[key] = embedding_usage.get(key, 0.0) + float(value)
            for row in kb.connection.execute(
                """
                SELECT rowid, doc_id, passage_no, path, title, heading,
                       start_line, end_line, body
                FROM passages ORDER BY rowid
                """
            ):
                item = dict(row)
                item["passage_no"] = int(item["passage_no"])
                item["start_line"] = int(item["start_line"])
                item["end_line"] = int(item["end_line"])
                item["snippet"] = make_snippet(item["body"], "")
                documents_by_id[path_to_doc_id[item["path"]]] = item
        for number, query_id in enumerate(query_ids, 1):
            query = queries[query_id]
            started = time.perf_counter()
            try:
                candidates = kb.lexical_search(query, args.top_k)
            except ValueError:
                candidates = []
            retrieval_meta: dict[str, Any] = {"mode": "bm25"}
            if args.embedding_model:
                dense_ids = vector_rank(
                    query_vectors_by_id[query_id],
                    corpus_vectors,
                    vector_doc_ids,
                    args.vector_top_k,
                )
                candidates = reciprocal_rank_fusion(
                    candidates,
                    dense_ids,
                    documents_by_id,
                    path_to_doc_id,
                    args.top_k,
                    args.rrf_k,
                )
                retrieval_meta = {
                    "mode": "bm25+embedding+rrf",
                    "embedding_model": args.embedding_model,
                    "vector_top_k": args.vector_top_k,
                    "rrf_k": args.rrf_k,
                }
            retrieval_ranked_ids = [path_to_doc_id[item["path"]] for item in candidates]
            retrieval_metrics = metrics_for_query(
                retrieval_ranked_ids, qrels[query_id], cutoffs
            )
            retrieval_latency_ms = round((time.perf_counter() - started) * 1000, 2)
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
            rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "ranked_doc_ids": ranked_ids,
                    "candidate_count": len(ranked_ids),
                    "metrics": metrics_for_query(ranked_ids, qrels[query_id], cutoffs),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "retrieval_ranked_doc_ids": retrieval_ranked_ids,
                    "retrieval_metrics": retrieval_metrics,
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "retrieval": retrieval_meta,
                    "jev": jev_meta,
                }
            )
            if number % 25 == 0 or number == len(query_ids):
                print(f"Evaluated {number}/{len(query_ids)} queries", file=sys.stderr)
    finally:
        kb.close()

    summary = aggregate(rows, cutoffs)
    retrieval_summary = aggregate(
        [
            {"metrics": row["retrieval_metrics"], "latency_ms": row["retrieval_latency_ms"]}
            for row in rows
        ],
        cutoffs,
    )
    result = {
        "benchmark": "BEIR",
        "dataset": args.dataset,
        "split": args.split,
        "mode": (
            f"bm25+embedding+rrf+jev:{args.provider}"
            if args.embedding_model and args.use_jev
            else "bm25+embedding+rrf"
            if args.embedding_model
            else f"bm25+jev:{args.provider}"
            if args.use_jev
            else "bm25"
        ),
        "top_k": args.top_k,
        "vector_top_k": args.vector_top_k if args.embedding_model else None,
        "embedding_model": args.embedding_model,
        "rrf_k": args.rrf_k if args.embedding_model else None,
        "limited": args.limit_queries is not None,
        "dataset_sha256": DATASETS[args.dataset]["sha256"],
        "index": index_stats,
        "summary": summary,
        "retrieval_summary": retrieval_summary,
        "usage": total_usage,
        "embedding_index": embedding_index,
        "embedding_queries": embedding_queries,
        "embedding_usage": embedding_usage,
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
