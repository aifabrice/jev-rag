#!/usr/bin/env python3
"""Offline sweep of the Jev retrieval-prior fusion hyperparameters (w, k).

Zero-cost analysis: this script only reads cached benchmark result JSON files
and the qrels TSV. It NEVER calls any external API / model.

Background
----------
The production pipeline fuses the Jev rerank order with the original retrieval
order via ``apply_retrieval_prior()`` (see ``local_kb.py`` ~L2038-2057)::

    score = 1.0 / (k + jev_rank) + w / (k + retrieval_rank)

then sorts by ``(-score, jev_rank, retrieval_rank)`` and recomputes nDCG@10.

IMPORTANT data note (discovered by inspecting the cached files, not assumed)
--------------------------------------------------------------------------
``apply_retrieval_prior()`` consumes the *pure* Jev rerank order (pre-fusion)
as ``jev_rank``. Two cached dev files exist:

* ``nfcorpus-agentic-hybrid-jev-dev.json``
      -> ``jev.retrieval_prior_applied`` is None  => ``ranked_doc_ids`` is the
         PURE Jev rerank order (no prior fused). This is the correct
         ``jev_rank`` source for re-running the fusion offline.
* ``nfcorpus-agentic-hybrid-jev-prior-dev.json``
      -> ``jev.retrieval_prior_applied`` is True  => ``ranked_doc_ids`` is the
         POST-fusion order (already fused with w=0.25, k=60). Its reported
         summary nDCG@10 = 0.412509 is the current baseline. Re-fusing THIS
         order would be a "double fusion" and does NOT reproduce the baseline.

``retrieval_ranked_doc_ids`` is byte-identical across both files (verified), so
it is used directly as ``retrieval_rank``.

This script therefore:
  1. reads the PURE Jev order as ``jev_rank`` and ``retrieval_ranked_doc_ids``
     as ``retrieval_rank``;
  2. SELF-VALIDATES that fusing with (w=0.25, k=60) reproduces, for every
     query, the post-fusion ``ranked_doc_ids`` of the prior file AND the
     baseline mean nDCG@10 (0.412509);
  3. sweeps the (w, k) grid and reports the optimum vs. the baseline.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Defaults (resolved relative to the repository root)
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / ".knowledge" / "public-benchmarks" / "results"
QRELS_PATH = (
    REPO_ROOT / ".knowledge" / "public-benchmarks" / "nfcorpus" / "qrels" / "dev.tsv"
)

# Pure Jev rerank order (prior NOT applied) -> source of jev_rank.
PURE_JEV_FILE = RESULTS_DIR / "nfcorpus-agentic-hybrid-jev-dev.json"
# Post-fusion baseline (prior applied w=0.25, k=60) -> validation reference.
PRIOR_FILE = RESULTS_DIR / "nfcorpus-agentic-hybrid-jev-prior-dev.json"

# Sweep grid (task spec).
W_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75]
K_GRID = [10, 20, 30, 40, 60, 80, 90, 120]

BASELINE_W = 0.25
BASELINE_K = 60
NDCG_K = 10


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Load BEIR-style qrels TSV: query-id <tab> corpus-id <tab> score (header)."""
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as fh:
        fh.readline()  # drop header line
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], parts[2]
            try:
                rel = int(score)
            except ValueError:
                rel = int(float(score))
            qrels.setdefault(qid, {})[did] = rel
    return qrels


def load_results(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load a benchmark result JSON; return (full_data, results_by_query_id)."""
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    by_q = {r["query_id"]: r for r in data["results"]}
    return data, by_q


# --------------------------------------------------------------------------- #
# Metrics + fusion (faithful replica of apply_retrieval_prior)
# --------------------------------------------------------------------------- #
def dcg(gains: list[float], k: int = NDCG_K) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked_ids: list[str], qrels_q: dict[str, int], k: int = NDCG_K) -> float:
    """Standard BEIR nDCG@k: gain = 2**rel - 1, discount = 1/log2(rank+1)."""
    gains = [2 ** qrels_q.get(d, 0) - 1 for d in ranked_ids]
    ideal = sorted((2 ** v - 1 for v in qrels_q.values()), reverse=True)
    actual = dcg(gains, k)
    best = dcg(ideal, k)
    return actual / best if best > 0 else 0.0


def fuse(jev_ids: list[str], ret_ids: list[str], w: float, k: int) -> list[str]:
    """Replicate apply_retrieval_prior().

    score = 1/(k + jev_rank) + w/(k + retrieval_rank), sorted by
    (-score, jev_rank, retrieval_rank). Ranks are 1-based.
    """
    ret_rank = {d: i + 1 for i, d in enumerate(ret_ids)}
    scored: list[tuple[float, int, int, str]] = []
    for jev_rank, doc in enumerate(jev_ids, start=1):
        rr = ret_rank.get(doc, 10**9)
        score = 1.0 / (k + jev_rank) + w / (k + rr)
        scored.append((score, jev_rank, rr, doc))
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [x[3] for x in scored]


def evaluate(
    jev_by_q: dict[str, dict[str, Any]],
    qrels: dict[str, dict[str, int]],
    query_ids: list[str],
    w: float,
    k: int,
) -> float:
    """Mean nDCG@10 over the given queries for a single (w, k) combination."""
    total = 0.0
    n = 0
    for qid in query_ids:
        r = jev_by_q[qid]
        fused = fuse(r["ranked_doc_ids"], r["retrieval_ranked_doc_ids"], w, k)
        total += ndcg_at_k(fused, qrels.get(qid, {}), NDCG_K)
        n += 1
    return total / n if n else 0.0


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_baseline(
    pure_by_q: dict[str, dict[str, Any]],
    prior_by_q: dict[str, dict[str, Any]],
    qrels: dict[str, dict[str, int]],
    query_ids: list[str],
) -> float:
    """Confirm fuse(pure, 0.25, 60) reproduces the prior file's order + nDCG.

    Returns the reproduced mean nDCG@10. Raises on any order mismatch.
    """
    order_mismatch = 0
    ndcg_sum = 0.0
    for qid in query_ids:
        pu = pure_by_q[qid]
        pr = prior_by_q[qid]
        fused = fuse(
            pu["ranked_doc_ids"], pu["retrieval_ranked_doc_ids"], BASELINE_W, BASELINE_K
        )
        if fused != pr["ranked_doc_ids"]:
            order_mismatch += 1
        ndcg_sum += ndcg_at_k(fused, qrels.get(qid, {}), NDCG_K)
    reproduced = ndcg_sum / len(query_ids)
    if order_mismatch:
        raise AssertionError(
            f"baseline reproduction FAILED: {order_mismatch}/{len(query_ids)} "
            "queries have a different order than the prior file."
        )
    return reproduced


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def fmt_w(w: float) -> str:
    return f"{w:.2f}"


def print_grid(grid: dict[tuple[float, int], float]) -> None:
    """Print the (w x k) nDCG@10 matrix as an aligned table."""
    col_w = 10
    header = "  w \\ k  " + "".join(f"{k:>{col_w}}" for k in K_GRID)
    print(header)
    print("  " + "-" * (len(header) - 2))
    best_val = max(grid.values())
    for w in W_GRID:
        row = f"  {fmt_w(w):>5}  "
        for k in K_GRID:
            v = grid[(w, k)]
            mark = "*" if abs(v - best_val) < 1e-12 else " "
            row += f"{v:>{col_w - 1}.6f}{mark}"
        print(row)
    print("  (* = global optimum)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline (w, k) sweep for the Jev retrieval prior."
    )
    parser.add_argument("--pure-jev", type=Path, default=PURE_JEV_FILE,
                        help="Result JSON with the PURE (pre-fusion) Jev order.")
    parser.add_argument("--prior", type=Path, default=PRIOR_FILE,
                        help="Result JSON with the POST-fusion baseline order.")
    parser.add_argument("--qrels", type=Path, default=QRELS_PATH,
                        help="qrels TSV for the dev split.")
    args = parser.parse_args(argv)

    for p in (args.pure_jev, args.prior, args.qrels):
        if not p.exists():
            print(f"[ERROR] required file not found: {p}", file=sys.stderr)
            return 2

    print("=" * 78)
    print("Jev retrieval-prior offline sweep  (nDCG@10, dev split)")
    print("=" * 78)
    print(f"pure Jev file : {args.pure_jev.name}")
    print(f"prior file    : {args.prior.name}")
    print(f"qrels file    : {args.qrels}")
    print()

    qrels = load_qrels(args.qrels)
    pure_data, pure_by_q = load_results(args.pure_jev)
    prior_data, prior_by_q = load_results(args.prior)

    # Sanity: pure file must NOT have the prior applied.
    sample_pure = next(iter(pure_by_q.values()))
    pure_applied = sample_pure.get("jev", {}).get("retrieval_prior_applied")
    sample_prior = next(iter(prior_by_q.values()))
    prior_applied = sample_prior.get("jev", {}).get("retrieval_prior_applied")
    print(f"pure  retrieval_prior_applied = {pure_applied!r} (expected falsy)")
    print(f"prior retrieval_prior_applied = {prior_applied!r} (expected True)")
    if pure_applied:
        print("[ERROR] the 'pure' file already has the prior applied; "
              "jev_rank would be post-fusion. Aborting.", file=sys.stderr)
        return 3
    print()

    # Queries evaluated: those present in both files (order taken from prior).
    query_ids = [qid for qid in prior_by_q if qid in pure_by_q]
    only_prior = [qid for qid in prior_by_q if qid not in pure_by_q]
    only_pure = [qid for qid in pure_by_q if qid not in prior_by_q]
    print(f"queries: prior={len(prior_by_q)}  pure={len(pure_by_q)}  "
          f"evaluated={len(query_ids)}")
    if only_prior or only_pure:
        print(f"  [warn] only-in-prior={len(only_prior)}  only-in-pure={len(only_pure)}")

    # ---- Self-validation: reproduce the baseline exactly ----
    print()
    print("-" * 78)
    print("SELF-VALIDATION  (fuse pure Jev with w=0.25, k=60)")
    print("-" * 78)
    reproduced = validate_baseline(pure_by_q, prior_by_q, qrels, query_ids)
    prior_summary_ndcg = prior_data["summary"]["metrics"].get("ndcg@10")
    print(f"reproduced order matches prior file : YES (all {len(query_ids)} queries)")
    print(f"reproduced mean nDCG@10            : {reproduced:.6f}")
    print(f"prior file summary nDCG@10         : {prior_summary_ndcg:.6f}")
    delta = abs(reproduced - (prior_summary_ndcg or 0.0))
    print(f"abs difference                     : {delta:.2e}  "
          f"({'OK' if delta < 1e-6 else 'MISMATCH'})")
    if delta >= 1e-6:
        print("[ERROR] baseline not reproduced; results would be unreliable.",
              file=sys.stderr)
        return 4
    baseline_ndcg = reproduced

    # ---- Sweep ----
    print()
    print("-" * 78)
    print(f"GRID SWEEP  w in {W_GRID}")
    print(f"            k in {K_GRID}")
    print("-" * 78)
    grid: dict[tuple[float, int], float] = {}
    for w in W_GRID:
        for k in K_GRID:
            grid[(w, k)] = evaluate(pure_by_q, qrels, query_ids, w, k)

    print_grid(grid)

    # ---- Optimum + baseline comparison ----
    (w_star, k_star), best_ndcg = max(grid.items(), key=lambda kv: kv[1])
    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"optimal (w*, k*)      : ({fmt_w(w_star)}, {k_star})")
    print(f"optimal dev nDCG@10   : {best_ndcg:.6f}")
    print(f"baseline (0.25, 60)   : {baseline_ndcg:.6f}")
    gain = best_ndcg - baseline_ndcg
    pct = (gain / baseline_ndcg * 100) if baseline_ndcg else 0.0
    print(f"absolute gain         : {gain:+.6f}  ({pct:+.3f}% relative)")
    print()

    # Top-5 combinations for context.
    print("top-5 combinations:")
    ranked = sorted(grid.items(), key=lambda kv: kv[1], reverse=True)[:5]
    for rank, ((w, k), v) in enumerate(ranked, start=1):
        tag = "  <-- baseline" if (w == BASELINE_W and k == BASELINE_K) else ""
        print(f"  {rank}. w={fmt_w(w):>4}  k={k:>3}  nDCG@10={v:.6f}{tag}")

    # Where does the baseline rank?
    sorted_vals = sorted(grid.values(), reverse=True)
    baseline_rank = sorted_vals.index(baseline_ndcg) + 1
    print()
    print(f"baseline (0.25,60) rank in grid: {baseline_rank}/{len(sorted_vals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
