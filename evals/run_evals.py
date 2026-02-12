"""Evaluation runner for the RAG pipeline.

Loads the eval set and existing index, runs the full retrieval pipeline
(filter extraction + hybrid search) on each query, and computes retrieval
quality metrics: Recall@5, Recall@10, and MRR (Mean Reciprocal Rank).
Results are printed as a per-category summary table and saved to
``evals/results/run_TIMESTAMP.json``.

Run:
    uv run python -m evals.run_evals
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table

from src.filter_extractor import convert_to_chromadb_where, extract_filters
from src.indexer import load_existing
from src.retriever import search_hybrid

logger = logging.getLogger(__name__)
console = Console()

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Compute Recall@k: fraction of expected doc_ids found in the top-k results.

    Args:
        retrieved_ids: Ordered list of retrieved doc_ids.
        expected_ids: List of expected (relevant) doc_ids.
        k: Cutoff rank.

    Returns:
        Recall score between 0.0 and 1.0.
    """
    if not expected_ids:
        return 1.0
    top_k_set = set(retrieved_ids[:k])
    found = sum(1 for eid in expected_ids if eid in top_k_set)
    return found / len(expected_ids)


def reciprocal_rank(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """Compute Reciprocal Rank: 1/rank of the first relevant document.

    Args:
        retrieved_ids: Ordered list of retrieved doc_ids.
        expected_ids: List of expected (relevant) doc_ids.

    Returns:
        Reciprocal rank (0.0 if no relevant doc is found).
    """
    expected_set = set(expected_ids)
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in expected_set:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def load_eval_set() -> list[dict[str, Any]]:
    """Load the evaluation query set from disk.

    Returns:
        List of eval query dicts.

    Raises:
        SystemExit: If the eval set file is missing.
    """
    if not EVAL_SET_PATH.exists():
        console.print(
            f"[red]Error:[/red] {EVAL_SET_PATH} not found. "
            "Run `uv run python -m data.generate_dataset` first."
        )
        sys.exit(1)

    with open(EVAL_SET_PATH) as f:
        return json.load(f)


def run_eval_query(
    query_entry: dict[str, Any],
    collection: Any,
    bm25_index: Any,
) -> dict[str, Any]:
    """Run one eval query through the pipeline and compute metrics.

    Args:
        query_entry: Eval query dict with ``query``, ``expected_doc_ids``, etc.
        collection: ChromaDB collection.
        bm25_index: BM25Index instance.

    Returns:
        Result dict with query info, retrieved doc_ids, and per-query metrics.
    """
    query = query_entry["query"]
    expected_ids = query_entry["expected_doc_ids"]

    # Extract filters
    try:
        filters = extract_filters(query)
        where = convert_to_chromadb_where(filters)
    except Exception:
        logger.warning("Filter extraction failed for %s; using no filters", query)
        filters = {}
        where = {}

    # Hybrid search
    results = search_hybrid(collection, bm25_index, query, top_k=10, where=where or None)
    retrieved_ids = [r["doc_id"] for r in results]

    # Metrics
    r_at_5 = recall_at_k(retrieved_ids, expected_ids, k=5)
    r_at_10 = recall_at_k(retrieved_ids, expected_ids, k=10)
    rr = reciprocal_rank(retrieved_ids, expected_ids)

    return {
        "query_id": query_entry["query_id"],
        "query": query,
        "category": query_entry["category"],
        "expected_doc_ids": expected_ids,
        "extracted_filters": filters,
        "chromadb_where": where,
        "retrieved_doc_ids": retrieved_ids,
        "recall_at_5": r_at_5,
        "recall_at_10": r_at_10,
        "mrr": rr,
    }


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print a per-category metrics summary table.

    Args:
        results: List of per-query result dicts from ``run_eval_query``.
    """
    # Aggregate by category
    categories: dict[str, list[dict]] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)

    table = Table(
        title="Evaluation Results Summary",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("Category", width=20)
    table.add_column("Queries", justify="right", width=8)
    table.add_column("Recall@5", justify="right", width=10)
    table.add_column("Recall@10", justify="right", width=10)
    table.add_column("MRR", justify="right", width=10)

    all_r5: list[float] = []
    all_r10: list[float] = []
    all_mrr: list[float] = []

    for cat in sorted(categories):
        cat_results = categories[cat]
        n = len(cat_results)
        avg_r5 = sum(r["recall_at_5"] for r in cat_results) / n
        avg_r10 = sum(r["recall_at_10"] for r in cat_results) / n
        avg_mrr = sum(r["mrr"] for r in cat_results) / n

        all_r5.extend(r["recall_at_5"] for r in cat_results)
        all_r10.extend(r["recall_at_10"] for r in cat_results)
        all_mrr.extend(r["mrr"] for r in cat_results)

        table.add_row(
            cat,
            str(n),
            f"{avg_r5:.3f}",
            f"{avg_r10:.3f}",
            f"{avg_mrr:.3f}",
        )

    # Overall row
    total_n = len(results)
    table.add_row(
        "[bold]OVERALL[/bold]",
        f"[bold]{total_n}[/bold]",
        f"[bold]{sum(all_r5) / total_n:.3f}[/bold]",
        f"[bold]{sum(all_r10) / total_n:.3f}[/bold]",
        f"[bold]{sum(all_mrr) / total_n:.3f}[/bold]",
        style="bold",
    )

    console.print()
    console.print(table)
    console.print()


def save_results(results: list[dict[str, Any]]) -> Path:
    """Save detailed eval results to a timestamped JSON file.

    Args:
        results: List of per-query result dicts.

    Returns:
        Path to the saved results file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"run_{timestamp}.json"

    # Compute summary stats for the output
    total = len(results)
    summary = {
        "timestamp": timestamp,
        "total_queries": total,
        "avg_recall_at_5": sum(r["recall_at_5"] for r in results) / total,
        "avg_recall_at_10": sum(r["recall_at_10"] for r in results) / total,
        "avg_mrr": sum(r["mrr"] for r in results) / total,
    }

    output = {
        "summary": summary,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output_path


def main() -> None:
    """Run the full evaluation pipeline."""
    console.print("[bold]Loading eval set...[/bold]")
    eval_set = load_eval_set()
    console.print(f"Loaded [cyan]{len(eval_set)}[/cyan] eval queries")

    console.print("[bold]Loading index...[/bold]")
    collection, bm25_index = load_existing()

    console.print("[bold]Running evaluations...[/bold]")
    results: list[dict[str, Any]] = []
    for i, query_entry in enumerate(eval_set, start=1):
        console.print(
            f"  [{i}/{len(eval_set)}] {query_entry['query_id']}: "
            f"{query_entry['query'][:60]}..."
        )
        result = run_eval_query(query_entry, collection, bm25_index)
        results.append(result)

    # Print summary
    print_summary(results)

    # Save detailed results
    output_path = save_results(results)
    console.print(f"Detailed results saved to [cyan]{output_path}[/cyan]")


if __name__ == "__main__":
    main()
