"""Demo 1 -- Naive vs. Hybrid Retrieval (Interactive).

Demonstrates that dense-only search retrieves semantically similar but
potentially wrong documents, while hybrid search with RRF finds the
exact match.  Optionally applies cross-encoder reranking.

Each step pauses for the presenter to narrate before continuing.

Run::

    uv run python -m demos.demo1_retrieval
"""

import json
from pathlib import Path

from rich.console import Console

from src.indexer import load_existing
from src.reranker import Reranker
from src.retriever import search_dense, search_hybrid, search_sparse
from src.utils import (
    print_results_table,
    prompt_query,
    wait_for_enter,
)

console = Console()

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "kb_articles.json"

DEFAULT_QUERY = "authentication stack timeout error E-4011 in EU"

# KB-0031: the non-deprecated EU article for E-4011.
# Dense search buries this under E-4012/E-1009/E-1015 articles that live
# in the same "auth timeout" embedding neighbourhood but have wrong codes.
TARGET_DOC_ID = "KB-0031"


def _find_target_doc_id(query: str) -> str | None:
    """Return the hardcoded target for the default query, or try to find one dynamically."""
    import re as _re

    query_upper = query.upper()

    code_match = _re.search(r"E-\d{4}", query_upper)
    if not code_match:
        return None
    error_code = code_match.group()

    region = None
    for r in ("EU", "US", "APAC"):
        if r in query_upper:
            region = r
            break
    if region is None:
        return None

    # Fast-path for the default query
    if error_code == "E-4011" and region == "EU":
        return TARGET_DOC_ID

    articles = json.loads(DATA_PATH.read_text())
    for a in articles:
        if (
            error_code in a.get("error_codes", [])
            and a.get("region") == region
            and not a.get("deprecated", False)
        ):
            return a["doc_id"]
    return None


def run_demo(collection, bm25_index, query: str, reranker: Reranker) -> None:
    """Run a single retrieval comparison for the given query."""
    target = _find_target_doc_id(query)
    if target:
        console.print(f"[bold]Expected doc:[/bold] [green]{target}[/green]\n")

    # Dense only
    wait_for_enter("dense search (embeddings only)")
    results_dense = search_dense(collection, query, top_k=5)
    print_results_table(
        results_dense, title="Dense Search (Embeddings Only)", highlight_doc_id=target
    )

    # BM25 only
    wait_for_enter("BM25 search (keyword only)")
    results_sparse = search_sparse(bm25_index, query, top_k=5)
    print_results_table(
        results_sparse, title="Sparse Search (BM25 Only)", highlight_doc_id=target
    )

    # Hybrid RRF
    wait_for_enter("hybrid search (RRF fusion)")
    results_hybrid = search_hybrid(collection, bm25_index, query, top_k=5)
    print_results_table(
        results_hybrid, title="Hybrid Search (RRF Fusion)", highlight_doc_id=target
    )

    # Rerank
    wait_for_enter("reranking")
    results_reranked = reranker.rerank(query, results_hybrid, top_k=5)
    print_results_table(
        results_reranked, title="After Reranking", highlight_doc_id=target
    )

    # Summary
    console.rule("[bold cyan]Summary[/bold cyan]")
    console.print()
    methods = [
        ("Dense", results_dense),
        ("BM25", results_sparse),
        ("Hybrid (RRF)", results_hybrid),
        ("Hybrid + Rerank", results_reranked),
    ]
    for method_name, results in methods:
        doc_ids = []
        for r in results:
            did = r["doc_id"]
            if did == target:
                doc_ids.append(f"[green]{did}[/green]")
            else:
                doc_ids.append(did)
        rank = next(
            (i + 1 for i, r in enumerate(results) if r["doc_id"] == target),
            None,
        )
        rank_str = f"rank {rank}" if rank else "[red]not in top 5[/red]"
        console.print(
            f"  [bold]{method_name:20s}[/bold]  {rank_str:>20s}  │  {', '.join(doc_ids)}"
        )
    console.print()


def main() -> None:
    """Run the naive-vs-hybrid retrieval demo."""
    console.rule("[bold cyan]Demo 1 — Naive vs. Hybrid Retrieval[/bold cyan]")
    console.print()

    # Load indices and reranker once
    collection, bm25_index = load_existing()
    console.print()

    console.print("Loading reranker model...")
    reranker = Reranker()
    console.print()

    # First run with default (or custom) query
    query = prompt_query(DEFAULT_QUERY)
    run_demo(collection, bm25_index, query, reranker)

    # Interactive loop
    while True:
        console.rule("[dim]Try another query?[/dim]")
        custom = console.input(
            "[dim]Enter a new query (or press Enter to exit demo):[/dim] "
        ).strip()
        if not custom:
            break
        console.print(f"\n[bold]Query:[/bold] {custom}\n")
        run_demo(collection, bm25_index, custom, reranker)

    console.print("[bold cyan]Demo 1 complete.[/bold cyan]\n")


if __name__ == "__main__":
    main()
