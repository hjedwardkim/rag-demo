"""Demo 2 -- Metadata Filtering (Interactive).

Shows that hybrid search without metadata filters returns wrong-scope
results, and that adding filters produces clean, scoped results.  Then
demonstrates LLM-based filter extraction from natural language.

Each step pauses for the presenter to narrate before continuing.

Run::

    uv run python -m demos.demo2_filtering
"""

from rich.console import Console

from src.filter_extractor import convert_to_chromadb_where, extract_filters
from src.indexer import load_existing
from src.retriever import search_hybrid
from src.utils import print_filters, print_results_table, prompt_query, wait_for_enter

console = Console()

DEFAULT_QUERY_A = "What's the authentication setup process?"
DEFAULT_QUERY_B = (
    "How do I set up authentication for our EU deployment on v3? "
    "We upgraded last month."
)


def main() -> None:
    """Run the metadata-filtering demo."""
    console.rule("[bold cyan]Demo 2 — Metadata Filtering[/bold cyan]")
    console.print()

    # Load existing indices
    collection, bm25_index = load_existing()
    console.print()

    # ------------------------------------------------------------------
    # Part A — Manual filters
    # ------------------------------------------------------------------
    console.rule("[bold]Part A — Manual Filters[/bold]")
    console.print()

    query_a = prompt_query(DEFAULT_QUERY_A)

    # Unfiltered
    wait_for_enter("unfiltered hybrid search")
    results_unfiltered = search_hybrid(collection, bm25_index, query_a, top_k=5)
    print_results_table(results_unfiltered, title="Hybrid Search — No Filters")

    # Filtered
    console.print(
        "[bold]Now adding filters:[/bold] region=EU, version=v3.0, deprecated=false\n"
    )
    wait_for_enter("filtered hybrid search")
    where = {
        "$and": [
            {"region": {"$eq": "EU"}},
            {"product_version": {"$eq": "v3.0"}},
            {"deprecated": {"$eq": False}},
        ]
    }
    results_filtered = search_hybrid(
        collection, bm25_index, query_a, top_k=5, where=where
    )
    print_results_table(
        results_filtered,
        title="Hybrid Search — Filtered (EU, v3.0, active)",
    )

    # ------------------------------------------------------------------
    # Part B — LLM filter extraction
    # ------------------------------------------------------------------
    console.rule("[bold]Part B — LLM Filter Extraction[/bold]")
    console.print()

    query_b = prompt_query(DEFAULT_QUERY_B)

    # Extract filters
    wait_for_enter("LLM filter extraction")
    filters = extract_filters(query_b)
    print_filters(filters)

    # Search with extracted filters
    wait_for_enter("search with auto-extracted filters")
    auto_where = convert_to_chromadb_where(filters)
    results_auto = search_hybrid(
        collection, bm25_index, query_b, top_k=5, where=auto_where
    )
    print_results_table(
        results_auto, title="Hybrid Search — Auto-extracted Filters"
    )

    # ------------------------------------------------------------------
    # Interactive loop — try your own queries with LLM filter extraction
    # ------------------------------------------------------------------
    while True:
        console.rule("[dim]Try another query with auto-filtering?[/dim]")
        custom = console.input(
            "[dim]Enter a new query (or press Enter to exit demo):[/dim] "
        ).strip()
        if not custom:
            break
        console.print(f"\n[bold]Query:[/bold] {custom}\n")

        console.print("[dim]Extracting filters...[/dim]")
        filters = extract_filters(custom)
        print_filters(filters)

        auto_where = convert_to_chromadb_where(filters)
        results = search_hybrid(
            collection, bm25_index, custom, top_k=5, where=auto_where
        )
        print_results_table(results, title="Hybrid Search — Auto-extracted Filters")

    console.print("[bold cyan]Demo 2 complete.[/bold cyan]\n")


if __name__ == "__main__":
    main()
