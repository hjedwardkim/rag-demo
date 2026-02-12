"""Demo 3 -- Full Pipeline with Langfuse Tracing (Interactive).

Runs the full RAG pipeline (filter extraction -> hybrid search -> rerank
-> LLM generation) with Langfuse tracing for each query.  After
execution, directs the user to the Langfuse dashboard to inspect traces.

Each pipeline step pauses for the presenter to narrate before continuing.
After the pre-set queries, an interactive loop lets the presenter try
custom queries.

Run::

    uv run python -m demos.demo3_tracing
"""

from rich.console import Console

from src.config import LANGFUSE_HOST, LLM_MODEL
from src.filter_extractor import convert_to_chromadb_where, extract_filters
from src.generator import generate_answer
from src.indexer import load_existing
from src.reranker import Reranker
from src.retriever import search_hybrid
from src.tracing import NullLangfuse, init_langfuse
from src.utils import (
    print_answer,
    print_filters,
    print_results_table,
    prompt_query,
    wait_for_enter,
)

console = Console()

SUGGESTED_QUERIES = [
    {
        "query": "How do I fix error E-4012 in the EU region?",
        "description": "Clean hit -- correct doc should be rank 1 after filtering",
    },
    {
        "query": "What's the billing process?",
        "description": "Broad query -- shows how lack of filters leads to mixed results",
    },
    {
        "query": "How do I set up SSO for our APAC v3 deployment?",
        "description": "Multi-doc topic -- may need multiple chunks",
    },
]


def run_traced_query(
    query: str,
    collection,
    bm25_index,
    reranker: Reranker,
    langfuse,
    *,
    step_by_step: bool = True,
) -> None:
    """Run a single query through the full traced pipeline.

    Args:
        query: The user query.
        collection: ChromaDB collection.
        bm25_index: BM25 index.
        reranker: Reranker instance.
        langfuse: Langfuse client (or NullLangfuse).
        step_by_step: If True, pause between each pipeline step.
    """
    pause = wait_for_enter if step_by_step else (lambda _label: None)

    with langfuse.start_as_current_observation(
        as_type="span",
        name="rag-pipeline",
        input={"query": query},
    ) as trace_span:
        # Filter extraction
        pause("filter extraction")
        with trace_span.start_as_current_observation(
            as_type="span",
            name="filter-extraction",
            input={"query": query},
        ) as filter_span:
            filters = extract_filters(query)
            where = convert_to_chromadb_where(filters)
            filter_span.update(output={"filters": filters, "where": where})
        print_filters(filters)

        # Hybrid retrieval
        pause("hybrid retrieval")
        with trace_span.start_as_current_observation(
            as_type="span",
            name="hybrid-retrieval",
            input={"query": query, "where": str(where)},
        ) as retrieval_span:
            results = search_hybrid(
                collection, bm25_index, query, top_k=10, where=where
            )
            retrieval_span.update(
                output={
                    "result_count": len(results),
                    "doc_ids": [r["doc_id"] for r in results],
                    "top_scores": [r["score"] for r in results[:5]],
                }
            )
        print_results_table(results, title="Retrieved (top 10)")

        # Reranking
        pause("reranking")
        with trace_span.start_as_current_observation(
            as_type="span",
            name="reranking",
            input={"candidate_count": len(results)},
        ) as rerank_span:
            reranked = reranker.rerank(query, results, top_k=5)
            rerank_span.update(
                output={"doc_ids": [r["doc_id"] for r in reranked]}
            )
        print_results_table(reranked, title="After Reranking (top 5)")

        # Answer generation
        pause("answer generation (LLM call)")
        with trace_span.start_as_current_observation(
            as_type="generation",
            name="answer-generation",
            model=LLM_MODEL,
            input={
                "query": query,
                "context_doc_ids": [r["doc_id"] for r in reranked],
            },
        ) as gen_span:
            answer = generate_answer(query, reranked)
            gen_span.update(output={"answer": answer})
        print_answer(answer, [r["doc_id"] for r in reranked])

    langfuse.flush()


def main() -> None:
    """Run the full traced RAG pipeline demo."""
    console.rule("[bold cyan]Demo 3 — Full Pipeline with Tracing[/bold cyan]")
    console.print()

    # 1. Load existing indices
    collection, bm25_index = load_existing()
    console.print()

    # 2. Initialize Langfuse
    langfuse = init_langfuse()
    is_null = isinstance(langfuse, NullLangfuse)
    if is_null:
        console.print(
            "[yellow]Warning:[/yellow] Langfuse credentials not configured. "
            "Tracing will be no-op. Set LANGFUSE_PUBLIC_KEY and "
            "LANGFUSE_SECRET_KEY in .env to enable tracing.\n"
        )
    else:
        console.print("[green]Langfuse tracing enabled.[/green]\n")

    # 3. Initialize reranker
    console.print("Loading reranker model...")
    reranker = Reranker()
    console.print()

    # 4. Walk through suggested queries one at a time
    for i, q in enumerate(SUGGESTED_QUERIES, start=1):
        console.rule(
            f"[bold]Query {i}/{len(SUGGESTED_QUERIES)}[/bold]"
        )
        console.print(f"[dim]{q['description']}[/dim]\n")

        query = prompt_query(q["query"])
        run_traced_query(
            query, collection, bm25_index, reranker, langfuse, step_by_step=True
        )

        # Pause between suggested queries (unless it's the last one)
        if i < len(SUGGESTED_QUERIES):
            wait_for_enter("next query")

    # 5. Interactive loop
    while True:
        console.rule("[dim]Try another query through the full pipeline?[/dim]")
        custom = console.input(
            "[dim]Enter a new query (or press Enter to exit demo):[/dim] "
        ).strip()
        if not custom:
            break
        console.print(f"\n[bold]Query:[/bold] {custom}\n")
        run_traced_query(
            custom, collection, bm25_index, reranker, langfuse, step_by_step=True
        )

    # 6. Print Langfuse dashboard URL
    console.print()
    if not is_null:
        console.print(
            f"[bold green]View traces at:[/bold green] {LANGFUSE_HOST}/traces"
        )
    else:
        console.print(
            "[dim]Tracing was disabled. Configure Langfuse to view traces.[/dim]"
        )

    console.print("[bold cyan]Demo 3 complete.[/bold cyan]\n")


if __name__ == "__main__":
    main()
