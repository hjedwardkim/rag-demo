"""Pretty-printing helpers for RAG demo terminal output.

Uses the ``rich`` library for formatted tables, panels, and highlighted text.
"""

from __future__ import annotations

import re

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def wait_for_enter(label: str = "next step") -> None:
    """Pause execution until the presenter presses Enter.

    Args:
        label: Short description shown in the prompt.
    """
    console.print(f"  [dim]Press Enter for {label}...[/dim]", end="")
    input()
    console.print()


def prompt_query(default: str) -> str:
    """Prompt the presenter for a query, with a default pre-filled.

    Args:
        default: The default query shown if the presenter just presses Enter.

    Returns:
        The query string to use.
    """
    console.print(f"[bold]Default query:[/bold] [cyan]{default}[/cyan]")
    custom = console.input("[dim]Enter a custom query (or press Enter for default):[/dim] ").strip()
    query = custom if custom else default
    console.print(f"\n[bold]Query:[/bold] {query}\n")
    return query


def print_results_table(
    results: list[dict],
    title: str = "Results",
    highlight_doc_id: str | None = None,
) -> None:
    """Render a retrieval results table to the console.

    Args:
        results: List of result dicts, each containing at minimum
            ``doc_id``, ``title``, ``region``, ``product_version``,
            ``deprecated``, and ``score``.
        title: Table title.
        highlight_doc_id: If provided, the row matching this doc_id is
            rendered in bold green.
    """
    table = Table(title=title, box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Rank", justify="right", width=5)
    table.add_column("Doc ID", width=9)
    table.add_column("Title", max_width=50)
    table.add_column("Region", width=7)
    table.add_column("Version", width=8)
    table.add_column("Deprecated", width=11)
    table.add_column("Score", justify="right", width=10)

    for i, r in enumerate(results, start=1):
        rank = str(r.get("rank", i))
        doc_id = r.get("doc_id", "")
        doc_title = r.get("title", "")
        region = r.get("region", "")
        version = r.get("product_version", "")
        deprecated = r.get("deprecated", False)
        score = r.get("score", 0.0)

        dep_text = Text("Yes", style="bold red") if deprecated else Text("No", style="dim")
        score_str = f"{score:.4f}"

        row_style = "bold green" if doc_id == highlight_doc_id else None

        table.add_row(
            rank,
            doc_id,
            doc_title,
            region,
            version,
            dep_text,
            score_str,
            style=row_style,
        )

    console.print(table)
    console.print()


def print_comparison(
    results_sets: list[tuple[str, list[dict]]],
    highlight_doc_id: str | None = None,
) -> None:
    """Print multiple result sets as sequential tables for comparison.

    Args:
        results_sets: List of ``(label, results)`` tuples, e.g.
            ``[("Dense", dense_results), ("BM25", sparse_results)]``.
        highlight_doc_id: Optional doc_id to highlight across all tables.
    """
    for label, results in results_sets:
        print_results_table(results, title=label, highlight_doc_id=highlight_doc_id)


def print_filters(filters: dict) -> None:
    """Pretty-print extracted metadata filters in a rich Panel.

    Args:
        filters: Dict of extracted filter key-value pairs. An empty dict
            indicates no filters were extracted.
    """
    if not filters:
        console.print(
            Panel("No metadata filters extracted.", title="Extracted Filters", box=box.ROUNDED)
        )
        return

    lines: list[str] = []
    for key, value in filters.items():
        lines.append(f"[bold cyan]{key}[/bold cyan]: {value}")

    console.print(
        Panel("\n".join(lines), title="Extracted Filters", box=box.ROUNDED)
    )
    console.print()


def print_answer(answer: str, doc_ids: list[str]) -> None:
    """Print a generated answer with referenced doc IDs highlighted.

    Args:
        answer: The generated answer text.
        doc_ids: List of doc IDs that were provided as context. Occurrences
            of these IDs in the answer text are highlighted.
    """
    highlighted = answer
    for doc_id in doc_ids:
        highlighted = re.sub(
            re.escape(doc_id),
            f"[bold green]{doc_id}[/bold green]",
            highlighted,
        )

    console.print(
        Panel(
            highlighted,
            title="Generated Answer",
            subtitle=f"Context: {', '.join(doc_ids)}",
            box=box.ROUNDED,
        )
    )
    console.print()
