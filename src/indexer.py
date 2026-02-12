"""Indexer module. Reads KB articles, builds ChromaDB and BM25 indices.

Run once before any demo::

    uv run python -m src.indexer
"""

import json
import sys

import chromadb
from chromadb.errors import NotFoundError
from rich.console import Console

from src.config import CHROMA_PERSIST_DIR, COLLECTION_NAME
from src.embeddings import embed_documents
from src.sparse import BM25Index

console = Console()

_DATA_PATH = "data/kb_articles.json"
_UPSERT_BATCH = 100


def _load_articles() -> list[dict]:
    """Load KB articles from the JSON dataset.

    Returns:
        List of article dicts.

    Raises:
        SystemExit: If the data file is missing.
    """
    try:
        with open(_DATA_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        console.print(
            f"[red]Error:[/red] {_DATA_PATH} not found. "
            "Run `uv run python -m data.generate_dataset` first."
        )
        sys.exit(1)


def index_all() -> tuple[chromadb.Collection, BM25Index]:
    """Create ChromaDB collection and BM25 index from scratch.

    Embeds every document (title + body) via the dense embedding client
    and upserts them into a persistent ChromaDB collection with metadata.
    Also builds an in-memory BM25 index over the same corpus.

    Returns:
        Tuple of (ChromaDB collection, BM25Index).
    """
    articles = _load_articles()
    console.print(f"Loaded [cyan]{len(articles)}[/cyan] articles from {_DATA_PATH}")

    # --- ChromaDB setup ---
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    # Delete existing collection if present so we start fresh
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except (ValueError, NotFoundError):
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # --- Dense embeddings ---
    texts = [a["title"] + " " + a["body"] for a in articles]
    console.print("Embedding documents...")
    embeddings = embed_documents(texts)
    console.print(f"Embedded [cyan]{len(embeddings)}[/cyan] documents")

    # --- Upsert into ChromaDB in batches ---
    for i in range(0, len(articles), _UPSERT_BATCH):
        batch_articles = articles[i : i + _UPSERT_BATCH]
        batch_embeddings = embeddings[i : i + _UPSERT_BATCH]

        ids = [a["doc_id"] for a in batch_articles]
        documents = [a["title"] + "\n\n" + a["body"] for a in batch_articles]
        metadatas = [
            {
                "region": a["region"],
                "product_version": a["product_version"],
                "effective_date": a["effective_date"],
                "category": a["category"],
                "deprecated": a["deprecated"],
                "error_codes_str": ",".join(a.get("error_codes", [])),
            }
            for a in batch_articles
        ]

        collection.upsert(
            ids=ids,
            embeddings=batch_embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    console.print(
        f"Upserted [cyan]{collection.count()}[/cyan] documents into ChromaDB"
    )

    # --- BM25 index ---
    bm25_index = BM25Index(articles)
    console.print("Built BM25 index")

    return collection, bm25_index


def load_existing() -> tuple[chromadb.Collection, BM25Index]:
    """Load an existing ChromaDB collection and rebuild BM25 from disk.

    Use this in demo scripts to avoid re-embedding the entire corpus.

    Returns:
        Tuple of (ChromaDB collection, BM25Index).

    Raises:
        SystemExit: If the ChromaDB collection does not exist.
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except (ValueError, NotFoundError):
        console.print(
            "[red]Error:[/red] ChromaDB collection not found. "
            "Run `uv run python -m src.indexer` first."
        )
        sys.exit(1)

    articles = _load_articles()
    bm25_index = BM25Index(articles)

    console.print(
        f"Loaded collection with [cyan]{collection.count()}[/cyan] documents "
        f"and built BM25 index over [cyan]{len(articles)}[/cyan] articles"
    )
    return collection, bm25_index


if __name__ == "__main__":
    index_all()
    console.print("[green]Indexing complete.[/green]")
