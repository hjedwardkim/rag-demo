"""Core retrieval module supporting dense, sparse, and hybrid (RRF) search.

Provides three search modes plus a reciprocal-rank-fusion helper and
post-hoc BM25 metadata filtering for ChromaDB-style where clauses.
"""

import logging

import chromadb

from src.config import DEFAULT_TOP_K, RRF_K
from src.embeddings import embed_query
from src.sparse import BM25Index

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _chroma_to_results(query_result: dict) -> list[dict]:
    """Convert ChromaDB query output into the standard result format.

    Args:
        query_result: Dict returned by ``collection.query()``.

    Returns:
        List of result dicts with ``doc_id``, ``title``, ``body``,
        ``region``, ``product_version``, ``category``, ``deprecated``,
        ``score``, and ``rank``.
    """
    results: list[dict] = []
    ids = query_result["ids"][0]
    documents = query_result["documents"][0]
    metadatas = query_result["metadatas"][0]
    distances = query_result["distances"][0]

    for rank, (doc_id, doc_text, meta, dist) in enumerate(
        zip(ids, documents, metadatas, distances), start=1
    ):
        # ChromaDB cosine distance: lower is better.  Convert to a
        # similarity score so higher == better.
        score = 1.0 - dist

        parts = doc_text.split("\n\n", 1)
        title = parts[0]
        body = parts[1] if len(parts) > 1 else ""

        results.append(
            {
                "doc_id": doc_id,
                "title": title,
                "body": body,
                "region": meta.get("region", ""),
                "product_version": meta.get("product_version", ""),
                "category": meta.get("category", ""),
                "deprecated": meta.get("deprecated", False),
                "score": score,
                "rank": rank,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Post-hoc metadata filtering for BM25 results
# ---------------------------------------------------------------------------

def _evaluate_condition(meta: dict, condition: dict) -> bool:
    """Evaluate a single ChromaDB where condition against metadata.

    Supports ``$eq``, ``$ne``, ``$gt``, ``$gte``, ``$lt``, ``$lte``,
    ``$in``, ``$nin``, as well as nested ``$and`` / ``$or``.

    Args:
        meta: Metadata dict from an article.
        condition: A single ChromaDB where condition.

    Returns:
        True if the metadata satisfies the condition.
    """
    if "$and" in condition:
        return all(_evaluate_condition(meta, c) for c in condition["$and"])
    if "$or" in condition:
        return any(_evaluate_condition(meta, c) for c in condition["$or"])

    for field, constraint in condition.items():
        value = meta.get(field)
        if isinstance(constraint, dict):
            for op, target in constraint.items():
                if op == "$eq" and value != target:
                    return False
                if op == "$ne" and value == target:
                    return False
                if op == "$gt" and not (value is not None and value > target):
                    return False
                if op == "$gte" and not (value is not None and value >= target):
                    return False
                if op == "$lt" and not (value is not None and value < target):
                    return False
                if op == "$lte" and not (value is not None and value <= target):
                    return False
                if op == "$in" and value not in target:
                    return False
                if op == "$nin" and value in target:
                    return False
        else:
            # Shorthand: {"field": value} => {"field": {"$eq": value}}
            if value != constraint:
                return False
    return True


def apply_filters(
    results: list[dict],
    where: dict,
    articles_by_id: dict[str, dict],
) -> list[dict]:
    """Filter BM25 results against a ChromaDB-style where clause.

    Args:
        results: BM25 result dicts (must contain ``doc_id``).
        where: ChromaDB where clause dict.
        articles_by_id: Mapping from ``doc_id`` to full article dict
            containing metadata fields.

    Returns:
        Subset of *results* whose metadata satisfies *where*.
    """
    if not where:
        return results

    filtered: list[dict] = []
    for r in results:
        article = articles_by_id.get(r["doc_id"])
        if article is None:
            continue
        meta = {
            "region": article.get("region", ""),
            "product_version": article.get("product_version", ""),
            "category": article.get("category", ""),
            "deprecated": article.get("deprecated", False),
            "effective_date": article.get("effective_date", ""),
            "error_codes_str": ",".join(article.get("error_codes", [])),
        }
        if _evaluate_condition(meta, where):
            filtered.append(r)
    return filtered


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        result_lists: List of result lists, each sorted by relevance.
        k: RRF smoothing constant (default 60).

    Returns:
        Fused results sorted by descending RRF score, with updated
        ``score`` and ``rank`` fields.
    """
    scores: dict[str, float] = {}
    doc_data: dict[str, dict] = {}

    for results in result_lists:
        for r in results:
            doc_id = r["doc_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + r["rank"])
            # Keep the richest version of the document data
            if doc_id not in doc_data:
                doc_data[doc_id] = {key: r[key] for key in r if key not in ("score", "rank")}

    sorted_ids = sorted(scores, key=lambda d: scores[d], reverse=True)

    fused: list[dict] = []
    for rank, doc_id in enumerate(sorted_ids, start=1):
        entry = dict(doc_data[doc_id])
        entry["score"] = scores[doc_id]
        entry["rank"] = rank
        fused.append(entry)
    return fused


# ---------------------------------------------------------------------------
# Search functions
# ---------------------------------------------------------------------------

def search_dense(
    collection: chromadb.Collection,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    where: dict | None = None,
) -> list[dict]:
    """Dense (embedding-based) search via ChromaDB.

    Args:
        collection: ChromaDB collection.
        query: Natural-language query.
        top_k: Number of results to return.
        where: Optional ChromaDB where clause for metadata filtering.

    Returns:
        List of result dicts sorted by descending similarity.
    """
    query_embedding = embed_query(query)

    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    try:
        query_result = collection.query(**kwargs)
    except Exception:
        if where:
            logger.warning(
                "Filtered dense search returned no results; falling back to unfiltered."
            )
            kwargs.pop("where", None)
            query_result = collection.query(**kwargs)
        else:
            raise

    return _chroma_to_results(query_result)


def search_sparse(
    bm25_index: BM25Index,
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Sparse (BM25) keyword search.

    Args:
        bm25_index: Initialised BM25Index instance.
        query: Natural-language query.
        top_k: Number of results to return.

    Returns:
        List of result dicts sorted by descending BM25 score.
    """
    bm25_results = bm25_index.search(query, top_k=top_k)

    # Enrich with article metadata
    articles_by_id = {a["doc_id"]: a for a in bm25_index.documents}

    enriched: list[dict] = []
    for r in bm25_results:
        article = articles_by_id.get(r["doc_id"], {})
        enriched.append(
            {
                "doc_id": r["doc_id"],
                "title": article.get("title", ""),
                "body": article.get("body", ""),
                "region": article.get("region", ""),
                "product_version": article.get("product_version", ""),
                "category": article.get("category", ""),
                "deprecated": article.get("deprecated", False),
                "score": r["score"],
                "rank": r["rank"],
            }
        )
    return enriched


def search_hybrid(
    collection: chromadb.Collection,
    bm25_index: BM25Index,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    where: dict | None = None,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """Hybrid search combining dense and sparse retrieval with RRF fusion.

    Over-retrieves from both sources, optionally applies metadata filters
    to BM25 results post-hoc, then fuses via Reciprocal Rank Fusion.

    Args:
        collection: ChromaDB collection.
        bm25_index: Initialised BM25Index.
        query: Natural-language query.
        top_k: Number of final results to return.
        where: Optional ChromaDB where clause for metadata filtering.
        rrf_k: RRF smoothing constant.

    Returns:
        Fused result list sorted by descending RRF score.
    """
    over_k = top_k * 3

    # Dense branch (ChromaDB handles filtering natively)
    dense_results = search_dense(collection, query, top_k=over_k, where=where)

    # Sparse branch (post-hoc filtering required)
    if where:
        sparse_raw = bm25_index.search(query, top_k=top_k * 10)
        articles_by_id = {a["doc_id"]: a for a in bm25_index.documents}
        sparse_filtered = apply_filters(sparse_raw, where, articles_by_id)

        # Re-rank after filtering
        for new_rank, item in enumerate(sparse_filtered, start=1):
            item["rank"] = new_rank
        sparse_results_raw = sparse_filtered[:over_k]

        if not sparse_results_raw:
            logger.warning(
                "BM25 post-hoc filter returned 0 results; using unfiltered BM25."
            )
            sparse_results_raw = bm25_index.search(query, top_k=over_k)
    else:
        sparse_results_raw = bm25_index.search(query, top_k=over_k)

    # Enrich sparse results with article data
    articles_by_id = {a["doc_id"]: a for a in bm25_index.documents}
    sparse_results: list[dict] = []
    for r in sparse_results_raw:
        article = articles_by_id.get(r["doc_id"], {})
        sparse_results.append(
            {
                "doc_id": r["doc_id"],
                "title": article.get("title", ""),
                "body": article.get("body", ""),
                "region": article.get("region", ""),
                "product_version": article.get("product_version", ""),
                "category": article.get("category", ""),
                "deprecated": article.get("deprecated", False),
                "score": r["score"],
                "rank": r["rank"],
            }
        )

    # Fuse
    fused = reciprocal_rank_fusion([dense_results, sparse_results], k=rrf_k)

    # If filters produced no results at all, fall back to unfiltered
    if not fused and where:
        logger.warning(
            "Hybrid search with filters returned 0 results; "
            "retrying without filters."
        )
        return search_hybrid(
            collection, bm25_index, query, top_k=top_k, where=None, rrf_k=rrf_k
        )

    return fused[:top_k]
