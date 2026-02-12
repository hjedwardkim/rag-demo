"""BM25 sparse index for keyword-based document retrieval.

Tokenisation preserves hyphenated tokens (e.g. ``E-4012`` -> ``e-4012``) so
that error-code searches work correctly.
"""

import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 indexing and search.

    Rules:
    - Lowercase the input.
    - Extract alphanumeric tokens, preserving internal hyphens
      (e.g. ``E-4012`` becomes ``e-4012``).
    - No stemming.

    Args:
        text: Raw text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    return re.findall(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", text.lower())


class BM25Index:
    """In-memory BM25 index over a collection of KB articles.

    Args:
        documents: List of article dicts, each containing at least
            ``doc_id``, ``title``, and ``body`` keys.
    """

    def __init__(self, documents: list[dict]) -> None:
        self._doc_ids: list[str] = []
        self._documents: list[dict] = documents
        corpus: list[list[str]] = []

        for doc in documents:
            self._doc_ids.append(doc["doc_id"])
            text = doc["title"] + " " + doc["body"]
            corpus.append(tokenize(text))

        self._bm25 = BM25Okapi(corpus)

    @property
    def documents(self) -> list[dict]:
        """Return the underlying document list."""
        return self._documents

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the index and return the top-k results.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of results to return.

        Returns:
            List of dicts with keys ``doc_id``, ``score``, and ``rank``,
            sorted by descending BM25 score.
        """
        tokenized_query = tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Pair each doc_id with its score, sort descending
        scored = sorted(
            zip(self._doc_ids, scores), key=lambda x: x[1], reverse=True
        )

        results: list[dict] = []
        for rank, (doc_id, score) in enumerate(scored[:top_k], start=1):
            results.append({"doc_id": doc_id, "score": float(score), "rank": rank})
        return results
