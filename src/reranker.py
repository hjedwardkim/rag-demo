"""Cross-encoder reranking module.

Uses a self-hosted BAAI/bge-reranker-v2-m3 model via a Text Embeddings
Inference (TEI) endpoint to rerank retrieval results.
"""

import logging
import time

import httpx

from src.config import RERANKER_API_BASE, RERANKER_API_KEY

logger = logging.getLogger(__name__)


class Reranker:
    """Reranks retrieval results using a TEI rerank endpoint.

    Sends (query, document) pairs to the self-hosted TEI service,
    which scores them with a cross-encoder model.
    """

    def __init__(self, api_base: str = RERANKER_API_BASE, api_key: str = RERANKER_API_KEY) -> None:
        """Initialize the reranker with the TEI endpoint.

        Args:
            api_base: Base URL of the TEI service (e.g. "https://host:port").
            api_key: API key for the TEI service.
        """
        self.rerank_url = f"{api_base.rstrip('/')}/rerank"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def rerank(
        self, query: str, results: list[dict], top_k: int = 5
    ) -> list[dict]:
        """Rerank retrieval results using the TEI rerank endpoint.

        Sends the query and document texts to the TEI /rerank endpoint,
        then returns the top_k results sorted by descending score.

        Args:
            query: The user's search query.
            results: List of retrieval result dicts, each containing at least
                'title' and 'body' keys for building the document text.
            top_k: Number of top results to return after reranking.

        Returns:
            List of result dicts sorted by cross-encoder score, each with
            updated 'score' and 'rank' fields.
        """
        if not results:
            return []

        texts = [
            f"{r.get('title', '')} {r.get('body', '')}" for r in results
        ]

        payload = {"query": query, "texts": texts}

        scores = self._call_tei(payload)

        scored_results = []
        for item in scores:
            idx = item["index"]
            updated = dict(results[idx])
            updated["score"] = float(item["score"])
            scored_results.append(updated)

        scored_results.sort(key=lambda x: x["score"], reverse=True)

        for i, result in enumerate(scored_results[:top_k]):
            result["rank"] = i + 1

        return scored_results[:top_k]

    def _call_tei(self, payload: dict) -> list[dict]:
        """Call the TEI /rerank endpoint with retries.

        Args:
            payload: Request body with 'query' and 'texts' keys.

        Returns:
            List of dicts with 'index' and 'score' keys.

        Raises:
            httpx.HTTPStatusError: If all retries fail.
        """
        for attempt in range(3):
            try:
                response = httpx.post(
                    self.rerank_url,
                    json=payload,
                    headers=self.headers,
                    timeout=30.0,
                )
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt < 2:
                    wait = 2**attempt
                    logger.warning("TEI rerank attempt %d failed: %s. Retrying in %ds.", attempt + 1, e, wait)
                    time.sleep(wait)
                else:
                    raise
        return []  # unreachable, satisfies type checker
