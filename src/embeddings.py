"""Dense embedding client wrapping an OpenAI-compatible endpoint.

Uses the intfloat/multilingual-e5-large-instruct model (1024-dim) with
instruction prefixes applied internally so callers never need to add them.
"""

import time

from openai import OpenAI

from src.config import EMBEDDING_API_BASE, EMBEDDING_API_KEY, EMBEDDING_MODEL

_client = OpenAI(base_url=EMBEDDING_API_BASE, api_key=EMBEDDING_API_KEY)

_MAX_RETRIES = 3
_BACKOFF_BASE = 1  # seconds


def _embed_with_retry(texts: list[str]) -> list[list[float]]:
    """Call the embedding API with exponential-backoff retries.

    Args:
        texts: Pre-prefixed texts to embed in a single request.

    Returns:
        List of embedding vectors.

    Raises:
        Exception: After 3 consecutive failures.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            response = _client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            return [item.embedding for item in response.data]
        except Exception:
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(_BACKOFF_BASE * (2 ** attempt))
    return []  # unreachable, but keeps type checkers happy


def embed_documents(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed document texts with the ``passage: `` instruction prefix.

    Batches requests internally to respect API size limits.

    Args:
        texts: Raw document strings (no prefix needed).
        batch_size: Number of texts per API call.

    Returns:
        List of 1024-dimensional embedding vectors, one per input text.
    """
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = [f"passage: {t}" for t in texts[i : i + batch_size]]
        all_embeddings.extend(_embed_with_retry(batch))
    return all_embeddings


def embed_query(text: str) -> list[float]:
    """Embed a single query with the ``query: `` instruction prefix.

    Args:
        text: Raw query string (no prefix needed).

    Returns:
        A 1024-dimensional embedding vector.
    """
    return _embed_with_retry([f"query: {text}"])[0]
