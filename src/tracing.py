"""Langfuse tracing instrumentation for the RAG pipeline.

Provides optional tracing via Langfuse. When Langfuse credentials are not
configured, all tracing functions become no-ops using the NullLangfuse pattern.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from src.config import LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY


class _NullContext:
    """No-op context manager that accepts any method call and returns itself."""

    def __enter__(self) -> _NullContext:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def update(self, **kwargs: Any) -> None:
        """No-op update."""

    def end(self, **kwargs: Any) -> None:
        """No-op end."""

    def start_as_current_observation(self, **kwargs: Any) -> _NullContext:
        """Return another no-op context."""
        return _NullContext()

    def update_trace(self, **kwargs: Any) -> None:
        """No-op."""


class NullLangfuse:
    """No-op Langfuse client used when credentials are not configured."""

    def start_as_current_observation(self, **kwargs: Any) -> _NullContext:
        """Return a no-op context manager."""
        return _NullContext()

    def flush(self) -> None:
        """No-op flush."""


def init_langfuse() -> Any:
    """Initialize the Langfuse client.

    Returns:
        A ``Langfuse`` instance if keys are configured, otherwise a
        ``NullLangfuse`` instance that makes all tracing calls no-ops.
    """
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        return NullLangfuse()

    from langfuse import Langfuse

    return Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST,
    )


@contextmanager
def traced_step(
    parent: Any, name: str, input_data: dict[str, Any]
) -> Generator[Any, None, None]:
    """Context manager that wraps a pipeline step in a Langfuse span.

    Works with both real Langfuse spans and NullLangfuse no-ops.

    Args:
        parent: A Langfuse client/span or NullLangfuse/NullContext.
        name: Name for the span (e.g. ``"filter-extraction"``).
        input_data: Input data recorded on the span.

    Yields:
        A span context. Callers can call ``span.update(output=...)``
        before the context exits.

    Example::

        with traced_step(langfuse, "retrieval", {"query": q}) as span:
            results = search_hybrid(...)
            span.update(output={"doc_ids": [r["doc_id"] for r in results]})
    """
    with parent.start_as_current_observation(as_type="span", name=name, input=input_data) as span:
        yield span
