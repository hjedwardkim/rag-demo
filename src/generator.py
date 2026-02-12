"""LLM-based answer generation from retrieved context documents.

Uses the self-hosted LLM to produce grounded answers that reference
specific document IDs from the knowledge base.
"""

from openai import OpenAI

from src.config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL

client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)

SYSTEM_PROMPT = """You are a helpful IT support assistant. Answer the user's question
based ONLY on the provided context documents. If the context doesn't
contain enough information to answer, say so explicitly.

For each claim in your answer, reference the document ID (e.g., KB-0042)
that supports it."""


def _format_context(context_docs: list[dict]) -> str:
    """Format retrieved documents into a context block for the LLM prompt.

    Args:
        context_docs: List of document dicts with keys doc_id, title,
            body, region, and product_version.

    Returns:
        Formatted context string with each document's metadata and body.
    """
    parts: list[str] = []
    for doc in context_docs:
        doc_id = doc.get("doc_id", "UNKNOWN")
        title = doc.get("title", "Untitled")
        region = doc.get("region", "N/A")
        version = doc.get("product_version", "N/A")
        body = doc.get("body", "")
        parts.append(f"[{doc_id}] {title} ({region}, {version})\n{body}")
    return "\n\n".join(parts)


def generate_answer(query: str, context_docs: list[dict]) -> str:
    """Generate an answer to the query grounded in the provided context documents.

    Formats the context documents and calls the self-hosted LLM with a system
    prompt that instructs it to answer only from context and cite document IDs.

    Args:
        query: The user's question.
        context_docs: List of retrieved document dicts, each containing at
            least doc_id, title, body, region, and product_version.

    Returns:
        The LLM-generated answer string.
    """
    context_block = _format_context(context_docs)

    user_message = f"""Context documents:

{context_block}

---
Question: {query}"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()
