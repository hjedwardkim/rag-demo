"""LLM-based metadata filter extraction from natural language queries.

Uses the self-hosted LLM to parse user queries and extract structured
metadata filters compatible with ChromaDB's where clause syntax.
"""

import json
import logging

from openai import OpenAI

from src.config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)

SYSTEM_PROMPT = """You are a metadata filter extractor for an IT support knowledge base.
Given a user query, extract any metadata filters that should be applied
to narrow the search. Return ONLY a JSON object with the following
optional fields:

- "region": one of "EU", "US", "APAC" (only if explicitly mentioned or clearly implied)
- "product_version": one of "v1.0", "v2.0", "v3.0" (only if mentioned)
- "category": one of "authentication", "billing", "deployment", "networking" (only if clearly about one category)
- "deprecated": false (include this filter to exclude deprecated docs unless the user explicitly asks for old/deprecated content)
- "error_codes": a specific error code like "E-4012" (only if mentioned)

If no filters can be extracted, return an empty JSON object: {}
Return ONLY valid JSON. No explanation, no markdown."""


def extract_filters(query: str) -> dict:
    """Extract structured metadata filters from a natural language query.

    Calls the self-hosted LLM to identify region, product version, category,
    deprecation status, and error codes mentioned in the query.

    Args:
        query: The user's natural language search query.

    Returns:
        A dict of extracted filter fields. Empty dict if no filters detected
        or if the LLM response cannot be parsed.
    """
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=256,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fencing if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [line for line in lines if not line.startswith("```")]
        raw = "\n".join(lines).strip()

    try:
        filters = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON for filter extraction: %s", raw)
        return {}

    if not isinstance(filters, dict):
        logger.warning("LLM returned non-dict JSON for filter extraction: %s", raw)
        return {}

    return filters


def convert_to_chromadb_where(filters: dict) -> dict:
    """Convert extracted filters to a ChromaDB-compatible where clause.

    Translates the flat filter dict from extract_filters() into ChromaDB's
    $and-based where clause format. Error codes are excluded from the
    ChromaDB where clause since they require post-filtering on the
    error_codes_str metadata field.

    Args:
        filters: Dict of extracted filters from extract_filters().

    Returns:
        A ChromaDB where clause dict, or empty dict if no filters apply.
        Error codes are stored under a separate key convention if present.
    """
    if not filters:
        return {}

    conditions: list[dict] = []

    if "region" in filters:
        conditions.append({"region": {"$eq": filters["region"]}})

    if "product_version" in filters:
        conditions.append({"product_version": {"$eq": filters["product_version"]}})

    if "category" in filters:
        conditions.append({"category": {"$eq": filters["category"]}})

    if "deprecated" in filters:
        conditions.append({"deprecated": {"$eq": filters["deprecated"]}})

    if not conditions:
        return {}

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


def get_error_code_filter(filters: dict) -> str | None:
    """Extract the error code from filters for post-filtering.

    Since ChromaDB metadata filters don't support substring matching,
    error code filtering must be applied in Python after retrieval
    by checking if the code appears in the error_codes_str field.

    Args:
        filters: Dict of extracted filters from extract_filters().

    Returns:
        The error code string (e.g., "E-4012") if present, else None.
    """
    return filters.get("error_codes")
