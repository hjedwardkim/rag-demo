# Context is King — RAG Demo

A hands-on demonstration of retrieval-augmented generation (RAG) techniques, showing how naive dense search fails and how hybrid retrieval, metadata filtering, reranking, and observability combine to produce accurate results.

Built on a synthetic IT support knowledge base (200 articles) with intentional failure modes: cross-region error code overlap, deprecated documents, multi-chunk topics, and version drift.

## Architecture

```
Query
  │
  ├─► Filter Extraction (LLM)     → ChromaDB where clause
  │
  ├─► Dense Search (embeddings)    → cosine similarity via ChromaDB
  ├─► Sparse Search (BM25)         → lexical match via rank_bm25
  │
  ├─► Reciprocal Rank Fusion       → merged ranked list
  ├─► Cross-Encoder Reranking      → rescored top results
  │
  └─► Answer Generation (LLM)      → grounded response with doc references
```

## Project Structure

```
rag-demo/
├── data/
│   ├── generate_dataset.py       # Deterministic dataset generator (200 articles)
│   └── kb_articles.json          # Generated KB articles
├── src/
│   ├── config.py                 # Environment config (.env loader)
│   ├── embeddings.py             # Dense embedding client (OpenAI-compatible)
│   ├── sparse.py                 # BM25 index and search
│   ├── indexer.py                # ChromaDB + BM25 indexing
│   ├── retriever.py              # Dense, sparse, and hybrid search with RRF
│   ├── reranker.py               # Cross-encoder reranking (via TEI)
│   ├── filter_extractor.py       # LLM-based metadata filter extraction
│   ├── generator.py              # LLM-based answer generation
│   ├── tracing.py                # Langfuse instrumentation (optional)
│   └── utils.py                  # Rich terminal output helpers
├── demos/
│   ├── demo1_retrieval.py        # Naive vs. hybrid retrieval
│   ├── demo2_filtering.py        # Metadata filtering
│   └── demo3_tracing.py          # Full pipeline with Langfuse tracing
├── evals/
│   ├── eval_set.json             # 20 query-answer pairs with expected doc_ids
│   ├── run_evals.py              # Evaluation runner (Recall@K, MRR)
│   └── results/                  # Eval output logs
├── .env.example                  # Environment variable template
└── pyproject.toml
```

## Setup

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys and endpoints
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `EMBEDDING_API_BASE` | Yes | OpenAI-compatible embedding endpoint |
| `EMBEDDING_API_KEY` | Yes | API key for embedding endpoint |
| `EMBEDDING_MODEL` | No | Defaults to `intfloat/multilingual-e5-large-instruct` |
| `LLM_API_BASE` | Yes | OpenAI-compatible LLM endpoint |
| `LLM_API_KEY` | Yes | API key for LLM endpoint |
| `LLM_MODEL` | No | Defaults to `cm-llm` |
| `RERANKER_API_BASE` | Yes | TEI reranker endpoint base URL |
| `RERANKER_API_KEY` | Yes | API key for TEI reranker endpoint |
| `RERANKER_MODEL` | No | Defaults to `BAAI/bge-reranker-v2-m3` |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse public key (demo3 only) |
| `LANGFUSE_SECRET_KEY` | No | Langfuse secret key (demo3 only) |
| `LANGFUSE_HOST` | No | Defaults to `https://cloud.langfuse.com` |

## Usage

### 1. Generate the dataset

```bash
uv run python -m data.generate_dataset
```

Produces `data/kb_articles.json` (200 articles) and `evals/eval_set.json` (20 eval queries).

### 2. Index documents

```bash
uv run python -m src.indexer
```

Embeds all articles into ChromaDB (dense vectors) and builds a BM25 index (sparse).

### 3. Run demos

```bash
# Demo 1: Dense vs. BM25 vs. Hybrid retrieval
uv run python -m demos.demo1_retrieval

# Demo 2: Metadata filtering (manual + LLM-extracted)
uv run python -m demos.demo2_filtering

# Demo 3: Full RAG pipeline with Langfuse tracing
uv run python -m demos.demo3_tracing
```

### 4. Run evaluations

```bash
uv run python -m evals.run_evals
```

Computes Recall@5, Recall@10, and MRR across 20 eval queries with per-category breakdowns.

## Demos

### Demo 1 — Naive vs. Hybrid Retrieval

Shows that dense-only search retrieves semantically similar but wrong documents (e.g., wrong region for an error code), while hybrid search with RRF fusion finds the exact match. Cross-encoder reranking further sharpens the ranking.

**Query:** `"How do I fix error E-4012 in the EU region?"`

### Demo 2 — Metadata Filtering

Shows that unfiltered search returns mixed regions, versions, and deprecated docs. Manual filters scope results correctly. Then demonstrates LLM-based filter extraction from natural language queries.

**Query:** `"How do I set up authentication for our EU deployment on v3?"`

### Demo 3 — Full Pipeline with Tracing

Runs the complete pipeline (filter extraction → hybrid search → rerank → LLM generation) with Langfuse tracing. Each step is instrumented as a span, viewable in the Langfuse dashboard.

## Key Techniques

- **Hybrid Search (RRF):** Fuses dense (semantic) and sparse (BM25) retrieval using Reciprocal Rank Fusion for better recall than either alone
- **Metadata Filtering:** ChromaDB `where` clauses scope results by region, version, category, and deprecation status
- **LLM Filter Extraction:** Automatically extracts structured filters from natural language queries
- **Cross-Encoder Reranking:** Rescores retrieved documents via a self-hosted TEI endpoint (`BAAI/bge-reranker-v2-m3`) for precision
- **Observability:** Langfuse tracing captures the full pipeline with spans, timings, and I/O for each step

## Dataset

### Overview

The dataset is a synthetic IT support knowledge base of 200 articles for a fictitious SaaS product. It is generated deterministically (seed=42) via `data/generate_dataset.py` — no LLM calls are needed. Running the script also produces `evals/eval_set.json` (20 query-answer pairs).

### Article Schema

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `doc_id` | string | Unique ID, format `KB-XXXX` | `"KB-0042"` |
| `title` | string | Article title | `"Resolving E-4012: Networking Issue in EU"` |
| `body` | string | 2-4 paragraphs of technical content | *(template-generated)* |
| `region` | string | `"EU"`, `"US"`, or `"APAC"` | `"EU"` |
| `product_version` | string | `"v1.0"`, `"v2.0"`, or `"v3.0"` | `"v2.0"` |
| `effective_date` | string | ISO date aligned to version (v1.0→2023, v2.0→2024, v3.0→2025) | `"2025-01-15"` |
| `error_codes` | list[string] | 0-3 codes from a pool of 60, format `E-XXXX` | `["E-4012"]` |
| `category` | string | `"authentication"`, `"billing"`, `"deployment"`, or `"networking"` | `"authentication"` |
| `deprecated` | boolean | Whether superseded by a newer doc | `false` |
| `topic_group` | string \| null | Bookkeeping field for multi-chunk topics (not indexed) | `"sso_setup"` |

### Generation Phases

The generator builds articles in four phases to guarantee the data properties needed for the demos:

**Phase 1 — Cross-region error code articles (30 docs).** 10 error codes (e.g. E-4012, E-1001) are each generated for all 3 regions. Each region variant has a different resolution procedure. This creates the failure mode shown in Demo 1: dense search retrieves the wrong region's article because the text is semantically similar.

**Phase 2 — Deprecated/replacement pairs (60 docs).** 30 deprecated articles (v1.0 or v2.0) are paired with 30 non-deprecated replacements (one version newer). The pairs share similar titles and body text but have different instructions. This creates the failure mode shown in Demo 2: unfiltered search returns outdated documents.

**Phase 3 — Multi-chunk topic articles (25 docs).** 10 topic groups (e.g. SSO setup, VPN configuration, container deployment) are each covered by 2-3 separate documents with complementary partial information. This tests whether retrieval can co-retrieve all chunks needed for a complete answer.

**Phase 4 — Fill articles (85 docs).** Remaining articles are generated to reach exactly 200, distributed evenly across categories. Each picks the category with the fewest articles so far to maintain balance.

### Enforced Data Properties

The generator validates all of the following after generation (fails with an assertion error if any check fails):

1. **Cross-region error code overlap:** 10 error codes appear across 2+ regions with region-specific resolution steps (minimum 8 required)
2. **Deprecated documents:** 30 articles (15%) have `deprecated: true`, each with a newer non-deprecated counterpart
3. **Multi-chunk topics:** 10 topic groups, each with 2-3 documents containing partial information
4. **Version-date alignment:** v1.0 dates fall in 2023, v2.0 in 2024, v3.0 in 2025
5. **Category distribution:** ~50 articles per category (40-60 allowed)
6. **Error code validity:** All codes come from a fixed pool of 60 (15 per category), max 3 per document

### Eval Set

The eval set (`evals/eval_set.json`) contains 20 queries generated alongside the dataset, referencing real `doc_id` values:

| Category | Count | Purpose |
|----------|-------|---------|
| `exact_match` | 5 | Specific error code + region — tests hybrid precision |
| `scoped` | 5 | Needs correct region/version/deprecation filter — tests filtering |
| `multi_doc` | 4 | Answer spans multiple docs — tests recall |
| `broad` | 3 | Vague query, no filters — tests baseline retrieval quality |
| `deprecated_trap` | 3 | Query that would match a deprecated doc — tests filter correctness |
