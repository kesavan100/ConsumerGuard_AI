"""
ConsumerGuard AI — Retrieval Agent (Agent 2)
Generates a query embedding and retrieves the most relevant policy chunks
from our numpy VectorStore using cosine similarity search.

Key design decisions:
- top_k=3: Small enough to avoid irrelevant context, large enough for good coverage
- Platform filtering: When platform is known (amazon/flipkart), we filter results
  to prefer platform-specific policies. We always include "both" docs (CP Act, E-Comm Rules).
- Similarity score: Converts cosine distance (1 - similarity) to a 0–1 similarity score
  used by the pipeline to compute confidence level.
"""

import streamlit as st
from typing import Dict, Optional

from consumerguard.config import (
    CHROMA_DIR,
    TOP_K,
    MIN_RETRIEVAL_SIMILARITY,
)
from consumerguard.embedder import generate_embedding
from consumerguard.vectorstore import VectorStore


@st.cache_resource
def _get_store() -> VectorStore:
    """
    Return the VectorStore, initializing it on the first call.
    st.cache_resource avoids reloading disk data on every Streamlit query.
    """
    store = VectorStore(persist_dir=str(CHROMA_DIR))
    if store.count() == 0:
        raise RuntimeError(
            "VectorStore is empty. Run ingestion first:\n"
            "    python -m consumerguard.ingest"
        )
    return store


# ─── Distance → Similarity Conversion ────────────────────────────────────────

def _distance_to_similarity(distance: float) -> float:
    """
    Convert a cosine distance (1 - cosine_similarity) to a similarity score.
    Our VectorStore returns distances = 1 - similarity, so:
        similarity = 1 - distance
    Clamped to [0, 1] to handle floating-point rounding.
    """
    return max(0.0, min(1.0, 1.0 - distance))


# ─── Main Retrieval Function ──────────────────────────────────────────────────

def retrieve(
    query: str,
    platform_hint: str = "unknown",
    top_k: int = TOP_K,
) -> Dict:
    """
    Retrieve the most relevant policy chunks for a given query.

    Strategy:
    1. Generate embedding for the query using all-MiniLM-L6-v2
    2. Query VectorStore for nearest vectors by cosine similarity
    3. If platform is known, prefer platform-specific docs + shared docs
    4. Return structured result with text, metadata, similarity scores, and sources

    Args:
        query:         The complaint text or derived query string.
        platform_hint: "amazon", "flipkart", or "unknown".
        top_k:         Number of chunks to retrieve.

    Returns:
        {
            "context_text": str,        # formatted text for LLM prompt
            "sources": List[str],       # deduplicated list of source policy names
            "chunks": List[Dict],       # raw chunk data with scores
            "best_similarity": float,   # highest similarity score (for confidence)
            "chunk_count": int,         # number of chunks retrieved
        }
    """
    store = _get_store()

    # Step 1: Generate query embedding
    query_embedding = generate_embedding(query)

    # Step 2: Build metadata filter
    # When platform is known, restrict to platform-specific docs + docs for "both" platforms
    where_filter = None
    if platform_hint in ("amazon", "flipkart"):
        where_filter = {
            "$or": [
                {"platform": {"$eq": platform_hint}},
                {"platform": {"$eq": "both"}},
            ]
        }

    # Step 3: Query the VectorStore
    results = store.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    # Step 4: Parse results
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return {
            "context_text": "No relevant policy information found.",
            "sources": [],
            "chunks": [],
            "best_similarity": 0.0,
            "chunk_count": 0,
        }

    # Step 5: Build structured chunk list with similarity scores, filtering out low similarity
    chunks = []
    seen_sources = []  # preserve insertion order for deduplication

    for doc, meta, dist in zip(documents, metadatas, distances):
        similarity = _distance_to_similarity(dist)
        
        if similarity < MIN_RETRIEVAL_SIMILARITY:
            continue

        policy_name = meta.get("policy_name", "Unknown Policy")

        chunks.append({
            "text": doc,
            "policy_name": policy_name,
            "platform": meta.get("platform", "unknown"),
            "chunk_id": meta.get("chunk_index", -1),
            "source_file": meta.get("source", ""),
            "similarity_score": round(similarity, 3),
        })

        # Deduplicate sources while preserving insertion order
        if policy_name not in seen_sources:
            seen_sources.append(policy_name)
            
    if not chunks:
        return {
            "context_text": "No relevant policy information found.",
            "sources": [],
            "chunks": [],
            "best_similarity": 0.0,
            "chunk_count": 0,
        }

    # Step 6: Build formatted context string for the LLM prompt
    # Each chunk is clearly labelled with its source policy name and similarity score
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Source {i}: {chunk['policy_name']} (similarity: {chunk['similarity_score']})]\n{chunk['text']}"
        )
    context_text = "\n\n---\n\n".join(context_parts)

    # Step 7: Compute best (highest) similarity score for confidence scoring
    best_similarity = max(c["similarity_score"] for c in chunks) if chunks else 0.0

    return {
        "context_text": context_text,
        "sources": seen_sources,
        "chunks": chunks,
        "best_similarity": best_similarity,
        "chunk_count": len(chunks),
    }
