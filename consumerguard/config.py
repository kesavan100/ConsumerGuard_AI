"""
ConsumerGuard AI — Central Configuration
All constants, paths, and model names are defined here.
Change values in this file to tune the application without touching other modules.
"""

import os
from pathlib import Path

# ─── Project Paths ───────────────────────────────────────────────────────────

# Root directory of the project (the folder containing this package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Folder containing the 9 source PDF files
PDF_DIR = PROJECT_ROOT / "product_data"

# Folder where the numpy VectorStore persists its data (pre-built, shipped with the app)
# Files: embeddings.npy, documents.json, metadatas.json, ids.json
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# ─── Embedding Model ─────────────────────────────────────────────────────────

# sentence-transformers model — runs locally, no API key needed
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ─── Text Chunking Settings ──────────────────────────────────────────────────

# Larger chunks preserve context across verbose policy paragraphs
CHUNK_SIZE = 700        # characters per chunk
CHUNK_OVERLAP = 100     # overlapping characters between consecutive chunks

# ─── Retrieval Settings ──────────────────────────────────────────────────────

# Number of chunks to retrieve per query
# Kept small (3) to reduce irrelevant context sent to Gemini
TOP_K = 3

# ─── Confidence Scoring Thresholds ───────────────────────────────────────────

# ChromaDB returns L2 distance scores (lower = more similar)
# We convert them to similarity = 1 / (1 + distance) for readability
MIN_RETRIEVAL_SIMILARITY = 0.45    # Hard filter: drop chunks below this
HIGH_SIMILARITY_THRESHOLD = 0.75   # similarity > this → HIGH confidence
MEDIUM_SIMILARITY_THRESHOLD = 0.55 # similarity > this → MEDIUM confidence

# ─── Gemini API Settings ─────────────────────────────────────────────────────

# Model used for response generation (Agent 3)
# We tested all available models for this API key, and the Gemini models
# returned a 'limit: 0' quota error. However, the Gemma open model works perfectly!
GEMINI_MODEL = "gemini-3.5-flash"

# API key is read from environment variable (set in .env or Azure App Settings)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ─── Document Metadata Map ───────────────────────────────────────────────────
# Maps each PDF filename to its platform tag and human-readable policy name.
# Used during ingestion to tag each chunk with source metadata.

DOCUMENT_METADATA = {
    "Amazon_return_policy.pdf": {
        "platform": "amazon",
        "policy_name": "Amazon Return Policy",
    },
    "Amazon Refund Policy.pdf": {
        "platform": "amazon",
        "policy_name": "Amazon Refund Policy",
    },
    "Amazon Return Pick-up Issues.pdf": {
        "platform": "amazon",
        "policy_name": "Amazon Return Pickup Issues",
    },
    "Amazon. Returns and Replacements Frequently Asked Questions.pdf": {
        "platform": "amazon",
        "policy_name": "Amazon Returns & Replacements FAQ",
    },
    "Return Orders Placed with a Third-party.pdf": {
        "platform": "amazon",
        "policy_name": "Amazon Third-Party Seller Returns",
    },
    "flipkart Order Cancellation and Return Policy.pdf": {
        "platform": "flipkart",
        "policy_name": "Flipkart Return & Cancellation Policy",
    },
    "flipkart Open BOX Delivery.pdf": {
        "platform": "flipkart",
        "policy_name": "Flipkart Open Box Delivery Policy",
    },
    "CP Act 2019_1732700731.pdf": {
        "platform": "both",
        "policy_name": "Consumer Protection Act 2019",
    },
    "e commerce rules.pdf": {
        "platform": "both",
        "policy_name": "Consumer Protection E-Commerce Rules 2020",
    },
}
