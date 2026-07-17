"""
ConsumerGuard AI — Knowledge Base Ingestion Pipeline
Reads all 9 policy PDF files, splits them into overlapping chunks,
generates embeddings, and stores everything in our numpy VectorStore.

Run this script ONCE locally before deploying:
    python -m consumerguard.ingest

The resulting chroma_db/ folder (~5-10 MB) should be committed to Git
and deployed with the application. This avoids rebuilding on Azure cold starts.
"""

import uuid
from pathlib import Path
from typing import List

from pypdf import PdfReader

from consumerguard.config import (
    PDF_DIR,
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DOCUMENT_METADATA,
)
from consumerguard.embedder import generate_embeddings_batch
from consumerguard.vectorstore import VectorStore


# ─── Text Extraction ──────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract all text from a PDF file using pypdf.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        Concatenated text from all pages of the PDF.
    """
    reader = PdfReader(str(pdf_path))
    pages_text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages_text.append(page_text.strip())
    return "\n\n".join(pages_text)


# ─── Text Chunking ────────────────────────────────────────────────────────────

def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split a long text into overlapping fixed-size character chunks.

    Larger chunks (700 chars) are used here because policy documents often
    reference previous paragraphs, so keeping context together improves retrieval.

    Args:
        text:       The full document text.
        chunk_size: Maximum characters per chunk.
        overlap:    Number of characters shared between consecutive chunks.

    Returns:
        List of text chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:  # skip empty chunks
            chunks.append(chunk)
        # Move forward by (chunk_size - overlap) to create the sliding window
        start += chunk_size - overlap

    return chunks


# ─── Main Ingestion Function ──────────────────────────────────────────────────

def ingest_all_pdfs() -> None:
    """
    Main ingestion function.
    1. Reads each PDF from product_data/
    2. Splits into chunks (700 chars, 100 overlap)
    3. Generates embeddings in batch using all-MiniLM-L6-v2
    4. Stores in the numpy VectorStore (saved to chroma_db/)

    The VectorStore is cleared and rebuilt on each run to ensure
    the database stays consistent with the source PDFs.
    """
    # Create a fresh VectorStore (clears any existing data first)
    store = VectorStore(persist_dir=str(CHROMA_DIR))
    store.clear()
    print(f"[Ingest] VectorStore cleared. Starting fresh ingestion into: {CHROMA_DIR}")

    total_chunks = 0

    for filename, meta in DOCUMENT_METADATA.items():
        pdf_path = PDF_DIR / filename

        if not pdf_path.exists():
            print(f"[Ingest] WARNING: File not found — {filename}. Skipping.")
            continue

        print(f"\n[Ingest] Processing: {filename}")

        # Step 1: Extract text from PDF
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text.strip():
            print(f"[Ingest] WARNING: No text extracted from {filename}. Skipping.")
            continue

        # Step 2: Split into overlapping chunks
        chunks = split_into_chunks(raw_text)
        print(
            f"[Ingest] -> {len(chunks)} chunks created "
            f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
        )

        # Step 3: Generate embeddings for all chunks in one batch call
        print(f"[Ingest] → Generating embeddings...")
        embeddings = generate_embeddings_batch(chunks)

        # Step 4: Build record lists for VectorStore.add()
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {
                "source": filename,
                "policy_name": meta["policy_name"],
                "platform": meta["platform"],
                "chunk_index": i,
            }
            for i, _ in enumerate(chunks)
        ]

        # Step 5: Add to the VectorStore (saved to disk automatically)
        store.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        total_chunks += len(chunks)
        print(f"[Ingest] [OK] {filename} -- {len(chunks)} chunks stored.")

    print(f"\n[Ingest] Ingestion complete.")
    print(f"[Ingest] Total chunks stored: {total_chunks}")
    print(f"[Ingest] VectorStore saved to: {CHROMA_DIR}")
    print("[Ingest] You can now commit the chroma_db/ folder to Git.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("ConsumerGuard AI -- Knowledge Base Ingestion")
    print("=" * 60)
    ingest_all_pdfs()
