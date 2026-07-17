"""
ConsumerGuard AI — Pure-Numpy Vector Store
A lightweight, zero-compilation replacement for ChromaDB.

Why not ChromaDB?
    ChromaDB depends on `chroma-hnswlib`, a C++ extension that has no pre-built
    Windows wheel for Python 3.12. It requires Microsoft C++ Build Tools to compile.
    For a ~300 chunk dataset, numpy cosine similarity is equally fast (~1ms per query)
    and requires no compilation at all.

How it works:
    - Embeddings are stored as a numpy float32 matrix (N × 384).
    - Documents and metadata are stored as JSON lists.
    - Both are saved to disk inside the `chroma_db/` folder.
    - Cosine similarity is computed as a matrix-vector dot product (fast with numpy).
    - Optional metadata filtering is applied before similarity scoring.

Storage layout:
    chroma_db/
    ├── embeddings.npy   ← numpy float32 array, shape (N, 384)
    ├── documents.json   ← list of N text strings
    ├── metadatas.json   ← list of N metadata dicts
    └── ids.json         ← list of N unique string IDs

API is compatible with the ChromaDB collection interface used in ingest.py and retriever.py.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


class VectorStore:
    """
    A simple, file-backed vector store using numpy cosine similarity.

    Usage:
        store = VectorStore(persist_dir="./chroma_db")
        store.add(ids=[...], documents=[...], embeddings=[...], metadatas=[...])
        results = store.query(query_embedding=[...], n_results=3)
    """

    def __init__(self, persist_dir: str):
        """
        Initialize the vector store. Loads existing data from disk if found.

        Args:
            persist_dir: Path to the folder where the store is persisted.
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # In-memory state
        self._embeddings: Optional[np.ndarray] = None  # shape (N, D)
        self._documents: List[str] = []
        self._metadatas: List[Dict] = []
        self._ids: List[str] = []

        # Load from disk if data already exists
        self._load()

    # ─── Persistence ──────────────────────────────────────────────────────────

    def _embeddings_path(self) -> Path:
        return self.persist_dir / "embeddings.npy"

    def _documents_path(self) -> Path:
        return self.persist_dir / "documents.json"

    def _metadatas_path(self) -> Path:
        return self.persist_dir / "metadatas.json"

    def _ids_path(self) -> Path:
        return self.persist_dir / "ids.json"

    def _save(self) -> None:
        """Persist current in-memory state to disk."""
        if self._embeddings is not None:
            np.save(str(self._embeddings_path()), self._embeddings)
        with open(self._documents_path(), "w", encoding="utf-8") as f:
            json.dump(self._documents, f, ensure_ascii=False)
        with open(self._metadatas_path(), "w", encoding="utf-8") as f:
            json.dump(self._metadatas, f, ensure_ascii=False)
        with open(self._ids_path(), "w", encoding="utf-8") as f:
            json.dump(self._ids, f)

    def _load(self) -> None:
        """Load state from disk if all four files exist."""
        if not all([
            self._embeddings_path().exists(),
            self._documents_path().exists(),
            self._metadatas_path().exists(),
            self._ids_path().exists(),
        ]):
            return  # Fresh store — nothing to load

        self._embeddings = np.load(str(self._embeddings_path()))
        with open(self._documents_path(), "r", encoding="utf-8") as f:
            self._documents = json.load(f)
        with open(self._metadatas_path(), "r", encoding="utf-8") as f:
            self._metadatas = json.load(f)
        with open(self._ids_path(), "r", encoding="utf-8") as f:
            self._ids = json.load(f)

        print(f"[VectorStore] Loaded {len(self._documents)} chunks from {self.persist_dir}")

    def clear(self) -> None:
        """Delete all stored data (in-memory and on disk). Used before re-ingesting."""
        self._embeddings = None
        self._documents = []
        self._metadatas = []
        self._ids = []
        for path in [
            self._embeddings_path(),
            self._documents_path(),
            self._metadatas_path(),
            self._ids_path(),
        ]:
            if path.exists():
                path.unlink()

    # ─── Write ────────────────────────────────────────────────────────────────

    def add(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
    ) -> None:
        """
        Add a batch of documents with their embeddings and metadata to the store.
        Appends to existing data and immediately saves to disk.

        Args:
            ids:        Unique string identifier for each document.
            documents:  List of text strings.
            embeddings: List of float lists (384-dim each).
            metadatas:  List of metadata dicts (e.g. {"platform": "amazon", ...}).
        """
        new_embeddings = np.array(embeddings, dtype=np.float32)

        # Normalise each embedding to unit length for cosine similarity via dot product
        norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)  # avoid division by zero
        new_embeddings = new_embeddings / norms

        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

        self._documents.extend(documents)
        self._metadatas.extend(metadatas)
        self._ids.extend(ids)

        self._save()

    # ─── Read / Query ─────────────────────────────────────────────────────────

    def _apply_filter(self, where: Optional[Dict]) -> List[int]:
        """
        Return indices of records matching the metadata filter.

        Supported operators:
            {"field": {"$eq": value}}           — exact match
            {"$or": [filter1, filter2, ...]}    — any filter matches

        Args:
            where: Metadata filter dict (ChromaDB-style).

        Returns:
            List of integer indices that pass the filter.
        """
        if where is None:
            return list(range(len(self._metadatas)))

        def matches_single(meta: Dict, condition: Dict) -> bool:
            for key, val in condition.items():
                if key == "$or":
                    # At least one sub-condition must match
                    return any(matches_single(meta, sub) for sub in val)
                elif key == "$and":
                    return all(matches_single(meta, sub) for sub in val)
                else:
                    # key is a field name, val is {"$eq": value} or similar
                    if isinstance(val, dict):
                        if "$eq" in val:
                            if meta.get(key) != val["$eq"]:
                                return False
                        # Add more operators here if needed ($ne, $in, etc.)
                    else:
                        # Direct equality: {"field": value}
                        if meta.get(key) != val:
                            return False
            return True

        return [
            i for i, meta in enumerate(self._metadatas)
            if matches_single(meta, where)
        ]

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 3,
        where: Optional[Dict] = None,
        include: Optional[List[str]] = None,
    ) -> Dict:
        """
        Find the n_results most similar documents using cosine similarity.
        Returns a result dict in ChromaDB-compatible format.

        Args:
            query_embeddings: List containing one query vector (we use only the first).
            n_results:        Number of results to return.
            where:            Optional metadata filter (ChromaDB-style).
            include:          List of fields to include in results (ignored here;
                              we always return documents, metadatas, distances).

        Returns:
            {
                "documents": [[doc1, doc2, ...]],
                "metadatas": [[meta1, meta2, ...]],
                "distances": [[dist1, dist2, ...]],  # cosine distance (1 - similarity)
                "ids":       [[id1, id2, ...]],
            }
        """
        if self._embeddings is None or len(self._documents) == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

        # Step 1: Apply metadata filter to get candidate indices
        candidate_indices = self._apply_filter(where)
        if not candidate_indices:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

        # Step 2: Normalise query vector
        query_vec = np.array(query_embeddings[0], dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm

        # Step 3: Compute cosine similarity for candidate rows only
        candidate_matrix = self._embeddings[candidate_indices]  # shape (C, D)
        similarities = candidate_matrix @ query_vec              # shape (C,)

        # Step 4: Get top-k by similarity (descending)
        k = min(n_results, len(candidate_indices))
        top_k_local = np.argsort(similarities)[::-1][:k]

        # Step 5: Map back to global indices
        top_k_global = [candidate_indices[i] for i in top_k_local]
        top_similarities = similarities[top_k_local]

        # Cosine distance = 1 - cosine similarity (lower is better, matches ChromaDB L2 convention)
        top_distances = (1.0 - top_similarities).tolist()

        return {
            "documents": [[self._documents[i] for i in top_k_global]],
            "metadatas": [[self._metadatas[i] for i in top_k_global]],
            "distances": [top_distances],
            "ids":       [[self._ids[i] for i in top_k_global]],
        }

    def count(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._documents)


# ─── Module-level helper (mirrors chromadb.PersistentClient pattern) ──────────

def get_store(persist_dir: str) -> VectorStore:
    """
    Load (or create) a VectorStore at the given directory.
    Equivalent to chromadb.PersistentClient(path=...).get_collection(...).
    """
    return VectorStore(persist_dir=persist_dir)
