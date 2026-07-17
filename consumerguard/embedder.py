"""
ConsumerGuard AI — Embedder Module
Wraps sentence-transformers/all-MiniLM-L6-v2 for generating text embeddings.

This model runs LOCALLY — no API key or internet connection required at runtime.
Output: 384-dimensional float vectors.

The model is loaded once and cached in memory to avoid reloading on every call.
"""

import streamlit as st
from typing import List
import os

import torch
# Force PyTorch to allocate tensors on CPU by default.
# This prevents the transformers library from temporarily using the 'meta' device 
# on Windows which causes the "Cannot copy out of meta tensor" error.
torch.set_default_device('cpu')

from sentence_transformers import SentenceTransformer

from consumerguard.config import EMBEDDING_MODEL_NAME


@st.cache_resource
def _load_model() -> SentenceTransformer:
    """
    Load the embedding model exactly once and cache it.
    st.cache_resource ensures the model is not reloaded on Streamlit reruns.

    device="cpu"            — skip auto GPU detection (prevents meta tensor on torch 2.x)
    low_cpu_mem_usage=False — disable accelerate's meta-device trick during loading.
                              Without this, transformers 4.5x + torch 2.13 places weights
                              on a virtual 'meta' device, then fails when copying to CPU.
    """
    print(f"[Embedder] Loading model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device="cpu",
        model_kwargs={"low_cpu_mem_usage": False},
    )
    print("[Embedder] Model loaded successfully.")
    return model


def generate_embedding(text: str) -> List[float]:
    """
    Generate a 384-dimensional embedding vector for the given text.

    Args:
        text: The input string to embed.

    Returns:
        A list of 384 floats representing the text embedding.
    """
    model = _load_model()
    # encode() returns a numpy array; convert to plain Python list for ChromaDB
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts in a single batch call.
    More efficient than calling generate_embedding() in a loop.

    Args:
        texts: List of strings to embed.

    Returns:
        List of 384-dimensional float vectors, one per input text.
    """
    model = _load_model()
    embeddings = model.encode(texts, convert_to_numpy=True, batch_size=32, show_progress_bar=True)
    return [emb.tolist() for emb in embeddings]
