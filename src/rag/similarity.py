"""
Shared similarity code for the RAG project.
"""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Measure how similar two vectors are in *direction*, ignoring their
    length. Returns a value from -1 (opposite) to 1 (identical
    direction); in practice, real embeddings usually land between
    0 (unrelated) and 1 (very similar meaning).

    Formula: dot(a, b) / (length(a) * length(b))

    Note: our embeddings are already normalized to length 1 (see
    embedder.py), so length(a) and length(b) are both 1, and this
    formula simplifies to just dot(a, b). We still divide by the
    lengths explicitly here so the function is correct even if it's
    ever given vectors that AREN'T normalized.
    """
    dot_product = np.dot(a, b)
    length_a = np.linalg.norm(a)
    length_b = np.linalg.norm(b)
    return float(dot_product / (length_a * length_b))
