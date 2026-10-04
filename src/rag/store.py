"""
Shared vector storage and search code for the RAG project.

Instead of re-embedding every chunk every single time we want to
search (slow, and wasteful), we embed all chunks ONCE (04_build_index.py:
the "indexing" phase) and save the results to disk. Searching later
(05_search.py: the "query" phase) just loads those saved vectors back
- fast, no AI model needed for that part - and only embeds the new
question.
"""

import json
from pathlib import Path

import numpy as np

from rag.similarity import cosine_similarity

PROJECT_ROOT = Path(__file__).parent.parent.parent
INDEX_DIR = PROJECT_ROOT / "index"
VECTORS_PATH = INDEX_DIR / "vectors.npy"
METADATA_PATH = INDEX_DIR / "metadata.json"


def save_index(chunks: list[dict], vectors: np.ndarray) -> None:
    """
    Save chunk metadata (filename, chunk_id, text) and their embedding
    vectors to disk, as two aligned files:
      - metadata.json: a plain list of chunk info (human-readable)
      - vectors.npy:   a NumPy array of the matching vectors

    chunks[i] and vectors[i] always describe the same chunk.
    """
    INDEX_DIR.mkdir(exist_ok=True)
    METADATA_PATH.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    np.save(VECTORS_PATH, vectors)


def load_index() -> tuple[list[dict], np.ndarray]:
    """
    Load a previously saved index back into memory.
    """
    if not VECTORS_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "No index found. Run 04_build_index.py first to create one."
        )
    chunks = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    vectors = np.load(VECTORS_PATH)
    return chunks, vectors


def search(query_vector: np.ndarray, chunks: list[dict], vectors: np.ndarray, top_k: int = 3) -> list[dict]:
    """
    Brute-force vector search: compare query_vector against every
    stored vector, and return the top_k chunks with the highest
    cosine similarity, each annotated with its "score".

    This checks every single vector (O(n)) - fast enough for a
    handful, or even a few hundred thousand, chunks. Real vector
    databases (FAISS, Chroma, Pinecone, ...) use smarter index
    structures to avoid checking everything once you have millions
    of vectors, but they compute this exact same similarity score
    under the hood.
    """
    scored = []
    for chunk, vector in zip(chunks, vectors):
        score = cosine_similarity(query_vector, vector)
        scored.append({**chunk, "score": score})

    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:top_k]
