"""
Shared embedding code for the RAG project.

An embedding turns a piece of text into a list of numbers (a vector)
that captures its meaning. Pieces of text with similar meaning end up
with similar vectors — this is what lets us search "by meaning"
instead of by exact keyword matches.
"""

from sentence_transformers import SentenceTransformer

# "all-MiniLM-L6-v2" is a small, free, well-known embedding model.
# It turns any piece of text into a vector of 384 numbers.
_MODEL_NAME = "all-MiniLM-L6-v2"

# Loading the model from disk is slow (a few seconds) but only needs
# to happen once. We do it here, at import time, so it runs a single
# time no matter how many times embed_texts() is called afterwards.
_model = SentenceTransformer(_MODEL_NAME)


def embed_texts(texts: list[str]):
    """
    Turn a list of strings into a list of embedding vectors.

    Returns a NumPy array of shape (len(texts), 384): one 384-number
    vector per input string. normalize_embeddings=True scales every
    vector to length 1, which makes similarity comparisons simpler
    (see rag/similarity.py).
    """
    return _model.encode(texts, normalize_embeddings=True)
