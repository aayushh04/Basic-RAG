"""
Module 6, Step 1: Build the index

Loads documents, chunks them, embeds every chunk, and saves the
result to disk (index/vectors.npy + index/metadata.json). This is
the INDEXING phase from our architecture diagram - run this once,
and again whenever files in data/ change. It is NOT run every time
someone asks a question (that's 05_search.py).
"""

from rag.loader import load_documents
from rag.chunker import chunk_documents
from rag.embedder import embed_texts
from rag.store import save_index, INDEX_DIR


def main():
    documents = load_documents()
    chunks = chunk_documents(documents)
    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(chunks)} chunks from {len(documents)} document(s)...")
    vectors = embed_texts(texts)

    save_index(chunks, vectors)
    print(f"\nSaved index to {INDEX_DIR}/")
    print(f"  - metadata.json ({len(chunks)} chunks)")
    print(f"  - vectors.npy (shape {vectors.shape})")


if __name__ == "__main__":
    main()
