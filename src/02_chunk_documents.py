"""
Module 4: Chunking

Run this file to see documents from data/ split into overlapping
chunks. (The actual chunking logic now lives in rag/chunker.py, so
later scripts can reuse it too.)
"""

from rag.loader import load_documents
from rag.chunker import chunk_documents


def main():
    documents = load_documents()
    chunks = chunk_documents(documents)

    print(f"{len(documents)} document(s) -> {len(chunks)} chunk(s) total\n")

    for doc in documents:
        doc_chunks = [c for c in chunks if c["filename"] == doc["filename"]]
        print(f"{doc['filename']}: {len(doc['text'])} characters -> {len(doc_chunks)} chunk(s)")

    print("\n--- Example: first two chunks of gym.txt ---")
    gym_chunks = [c for c in chunks if c["filename"] == "gym.txt"]
    for c in gym_chunks[:2]:
        print(f"\n[chunk {c['chunk_id']}] ({len(c['text'])} chars)")
        print(c["text"])


if __name__ == "__main__":
    main()
