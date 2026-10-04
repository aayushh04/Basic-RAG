"""
Module 6, Step 2: Search

Loads the index built by 04_build_index.py (no re-embedding of
chunks needed - that already happened) and lets you type questions.
Each question is embedded and compared against every stored chunk
vector; the most similar chunks are printed with their similarity
scores. This is the QUERY phase from our architecture diagram.
"""

from rag.embedder import embed_texts
from rag.store import load_index, search


def main():
    chunks, vectors = load_index()
    print(f"Loaded index: {len(chunks)} chunks.\n")
    print("Type a question and press Enter (or type 'quit' to exit).\n")

    while True:
        question = input("> ").strip()
        if not question or question.lower() == "quit":
            break

        query_vector = embed_texts([question])[0]
        results = search(query_vector, chunks, vectors, top_k=3)

        print(f"\nTop {len(results)} matches:")
        for r in results:
            preview = r["text"][:100].replace("\n", " ")
            print(f"  [{r['score']:.3f}] {r['filename']} (chunk {r['chunk_id']})")
            print(f"          {preview}...")
        print()


if __name__ == "__main__":
    main()
