"""
Module 7: Ask - the full RAG pipeline, end to end

For each question you type:
  1. Embed the question              (Module 5/6)
  2. Retrieve the top-k chunks        (Module 6)
  3. Build a grounded prompt          (Module 7)
  4. Send it to Claude, get an answer (Module 7)

Run 04_build_index.py first if you haven't already.
"""

from rag.embedder import embed_texts
from rag.store import load_index, search
from rag.generator import ask_model


def main():
    chunks, vectors = load_index()
    print(f"Loaded index: {len(chunks)} chunks.\n")
    print("Type a question and press Enter (or type 'quit' to exit).\n")

    while True:
        question = input("> ").strip()
        if not question or question.lower() == "quit":
            break

        query_vector = embed_texts([question])[0]
        retrieved = search(query_vector, chunks, vectors, top_k=3)

        print("\nRetrieved chunks:")
        for r in retrieved:
            print(f"  [{r['score']:.3f}] {r['filename']} (chunk {r['chunk_id']})")

        print("\nAsking the model...\n")
        answer = ask_model(question, retrieved)
        print(answer)
        print()


if __name__ == "__main__":
    main()
