"""
Module 5: Embeddings

Turns our chunks into vectors (embeddings) and demonstrates that
cosine similarity between vectors reflects similarity of MEANING,
not just shared words. This is what makes "search by meaning"
possible, ahead of Module 6 (vector search).
"""

from rag.loader import load_documents
from rag.chunker import chunk_documents
from rag.embedder import embed_texts
from rag.similarity import cosine_similarity


def main():
    documents = load_documents()
    chunks = chunk_documents(documents)
    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(texts)} chunks...")
    vectors = embed_texts(texts)
    print(f"Done. Each chunk became a vector of shape: {vectors[0].shape}")
    print(f"First 8 numbers of chunk 0's vector:\n{vectors[0][:8]}\n")

    # --- Demo 1: two chunks from the SAME topic (both gym-related) ---
    gym_indices = [i for i, c in enumerate(chunks) if c["filename"] == "gym.txt"]
    a, b = gym_indices[0], gym_indices[1]
    sim = cosine_similarity(vectors[a], vectors[b])
    print("--- Same-topic comparison (gym chunk vs. another gym chunk) ---")
    print(f"Similarity: {sim:.3f}\n")

    # --- Demo 2: two chunks from DIFFERENT topics (gym vs. recipe) ---
    recipe_indices = [i for i, c in enumerate(chunks) if c["filename"] == "recipe_notes.txt"]
    a, b = gym_indices[0], recipe_indices[0]
    sim = cosine_similarity(vectors[a], vectors[b])
    print("--- Different-topic comparison (gym chunk vs. recipe chunk) ---")
    print(f"Similarity: {sim:.3f}\n")

    # --- Demo 3: a real QUESTION compared against chunks from every file ---
    # This is a preview of Module 6: we embed a question the same way
    # we embedded the chunks, then see which chunk it's closest to.
    question = "What is the rep range for bench press?"
    question_vector = embed_texts([question])[0]

    print(f"--- Question: \"{question}\" ---")
    print("Similarity to one chunk from each file:")
    seen_files = set()
    for i, c in enumerate(chunks):
        if c["filename"] in seen_files:
            continue
        seen_files.add(c["filename"])
        sim = cosine_similarity(question_vector, vectors[i])
        print(f"  {sim:.3f}  <-  {c['filename']} (chunk {c['chunk_id']})")


if __name__ == "__main__":
    main()
