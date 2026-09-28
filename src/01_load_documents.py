"""
Module 3: Loading Documents

Run this file to see what load_documents() found in data/.
(The actual loading logic now lives in rag/loader.py, so it can be
reused by later scripts too.)
"""

from rag.loader import load_documents, DATA_DIR


def main():
    documents = load_documents()

    print(f"Loaded {len(documents)} document(s) from {DATA_DIR}\n")

    for doc in documents:
        char_count = len(doc["text"])
        word_count = len(doc["text"].split())
        preview = doc["text"][:80].replace("\n", " ")

        print(f"- {doc['filename']}")
        print(f"    characters: {char_count}")
        print(f"    words:      {word_count}")
        print(f"    preview:    {preview}...")
        print()


if __name__ == "__main__":
    main()
