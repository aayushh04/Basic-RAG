"""
Shared chunking code for the RAG project. See Module 4 for the full
explanation of why we split documents into overlapping chunks.
"""


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Split `text` into overlapping chunks of roughly `chunk_size`
    characters each. `overlap` characters are repeated between
    consecutive chunks so a fact sitting on a chunk boundary doesn't
    get cut in half and lost.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start = end - overlap

    return chunks


def chunk_documents(documents: list[dict], chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    """
    Chunk every document and return one flat list covering all of them.
    Each item is a dict: {"filename": ..., "chunk_id": ..., "text": ...}
    """
    all_chunks = []
    for doc in documents:
        pieces = chunk_text(doc["text"], chunk_size, overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "filename": doc["filename"],
                "chunk_id": i,
                "text": piece,
            })
    return all_chunks
