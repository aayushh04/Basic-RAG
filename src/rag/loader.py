"""
Shared document-loading code for the RAG project.

Other scripts import load_documents() from here instead of each
redefining it. This keeps the file-reading logic in exactly one place.
"""

from pathlib import Path

# This file lives at: src/rag/loader.py
#   .parent       -> src/rag/
#   .parent.parent -> src/
#   .parent.parent.parent -> the project root (test/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_documents(folder: Path = DATA_DIR) -> list[dict]:
    """
    Read every .txt file in `folder` and return a list of documents.

    Each document is a dict: {"filename": ..., "text": ...}
    """
    documents = []
    for file_path in sorted(folder.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        documents.append({
            "filename": file_path.name,
            "text": text,
        })
    return documents
