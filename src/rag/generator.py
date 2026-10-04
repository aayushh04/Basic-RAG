"""
Shared answer-generation code for the RAG project.

This is the "G" in RAG: taking retrieved chunks and a question, and
asking an LLM to write an answer grounded in those chunks.

We're using Ollama here: it runs an open-weight model LOCALLY on
your own machine (a local server at http://localhost:11434), so
there's no API key and no per-question cost. Compare this file to
how it looked when it called the Anthropic API - notice build_prompt()
and SYSTEM_PROMPT don't change at all. Only the one function that
actually sends the request is different. That's the point: RAG's
retrieval pipeline (Modules 3-6) doesn't care which LLM writes the
final answer.
"""

import ollama

MODEL_NAME = "llama3.2"


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Combine the retrieved chunks and the question into a single
    prompt string, with each chunk clearly labeled with its source
    file. This labeling is what lets the model tell you WHERE an
    answer came from.
    """
    context_sections = []
    for chunk in retrieved_chunks:
        context_sections.append(
            f"[Source: {chunk['filename']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_sections)

    return f"""Here are some excerpts from the user's documents:

{context}

Question: {question}"""


SYSTEM_PROMPT = """You answer questions using ONLY the provided document excerpts.

Rules:
- Base your answer strictly on the excerpts given to you. Do not use outside knowledge.
- If the excerpts do not contain the answer, say clearly: "I don't know based on the provided documents."
- When you do answer, mention which file(s) the information came from.
- Be concise."""


def ask_model(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Send the question + retrieved context to the local model and
    return its answer as plain text.
    """
    prompt = build_prompt(question, retrieved_chunks)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    return response["message"]["content"]
