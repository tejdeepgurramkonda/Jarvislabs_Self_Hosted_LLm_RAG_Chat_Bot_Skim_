"""Prompt construction for grounded (RAG) answers.

The system prompt pins the model to the retrieved context and forbids outside
knowledge. Each context block is labeled with its source (filename, page) so the
model can ground its answer and we can cross-check the citations later.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided "
    "context from the user's documents.\n"
    "Rules:\n"
    "- Use only the information in the CONTEXT section. Do not use outside knowledge.\n"
    "- If the context does not contain enough information to answer, say exactly: "
    '"I don\'t know based on the provided documents."\n'
    "- Do not invent sources, page numbers, or facts.\n"
    "- Be concise and answer in plain language."
)


def format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks into a numbered, source-labeled context block."""
    if not chunks:
        return "(no context available)"
    blocks = []
    for i, ch in enumerate(chunks, start=1):
        src = ch.get("filename") or ch.get("doc_id", "unknown")
        page = ch.get("page")
        label = f"[{i}] source: {src}" + (f", page {page}" if page is not None else "")
        blocks.append(f"{label}\n{ch.get('text', '').strip()}")
    return "\n\n".join(blocks)


def build_messages(query: str, chunks: list[dict]) -> list[dict]:
    """Build OpenAI-style chat messages for the vLLM chat.completions call."""
    context = format_context(chunks)
    user_content = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer using only the context above."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
