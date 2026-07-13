"""Prompt-building tests (PB1, PB2)."""

from __future__ import annotations

from app.utils.prompt_templates import build_messages, format_context


CHUNKS = [
    {"doc_id": "d1", "filename": "paris.pdf", "page": 1, "chunk_idx": 0,
     "text": "The Eiffel Tower is located in Paris, France.", "score": 0.9},
]


def test_PB1_context_injected_and_grounding_rule(record):
    msgs = build_messages("Where is the Eiffel Tower?", CHUNKS)
    system = msgs[0]["content"]
    user = msgs[1]["content"]
    record("PB1", "Prompt", "context injected + 'answer only from context'",
           inp="build_messages('Where is the Eiffel Tower?', [paris chunk])",
           ideal="system forbids outside knowledge; user has CONTEXT with chunk text + QUESTION",
           actual=f"roles={[m['role'] for m in msgs]}; system_has_only_context={'ONLY the provided context' in system}; "
                  f"user_has_context={'CONTEXT:' in user and 'Eiffel Tower' in user}; user_has_question={'QUESTION:' in user}")
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    assert "ONLY the provided context" in system and "Do not use outside knowledge" in system
    assert "CONTEXT:" in user and "The Eiffel Tower is located in Paris" in user
    assert "QUESTION:" in user and "Where is the Eiffel Tower?" in user


def test_PB2_source_labels_and_empty(record):
    labeled = format_context(CHUNKS)
    empty = format_context([])
    record("PB2", "Prompt", "source labels rendered; empty context handled",
           inp="format_context([paris chunk]) and format_context([])",
           ideal="labels show filename + page; [] -> '(no context available)'",
           actual=f"labeled_has_src={'source: paris.pdf' in labeled and 'page 1' in labeled}; empty={empty!r}")
    assert "source: paris.pdf" in labeled and "page 1" in labeled
    assert empty == "(no context available)"
