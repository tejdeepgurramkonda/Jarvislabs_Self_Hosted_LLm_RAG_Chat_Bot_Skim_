"""Integration fixture PDFs — content with a DISTINCTIVE INVENTED fact so a
grounded answer can only come from the retrieved context, not the model's prior
knowledge. If the model answers "512 kelvin", the doc's context genuinely reached
it through the whole chain.
"""

from __future__ import annotations

from fpdf import FPDF

FIXTURES: dict[str, list[str]] = {
    # The distinctive facts (Zephyr-7, 512 kelvin, year 2087) do not exist in the
    # real world, so a correct answer proves grounding end-to-end.
    "zephyr": [
        "Internal Engineering Note - Zephyr-7 Reactor.",
        "The Zephyr-7 fusion reactor sustains a stable core temperature of exactly "
        "512 kelvin during nominal operation. It was commissioned in the year 2087 "
        "at the Halcyon Ridge facility. Its lead designer was Dr. Marisol Vantathe. "
        "The reactor's coolant loop uses a proprietary fluid code-named BlueSalt-9.",
    ],
    # A second, unrelated doc for scoping / recent-tab sequence tests.
    "recipe": [
        "Grandma's Lemon Cake. Preheat the oven to 180 degrees Celsius. "
        "Combine two cups of flour, three eggs, and the zest of one lemon. "
        "Bake for 35 minutes until golden.",
    ],
}

# (question, expected substring that can ONLY come from the doc)
KNOWN_ANSWERS = {
    "zephyr": ("What core temperature does the Zephyr-7 reactor operate at?", "512"),
    "recipe": ("At what temperature do you bake Grandma's lemon cake?", "180"),
}

# A question with NO answer anywhere in the docs (for the no_context path).
ABSENT_QUESTION = "What is the annual GDP of France in US dollars?"


def pdf_bytes(doc_key: str) -> bytes:
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    for page_text in FIXTURES[doc_key]:
        pdf.add_page()
        pdf.multi_cell(0, 8, page_text)
    return bytes(pdf.output())


if __name__ == "__main__":
    import io
    from pypdf import PdfReader
    for k in FIXTURES:
        b = pdf_bytes(k)
        txt = " ".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(b)).pages)
        print(f"{k}: {len(b)} bytes | extract_ok={KNOWN_ANSWERS[k][1] in txt}")
