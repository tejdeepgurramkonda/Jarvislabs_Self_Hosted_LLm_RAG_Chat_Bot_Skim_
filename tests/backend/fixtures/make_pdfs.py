"""Generate small fixture PDFs with KNOWN content, so retrieval answers are checkable.

The app never generates PDFs — it reads user-uploaded ones. These are test inputs
only: deterministic PDFs whose text we control, letting tests assert that ingestion
produced the right chunks and that a question retrieves the correct passage.

`fpdf2` produces standard text PDFs that `pypdf` extracts cleanly.
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

# doc_key -> list of pages (each page is one block of text)
FIXTURES: dict[str, list[str]] = {
    "paris": [
        "The Eiffel Tower is located in Paris, France. "
        "It was completed in 1889 and stands about 330 meters tall. "
        "It was designed by the engineer Gustave Eiffel."
    ],
    "biology": [
        "Photosynthesis is the process by which green plants convert sunlight, "
        "water, and carbon dioxide into glucose and oxygen. "
        "Chlorophyll in the leaves absorbs the light energy."
    ],
    # two pages -> exercises page metadata (page 1 and page 2)
    "policy": [
        "Company leave policy. Full-time employees are entitled to 20 days of "
        "paid annual leave each calendar year.",
        "Sick leave is separate from annual leave. Unused annual leave may carry "
        "over a maximum of 5 days into the next year.",
    ],
}

# A short phrase we know lives verbatim in each doc, for exact-match retrieval tests.
KNOWN_ANSWERS = {
    "paris": ("Where is the Eiffel Tower located?", "Paris"),
    "biology": ("What does photosynthesis produce?", "oxygen"),
    "policy": ("How many paid annual leave days do employees get?", "20 days"),
}


def _write_pdf(pages: list[str], dest: Path) -> None:
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    for text in pages:
        pdf.add_page()
        pdf.multi_cell(0, 8, text)
    pdf.output(str(dest))


def build_fixture_pdfs(out_dir: Path) -> dict[str, Path]:
    """Write all fixture PDFs into out_dir; return {doc_key: path}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, pages in FIXTURES.items():
        dest = out_dir / f"{key}.pdf"
        _write_pdf(pages, dest)
        paths[key] = dest
    return paths


def pdf_bytes(doc_key: str) -> bytes:
    """Return the raw bytes of one fixture PDF (built in-memory)."""
    import io

    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    for text in FIXTURES[doc_key]:
        pdf.add_page()
        pdf.multi_cell(0, 8, text)
    buf = pdf.output()  # fpdf2 returns a bytearray
    return bytes(buf)


if __name__ == "__main__":  # manual: python fixtures/make_pdfs.py
    out = Path(__file__).resolve().parent / "pdfs"
    for k, p in build_fixture_pdfs(out).items():
        print(f"{k}: {p} ({p.stat().st_size} bytes)")
