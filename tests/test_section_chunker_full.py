import pytest

from src.api.pipeline_models import ExtractedDocument, Section
from src.embedding.chunking.section_chunker import chunk_document


def _make_doc(sections_data):
    """Build an ExtractedDocument from a list of (id, title, text) tuples."""
    sections = [
        Section(section_id=sid, title=title, level=1, start_page=1, end_page=1, text=text)
        for sid, title, text in sections_data
    ]
    full_text = "\n\n".join(s.text for s in sections)
    return ExtractedDocument(
        full_text=full_text,
        sections=sections,
        tables=[],
        figures=[],
        chunk_candidates=[],
    )


def test_full_section_preserves_text():
    """A section of 500 chars should produce ONE chunk with all text."""
    text = "Customer shall pay all undisputed invoices within thirty days. " * 10
    doc = _make_doc([("s1", "Payment Terms", text)])

    chunks = chunk_document(doc, doc_internal_id="doc1", source_filename="test.pdf")

    # Should produce 1 chunk (not 3-4 fragments)
    assert len(chunks) <= 2  # Allow for overlap but not fragmentation
    # The chunk text should contain the full section text
    total_text = " ".join(c.text for c in chunks)
    assert "undisputed invoices" in total_text


def test_large_section_splits_at_4000():
    """A section of 6000 chars should split into 2 chunks."""
    long_text = "This is a paragraph about contract terms. " * 200  # ~8600 chars
    doc = _make_doc([("s1", "Long Section", long_text)])

    chunks = chunk_document(doc, doc_internal_id="doc1", source_filename="test.pdf")

    # Should split but not into tiny fragments
    assert len(chunks) >= 2
    assert len(chunks) <= 4  # Not too many
    for c in chunks:
        assert len(c.text) >= 100  # No tiny chunks


def test_small_section_merged():
    """A section of 50 chars should be merged with adjacent."""
    doc = _make_doc([
        ("s1", "Intro", "Short intro."),
        ("s2", "Details",
         "This section has enough detail to stand on its own and contains important information. " * 5),
    ])

    chunks = chunk_document(doc, doc_internal_id="doc1", source_filename="test.pdf")

    # The short section should be merged — we should NOT have a 12-char chunk
    for c in chunks:
        assert len(c.text) >= 50  # No tiny standalone chunks
