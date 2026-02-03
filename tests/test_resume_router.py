from __future__ import annotations

from dataclasses import dataclass

from src.tools.resume_router import should_route_resume_analyzer


@dataclass
class DummyChunk:
    text: str
    metadata: dict


def test_resume_query_triggers_resume_analyzer():
    chunks = [DummyChunk(text="Profile summary", metadata={})]
    assert should_route_resume_analyzer(
        query_text="extract skills and certifications for all profiles",
        chunks=chunks,
        doc_inventory=[],
    )


def test_invoice_query_does_not_trigger_resume_analyzer():
    chunks = [DummyChunk(text="Invoice total $100", metadata={"section_title": "Totals"})]
    assert not should_route_resume_analyzer(
        query_text="total amount due",
        chunks=chunks,
        doc_inventory=[],
    )
