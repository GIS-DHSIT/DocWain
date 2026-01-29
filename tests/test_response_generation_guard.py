from __future__ import annotations

from dataclasses import dataclass

from src.retrieval.intent_filter import filter_chunks_by_intent
from src.chat.opener_generator import generate_opener, contains_banned_opener


@dataclass
class DummyChunk:
    text: str
    metadata: dict


def test_intent_filter_education_removes_experience_chunks():
    chunks = [
        DummyChunk(
            text="Experience highlights: led a team of engineers.",
            metadata={"section_title": "Experience"},
        ),
        DummyChunk(
            text="Education: B.Tech in Information Technology, Karpagam College.",
            metadata={"section_title": "Education"},
        ),
    ]
    required_attrs = ["education", "degree", "college"]
    filtered = filter_chunks_by_intent(
        chunks,
        required_attrs,
        ["Ajay"],
        "factual",
    )
    assert len(filtered) == 1
    assert "Education" in (filtered[0].metadata.get("section_title") or "")


def test_citation_label_omits_section_and_page():
    from src.agentic.post_processor import PostProcessor

    label = PostProcessor._format_label(
        {"source_name": "resume.pdf", "section": "Intro", "page": 2}
    )
    assert "Section" not in label
    assert "Page" not in label
    assert "Source" in label


def test_companion_opener_avoids_banned_prefixes():
    opener = generate_opener(
        intent="factual",
        sentiment="neutral",
        follow_up=False,
        style_directives={},
        query="education details",
    )
    assert opener
    assert not contains_banned_opener(opener)
