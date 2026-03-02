"""Unit tests for tool dispatch with RAG context.

Tests that _dispatch_tools() passes reranked chunks to tool handlers,
verifies serialization format, chunk cap, and backward compatibility.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field

import pytest


# ---------------------------------------------------------------------------
# Minimal Chunk / ChunkSource stubs matching src/rag_v3/types.py
# ---------------------------------------------------------------------------


@dataclass
class ChunkSource:
    document_name: str = ""
    document_id: str = ""
    section_title: str = ""


@dataclass
class Chunk:
    id: str = ""
    text: str = ""
    score: float = 0.0
    source: Optional[ChunkSource] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def _make_chunks(n: int) -> List[Chunk]:
    """Create n test chunks."""
    return [
        Chunk(
            id=f"chunk_{i}",
            text=f"Content of chunk {i}",
            score=0.9 - i * 0.05,
            source=ChunkSource(document_name=f"doc_{i}.pdf"),
            meta={"doc_id": f"did_{i}", "domain": "resume"},
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# TestToolDispatchWithContext
# ---------------------------------------------------------------------------


class TestToolDispatchWithContext:
    """Tests that _dispatch_tools passes reranked chunks to tool handlers."""

    def test_tool_receives_chunks(self):
        """Tool payload includes serialized chunks."""
        from src.rag_v3.pipeline import _dispatch_tools, _MAX_TOOL_CONTEXT_CHUNKS

        chunks = _make_chunks(3)
        captured_payload = {}

        async def fake_invoke(name, payload, **kwargs):
            captured_payload.update(payload)
            return {
                "status": "success",
                "result": {"answer": "test"},
                "sources": [],
                "grounded": True,
                "warnings": [],
            }

        with patch("src.tools.base.registry") as mock_registry:
            mock_registry.invoke = fake_invoke
            result = _dispatch_tools(
                tool_names=["content_generate"],
                query="summarize this",
                profile_id="p1",
                subscription_id="s1",
                tool_inputs=None,
                correlation_id="cid-1",
                chunks=chunks,
            )

        assert len(result) == 1
        assert "chunks" in captured_payload["input"]
        assert len(captured_payload["input"]["chunks"]) == 3

    def test_chunks_have_correct_format(self):
        """Serialized chunks have id, text, score, source, meta fields."""
        from src.rag_v3.pipeline import _dispatch_tools

        chunks = _make_chunks(2)
        captured_payload = {}

        async def fake_invoke(name, payload, **kwargs):
            captured_payload.update(payload)
            return {"status": "success", "result": {}, "sources": [], "grounded": True, "warnings": []}

        with patch("src.tools.base.registry") as mock_registry:
            mock_registry.invoke = fake_invoke
            _dispatch_tools(
                tool_names=["test_tool"],
                query="q",
                profile_id="p",
                subscription_id="s",
                tool_inputs=None,
                correlation_id="cid",
                chunks=chunks,
            )

        serialized = captured_payload["input"]["chunks"]
        for sc in serialized:
            assert "id" in sc
            assert "text" in sc
            assert "score" in sc
            assert "source" in sc
            assert "meta" in sc

        assert serialized[0]["id"] == "chunk_0"
        assert serialized[0]["text"] == "Content of chunk 0"
        assert serialized[0]["source"] == "doc_0.pdf"
        assert serialized[0]["meta"]["domain"] == "resume"

    def test_chunks_capped_at_max(self):
        """Only _MAX_TOOL_CONTEXT_CHUNKS are forwarded to tools."""
        from src.rag_v3.pipeline import _dispatch_tools, _MAX_TOOL_CONTEXT_CHUNKS

        chunks = _make_chunks(20)  # More than the cap
        captured_payload = {}

        async def fake_invoke(name, payload, **kwargs):
            captured_payload.update(payload)
            return {"status": "success", "result": {}, "sources": [], "grounded": True, "warnings": []}

        with patch("src.tools.base.registry") as mock_registry:
            mock_registry.invoke = fake_invoke
            _dispatch_tools(
                tool_names=["test_tool"],
                query="q",
                profile_id="p",
                subscription_id="s",
                tool_inputs=None,
                correlation_id="cid",
                chunks=chunks,
            )

        assert len(captured_payload["input"]["chunks"]) == _MAX_TOOL_CONTEXT_CHUNKS

    def test_tool_works_without_chunks(self):
        """Tool dispatch works when chunks=None (backward compat)."""
        from src.rag_v3.pipeline import _dispatch_tools

        captured_payload = {}

        async def fake_invoke(name, payload, **kwargs):
            captured_payload.update(payload)
            return {"status": "success", "result": {"ok": True}, "sources": [], "grounded": True, "warnings": []}

        with patch("src.tools.base.registry") as mock_registry:
            mock_registry.invoke = fake_invoke
            result = _dispatch_tools(
                tool_names=["test_tool"],
                query="q",
                profile_id="p",
                subscription_id="s",
                tool_inputs=None,
                correlation_id="cid",
                chunks=None,
            )

        assert len(result) == 1
        assert captured_payload["input"]["chunks"] == []

    def test_empty_chunks_list(self):
        """Empty chunks list results in empty serialized chunks."""
        from src.rag_v3.pipeline import _dispatch_tools

        captured_payload = {}

        async def fake_invoke(name, payload, **kwargs):
            captured_payload.update(payload)
            return {"status": "success", "result": {}, "sources": [], "grounded": True, "warnings": []}

        with patch("src.tools.base.registry") as mock_registry:
            mock_registry.invoke = fake_invoke
            _dispatch_tools(
                tool_names=["test_tool"],
                query="q",
                profile_id="p",
                subscription_id="s",
                tool_inputs=None,
                correlation_id="cid",
                chunks=[],
            )

        assert captured_payload["input"]["chunks"] == []

    def test_chunk_without_source(self):
        """Chunks with source=None serialize source as empty string."""
        from src.rag_v3.pipeline import _dispatch_tools

        chunks = [Chunk(id="c1", text="text", score=0.8, source=None, meta={})]
        captured_payload = {}

        async def fake_invoke(name, payload, **kwargs):
            captured_payload.update(payload)
            return {"status": "success", "result": {}, "sources": [], "grounded": True, "warnings": []}

        with patch("src.tools.base.registry") as mock_registry:
            mock_registry.invoke = fake_invoke
            _dispatch_tools(
                tool_names=["test_tool"],
                query="q",
                profile_id="p",
                subscription_id="s",
                tool_inputs=None,
                correlation_id="cid",
                chunks=chunks,
            )

        assert captured_payload["input"]["chunks"][0]["source"] == ""

    def test_tool_inputs_merged_with_chunks(self):
        """Per-tool inputs are merged alongside chunks in the payload."""
        from src.rag_v3.pipeline import _dispatch_tools

        chunks = _make_chunks(2)
        captured_payload = {}

        async def fake_invoke(name, payload, **kwargs):
            captured_payload.update(payload)
            return {"status": "success", "result": {}, "sources": [], "grounded": True, "warnings": []}

        with patch("src.tools.base.registry") as mock_registry:
            mock_registry.invoke = fake_invoke
            _dispatch_tools(
                tool_names=["content_generate"],
                query="write a summary",
                profile_id="p1",
                subscription_id="s1",
                tool_inputs={"content_generate": {"content_type": "summary"}},
                correlation_id="cid",
                chunks=chunks,
            )

        # Both chunks and extra input should be present
        assert len(captured_payload["input"]["chunks"]) == 2
        assert captured_payload["input"]["content_type"] == "summary"
        assert captured_payload["input"]["query"] == "write a summary"
