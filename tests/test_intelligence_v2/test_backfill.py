"""Tests for the intelligence backfill script."""

from unittest.mock import MagicMock, patch

import pytest

from scripts.backfill_intelligence import backfill


def _make_doc(document_id, subscription_id="sub1", profile_id="prof1", filename="doc.pdf"):
    return {
        "document_id": document_id,
        "subscription_id": subscription_id,
        "profile_id": profile_id,
        "filename": filename,
    }


def test_backfill_processes_unanalyzed_documents():
    """Two unanalyzed docs should both be processed successfully."""
    mongodb = MagicMock()
    cursor = MagicMock()
    cursor.limit.return_value = [
        _make_doc("doc1"),
        _make_doc("doc2"),
    ]
    mongodb.find.return_value = cursor

    analyzer = MagicMock()
    analyzer.analyze.return_value = {"document_id": "x", "intelligence": {}}

    extracted_data = {"full_text": "some content"}
    load_fn = MagicMock(return_value=extracted_data)

    stats = backfill(mongodb, analyzer, load_fn, batch_size=50)

    assert stats["processed"] == 2
    assert stats["failed"] == 0
    assert stats["skipped"] == 0
    assert stats["total"] == 2
    assert analyzer.analyze.call_count == 2


def test_backfill_skips_already_analyzed():
    """When mongodb returns no docs, nothing should be processed."""
    mongodb = MagicMock()
    cursor = MagicMock()
    cursor.limit.return_value = []
    mongodb.find.return_value = cursor

    analyzer = MagicMock()
    load_fn = MagicMock()

    stats = backfill(mongodb, analyzer, load_fn, batch_size=50)

    assert stats["processed"] == 0
    assert stats["failed"] == 0
    assert stats["skipped"] == 0
    assert stats["total"] == 0
    analyzer.analyze.assert_not_called()
    load_fn.assert_not_called()


def test_backfill_continues_on_single_doc_failure():
    """If analyzer.analyze fails on one doc, processing should continue."""
    mongodb = MagicMock()
    cursor = MagicMock()
    cursor.limit.return_value = [
        _make_doc("doc1"),
        _make_doc("doc2"),
    ]
    mongodb.find.return_value = cursor

    analyzer = MagicMock()
    analyzer.analyze.side_effect = [RuntimeError("LLM down"), {"document_id": "doc2"}]

    extracted_data = {"full_text": "content"}
    load_fn = MagicMock(return_value=extracted_data)

    stats = backfill(mongodb, analyzer, load_fn, batch_size=50)

    assert stats["processed"] == 1
    assert stats["failed"] == 1
    assert stats["skipped"] == 0
    assert stats["total"] == 2


def test_backfill_skips_missing_pickle():
    """Documents with missing/empty extracted content should be skipped."""
    mongodb = MagicMock()
    cursor = MagicMock()
    cursor.limit.return_value = [
        _make_doc("doc1"),
        _make_doc("doc2"),
    ]
    mongodb.find.return_value = cursor

    analyzer = MagicMock()
    analyzer.analyze.return_value = {"document_id": "doc2"}

    load_fn = MagicMock(side_effect=[ValueError("not found"), {"full_text": "ok"}])

    stats = backfill(mongodb, analyzer, load_fn, batch_size=50)

    assert stats["processed"] == 1
    assert stats["skipped"] == 1
    assert stats["failed"] == 0
    assert stats["total"] == 2


def test_backfill_subscription_filter():
    """When subscription_id is provided, the query should include it."""
    mongodb = MagicMock()
    cursor = MagicMock()
    cursor.limit.return_value = []
    mongodb.find.return_value = cursor

    analyzer = MagicMock()
    load_fn = MagicMock()

    backfill(mongodb, analyzer, load_fn, batch_size=10, subscription_id="sub_abc")

    query = mongodb.find.call_args[0][0]
    assert query["subscription_id"] == "sub_abc"
