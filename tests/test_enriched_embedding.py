"""Verify embeddings include document intelligence context for better retrieval."""
import pytest


def test_stage_embed_accepts_doc_intelligence():
    """stage_embed must accept doc_intelligence parameter."""
    from src.teams.pipeline import TeamsDocumentPipeline

    pipeline = TeamsDocumentPipeline.__new__(TeamsDocumentPipeline)

    import inspect
    sig = inspect.signature(pipeline.stage_embed)
    params = list(sig.parameters.keys())
    assert "doc_intelligence" in params, (
        "stage_embed must accept doc_intelligence parameter "
        "to enrich embeddings with DI context"
    )
