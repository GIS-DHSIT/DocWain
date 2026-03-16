"""Embedding pipeline Celery task."""

import hashlib
import json
import time

from celery.exceptions import SoftTimeLimitExceeded
from src.celery_app import app
from src.api.document_status import (
    update_stage, update_pipeline_status, append_audit_log,
    get_document_record
)
from src.api.statuses import (
    STAGE_IN_PROGRESS, STAGE_COMPLETED, STAGE_FAILED,
    PIPELINE_EMBEDDING_IN_PROGRESS, PIPELINE_TRAINING_COMPLETED,
    PIPELINE_EMBEDDING_FAILED, KG_COMPLETED
)
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Blob helpers
# ---------------------------------------------------------------------------

def _download_blob_json(blob_path: str) -> dict:
    """Download and parse a JSON blob from Azure Blob storage."""
    from src.api.blob_content_store import get_blob_client

    container = get_blob_client()
    blob_client = container.get_blob_client(blob_path)
    raw = blob_client.download_blob().readall()
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Quality grading
# ---------------------------------------------------------------------------

def _quality_grade(score: float) -> str:
    """Convert a 0-1 quality score to a letter grade."""
    if score >= 0.8:
        return "A"
    if score >= 0.6:
        return "B"
    if score >= 0.4:
        return "C"
    if score >= 0.2:
        return "D"
    return "F"


# ---------------------------------------------------------------------------
# Dedup gate (Jaccard similarity)
# ---------------------------------------------------------------------------

def _dedup_chunks(chunks: list, threshold: float = 0.92) -> tuple:
    """Remove near-duplicate chunks using word-set Jaccard similarity.

    Returns (kept_chunks, removed_count).
    """
    if len(chunks) <= 1:
        return chunks, 0

    def _word_set(text: str) -> set:
        return set(text.lower().split())

    kept = []
    kept_sets = []
    removed = 0

    for chunk in chunks:
        text = chunk.get("text", "")
        ws = _word_set(text)
        is_dup = False
        for seen in kept_sets:
            if not ws or not seen:
                continue
            jaccard = len(ws & seen) / len(ws | seen)
            if jaccard >= threshold:
                is_dup = True
                break
        if is_dup:
            removed += 1
        else:
            kept.append(chunk)
            kept_sets.append(ws)

    return kept, removed


# ---------------------------------------------------------------------------
# KG node lookup
# ---------------------------------------------------------------------------

def _get_kg_node_ids(document_id: str) -> list:
    """Query Neo4j for KG node IDs linked to this document."""
    try:
        from src.kg.kg_store import KGStore
        store = KGStore()
        nodes = store.get_document_nodes(document_id)
        return [n.get("node_id") or n.get("id", "") for n in nodes] if nodes else []
    except Exception as exc:
        logger.debug("KG node lookup failed for %s: %s", document_id, exc)
        return []


# ---------------------------------------------------------------------------
# Main task
# ---------------------------------------------------------------------------

@app.task(bind=True, name="src.tasks.embedding.embed_document",
          max_retries=3, soft_time_limit=1500)
def embed_document(self, document_id: str, subscription_id: str,
                   profile_id: str):
    """Generate embeddings for validated document content."""
    try:
        update_stage(document_id, "embedding", status=STAGE_IN_PROGRESS,
                     celery_task_id=self.request.id)
        update_pipeline_status(document_id, PIPELINE_EMBEDDING_IN_PROGRESS)
        append_audit_log(document_id, "EMBEDDING_STARTED", by="user",
                        celery_task_id=self.request.id)

        # ── 1. Get document record from MongoDB ─────────────────────────
        record = get_document_record(document_id)
        if not record:
            raise ValueError(f"Document record not found: {document_id}")

        # ── 2. Load extraction JSON from Azure Blob ─────────────────────
        extraction = record.get("extraction", {})
        summary = extraction.get("summary", {})
        blob_path = summary.get("blob_path")
        if not blob_path:
            raise ValueError(f"No extraction blob_path for document {document_id}")

        extraction_data = _download_blob_json(blob_path)
        logger.info("Loaded extraction data from %s for document %s", blob_path, document_id)

        # ── 3. Load screening summary ───────────────────────────────────
        screening = record.get("screening", {})
        screening_summary = screening.get("summary", {}) or {}

        # ── 4. Check KG status and get node IDs ─────────────────────────
        kg_info = record.get("knowledge_graph", {})
        kg_status = kg_info.get("status", "")
        kg_node_ids = []
        kg_ready = (kg_status == KG_COMPLETED)
        if kg_ready:
            kg_node_ids = _get_kg_node_ids(document_id)

        # ── 5. Chunk the text ───────────────────────────────────────────
        from src.embedding.chunking.section_chunker import SectionChunker

        chunker = SectionChunker(
            target_chunk_chars=900,   # ~250 tokens
            min_chunk_chars=200,
            max_chunk_chars=1600,     # ~450 tokens
            overlap_sentences=2,      # ~60 token overlap
        )

        source_filename = record.get("source_file", "") or record.get("filename", "") or document_id
        raw_chunks = chunker.chunk_document(
            extraction_data,
            doc_internal_id=document_id,
            source_filename=source_filename,
        )

        # Convert Chunk dataclass objects to dicts for payload builder
        chunk_dicts = []
        for c in raw_chunks:
            chunk_hash = hashlib.sha256(c.text.encode("utf-8")).hexdigest()[:16]
            chunk_dicts.append({
                "text": c.text,
                "type": "text",
                "hash": chunk_hash,
                "token_count": len(c.text.split()),
                "section": {
                    "id": hashlib.sha1(
                        f"{c.section_title}|{c.chunk_index}".encode("utf-8")
                    ).hexdigest()[:12],
                    "title": c.section_title,
                    "path": c.section_path.split(" > ") if c.section_path else [],
                    "level": 1,
                },
                "provenance": {
                    "page_start": c.page_start or 0,
                    "page_end": c.page_end or 0,
                },
            })

        logger.info("Chunked document %s into %d chunks", document_id, len(chunk_dicts))

        # ── 6. Dedup gate (Jaccard >= 0.92) ─────────────────────────────
        chunk_dicts, dedup_removed = _dedup_chunks(chunk_dicts, threshold=0.92)
        if dedup_removed:
            logger.info("Dedup removed %d near-duplicate chunks for %s", dedup_removed, document_id)

        if not chunk_dicts:
            raise ValueError(f"No chunks remaining after dedup for {document_id}")

        # ── 7. Generate dense vectors ───────────────────────────────────
        from src.embedding.model_loader import encode_with_fallback

        texts = [c["text"] for c in chunk_dicts]
        vectors = encode_with_fallback(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        logger.info("Generated %d embedding vectors for %s", len(vectors), document_id)

        # ── 8. Build enriched payloads and quality grades ───────────────
        from src.embedding.payload_builder import build_enriched_payload
        from src.embedding.pipeline_enhancement import calculate_chunk_quality

        quality_scores = []
        payloads = []
        for idx, chunk in enumerate(chunk_dicts):
            q_score = calculate_chunk_quality(
                chunk["text"],
                section_type=chunk.get("type", "text"),
                has_entities=bool(extraction_data.get("entities")),
            )
            quality_scores.append(q_score)
            grade = _quality_grade(q_score)

            payload = build_enriched_payload(
                chunk=chunk,
                chunk_index=idx,
                document_id=document_id,
                subscription_id=subscription_id,
                profile_id=profile_id,
                extraction_data=extraction_data,
                screening_summary=screening_summary,
                kg_node_ids=kg_node_ids if kg_ready else [],
                quality_grade=grade,
            )
            payloads.append(payload)

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        avg_grade = _quality_grade(avg_quality)

        # ── 9. Upsert to Qdrant ─────────────────────────────────────────
        from src.api.vector_store import QdrantVectorStore, build_collection_name
        from src.api.pipeline_models import ChunkRecord

        collection_name = build_collection_name(subscription_id)
        vector_dim = len(vectors[0])

        store = QdrantVectorStore()
        store.ensure_collection(collection_name, vector_dim)

        # Delete previous embeddings for this document first
        try:
            store.delete_document(subscription_id, profile_id, document_id)
            logger.info("Cleared old embeddings for %s in %s", document_id, collection_name)
        except Exception as exc:
            logger.debug("Old embedding cleanup skipped for %s: %s", document_id, exc)

        records = []
        for idx, (payload, vector) in enumerate(zip(payloads, vectors)):
            chunk_id = payload.get("chunk", {}).get("id", f"{document_id}_chunk_{idx}")
            records.append(ChunkRecord(
                chunk_id=chunk_id,
                dense_vector=vector.tolist(),
                sparse_vector=None,
                payload=payload,
            ))

        upserted = store.upsert_records(collection_name, records)
        logger.info("Upserted %d records to Qdrant collection %s for %s",
                     upserted, collection_name, document_id)

        # ── 10. Update MongoDB: COMPLETED ────────────────────────────────
        embedding_summary = {
            "chunk_count": len(chunk_dicts),
            "dedup_removed": dedup_removed,
            "avg_quality_grade": avg_grade,
            "avg_quality_score": round(avg_quality, 3),
            "vector_dim": vector_dim,
            "collection": collection_name,
        }
        update_stage(document_id, "embedding", status=STAGE_COMPLETED,
                     summary=embedding_summary, error=None)
        update_pipeline_status(document_id, PIPELINE_TRAINING_COMPLETED)
        append_audit_log(document_id, "EMBEDDING_COMPLETED",
                        chunk_count=len(chunk_dicts),
                        dedup_removed=dedup_removed,
                        avg_quality_grade=avg_grade)

        # ── 11. If KG was not ready, dispatch backfill ───────────────────
        if not kg_ready:
            try:
                from src.tasks.backfill import backfill_kg_refs
                backfill_kg_refs.apply_async(
                    args=[document_id, subscription_id, profile_id],
                    countdown=30,
                )
                logger.info("Dispatched KG backfill for %s", document_id)
            except Exception as exc:
                logger.warning("Failed to dispatch KG backfill for %s: %s", document_id, exc)

        logger.info("Embedding pipeline completed for document %s: %d chunks, %d deduped, grade=%s",
                     document_id, len(chunk_dicts), dedup_removed, avg_grade)

    except SoftTimeLimitExceeded:
        error = {"message": "Embedding timed out", "code": "TIMEOUT"}
        update_stage(document_id, "embedding", status=STAGE_FAILED, error=error)
        update_pipeline_status(document_id, PIPELINE_EMBEDDING_FAILED)
        append_audit_log(document_id, "EMBEDDING_FAILED", error="timeout")

    except Exception as exc:
        error = {"message": str(exc), "code": "EMBEDDING_ERROR"}
        update_stage(document_id, "embedding", status=STAGE_FAILED, error=error)
        update_pipeline_status(document_id, PIPELINE_EMBEDDING_FAILED)
        append_audit_log(document_id, "EMBEDDING_FAILED", error=str(exc))
        self.retry(exc=exc)
