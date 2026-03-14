"""
Isolated MongoDB storage layer for Teams document pipeline.

Uses a dedicated `teams_documents` collection, completely separate
from the main app's `documents` collection.
"""

from datetime import datetime, timezone
from typing import Dict, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Status constants ────────────────────────────────────────────────
STATUS_EXTRACTING = "EXTRACTING"
STATUS_EXTRACTION_COMPLETED = "EXTRACTION_COMPLETED"
STATUS_SCREENING = "SCREENING"
STATUS_SCREENING_COMPLETED = "SCREENING_COMPLETED"
STATUS_SCREENING_PENDING_CONSENT = "SCREENING_PENDING_CONSENT"
STATUS_SCREENING_REJECTED = "SCREENING_REJECTED"
STATUS_EMBEDDING = "EMBEDDING"
STATUS_EMBEDDING_COMPLETED = "EMBEDDING_COMPLETED"
STATUS_EMBEDDING_FAILED = "EMBEDDING_FAILED"

CONSENT_REQUIRED_RISKS = {"MEDIUM", "HIGH", "CRITICAL"}

TEAMS_DOCUMENTS_COLLECTION = "teams_documents"


def _get_teams_collection():
    """Return the teams_documents MongoDB collection, or None if unavailable."""
    try:
        from src.api.dataHandler import db
        return db[TEAMS_DOCUMENTS_COLLECTION]
    except Exception as exc:
        logger.warning("teams_documents collection unavailable: %s", exc)
        return None


class TeamsDocumentStorage:
    """CRUD wrapper around the ``teams_documents`` MongoDB collection."""

    def __init__(self):
        self._collection = _get_teams_collection()

    # ── writes ──────────────────────────────────────────────────────

    def create_document(
        self,
        document_id: str,
        filename: str,
        subscription_id: str,
        profile_id: str,
        content_type: str = "",
        content_size: int = 0,
    ) -> str:
        """Insert a new teams document record. Returns the document_id."""
        if self._collection is None:
            logger.error("Cannot create document – collection unavailable")
            return document_id

        now = datetime.now(timezone.utc)
        record = {
            "document_id": document_id,
            "filename": filename,
            "subscription_id": subscription_id,
            "profile_id": profile_id,
            "content_type": content_type,
            "content_size": content_size,
            "status": STATUS_EXTRACTING,
            "source": "teams",
            "extraction": {"status": "IN_PROGRESS", "started_at": now},
            "screening": {"status": "PENDING"},
            "embedding": {"status": "PENDING"},
            "created_at": now,
            "updated_at": now,
        }
        self._collection.insert_one(record)
        logger.info("Created teams document %s (file=%s)", document_id, filename)
        return document_id

    def update_status(self, document_id: str, status: str, **extra_fields) -> None:
        """Set the document status and any additional fields."""
        if self._collection is None:
            logger.error("Cannot update status – collection unavailable")
            return

        update: Dict = {
            "$set": {
                "status": status,
                "updated_at": datetime.now(timezone.utc),
                **extra_fields,
            }
        }
        self._collection.update_one({"document_id": document_id}, update)
        logger.info("Document %s status → %s", document_id, status)

    def store_extraction_result(
        self,
        document_id: str,
        doc_type: str,
        summary: str,
        key_entities: list,
        key_facts: list,
        intent_tags: list,
        confidence: float = 0.0,
    ) -> None:
        """Persist document-intelligence extraction results."""
        if self._collection is None:
            logger.error("Cannot store extraction – collection unavailable")
            return

        self._collection.update_one(
            {"document_id": document_id},
            {
                "$set": {
                    "status": STATUS_EXTRACTION_COMPLETED,
                    "extraction.doc_type": doc_type,
                    "extraction.summary": summary,
                    "extraction.key_entities": key_entities,
                    "extraction.key_facts": key_facts,
                    "extraction.intent_tags": intent_tags,
                    "extraction.confidence": confidence,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )
        logger.info("Extraction stored for %s (type=%s)", document_id, doc_type)

    def store_screening_result(
        self,
        document_id: str,
        risk_level: str,
        risk_score: float,
        findings: list,
    ) -> None:
        """Store screening results; set consent-pending if risk requires it."""
        if self._collection is None:
            logger.error("Cannot store screening – collection unavailable")
            return

        needs_consent = risk_level.upper() in CONSENT_REQUIRED_RISKS
        status = (
            STATUS_SCREENING_PENDING_CONSENT if needs_consent
            else STATUS_SCREENING_COMPLETED
        )

        self._collection.update_one(
            {"document_id": document_id},
            {
                "$set": {
                    "status": status,
                    "screening.risk_level": risk_level,
                    "screening.risk_score": risk_score,
                    "screening.findings": findings,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )
        logger.info(
            "Screening stored for %s (risk=%s, consent_needed=%s)",
            document_id, risk_level, needs_consent,
        )

    def consent_proceed(self, document_id: str) -> None:
        """User consented – allow embedding to proceed."""
        self.update_status(document_id, STATUS_SCREENING_COMPLETED)

    def consent_reject(self, document_id: str) -> None:
        """User rejected the document after screening."""
        self.update_status(document_id, STATUS_SCREENING_REJECTED)

    def mark_embedding_started(self, document_id: str) -> None:
        self.update_status(document_id, STATUS_EMBEDDING)

    def mark_embedding_completed(
        self, document_id: str, chunks_count: int = 0, quality_grade: str = ""
    ) -> None:
        self.update_status(
            document_id,
            STATUS_EMBEDDING_COMPLETED,
            **{
                "embedding.chunks_count": chunks_count,
                "embedding.quality_grade": quality_grade,
            },
        )

    def mark_embedding_failed(self, document_id: str, error: str = "") -> None:
        self.update_status(
            document_id,
            STATUS_EMBEDDING_FAILED,
            **{"embedding.error": error},
        )

    # ── reads ───────────────────────────────────────────────────────

    def get_document(self, document_id: str) -> Optional[Dict]:
        """Retrieve a single teams document by its document_id."""
        if self._collection is None:
            logger.error("Cannot get document – collection unavailable")
            return None

        return self._collection.find_one({"document_id": document_id})
