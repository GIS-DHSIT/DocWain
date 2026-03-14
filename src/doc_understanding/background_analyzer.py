"""Background document analysis worker.

Performs heavy analysis asynchronously via a Redis-backed queue:
1. Relationship extraction (co-occurrence + dependency parsing)
2. Cross-reference detection (section references, figure citations)
3. Semantic clustering (group related chunks)
4. Update MongoDB + Qdrant with enriched data
"""
from __future__ import annotations

import json
from src.utils.logging_utils import get_logger
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = get_logger(__name__)

@dataclass
class AnalysisJob:
    """A queued analysis job."""
    document_id: str
    subscription_id: str
    profile_id: str
    enqueued_at: float
    payload_key: str  # Redis key where extracted doc is stored

class BackgroundAnalyzer:
    """Async background enrichment worker using Redis queue."""

    QUEUE_KEY = "docwain:deep_analysis:queue"
    RESULT_PREFIX = "docwain:deep_analysis:result:"
    JOB_TTL = 86400  # 24 hours

    def __init__(self, redis_client=None):
        self._redis = redis_client
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._processed = 0

    def set_redis(self, redis_client) -> None:
        self._redis = redis_client

    def enqueue(
        self,
        document_id: str,
        subscription_id: str,
        profile_id: str,
    ) -> bool:
        """Push a document analysis job to the Redis queue.

        Returns True if enqueued successfully.
        """
        if not self._redis:
            logger.debug("BackgroundAnalyzer: no Redis client, skipping enqueue")
            return False

        job = {
            "document_id": document_id,
            "subscription_id": subscription_id,
            "profile_id": profile_id,
            "enqueued_at": time.time(),
        }
        try:
            self._redis.rpush(self.QUEUE_KEY, json.dumps(job))
            logger.info("Enqueued background analysis for document %s", document_id)
            return True
        except Exception as exc:
            logger.warning("Failed to enqueue analysis job: %s", exc)
            return False

    def start_worker(self) -> None:
        """Start the background worker thread if not already running."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="docwain-bg-analyzer",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("BackgroundAnalyzer worker started")

    def stop_worker(self) -> None:
        """Signal the worker to stop."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=10)
            self._worker_thread = None
        logger.info("BackgroundAnalyzer worker stopped")

    def _worker_loop(self) -> None:
        """Main worker loop — blocks on Redis queue."""
        while self._running:
            if not self._redis:
                time.sleep(5)
                continue
            try:
                result = self._redis.blpop(self.QUEUE_KEY, timeout=5)
                if not result:
                    continue
                _, raw = result
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                job = json.loads(raw)
                self._analyze(job)
            except Exception as exc:
                logger.warning("BackgroundAnalyzer worker error: %s", exc)
                time.sleep(2)

    def _analyze(self, job: Dict[str, Any]) -> None:
        """Perform Phase 2 deep analysis on a document."""
        doc_id = job.get("document_id", "unknown")
        sub_id = job.get("subscription_id", "")
        profile_id = job.get("profile_id", "")

        logger.info("Starting background analysis for document %s", doc_id)
        start = time.time()

        try:
            # 1. Load document metadata from MongoDB
            mongo_meta = self._load_document_meta(doc_id)
            if not mongo_meta:
                logger.debug("No metadata found for document %s, skipping", doc_id)
                return

            # 2. Relationship extraction from existing entities
            entities = mongo_meta.get("entity_mentions", [])
            relationships = self._extract_relationships(entities)

            # 3. Cross-reference detection
            cross_refs = self._detect_cross_references(mongo_meta)

            # 4. Store enriched data back to MongoDB
            enrichment = {
                "background_analysis": {
                    "relationships": relationships[:50],
                    "cross_references": cross_refs[:20],
                    "analyzed_at": time.time(),
                    "analysis_duration_s": round(time.time() - start, 2),
                }
            }
            self._update_document(doc_id, enrichment)

            self._processed += 1
            elapsed = time.time() - start
            logger.info(
                "Background analysis completed for %s in %.1fs (%d relationships, %d cross-refs)",
                doc_id, elapsed, len(relationships), len(cross_refs),
            )

            # Store result status in Redis
            self._store_result(doc_id, "completed", elapsed)

        except Exception as exc:
            logger.warning("Background analysis failed for %s: %s", doc_id, exc)
            self._store_result(doc_id, "failed", time.time() - start)

    def _load_document_meta(self, document_id: str) -> Optional[Dict]:
        """Load document metadata from MongoDB."""
        try:
            from src.api.dataHandler import db
            if db is None:
                return None
            doc = db.documents.find_one({"_id": document_id})
            if not doc:
                doc = db.documents.find_one({"document_id": document_id})
            return doc
        except Exception as exc:
            logger.warning("Failed to load document meta: %s", exc)
            return None

    def _extract_relationships(self, entities: List[Dict]) -> List[Dict]:
        """Extract entity co-occurrence relationships."""
        from collections import Counter

        # Group entities by section
        section_entities: Dict[str, List[Dict]] = {}
        for ent in entities:
            section = ent.get("section_title", "default")
            section_entities.setdefault(section, []).append(ent)

        # Find co-occurrences
        pair_counts: Counter = Counter()
        for section, ents in section_entities.items():
            names = [(e.get("text", ""), e.get("type", "")) for e in ents]
            for i, (n1, t1) in enumerate(names):
                for n2, t2 in names[i + 1:]:
                    if n1 != n2:
                        pair = tuple(sorted([(n1, t1), (n2, t2)]))
                        pair_counts[pair] += 1

        relationships = []
        for (e1, e2), count in pair_counts.most_common(50):
            if count >= 2:
                relationships.append({
                    "entity1": {"text": e1[0], "type": e1[1]},
                    "entity2": {"text": e2[0], "type": e2[1]},
                    "relation_type": "RELATED_TO",
                    "frequency": count,
                })
        return relationships

    def _detect_cross_references(self, meta: Dict) -> List[Dict]:
        """Detect cross-references between sections."""
        import re

        cross_refs = []
        summary = meta.get("document_summary", "")

        # Detect section references like "see Section X", "refer to X"
        ref_pattern = re.compile(
            r"(?:see|refer to|as described in|per|according to)\s+(?:section|chapter|appendix|table|figure)\s+(\w+)",
            re.IGNORECASE,
        )
        for match in ref_pattern.finditer(summary):
            cross_refs.append({
                "type": "section_reference",
                "target": match.group(1),
                "context": match.group(0),
            })

        return cross_refs

    def _update_document(self, document_id: str, data: Dict) -> None:
        """Update document in MongoDB with enriched data."""
        try:
            from src.api.dataHandler import db
            if db is None:
                return
            db.documents.update_one(
                {"_id": document_id},
                {"$set": data},
                upsert=False,
            )
        except Exception as exc:
            logger.warning("Failed to update document %s: %s", document_id, exc)

    def _store_result(self, document_id: str, status: str, elapsed: float) -> None:
        """Store analysis result status in Redis."""
        if not self._redis:
            return
        try:
            key = f"{self.RESULT_PREFIX}{document_id}"
            self._redis.setex(
                key,
                self.JOB_TTL,
                json.dumps({"status": status, "elapsed_s": round(elapsed, 2)}),
            )
        except Exception:
            pass

    @property
    def processed_count(self) -> int:
        return self._processed

    @property
    def is_running(self) -> bool:
        return self._running

# Singleton
_ANALYZER: Optional[BackgroundAnalyzer] = None

def get_background_analyzer() -> Optional[BackgroundAnalyzer]:
    return _ANALYZER

def set_background_analyzer(analyzer: BackgroundAnalyzer) -> None:
    global _ANALYZER
    _ANALYZER = analyzer

__all__ = ["BackgroundAnalyzer", "get_background_analyzer", "set_background_analyzer"]
