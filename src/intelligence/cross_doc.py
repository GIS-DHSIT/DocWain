"""Cross-document intelligence for the Document Intelligence Graph.

Runs as a post-ingestion step (non-blocking) to:
1. Detect near-duplicate documents via doc-vector cosine similarity
2. Detect version/amendment chains via filename + summary patterns
3. Compute entity overlap between documents in the same profile
4. Persist results to MongoDB and Neo4j (SIMILAR_TO, SHARED_ENTITY edges)
"""
from __future__ import annotations

import hashlib
from src.utils.logging_utils import get_logger
import re
from typing import Any, Dict, List, Optional, Tuple

logger = get_logger(__name__)

# ── Version/amendment patterns ──
_VERSION_RE = re.compile(
    r"(?:v(?:ersion)?|rev(?:ision)?|draft|amendment|update|ed(?:ition)?)"
    r"[\s._-]*(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)
_DATED_VERSION_RE = re.compile(
    r"(?:20\d{2}[-_]\d{2}[-_]\d{2}|20\d{6})",
)
_FILENAME_STEM_RE = re.compile(
    r"^(.*?)[\s._-]*(?:v\d|rev\d|draft|update|final|latest|\(\d+\)|copy)",
    re.IGNORECASE,
)

def _normalize_filename_stem(name: str) -> str:
    """Extract a canonical stem from a filename for version chain matching."""
    if not name:
        return ""
    stem = re.sub(r"\.[^.]+$", "", name)  # strip extension
    m = _FILENAME_STEM_RE.match(stem)
    return (m.group(1).strip().lower() if m else stem.strip().lower())

def detect_version_chain(
    doc_name: str,
    doc_summary: str,
    existing_docs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Find documents that are likely versions/amendments of the same base document.

    Returns list of {document_id, doc_name, relationship: 'version_of'|'amendment_of'}.
    """
    matches: List[Dict[str, Any]] = []
    if not doc_name:
        return matches

    stem = _normalize_filename_stem(doc_name)
    if not stem or len(stem) < 3:
        return matches

    for doc in existing_docs:
        other_name = doc.get("doc_name") or doc.get("source_name") or ""
        other_stem = _normalize_filename_stem(other_name)
        if not other_stem or len(other_stem) < 3:
            continue
        # Same stem = likely version chain
        if stem == other_stem and other_name != doc_name:
            rel = "amendment_of" if "amendment" in doc_name.lower() else "version_of"
            matches.append({
                "document_id": doc.get("document_id") or doc.get("_id"),
                "doc_name": other_name,
                "relationship": rel,
            })

    return matches

def compute_entity_overlap(
    doc_entities: List[str],
    other_doc_entities: Dict[str, List[str]],
) -> List[Tuple[str, float, List[str]]]:
    """Compute Jaccard overlap between this doc's entities and other docs.

    Args:
        doc_entities: normalized entity names for current document
        other_doc_entities: {document_id: [normalized entity names]}

    Returns: list of (document_id, overlap_score, shared_entities) where overlap >= 0.15
    """
    results: List[Tuple[str, float, List[str]]] = []
    if not doc_entities:
        return results

    doc_set = set(doc_entities)
    for other_id, other_entities in other_doc_entities.items():
        other_set = set(other_entities)
        intersection = doc_set & other_set
        if not intersection:
            continue
        union = doc_set | other_set
        jaccard = len(intersection) / len(union) if union else 0.0
        if jaccard >= 0.15:
            results.append((other_id, jaccard, sorted(intersection)))

    return results

def run_cross_document_intelligence(
    *,
    subscription_id: str,
    profile_id: str,
    document_id: str,
    doc_name: str,
    doc_summary: str = "",
    doc_entities: Optional[List[str]] = None,
    doc_vector: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Run cross-document intelligence for a newly ingested document.

    This is a post-ingestion step that queries existing docs in the same profile
    to find duplicates, versions, and entity overlaps.

    Returns dict with keys: duplicates, version_chain, entity_overlaps, cluster_id
    """
    result: Dict[str, Any] = {
        "duplicates": [],
        "version_chain": [],
        "entity_overlaps": [],
        "cluster_id": None,
    }

    try:
        from src.api.document_status import get_document_record
        from src.api.config import Config

        # Get all other documents in same profile from MongoDB
        mongo_db = None
        try:
            from src.api.dataHandler import db
            mongo_db = db
        except Exception:
            pass

        if not mongo_db:
            logger.debug("cross_doc: MongoDB unavailable, skipping")
            return result

        docs_coll = mongo_db.get_collection("documents")
        profile_docs = list(docs_coll.find(
            {
                "subscription_id": subscription_id,
                "profile_id": profile_id,
                "document_id": {"$ne": document_id},
                "status": {"$in": ["TRAINING_COMPLETED", "training_completed"]},
            },
            {
                "document_id": 1,
                "doc_name": 1,
                "source_name": 1,
                "key_entities": 1,
                "document_summary": 1,
                "_id": 0,
            },
        ))

        if not profile_docs:
            logger.debug("cross_doc: no other docs in profile, skipping")
            return result

        # ── Version chain detection ──
        version_chain = detect_version_chain(doc_name, doc_summary, profile_docs)
        result["version_chain"] = version_chain
        if version_chain:
            logger.info(
                "cross_doc: %s is related to %d other version(s)",
                doc_name, len(version_chain),
            )

        # ── Entity overlap ──
        doc_entities = doc_entities or []
        if doc_entities:
            other_doc_entities: Dict[str, List[str]] = {}
            for pdoc in profile_docs:
                other_id = pdoc.get("document_id")
                other_ents = pdoc.get("key_entities") or []
                if other_id and other_ents:
                    normalized = []
                    for e in other_ents:
                        if isinstance(e, dict):
                            normalized.append(str(e.get("text", "")).strip().lower())
                        elif isinstance(e, str):
                            normalized.append(e.strip().lower())
                    other_doc_entities[other_id] = [n for n in normalized if n]

            overlaps = compute_entity_overlap(
                [e.strip().lower() if isinstance(e, str) else str(e.get("text", "")).strip().lower()
                 for e in doc_entities if e],
                other_doc_entities,
            )
            result["entity_overlaps"] = [
                {"document_id": did, "overlap_score": score, "shared_entities": shared[:20]}
                for did, score, shared in overlaps
            ]
            if overlaps:
                logger.info(
                    "cross_doc: %s has entity overlap with %d doc(s)",
                    doc_name, len(overlaps),
                )

        # ── Doc-vector cosine similarity (duplicate detection) ──
        if doc_vector and len(doc_vector) > 10:
            try:
                from src.api.dataHandler import get_qdrant_client
                from src.api.vector_store import build_collection_name, build_qdrant_filter
                from qdrant_client.models import Filter, FieldCondition, MatchValue

                client = get_qdrant_client()
                collection_name = build_collection_name(subscription_id)

                # Search for similar doc-resolution vectors in same profile
                search_filter = Filter(must=[
                    FieldCondition(key="profile_id", match=MatchValue(value=profile_id)),
                    FieldCondition(key="resolution", match=MatchValue(value="doc")),
                ])

                hits = client.search(
                    collection_name=collection_name,
                    query_vector=doc_vector,
                    query_filter=search_filter,
                    limit=10,
                    score_threshold=0.85,
                )

                for hit in hits:
                    payload = hit.payload or {}
                    other_doc_id = payload.get("document_id", "")
                    if other_doc_id == document_id:
                        continue
                    similarity = float(hit.score)
                    is_duplicate = similarity >= 0.95
                    result["duplicates"].append({
                        "document_id": other_doc_id,
                        "doc_name": payload.get("source_name", ""),
                        "similarity": round(similarity, 4),
                        "is_duplicate": is_duplicate,
                    })

                if result["duplicates"]:
                    logger.info(
                        "cross_doc: %s has %d similar doc(s) (duplicates=%d)",
                        doc_name,
                        len(result["duplicates"]),
                        sum(1 for d in result["duplicates"] if d["is_duplicate"]),
                    )
            except Exception as sim_exc:
                logger.debug("cross_doc: similarity search failed: %s", sim_exc)

        # ── Persist results to MongoDB ──
        cross_doc_data = {
            k: v for k, v in result.items() if v
        }
        if cross_doc_data:
            try:
                from src.api.document_status import update_document_fields
                update_document_fields(document_id, {"cross_doc_intelligence": cross_doc_data})
            except Exception as persist_exc:
                logger.debug("cross_doc: MongoDB persist failed: %s", persist_exc)

        # ── Write SIMILAR_TO edges to Neo4j ──
        if result["duplicates"]:
            try:
                from src.kg.neo4j_store import Neo4jStore
                store = Neo4jStore()
                for dup in result["duplicates"]:
                    store.create_document_similarity(
                        doc1_id=document_id,
                        doc2_id=dup["document_id"],
                        similarity=dup["similarity"],
                    )
            except Exception as kg_exc:
                logger.debug("cross_doc: Neo4j similarity write failed: %s", kg_exc)

    except Exception as exc:
        logger.debug("cross_doc: intelligence failed for %s: %s", document_id, exc)

    return result

__all__ = [
    "detect_version_chain",
    "compute_entity_overlap",
    "run_cross_document_intelligence",
]
