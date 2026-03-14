from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from src.intelligence.redis_intel_cache import RedisIntelCache
from src.kg.entity_extractor import EntityExtractor, ExtractedEntity, normalize_entity_name
from src.utils.payload_utils import token_count

logger = get_logger(__name__)

DOMAIN_KEYWORDS = {
    "resume": [
        "resume",
        "curriculum vitae",
        "cv",
        "experience",
        "employment",
        "work history",
        "education",
        "skills",
        "projects",
        "certification",
    ],
    "tax": [
        "tax",
        "irs",
        "w-2",
        "w2",
        "1099",
        "schedule c",
        "deduction",
        "refund",
        "filing",
        "return",
        "taxable",
    ],
    "invoice": [
        "invoice",
        "bill to",
        "amount due",
        "subtotal",
        "total",
        "purchase order",
        "po",
        "due date",
        "balance due",
        "unit price",
    ],
    "legal": [
        "agreement",
        "contract",
        "clause",
        "indemnify",
        "hereby",
        "witnesseth",
        "liability",
        "governing law",
        "arbitration",
    ],
    "medical": [
        "patient",
        "diagnosis",
        "treatment",
        "medical",
        "clinical",
        "prescription",
        "symptom",
        "history of present illness",
    ],
}

def infer_domain(text: str, *, doc_type: Optional[str] = None, source_name: Optional[str] = None) -> str:
    try:
        from src.intelligence.domain_classifier import classify_domain

        classification = classify_domain(
            text,
            metadata={"doc_type": doc_type, "source_name": source_name},
        )
        return classification.domain
    except Exception:  # noqa: BLE001
        lowered = (text or "").lower()
        scores = {domain: 0 for domain in DOMAIN_KEYWORDS}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in lowered:
                    scores[domain] += 1
        if source_name:
            lower_name = source_name.lower()
            for domain, keywords in DOMAIN_KEYWORDS.items():
                if any(keyword.replace(" ", "") in lower_name.replace(" ", "") for keyword in keywords[:3]):
                    scores[domain] += 1
        if doc_type:
            doc_type_lower = str(doc_type).lower()
            if doc_type_lower in {"invoice", "tax", "resume", "cv"}:
                scores[doc_type_lower if doc_type_lower != "cv" else "resume"] += 2
            if "medical" in doc_type_lower:
                scores["medical"] += 2
            if "contract" in doc_type_lower or "legal" in doc_type_lower:
                scores["legal"] += 2

        top_domain = max(scores.items(), key=lambda kv: kv[1])
        if top_domain[1] == 0:
            return "unknown"
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[1] >= max(1, sorted_scores[0] - 1):
            return "mixed"
        return top_domain[0]

def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = []
    for piece in pieces:
        cleaned = piece.strip()
        if len(cleaned) < 8:
            continue
        sentences.append(cleaned)
    return sentences

def summarize_text_bullets(text: str, max_bullets: int = 6) -> List[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    bullets: List[str] = []
    seen = set()
    for sentence in sentences:
        normalized = sentence.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        bullets.append(sentence)
        if len(bullets) >= max_bullets:
            break
    return bullets

@dataclass
class DocumentQuality:
    extracted_chars: int
    token_count: int
    coverage: float
    ocr_used: bool
    valid_chunks: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "extracted_chars": self.extracted_chars,
            "token_count": self.token_count,
            "coverage": self.coverage,
            "ocr_used": self.ocr_used,
            "valid_chunks": self.valid_chunks,
        }

class DomainIndexer:
    def __init__(
        self,
        *,
        redis_cache: Optional[RedisIntelCache] = None,
        entity_extractor: Optional[EntityExtractor] = None,
    ) -> None:
        self.redis_cache = redis_cache
        self.entity_extractor = entity_extractor or EntityExtractor()

    def index_document(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        document_id: str,
        source_name: str,
        doc_type: Optional[str],
        full_text: str,
        chunk_texts: Iterable[str],
        chunk_metadata: Iterable[Dict[str, Any]],
        ocr_used: bool = False,
        profile_name: Optional[str] = None,
        doc_version_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        chunk_meta_list = list(chunk_metadata or [])
        canonical_text = full_text or " ".join([t for t in chunk_texts if t])
        domain = infer_domain(canonical_text, doc_type=doc_type, source_name=source_name)
        summary_bullets = summarize_text_bullets(canonical_text, max_bullets=6)
        doc_summary_short = summary_bullets[0] if summary_bullets else ""

        chunks = [t for t in chunk_texts if t]
        valid_chunks = len(chunks)
        total_chars = sum(len(t) for t in chunks)
        total_tokens = token_count(canonical_text)
        coverage = float(valid_chunks) / max(len(chunk_meta_list) or valid_chunks or 1, 1)

        quality = DocumentQuality(
            extracted_chars=total_chars,
            token_count=total_tokens,
            coverage=coverage,
            ocr_used=ocr_used,
            valid_chunks=valid_chunks,
        )

        entities = self._extract_entities(chunks, document_id=document_id)
        top_entities = entities[:8]

        updated_at = time.time()
        doc_entry = {
            "document_id": str(document_id),
            "source_name": source_name,
            "doc_domain": domain,
            "doc_type": doc_type or "",
            "quality": quality.to_dict(),
            "doc_summary_short": doc_summary_short,
            "top_entities": top_entities,
            "doc_version_hash": doc_version_hash,
            "updated_at": updated_at,
        }

        if self.redis_cache:
            self._update_catalog(
                subscription_id=subscription_id,
                profile_id=profile_id,
                profile_name=profile_name,
                doc_entry=doc_entry,
            )
            if summary_bullets:
                docsum_key = self.redis_cache.docsum_key(document_id)
                self.redis_cache.set_json(
                    docsum_key,
                    {
                        "document_id": str(document_id),
                        "summary_bullets": summary_bullets,
                        "extracted_at": updated_at,
                    },
                    ttl_seconds=self.redis_cache.summary_ttl,
                )
            self._update_entities(subscription_id, profile_id, entities, document_id=document_id)

        return {
            "doc_domain": domain,
            "doc_summary_short": doc_summary_short,
            "summary_bullets": summary_bullets,
            "quality": quality.to_dict(),
            "top_entities": top_entities,
        }

    def _extract_entities(self, texts: Iterable[str], *, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        counts: Dict[str, int] = {}
        entity_map: Dict[str, ExtractedEntity] = {}
        for text in texts:
            for ent in self.entity_extractor.extract_with_metadata(text):
                counts[ent.entity_id] = counts.get(ent.entity_id, 0) + 1
                entity_map[ent.entity_id] = ent
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        results: List[Dict[str, Any]] = []
        for entity_id, count in ranked:
            ent = entity_map.get(entity_id)
            if not ent:
                continue
            results.append(
                {
                    "entity_id": ent.entity_id,
                    "type": ent.type,
                    "value": ent.name,
                    "aliases": [normalize_entity_name(ent.name)],
                    "salience": float(count),
                    "document_ids": [str(document_id)] if document_id else [],
                }
            )
        return results

    def _update_catalog(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        profile_name: Optional[str],
        doc_entry: Dict[str, Any],
    ) -> None:
        cache = self.redis_cache
        if not cache:
            return
        key = cache.catalog_key(subscription_id, profile_id)
        payload = cache.get_json(key) or {
            "profile_id": str(profile_id),
            "profile_name": profile_name,
            "documents": [],
            "dominant_domains": {},
            "updated_at": time.time(),
        }

        documents = payload.get("documents") or []
        updated = False
        for idx, entry in enumerate(documents):
            if entry.get("document_id") == doc_entry.get("document_id"):
                documents[idx] = doc_entry
                updated = True
                break
        if not updated:
            documents.append(doc_entry)
        payload["documents"] = documents

        domain_counts: Dict[str, int] = {}
        for doc in documents:
            domain = doc.get("doc_domain") or "unknown"
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        payload["dominant_domains"] = domain_counts
        payload["updated_at"] = time.time()
        if profile_name and not payload.get("profile_name"):
            payload["profile_name"] = profile_name
        cache.set_json(key, payload, ttl_seconds=cache.catalog_ttl)

        if documents:
            bullets = [doc.get("doc_summary_short", "") for doc in documents if doc.get("doc_summary_short")]
            if bullets:
                profilesum_key = cache.profilesum_key(subscription_id, profile_id)
                cache.set_json(
                    profilesum_key,
                    {
                        "profile_id": str(profile_id),
                        "summary_bullets": bullets[:8],
                        "extracted_at": time.time(),
                    },
                    ttl_seconds=cache.summary_ttl,
                )

    def _update_entities(
        self,
        subscription_id: str,
        profile_id: str,
        entities: List[Dict[str, Any]],
        *,
        document_id: Optional[str] = None,
    ) -> None:
        cache = self.redis_cache
        if not cache or not entities:
            return
        key = cache.entities_key(subscription_id, profile_id)
        payload = cache.get_json(key) or {"entities": [], "updated_at": time.time()}
        entity_map = {ent.get("entity_id"): ent for ent in (payload.get("entities") or []) if ent.get("entity_id")}
        for ent in entities:
            entity_id = ent.get("entity_id")
            if not entity_id:
                continue
            existing = entity_map.get(entity_id)
            if existing:
                existing["salience"] = float(existing.get("salience", 0.0)) + float(ent.get("salience", 0.0))
                if document_id:
                    doc_ids = set(existing.get("document_ids") or [])
                    doc_ids.add(str(document_id))
                    existing["document_ids"] = sorted(doc_ids)
                continue
            if document_id and not ent.get("document_ids"):
                ent["document_ids"] = [str(document_id)]
            entity_map[entity_id] = ent
        merged = list(entity_map.values())
        merged.sort(key=lambda e: float(e.get("salience", 0.0)), reverse=True)
        payload["entities"] = merged[:100]
        payload["updated_at"] = time.time()
        cache.set_json(key, payload, ttl_seconds=cache.entities_ttl)

    def update_entities_only(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        document_id: str,
        chunk_texts: Iterable[str],
    ) -> None:
        if not self.redis_cache:
            return
        entities = self._extract_entities(chunk_texts, document_id=document_id)
        self._update_entities(subscription_id, profile_id, entities, document_id=document_id)

__all__ = ["DomainIndexer", "infer_domain", "summarize_text_bullets", "DocumentQuality"]
