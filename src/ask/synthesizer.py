from __future__ import annotations

from typing import Any, Dict, List

from src.embed.entity_extractor import EntityExtractor
from .models import EvidenceChunk


class EvidenceSynthesizer:
    def __init__(self) -> None:
        self.extractor = EntityExtractor()

    def synthesize(self, *, evidence: List[EvidenceChunk]) -> Dict[str, Any]:
        docs: Dict[str, Dict[str, Any]] = {}
        for chunk in evidence:
            file_name = chunk.file_name or "document"
            doc = docs.setdefault(
                file_name,
                {
                    "file_name": file_name,
                    "doc_domain_hint": chunk.metadata.get("doc_domain") or chunk.metadata.get("document_type") or "generic",
                    "objects": [],
                    "_entities": {},
                    "_evidence": {},
                },
            )
            entities = self.extractor.extract(chunk.text)
            for ent in entities:
                ent_list = doc["_entities"].setdefault(ent.entity_type, [])
                if ent.surface_form not in ent_list:
                    ent_list.append(ent.surface_form)
                ev_list = doc["_evidence"].setdefault(ent.entity_type, [])
                ev_list.append(_evidence_pointer(chunk))

        documents = []
        for doc in docs.values():
            entities = doc.pop("_entities")
            evidence_map = doc.pop("_evidence")
            fields: Dict[str, Any] = {}
            gaps: List[str] = []

            names = _normalize_list(entities.get("person", []))
            if names:
                fields["names"] = names
            else:
                gaps.append("names not explicitly stated")

            skills = _normalize_list(entities.get("skill", []))
            if skills:
                fields["skills"] = skills
            else:
                gaps.append("skills not explicitly stated")

            orgs = _normalize_list(entities.get("organization", []))
            if orgs:
                fields["organizations"] = orgs

            dates = _normalize_list(entities.get("date", []))
            if dates:
                fields["dates"] = dates

            object_type = _object_type(doc.get("doc_domain_hint"))
            doc["objects"].append(
                {
                    "object_type": object_type,
                    "fields": fields,
                    "evidence_map": {
                        key: _dedupe_evidence(value)
                        for key, value in evidence_map.items()
                        if key in {"person", "skill", "organization", "date"}
                    },
                    "gaps": gaps,
                }
            )
            documents.append(doc)

        return {
            "bundle_type": "DocWainEvidenceBundle",
            "documents": documents,
            "cross_doc": {"comparables": [], "notes": [], "gaps": []},
        }


def _normalize_list(items: List[str]) -> List[str]:
    cleaned = []
    seen = set()
    for item in items:
        if not item:
            continue
        normalized = " ".join(str(item).split())
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized)
    return cleaned


def _object_type(domain_hint: str) -> str:
    hint = (domain_hint or "").lower()
    if "resume" in hint or "profile" in hint:
        return "ProfileLike"
    if "invoice" in hint:
        return "InvoiceLike"
    if "medical" in hint:
        return "MedicalLike"
    return "Generic"


def _evidence_pointer(chunk: EvidenceChunk) -> Dict[str, Any]:
    return {
        "file_name": chunk.file_name,
        "page": chunk.page,
        "snippet": chunk.snippet,
        "snippet_sha": chunk.snippet_sha,
    }


def _dedupe_evidence(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        if not item:
            continue
        key = (item.get("file_name"), item.get("page"), item.get("snippet_sha"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


__all__ = ["EvidenceSynthesizer"]
