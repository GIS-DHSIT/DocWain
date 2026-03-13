from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from src.utils.logging_utils import get_logger

from src.retrieval.evidence_extractors import (
    EvidenceItem,
    extract_contacts,
    extract_dates,
    extract_entities,
    extract_identifiers,
    extract_sections,
    extract_tables,
)

logger = get_logger(__name__)


@dataclass
class DocumentEvidence:
    document_id: str
    source_name: Optional[str]
    contacts: Dict[str, List[EvidenceItem]]
    dates: List[EvidenceItem]
    identifiers: List[EvidenceItem]
    entities: List[EvidenceItem]
    sections: List[EvidenceItem]
    tables: List[EvidenceItem]


@dataclass
class EvidenceEdge:
    source_document_id: str
    target_document_id: str
    shared_identifiers: List[str] = field(default_factory=list)
    shared_entities: List[str] = field(default_factory=list)


@dataclass
class ProfileEvidenceGraph:
    documents: Dict[str, DocumentEvidence]
    edges: List[EvidenceEdge]


def build_document_evidence(document_id: str, chunks: Iterable[Dict[str, str]]) -> DocumentEvidence:
    logger.debug("build_document_evidence: document_id=%s", document_id)
    chunk_list = list(chunks)
    source_name = chunk_list[0].get("source_name") if chunk_list else None
    contacts = extract_contacts(chunk_list)
    dates = extract_dates(chunk_list)
    identifiers = extract_identifiers(chunk_list)
    entities = extract_entities(chunk_list)
    sections = extract_sections(chunk_list)
    tables = extract_tables(chunk_list)
    result = DocumentEvidence(
        document_id=document_id,
        source_name=source_name,
        contacts=contacts,
        dates=dates,
        identifiers=identifiers,
        entities=entities,
        sections=sections,
        tables=tables,
    )
    logger.debug("build_document_evidence: document_id=%s, dates=%d, identifiers=%d, entities=%d, sections=%d, tables=%d",
                 document_id, len(dates), len(identifiers), len(entities), len(sections), len(tables))
    return result


def build_profile_evidence_graph(corpora: Dict[str, List[Dict[str, str]]]) -> ProfileEvidenceGraph:
    logger.debug("build_profile_evidence_graph: documents=%d", len(corpora))
    documents: Dict[str, DocumentEvidence] = {}
    for doc_id, chunks in corpora.items():
        documents[doc_id] = build_document_evidence(doc_id, chunks)

    edges: List[EvidenceEdge] = []
    doc_ids = list(documents.keys())
    for idx, doc_id in enumerate(doc_ids):
        for other_id in doc_ids[idx + 1 :]:
            doc = documents[doc_id]
            other = documents[other_id]
            shared_identifiers = _shared_values(doc.identifiers, other.identifiers)
            shared_entities = _shared_values(doc.entities, other.entities)
            if shared_identifiers or shared_entities:
                edges.append(
                    EvidenceEdge(
                        source_document_id=doc_id,
                        target_document_id=other_id,
                        shared_identifiers=shared_identifiers,
                        shared_entities=shared_entities,
                    )
                )
    logger.debug("build_profile_evidence_graph: returning %d documents, %d edges", len(documents), len(edges))
    return ProfileEvidenceGraph(documents=documents, edges=edges)


def _shared_values(left: List[EvidenceItem], right: List[EvidenceItem]) -> List[str]:
    left_vals = {item.value.lower() for item in left}
    right_vals = {item.value.lower() for item in right}
    shared = left_vals.intersection(right_vals)
    return sorted(shared)


__all__ = [
    "DocumentEvidence",
    "EvidenceEdge",
    "ProfileEvidenceGraph",
    "build_document_evidence",
    "build_profile_evidence_graph",
]
