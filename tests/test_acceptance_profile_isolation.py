from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from qdrant_client.models import Filter, MatchAny, MatchValue

from src.intelligence.ask_pipeline import answer_with_section_intelligence
from src.intelligence.facts_store import FactsStore
from src.intelligence.retrieval import run_intelligent_pipeline
from src.intelligence.section_intelligence_builder import SectionIntelligenceBuilder
from src.services.retrieval.hybrid_retriever import HybridRetriever


@dataclass
class FakePoint:
    id: str
    score: float
    payload: Dict[str, Any]


class FakeQueryResult:
    def __init__(self, points: List[FakePoint]):
        self.points = points


class FakeQdrant:
    def __init__(self, points: List[FakePoint]):
        self._points = points

    def query_points(self, **kwargs):  # noqa: ANN003
        query_filter = kwargs.get("query_filter")
        points = [pt for pt in self._points if _match_filter(pt.payload or {}, query_filter)]
        return FakeQueryResult(points)

    def scroll(self, **kwargs):  # noqa: ANN003
        scroll_filter = kwargs.get("scroll_filter")
        points = [pt for pt in self._points if _match_filter(pt.payload or {}, scroll_filter)]
        return points, None


class FakeEmbedder:
    def encode(self, text: str, convert_to_numpy: bool = True, normalize_embeddings: bool = True):  # noqa: ANN001
        _ = (text, convert_to_numpy, normalize_embeddings)
        return [1.0, 1.0, 1.0, 1.0]


class FakeRedis:
    def __init__(self):
        self.store: Dict[str, Any] = {}

    def setex(self, key, ttl, value):  # noqa: ANN001, ARG002
        self.store[key] = value

    def get(self, key):  # noqa: ANN001
        return self.store.get(key)


class FakeCollection:
    def __init__(self):
        self.rows = []

    def update_one(self, filter_doc, update_doc, upsert=False):  # noqa: ANN001, ANN002
        for row in self.rows:
            if all(row.get(k) == v for k, v in filter_doc.items()):
                row.update(update_doc.get("$set", {}))
                return
        if upsert:
            record = dict(filter_doc)
            record.update(update_doc.get("$set", {}))
            self.rows.append(record)

    def find(self, filter_doc):  # noqa: ANN001
        return [row for row in self.rows if all(row.get(k) == v for k, v in filter_doc.items())]

    def find_one(self, filter_doc):  # noqa: ANN001
        for row in self.rows:
            if all(row.get(k) == v for k, v in filter_doc.items()):
                return row
        return None


class FakeDB(dict):
    def __getitem__(self, name):  # noqa: ANN001
        if name not in self:
            self[name] = FakeCollection()
        return dict.__getitem__(self, name)


def _payload_lookup(payload: Dict[str, Any], key: str) -> Optional[str]:
    parts = (key or "").split(".")
    current: Any = payload
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return None if current is None else str(current)


def _match_condition(payload: Dict[str, Any], cond: Any) -> bool:
    if cond is None:
        return True
    if isinstance(cond, Filter):
        must = getattr(cond, "must", []) or []
        should = getattr(cond, "should", []) or []
        min_should = getattr(cond, "min_should", None)
        if must and not all(_match_condition(payload, item) for item in must):
            return False
        if should:
            matched = sum(1 for item in should if _match_condition(payload, item))
            required = getattr(min_should, "min_count", None)
            required = int(required) if required is not None else 1
            return matched >= required
        return True
    key = getattr(cond, "key", None)
    match = getattr(cond, "match", None)
    if not key or match is None:
        return True
    value = _payload_lookup(payload, key)
    if isinstance(match, MatchValue):
        return str(value) == str(match.value)
    if isinstance(match, MatchAny):
        values = [str(v) for v in (match.any or [])]
        return str(value) in values
    return True


def _match_filter(payload: Dict[str, Any], filt: Optional[Filter]) -> bool:
    if filt is None:
        return True
    must = getattr(filt, "must", []) or []
    return all(_match_condition(payload, cond) for cond in must)


def _build_chunks(section_map):
    chunk_texts = []
    chunk_metadata = []
    for idx, (section_title, text) in enumerate(section_map):
        chunk_texts.append(text)
        chunk_metadata.append(
            {
                "section_title": section_title,
                "section_path": section_title,
                "chunk_id": f"chunk_{idx}",
                "page_start": idx + 1,
            }
        )
    return chunk_texts, chunk_metadata


def test_profile_isolation_qdrant():
    points = [
        FakePoint(
            id="1",
            score=0.9,
            payload={
                "subscription_id": "sub-1",
                "profile_id": "prof-A",
                "document_id": "doc-A",
                "text": "Alpha content " * 10,
            },
        ),
        FakePoint(
            id="2",
            score=0.8,
            payload={
                "subscription_id": "sub-1",
                "profileId": "prof-B",
                "document_id": "doc-B",
                "text": "Beta content " * 10,
            },
        ),
    ]
    retriever = HybridRetriever(client=FakeQdrant(points), embedder=FakeEmbedder())
    results = retriever.retrieve(
        collection_name="sub-1",
        query="alpha",
        profile_id="prof-A",
        subscription_id="sub-1",
        top_k=5,
    )
    assert results
    assert all(
        (item.metadata or {}).get("profile_id", (item.metadata or {}).get("profileId")) == "prof-A"
        for item in results
    )


def test_profile_isolation_redis():
    redis = FakeRedis()
    store = FactsStore(redis_client=redis, db=None)
    store.persist_document_sections(
        subscription_id="sub-1",
        profile_id="prof-A",
        document_id="doc-A",
        source_name="A.pdf",
        doc_domain="resume",
        sections=[],
        section_facts=[],
        section_summaries={},
    )
    store.persist_document_sections(
        subscription_id="sub-1",
        profile_id="prof-B",
        document_id="doc-B",
        source_name="B.pdf",
        doc_domain="resume",
        sections=[],
        section_facts=[],
        section_summaries={},
    )
    keys = list(redis.store.keys())
    assert "dwx:facts:doc:sub-1:prof-A:doc-A" in keys
    assert "dwx:facts:doc:sub-1:prof-B:doc-B" in keys
    assert "dwx:facts:profile:sub-1:prof-A" in keys
    assert "dwx:facts:profile:sub-1:prof-B" in keys


def test_domain_gate_no_po_fields_in_resume():
    resume_sections = [
        ("CONTACT", "Jane Doe\nEmail: jane@example.com\nPhone: 555-000-1111"),
        ("EXPERIENCE", "Worked on ML systems. PO #PO-123456 should be ignored."),
        ("EDUCATION", "B.S. Computer Science"),
        ("SKILLS", "Python, SQL, Spark"),
    ]
    chunk_texts, chunk_metadata = _build_chunks(resume_sections)
    builder = SectionIntelligenceBuilder(min_sections=3)
    result = builder.build(
        document_id="doc-resume",
        document_text="\n\n".join([t for _, t in resume_sections]),
        chunk_texts=chunk_texts,
        chunk_metadata=chunk_metadata,
        metadata={"doc_type": "resume", "source_name": "resume.pdf"},
    )
    for fact in result.section_facts:
        attrs = fact.get("attributes") or {}
        assert "purchase_order_number" not in attrs


def test_patient_details_format():
    med_sections = [
        ("PATIENT DETAILS", "Patient: John Smith\nID: P-7788\nAge: 45\nSex: Male"),
        ("DIAGNOSIS", "Diagnosis: Diabetes mellitus\nEncounter Date: 2024-02-01"),
        ("MEDICATIONS", "Medications: Metformin 500mg daily"),
        ("NOTES", "Follow-up in 3 months."),
    ]
    chunk_texts, chunk_metadata = _build_chunks(med_sections)
    builder = SectionIntelligenceBuilder(min_sections=3)
    result = builder.build(
        document_id="doc-med",
        document_text="\n\n".join([t for _, t in med_sections]),
        chunk_texts=chunk_texts,
        chunk_metadata=chunk_metadata,
        metadata={"doc_type": "medical", "source_name": "medical.pdf"},
    )

    redis = FakeRedis()
    db = FakeDB()
    store = FactsStore(redis_client=redis, db=db)
    store.persist_document_sections(
        subscription_id="sub-1",
        profile_id="prof-1",
        document_id="doc-med",
        source_name="medical.pdf",
        doc_domain=result.doc_domain,
        sections=[sec.__dict__ for sec in result.sections],
        section_facts=result.section_facts,
        section_summaries=result.section_summaries,
    )

    catalog = {
        "profile_id": "prof-1",
        "documents": [
            {"document_id": "doc-med", "source_name": "medical.pdf", "doc_domain": "medical"}
        ],
        "dominant_domains": {"medical": 1},
    }
    response = answer_with_section_intelligence(
        query="give me the patient's details",
        subscription_id="sub-1",
        profile_id="prof-1",
        session_state={},
        catalog=catalog,
        redis_client=redis,
        db=db,
    )
    assert response
    text = response.get("response") or ""
    assert "Patient Details:" in text
    assert "- Diagnosis:" in text
    assert "[" not in text
    assert "{" not in text


def test_greeting_short_circuit():
    response = answer_with_section_intelligence(
        query="hii",
        subscription_id="sub-1",
        profile_id="prof-1",
        session_state={},
        catalog={},
        redis_client=None,
        db=None,
    )
    assert response
    response_text = response.get("response") or ""
    assert "docwain" in response_text.lower()
    assert response.get("sources") == []
    assert response.get("metadata", {}).get("task") == "greet"


def test_rank_within_profile_only():
    points = [
        FakePoint(
            id="1",
            score=0.9,
            payload={
                "subscription_id": "sub-1",
                "profile_id": "prof-A",
                "document_id": "doc-A",
                "doc_domain": "resume",
                "text": "Python AWS Kubernetes experience " * 10,
            },
        ),
        FakePoint(
            id="2",
            score=0.8,
            payload={
                "subscription_id": "sub-1",
                "profile_id": "prof-B",
                "document_id": "doc-B",
                "doc_domain": "resume",
                "text": "Java Spring experience " * 10,
            },
        ),
    ]
    response = run_intelligent_pipeline(
        query="rank top candidates based on technical skills",
        subscription_id="sub-1",
        profile_id="prof-A",
        session_id="sess-rank",
        user_id="user",
        redis_client=None,
        qdrant_client=FakeQdrant(points),
        embedder=FakeEmbedder(),
    )
    assert response
    text = response.get("response") or ""
    assert "doc-B" not in text
    sources = response.get("sources") or []
    assert all(src.get("document_id") != "doc-B" for src in sources)
