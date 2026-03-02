from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from src.api import dw_newron
from src.api import rag_state
from src.intelligence.ask_pipeline import answer_with_section_intelligence
from src.intelligence.facts_store import FactsStore
from src.intelligence.section_intelligence_builder import SectionIntelligenceBuilder


@dataclass
class FakeRedis:
    store: dict

    def get(self, key):  # noqa: ANN001
        return self.store.get(key)

    def setex(self, key, ttl, value):  # noqa: ANN001, ARG002
        self.store[key] = value


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


def test_no_unknown_dump_for_skills_query():
    resume_sections = [
        ("CONTACT", "Jane Doe\nEmail: jane@example.com\nPhone: 555-000-1111"),
        ("SKILLS", "Python\nSQL\nSpark\nAWS"),
        ("EDUCATION", "B.S. Computer Science"),
    ]
    chunk_texts = [text for _, text in resume_sections]
    chunk_metadata = [
        {"section_title": title, "section_path": title, "chunk_id": f"chunk_{idx}", "page_start": idx + 1}
        for idx, (title, _) in enumerate(resume_sections)
    ]

    builder = SectionIntelligenceBuilder(min_sections=2)
    result = builder.build(
        document_id="doc-resume",
        document_text="\n\n".join(chunk_texts),
        chunk_texts=chunk_texts,
        chunk_metadata=chunk_metadata,
        metadata={"doc_type": "resume", "source_name": "resume.pdf"},
    )

    redis = FakeRedis(store={})
    db = FakeDB()
    store = FactsStore(redis_client=redis, db=db)
    store.persist_document_sections(
        subscription_id="sub-1",
        profile_id="prof-1",
        document_id="doc-resume",
        source_name="resume.pdf",
        doc_domain=result.doc_domain,
        sections=[sec.__dict__ for sec in result.sections],
        section_facts=result.section_facts,
        section_summaries=result.section_summaries,
    )

    catalog = {
        "profile_id": "prof-1",
        "documents": [{"document_id": "doc-resume", "source_name": "resume.pdf", "doc_domain": "resume"}],
        "dominant_domains": {"resume": 1},
    }
    response = answer_with_section_intelligence(
        query="list skills",
        subscription_id="sub-1",
        profile_id="prof-1",
        session_state={},
        catalog=catalog,
        redis_client=redis,
        db=db,
    )
    assert response
    text = response.get("response") or ""
    assert "Contact Details" not in text
    assert "unknown" not in text.lower()
    assert "Skills" in text


def test_singleton_init():
    calls = []

    class FakeRAG:
        def __init__(self, model_name=None, profile_id=None, backend_override=None, model_path=None):  # noqa: ANN001
            calls.append((model_name, profile_id, backend_override, model_path))

    rag_state.set_app_state(None)
    dw_newron._RAG_SYSTEM = None
    dw_newron._RAG_MODEL = None
    dw_newron._RAG_PROFILE = None
    dw_newron._RAG_BACKEND = None
    dw_newron._RAG_MODEL_PATH = None

    original = dw_newron.EnterpriseRAGSystem
    try:
        dw_newron.EnterpriseRAGSystem = FakeRAG
        first = dw_newron.get_rag_system(model_name="llama3.2", profile_id="prof-A")
        second = dw_newron.get_rag_system(model_name="llama3.2", profile_id="prof-B")
        assert first is second
        assert len(calls) == 1
    finally:
        dw_newron.EnterpriseRAGSystem = original
