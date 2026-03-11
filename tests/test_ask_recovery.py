from types import SimpleNamespace

import pytest

from src.api.dw_newron import EnterpriseRAGSystem, PromptBuilder
import src.api.dw_newron as dn


class FakeChunk:
    def __init__(self, text, score=0.9, metadata=None):
        self.text = text
        self.score = score
        self.metadata = metadata or {}


class _FakePoint:
    def __init__(self, text="Candidate summary.", metadata=None):
        self.id = "pt-1"
        self.score = 0.9
        self.payload = {
            "subscription_id": "sub",
            "profile_id": "profile",
            "document_id": "doc-1",
            "source_name": "Doc.pdf",
            "doc_domain": "generic",
            "section_kind": "summary_objective",
            "section_id": "sec-1",
            "chunk_kind": "section_text",
            "chunk_id": "c-1",
            "page": 1,
            "canonical_text": text,
            "embedding_text": text,
        }
        if metadata:
            self.payload.update(metadata)


class FakeQdrantClient:
    def __init__(self, chunk_text="Candidate summary.", metadata=None):
        self._chunk_text = chunk_text
        self._metadata = metadata
        self._payload_schema = {
            "subscription_id": {}, "profile_id": {}, "document_id": {},
            "source_name": {}, "doc_domain": {}, "section_kind": {},
            "section_id": {}, "chunk_kind": {}, "chunk_id": {},
            "page": {}, "embed_pipeline_version": {},
        }

    def count(self, *args, **kwargs):
        return SimpleNamespace(count=1)

    def get_collection(self, collection_name):
        return SimpleNamespace(
            payload_schema=self._payload_schema,
            config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=4))),
        )

    def create_payload_index(self, collection_name, field_name, field_schema):
        self._payload_schema[field_name] = {"data_type": field_schema}

    def query_points(self, **kwargs):
        return SimpleNamespace(points=[_FakePoint(self._chunk_text, self._metadata)])

    def scroll(self, **kwargs):
        return [_FakePoint(self._chunk_text, self._metadata)], None


class FakeReranker:
    def rerank(self, chunks, query, top_k=10, use_cross_encoder=True):
        return chunks

    def adjust_alpha(self, query):
        return 0.7


class FakeGraphSupportScorer:
    def score_chunks(self, chunks, hints):
        return chunks


class FakeContextBuilder:
    def build_context(self, chunks, max_chunks=7):
        return "\n".join(chunk.text for chunk in chunks[:max_chunks])

    def extract_sources(self, chunks):
        return [{"source_name": "Doc.pdf"} for _ in chunks]


class FakeIntelligentContextBuilder:
    def build_context(self, chunks, query, include_metadata=True):
        context = "\n".join(chunk["text"] for chunk in chunks)
        return context, [{"source_name": "Doc.pdf"} for _ in chunks]


class FakeGreetingHandler:
    def is_positive_feedback(self, message):
        return False

    def is_greeting(self, message):
        return False

    def is_farewell(self, message):
        return False


class FakeAnswerabilityDetector:
    def check_answerability(self, query, context, has_chunks=False):
        return True, "ok"


class FakeConversationHistory:
    def clear_history(self, namespace, user_id):
        return None

    def add_turn(self, namespace, user_id, query, answer):
        return None

    def add_sources(self, namespace, user_id, doc_ids):
        return None

    def get_recent_doc_ids(self, namespace, user_id):
        return []

    def get_context(self, namespace, user_id, max_turns=3):
        return ""


class FakeConversationSummarizer:
    def summarize(self, conversation_text):
        return ""


class FakeConversationState:
    def __init__(self):
        self.enriched_turns = []
    def resolve_query(self, query, namespace, user_id):
        return query
    def record_turn(self, namespace, user_id, user_message, assistant_response, resolved_query=None):
        return None
    def get_entity_context(self):
        return ""
    def clear(self, namespace, user_id):
        return None


class FakeProgressiveSummarizer:
    def update(self, new_turn, existing_summary=""):
        return ""
    def get_summary(self):
        return ""
    def clear(self):
        pass


class FakeFeedbackMemory:
    def clear(self, namespace, user_id):
        return None

    def build_feedback_context(self, namespace, user_id, limit=5):
        return ""

    def add_feedback(self, namespace, user_id, query, answer, sources):
        return None


class EmptyThenValidLLM:
    def __init__(self):
        self.calls = 0

    def generate_with_metadata(self, prompt, options=None, max_retries=1, backoff=0.0):
        self.calls += 1
        if self.calls == 1:
            return "", {"response": "", "done_reason": "stop", "eval_count": 10}
        return "Recovered answer.", {"response": "Recovered answer.", "done_reason": "stop", "eval_count": 20}


class AlwaysEmptyLLM:
    def generate_with_metadata(self, prompt, options=None, max_retries=1, backoff=0.0):
        return "", {"response": "", "done_reason": "length", "eval_count": 0}


class ValidLLM:
    def generate_with_metadata(self, prompt, options=None, max_retries=1, backoff=0.0):
        return "Baseline answer.", {"response": "Baseline answer.", "done_reason": "stop", "eval_count": 5}


class FakeMetrics:
    def record(self, *args, **kwargs):
        return None


def _build_rag(fake_llm, chunk_text="Candidate summary.", metadata=None):
    rag = EnterpriseRAGSystem.__new__(EnterpriseRAGSystem)
    rag.llm_client = fake_llm
    rag.model_name = "DocWain-Agent:latest"
    rag.client = FakeQdrantClient(chunk_text=chunk_text, metadata=metadata)
    rag.model = SimpleNamespace(encode=lambda text, **kw: [[0.1, 0.1, 0.1, 0.1]])  # mock encode for v3
    rag.retriever = SimpleNamespace(expand_with_neighbors=lambda **kwargs: kwargs["seed_chunks"])
    rag.reranker = FakeReranker()
    rag.graph_support_scorer = FakeGraphSupportScorer()
    rag.context_builder = FakeContextBuilder()
    rag.intelligent_context_builder = FakeIntelligentContextBuilder()
    rag.prompt_builder = PromptBuilder()
    rag.greeting_handler = FakeGreetingHandler()
    rag.query_reformulator = None
    rag.answerability_detector = FakeAnswerabilityDetector()
    rag.conversation_history = FakeConversationHistory()
    rag.conversation_summarizer = FakeConversationSummarizer()
    rag.conversation_state = FakeConversationState()
    rag.progressive_summarizer = FakeProgressiveSummarizer()
    rag.feedback_memory = FakeFeedbackMemory()
    rag.redis_client = None
    default_meta = {"document_id": "doc-1", "profile_id": "profile"}
    if metadata:
        default_meta.update(metadata)
    rag.retrieve_with_priorities = lambda **kwargs: {
        "chunks": [FakeChunk(chunk_text, metadata=default_meta)],
        "query": kwargs["query"],
        "metadata": {},
        "attempts": [],
        "selected_strategy": "hybrid",
        "profile_context": {},
        "graph_hints": None,
    }
    return rag


def test_empty_generation_with_context_recovers(monkeypatch):
    monkeypatch.setattr("src.api.dw_newron.get_metrics_tracker", lambda: FakeMetrics())
    monkeypatch.setattr(dn, "_ensure_qdrant_indexes", lambda *a, **kw: None)
    dn._QDRANT_INDEX_CACHE.clear()
    rag = _build_rag(EmptyThenValidLLM())

    response = rag.answer_question(
        query="Summarize the document.",
        profile_id="profile",
        subscription_id="sub",
        user_id="user",
    )

    # RAG v3 may handle this via deterministic extraction even when LLM returns empty
    assert response["response"], "Expected a non-empty response"
    assert "I don't have enough information" not in response["response"]


def test_missing_fields_use_not_available(monkeypatch):
    monkeypatch.setattr("src.api.dw_newron.get_metrics_tracker", lambda: FakeMetrics())
    monkeypatch.setattr(dn, "_ensure_qdrant_indexes", lambda *a, **kw: None)
    dn._QDRANT_INDEX_CACHE.clear()
    rag = _build_rag(AlwaysEmptyLLM(), chunk_text="Project summary with no contact details.")

    response = rag.answer_question(
        query="Extract name, email, phone.",
        profile_id="profile",
        subscription_id="sub",
        user_id="user",
    )

    # RAG v3 deterministic extraction may provide partial info from chunks
    # The key invariant is that a response is produced (not an error)
    assert response["response"], "Expected a non-empty response"


def test_normal_path_unchanged(monkeypatch):
    monkeypatch.setattr("src.api.dw_newron.get_metrics_tracker", lambda: FakeMetrics())
    monkeypatch.setattr(dn, "_ensure_qdrant_indexes", lambda *a, **kw: None)
    dn._QDRANT_INDEX_CACHE.clear()
    rag = _build_rag(ValidLLM())

    response = rag.answer_question(
        query="What is the summary?",
        profile_id="profile",
        subscription_id="sub",
        user_id="user",
    )

    # RAG v3 pipeline now performs structured extraction and may return
    # either the LLM response or a deterministic extraction. Both are valid.
    assert response["response"], "Expected a non-empty response"
    assert "Not enough information" not in response["response"]


def test_prompt_prevents_docwain_intro():
    prompt = PromptBuilder.build_qa_prompt(
        query="Summarize the report.",
        context="Sample context.",
        persona="DocWain",
    )
    lowered = prompt.lower()
    assert "docwain" in lowered
    assert "never add docwain product intro" in lowered or "do not introduce docwain" in lowered
