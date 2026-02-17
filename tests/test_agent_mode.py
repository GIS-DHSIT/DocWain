import types
from types import SimpleNamespace

import pytest

from src.agent.limits import LimitResult, check_and_count
from src.agent.orchestrator import AgentOrchestrator, DocumentContext
try:
    from src.execution.common import ExecutionResult
except ImportError:
    from src.execution import ExecutionResult
from src.mode.execution_mode import ExecutionMode


class FakePipeline:
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.ops = []

    def incr(self, key):
        self.ops.append(("incr", key))
        return self

    def expire(self, key, ttl):
        self.ops.append(("expire", key, ttl))
        return self

    def execute(self):
        for op in self.ops:
            if op[0] == "incr":
                key = op[1]
                self.redis_client.store[key] = str(int(self.redis_client.store.get(key, "0")) + 1)
            elif op[0] == "expire":
                continue
        self.ops = []
        return True


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.sets = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return False
        self.store[key] = value
        return True

    def sismember(self, key, member):
        return member in self.sets.get(key, set())

    def scard(self, key):
        return len(self.sets.get(key, set()))

    def sadd(self, key, member):
        self.sets.setdefault(key, set()).add(member)

    def get(self, key):
        return self.store.get(key)

    def pipeline(self):
        return FakePipeline(self)


class FakeLLM:
    def generate(self, prompt, max_retries=2):
        if "extraction worker" in prompt:
            return """{\n  \"doc_name\": \"Resume.pdf\",\n  \"category\": \"resume\",\n  \"language\": \"en\",\n  \"summary\": \"Candidate has experience in analytics and project delivery.\",\n  \"key_points\": [\"5+ years analytics\", \"Project leadership\"],\n  \"fields\": {\"experience\": [\"Analytics Lead\"], \"skills\": [\"Python\"]}\n}"""
        if "validation worker" in prompt:
            return "{\"supported\": [\"5+ years analytics\"], \"unsupported\": []}"
        if "SECTIONED SUMMARY" in prompt:
            return (
                "SECTIONED SUMMARY\n"
                "Overview\n"
                "Candidate summary across documents.\n"
                "Documents Covered\n"
                "- Resume.pdf — resume — 1-line\n"
                "Key Findings\n"
                "resume:\n"
                "- 5+ years analytics\n"
                "Evidence\n"
                "- Resume.pdf: summary evidence"
            )
        return "{}"


def test_agent_mode_summarize_returns_sectioned_summary(monkeypatch):
    fake_redis = FakeRedis()

    def fake_retrieve(*args, **kwargs):
        doc = DocumentContext(
            doc_id="doc-1",
            doc_name="Resume.pdf",
            category="resume",
            language="en",
            chunks=[{"text": "Worked 5 years in analytics.", "payload": {"source": {"name": "Resume.pdf"}}}],
        )
        chunks = [{"text": "Worked 5 years in analytics.", "payload": {"source": {"name": "Resume.pdf"}}}]
        return [doc], chunks

    monkeypatch.setattr("src.agent.orchestrator.get_redis_client", lambda: fake_redis)
    monkeypatch.setattr("src.agent.orchestrator.create_llm_client", lambda model_name: FakeLLM())
    monkeypatch.setattr("src.agent.orchestrator.check_and_count", lambda **kwargs: LimitResult(True, "ok", "ok"))
    monkeypatch.setattr("src.agent.orchestrator.add_message_to_history", lambda *args, **kwargs: (None, "session"))
    monkeypatch.setattr("src.agent.orchestrator.AgentOrchestrator._retrieve_profile_context", fake_retrieve)

    request = SimpleNamespace(
        query="summarize",
        user_id="user-1",
        profile_id="profile-1",
        subscription_id="sub-1",
        model_name="DocWain-Agent",
        persona="DocWain",
        session_id=None,
        new_session=False,
        agent_mode=True,
        stream=False,
        debug=False,
        tools=None,
        tool_inputs=None,
        use_tools=False,
    )

    response = AgentOrchestrator.run(request)
    assert "SECTIONED SUMMARY" in response["answer"]["response"]


def test_user_cap_enforced(monkeypatch):
    fake_redis = FakeRedis()
    fake_redis.sets["agent:users:sub-1"] = {"user-a"}

    monkeypatch.setenv("AGENT_MODE_ENABLED", "true")
    monkeypatch.setenv("AGENT_MODE_MAX_USERS_PER_SUBSCRIPTION", "1")

    result = check_and_count(
        subscription_id="sub-1",
        profile_id="profile-1",
        user_id="user-b",
        session_id="sess",
        query="summarize",
        redis_client=fake_redis,
    )
    assert result.allowed is False
    assert result.reason == "user_cap"


def test_daily_limit_enforced(monkeypatch):
    fake_redis = FakeRedis()
    day = "20991231"
    monkeypatch.setenv("AGENT_MODE_ENABLED", "true")
    monkeypatch.setenv("AGENT_MODE_MAX_PROMPTS_PER_USER_PER_DAY", "1")

    monkeypatch.setattr("src.agent.limits._today_key", lambda: day)
    fake_redis.store[f"agent:count:user:sub-1:user-1:daily:{day}"] = "1"

    result = check_and_count(
        subscription_id="sub-1",
        profile_id="profile-1",
        user_id="user-1",
        session_id="sess",
        query="summarize",
        redis_client=fake_redis,
    )
    assert result.allowed is False
    assert result.reason == "user_daily_limit"


def test_lock_prevents_duplicate_processing(monkeypatch):
    fake_redis = FakeRedis()

    def fake_retrieve(*args, **kwargs):
        doc = DocumentContext(
            doc_id="doc-1",
            doc_name="Resume.pdf",
            category="resume",
            language="en",
            chunks=[{"text": "Worked 5 years in analytics.", "payload": {"source": {"name": "Resume.pdf"}}}],
        )
        chunks = [{"text": "Worked 5 years in analytics.", "payload": {"source": {"name": "Resume.pdf"}}}]
        return [doc], chunks

    monkeypatch.setattr("src.agent.orchestrator.get_redis_client", lambda: fake_redis)
    monkeypatch.setattr("src.agent.orchestrator.create_llm_client", lambda model_name: FakeLLM())
    monkeypatch.setattr("src.agent.orchestrator.check_and_count", lambda **kwargs: LimitResult(True, "ok", "ok"))
    monkeypatch.setattr("src.agent.orchestrator.add_message_to_history", lambda *args, **kwargs: (None, "session"))
    monkeypatch.setattr("src.agent.orchestrator.AgentOrchestrator._retrieve_profile_context", fake_retrieve)

    request = SimpleNamespace(
        query="summarize",
        user_id="user-1",
        profile_id="profile-1",
        subscription_id="sub-1",
        model_name="DocWain-Agent",
        persona="DocWain",
        session_id=None,
        new_session=False,
        agent_mode=True,
        stream=False,
        debug=False,
        tools=None,
        tool_inputs=None,
        use_tools=False,
    )

    first = AgentOrchestrator.run(request)
    assert "SECTIONED SUMMARY" in first["answer"]["response"]

    second = AgentOrchestrator.run(request)
    second_text = second["answer"]["response"]
    assert "Agent mode is already processing this request" in second_text


def test_non_agent_path_unchanged(monkeypatch):
    import sys
    import types as pytypes

    def stub_module(name, **attrs):
        module = pytypes.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
        return module

    fake_datahandler = pytypes.ModuleType("src.api.dataHandler")
    fake_datahandler.db = {}
    fake_datahandler.clear_legacy_vetting_metadata = lambda: None
    fake_datahandler.delete_embeddings = lambda *args, **kwargs: None
    fake_datahandler.get_subscription_pii_setting = lambda *args, **kwargs: True
    fake_datahandler.trainData = lambda *args, **kwargs: None
    fake_datahandler.train_single_document = lambda *args, **kwargs: None
    fake_datahandler.get_qdrant_client = lambda: None
    fake_datahandler.get_redis_client = lambda: None
    fake_datahandler.encode_with_fallback = lambda *args, **kwargs: [[0.1]*4]
    fake_datahandler.fileProcessor = lambda *args, **kwargs: None
    fake_datahandler.train_on_document = lambda *args, **kwargs: None
    fake_datahandler.update_security_screening = lambda *args, **kwargs: None
    sys.modules["src.api.dataHandler"] = fake_datahandler

    from fastapi import APIRouter

    stub_module("src.api.documents_api", documents_router=APIRouter())
    stub_module("src.api.profile_documents_api", profile_docs_router=APIRouter())
    stub_module("src.api.profiles_api", profiles_router=APIRouter())
    stub_module("src.api.knowledge_graph", knowledge_graph_router=APIRouter())
    stub_module(
        "src.finetune",
        get_finetune_manager=lambda: None,
        list_models=lambda: [],
        resolve_model_for_profile=lambda *args, **kwargs: pytypes.SimpleNamespace(model_name=None),
    )
    stub_module(
        "src.finetune.agentic_orchestrator",
        AgenticFinetuneOrchestrator=object,
        OllamaModelMissing=Exception,
        OllamaUnavailable=Exception,
    )
    stub_module("src.finetune.dataset_builder", build_dataset_from_qdrant=lambda *args, **kwargs: None)
    stub_module(
        "src.finetune.models",
        CollectionOnlyFinetuneRequest=object,
        FinetuneRequest=object,
    )
    stub_module(
        "src.metrics.aggregation",
        aggregate_range=lambda *args, **kwargs: {},
        build_metrics_payload=lambda *args, **kwargs: {},
        compute_date_range=lambda *args, **kwargs: {},
        daily_boundaries=lambda *args, **kwargs: {},
        normalize_timezone=lambda *args, **kwargs: "UTC",
        normalize_week_start=lambda *args, **kwargs: "monday",
        weekly_boundaries=lambda *args, **kwargs: {},
    )
    stub_module("src.metrics.ai_metrics", get_metrics_store=lambda: pytypes.SimpleNamespace(available=False))
    stub_module("src.metrics.repository", MetricsRepository=object)
    stub_module("src.screening.api", screening_router=APIRouter())
    class _PermissiveScreeningConfig:
        """Stub that returns None/False for any attribute access."""
        ENABLED = False
        @classmethod
        def load(cls):
            return cls()
        def __getattr__(self, name):
            return None
    stub_module(
        "src.screening.config",
        log_legacy_vetting_notice_if_missing=lambda: None,
        ScreeningConfig=_PermissiveScreeningConfig,
    )
    stub_module(
        "src.storage.azure_blob_client",
        validate_containers_once=lambda: None,
        validate_storage_configured_once=lambda: None,
        get_blob_service_client=lambda: None,
        get_document_container_client=lambda: None,
        has_blob_credentials=lambda: False,
    )
    stub_module(
        "src.storage.blob_persistence",
        save_pickle_atomic=lambda *args, **kwargs: None,
        load_pickle=lambda *args, **kwargs: None,
    )
    stub_module("src.teams.adapter", TeamsAuthError=Exception, handle_teams_activity=lambda *args, **kwargs: {})
    stub_module(
        "src.teams.bot_app",
        BOT_CREDENTIALS_CONFIGURED=False,
        bot_adapter=pytypes.SimpleNamespace(),
        docwain_teams_bot=pytypes.SimpleNamespace(),
        MICROSOFT_APP_ID="",
        MICROSOFT_APP_PASSWORD="",
    )
    stub_module("src.tools.router", tools_router=APIRouter())
    stub_module("src.training.qdrant_profile_discovery", discover_profile_ids_from_collection=lambda *args, **kwargs: [])
    stub_module(
        "src.runtime.request_context",
        RequestContext=pytypes.SimpleNamespace(build=lambda **kwargs: pytypes.SimpleNamespace(request_id="req")),
    )
    stub_module(
        "src.execution.router",
        execute_request=lambda *args, **kwargs: ExecutionResult(
            answer={"response": "ok", "sources": [{"file_name": "doc.pdf", "page": 1}]},
            mode=ExecutionMode.NORMAL,
            debug={},
        ),
    )

    import src.main as main

    called = {"run": False}

    def fake_run(**kwargs):
        called["run"] = True
        return {"response": "ok", "sources": [{"file_name": "doc.pdf", "page": 1}]}

    monkeypatch.setattr(main.rag_v3, "run", fake_run)

    # When src.main is already imported by other tests, the `execute_request`
    # binding is already resolved. Monkeypatch it directly on the main module.
    fake_execute = lambda *args, **kwargs: ExecutionResult(
        answer={"response": "ok", "sources": [{"file_name": "doc.pdf", "page": 1}]},
        mode=ExecutionMode.NORMAL,
        debug={},
    )
    monkeypatch.setattr(main, "execute_request", fake_execute)

    from src.api.rag_state import AppState, set_app_state

    set_app_state(
        AppState(
            embedding_model=object(),
            reranker=None,
            qdrant_client=object(),
            redis_client=None,
            ollama_client=None,
            rag_system=None,
        )
    )

    request = main.QuestionRequest(
        query="What is in the document?",
        user_id="user-1",
        profile_id="profile-1",
        subscription_id="sub-1",
        document_id=None,
        tool_hint=None,
        model_name="llama3.2",
        persona="DocWain",
        session_id=None,
        new_session=False,
        agent_mode=False,
        stream=False,
        debug=False,
        tools=None,
        tool_inputs=None,
        use_tools=False,
    )
    request.stream = False

    response = main.ask_question_api(request, stream=False)
    # The execute_request stub returns a proper ExecutionResult
    answer_text = response.answer.response if hasattr(response.answer, "response") else str(response.answer)
    assert answer_text == "ok"
