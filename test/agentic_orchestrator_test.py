import json
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
from pathlib import Path

import pytest

# Provide lightweight stubs for heavy optional deps during import
sys.modules.setdefault("datasets", types.SimpleNamespace(Dataset=object, load_dataset=lambda *a, **k: types.SimpleNamespace()))
sys.modules.setdefault("trl", types.SimpleNamespace(SFTTrainer=object))
datasets_stub = sys.modules["datasets"]
setattr(datasets_stub, "__spec__", types.SimpleNamespace())
sys.modules["sentence_transformers"] = types.SimpleNamespace(
    SentenceTransformer=object,
    CrossEncoder=object,
    __spec__=types.SimpleNamespace(),
)


class _StubRouter:
    def __init__(self):
        self.routes = []
        self.dependencies = []
        self.on_startup = []
        self.on_shutdown = []
        self.default_response_class = None
        self.responses = {}
        self.callbacks = []
        self.route_class = None
        self.tags = []
        self.prefix = ""
        self.lifespan_context = None

# Stub heavy API modules to avoid importing optional dependencies during tests
sys.modules["src.api.dataHandler"] = types.SimpleNamespace(
    db={},
    delete_embeddings=lambda *a, **k: None,
    get_subscription_pii_setting=lambda *a, **k: None,
    trainData=lambda *a, **k: None,
    train_single_document=lambda *a, **k: None,
    fileProcessor=lambda *a, **k: None,
    train_on_document=lambda *a, **k: None,
    __spec__=types.SimpleNamespace(),
)
sys.modules["src.api.dw_chat"] = types.SimpleNamespace(
    add_message_to_history=lambda *a, **k: None,
    delete_chat_history=lambda *a, **k: None,
    delete_session=lambda *a, **k: None,
    get_chat_history=lambda *a, **k: None,
    get_session_by_id=lambda *a, **k: None,
    get_session_list=lambda *a, **k: None,
    __spec__=types.SimpleNamespace(),
)
sys.modules["src.api.documents_api"] = types.SimpleNamespace(documents_router=_StubRouter(), __spec__=types.SimpleNamespace())
sys.modules["src.screening.api"] = types.SimpleNamespace(screening_router=_StubRouter(), __spec__=types.SimpleNamespace())
sys.modules["src.tools.router"] = types.SimpleNamespace(tools_router=_StubRouter(), __spec__=types.SimpleNamespace())
sys.modules["src.teams.adapter"] = types.SimpleNamespace(__spec__=types.SimpleNamespace())
sys.modules["src.teams.bot_app"] = types.SimpleNamespace(
    BOT_CREDENTIALS_CONFIGURED=False,
    bot_adapter=None,
    teams_bot_adapter=types.SimpleNamespace(process_activity=lambda *a, **k: None),
    docwain_teams_bot=None,
    handle_teams_messages=lambda *a, **k: None,
    MICROSOFT_APP_ID="",
    MICROSOFT_APP_PASSWORD="",
    __spec__=types.SimpleNamespace(),
)
sys.modules["src.metrics.telemetry"] = types.SimpleNamespace(METRICS_V2_ENABLED=False, telemetry_store=None, __spec__=types.SimpleNamespace())
sys.modules["src.mode.execution_mode"] = types.SimpleNamespace(
    ExecutionMode=types.SimpleNamespace(AGENT="agent", NORMAL="normal", value="agent"),
    resolve_execution_mode=lambda *a, **k: None,
    __spec__=types.SimpleNamespace(),
)
sys.modules["src.execution.router"] = types.SimpleNamespace(execute_request=lambda *a, **k: None, __spec__=types.SimpleNamespace())
@dataclass
class _StubExecutionResult:
    answer: Dict[str, Any]
    mode: Any
    debug: Dict[str, Any]
    stream: Optional[Iterable[str]] = None


@dataclass
class _StubSessionState:
    preferred_execution_mode: Optional[Any] = None
    last_docs_used: List[str] = field(default_factory=list)
    last_intent: Optional[str] = None
    last_request_fingerprint: Optional[str] = None
    last_evidence_fingerprint: Optional[str] = None
    last_answer_hash: Optional[str] = None
    last_request_id: Optional[str] = None
    last_query: Optional[str] = None


sys.modules["src.execution.common"] = types.SimpleNamespace(
    ExecutionResult=_StubExecutionResult,
    normalize_answer=lambda x: x,
    chunk_text_stream=lambda x: iter([x]),
    __spec__=types.SimpleNamespace(),
)
sys.modules["src.mode.session_state"] = types.SimpleNamespace(
    SessionState=_StubSessionState,
    SessionStateStore=object,
    __spec__=types.SimpleNamespace(),
)
sys.modules["src.runtime.request_context"] = types.SimpleNamespace(RequestContext=object, __spec__=types.SimpleNamespace())

from src import main as docwain_api
from src.finetune import agentic_orchestrator as ao
from src.finetune.models import CollectionOnlyFinetuneRequest, FinetuneStatus


def test_agentic_off_legacy_path_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(docwain_api.Config.Path, "APP_HOME", tmp_path, raising=False)
    monkeypatch.setattr(docwain_api.Config.Finetune, "AGENTIC_ENABLED", False, raising=False)

    agent_called = {"count": 0}
    monkeypatch.setattr(docwain_api.AgenticFinetuneOrchestrator, "run", lambda self, req: agent_called.__setitem__("count", agent_called["count"] + 1))

    class _FakeQdrant:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(docwain_api, "QdrantClient", lambda **kwargs: _FakeQdrant())
    monkeypatch.setattr(docwain_api, "discover_profile_ids_from_collection", lambda client, collection_name, max_points: ["p1"])
    monkeypatch.setattr(docwain_api, "build_dataset_from_qdrant", lambda **kwargs: tmp_path / "ds.jsonl")
    status_obj = FinetuneStatus(job_id="j1", profile_id="p1", status="queued", message="", params={}, training_run_id="run", params_hash="hash")
    manager = types.SimpleNamespace(start_job=lambda req: status_obj, get_status=lambda job_id: status_obj)
    monkeypatch.setattr(docwain_api, "get_finetune_manager", lambda: manager)

    request = CollectionOnlyFinetuneRequest(collection_name="col1")
    result = docwain_api.finetune_from_collection(request)

    assert agent_called["count"] == 0
    assert result["profiles_processed"] == 1
    assert result["collection_name"] == "col1"


def test_agentic_on_uses_collection_only(monkeypatch, tmp_path):
    monkeypatch.setattr(ao.Config.Path, "APP_HOME", tmp_path, raising=False)
    monkeypatch.setattr(ao.Config.Finetune, "AGENTIC_ENABLED", True, raising=False)
    monkeypatch.setattr(ao, "discover_profile_ids_from_collection", lambda client, collection_name, max_points: ["p-ag"])

    calls = {"collections": set(), "subscription_id": None}

    def _fake_build_dataset(profile_id, subscription_id=None, collection_name=None, **kwargs):
        calls["collections"].add(collection_name)
        calls["subscription_id"] = subscription_id
        path = tmp_path / f"{profile_id}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "instruction": "Q?",
                    "output": "A.",
                    "input": "",
                    "metadata": {
                        "collection_name": collection_name,
                        "profile_id": profile_id,
                        "document_id": "doc1",
                        "source_file": "file.pdf",
                        "chunk_id": "c1",
                        "chunk_index": 1,
                        "page": 1,
                    },
                }
            )
            + "\n"
        )
        return path

    monkeypatch.setattr(ao, "build_dataset_from_qdrant", _fake_build_dataset)
    monkeypatch.setattr(ao.AgenticFinetuneOrchestrator, "_ensure_model_available", lambda self, model_name: None)
    monkeypatch.setattr(ao.AgenticFinetuneOrchestrator, "_plan_samples", lambda self, collection_name, profile_ids: [])
    monkeypatch.setattr(
        ao.NemotronOrchestratorAgent,
        "plan",
        lambda self, c, p, s: ao.FinetunePlan(
            acceptance_criteria=ao.AcceptanceCriteria(min_examples=1),
            dataset_plan=ao.DatasetPlan(max_points=1, questions_per_chunk=1),
        ),
    )
    status_obj = FinetuneStatus(job_id="j2", profile_id="p-ag", status="queued", message="", params={}, training_run_id="run", params_hash="hash")
    monkeypatch.setattr(ao, "get_finetune_manager", lambda: types.SimpleNamespace(start_job=lambda req: status_obj))
    orchestrator = ao.AgenticFinetuneOrchestrator(client=types.SimpleNamespace())

    request = CollectionOnlyFinetuneRequest(collection_name="onlycol", agentic=True)
    result = orchestrator.run(request)

    assert calls["collections"] == {"onlycol"}
    assert calls["subscription_id"] == "onlycol"
    assert result["overall_status"] in {"running", "succeeded", "queued"}


def test_snapshot_always_defined(monkeypatch, tmp_path):
    monkeypatch.setattr(ao.Config.Path, "APP_HOME", tmp_path, raising=False)
    monkeypatch.setattr(ao.AgenticFinetuneOrchestrator, "_ensure_model_available", lambda self, model_name: None)
    monkeypatch.setattr(ao, "discover_profile_ids_from_collection", lambda client, collection_name, max_points: [])
    orchestrator = ao.AgenticFinetuneOrchestrator(client=types.SimpleNamespace())
    request = CollectionOnlyFinetuneRequest(collection_name="empty-col", agentic=True)
    with pytest.raises(ValueError):
        orchestrator.run(request)


def test_ollama_model_missing_returns_clear_error(monkeypatch):
    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def list():
            return {"models": [{"name": "other"}]}

    monkeypatch.setattr(ao.ollama, "Client", lambda host=None: _Client())
    orchestrator = ao.AgenticFinetuneOrchestrator(client=types.SimpleNamespace())
    with pytest.raises(ao.OllamaModelMissing) as err:
        orchestrator._ensure_model_available("nemotron-3-nano")
    assert "ollama pull nemotron-3-nano" in str(err.value).lower()


def test_plan_json_parsing_and_validation(monkeypatch):
    class _Client:
        def chat(self, prompt: str):
            return json.dumps(
                {
                    "dataset_plan": {"max_points": 10, "questions_per_chunk": 1},
                    "training_plan": {"learning_rate": 1e-4, "max_steps": 50, "batch_size": 2, "gradient_accumulation": 1, "lora_r": 8, "lora_alpha": 8, "lora_dropout": 0.1},
                    "risk_flags": [{"code": "dup", "severity": "low", "mitigation": "dedup"}],
                    "acceptance_criteria": {"min_examples": 1, "duplicate_ratio_max": 0.5},
                }
            )

    monkeypatch.setattr(ao, "create_nemotron_client", lambda model_name=None: _Client())
    agent = ao.NemotronOrchestratorAgent("nemotron-3-nano")
    plan = agent.plan("col", ["p1"], [])
    assert plan.dataset_plan.max_points == 10
    assert plan.training_plan.learning_rate == 1e-4
    assert plan.acceptance_criteria.min_examples == 1
