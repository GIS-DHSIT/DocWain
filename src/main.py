import hashlib
import json
import logging
import datetime as dt
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import ollama
import uvicorn
from bson.objectid import ObjectId
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, constr
from qdrant_client import QdrantClient

from src.api.config import Config
from src.api.dataHandler import (
    db,
    clear_legacy_vetting_metadata,
    delete_embeddings,
    get_subscription_pii_setting,
    trainData,
    train_single_document,
)
from src.api.dw_chat import (
    add_message_to_history,
    delete_chat_history,
    delete_session,
    get_chat_history,
    get_session_by_id,
    get_session_list,
)
from src.api.schemas import ModelInfo, ModelsResponse
from src.api.documents_api import documents_router
from src.api.knowledge_graph import knowledge_graph_router
from src.finetune import get_finetune_manager, list_models
from src.finetune.agentic_orchestrator import AgenticFinetuneOrchestrator, OllamaModelMissing, OllamaUnavailable
from src.finetune.dataset_builder import build_dataset_from_qdrant
from src.finetune.models import CollectionOnlyFinetuneRequest, FinetuneRequest
from src.metrics.aggregation import (
    aggregate_range,
    build_metrics_payload,
    compute_date_range,
    daily_boundaries,
    normalize_timezone,
    normalize_week_start,
    weekly_boundaries,
)
from src.metrics.ai_metrics import get_metrics_store
from src.metrics.repository import MetricsRepository
from src.mode.execution_mode import ExecutionMode, resolve_execution_mode
from src.mode.session_state import SessionStateStore
from src.execution.router import execute_request
from src.execution.common import normalize_answer, chunk_text_stream
from src.screening.api import screening_router
from src.screening.config import log_legacy_vetting_notice_if_missing
from src.storage.azure_blob_client import validate_containers_once
from botbuilder.schema import Activity

from src.teams import adapter as teams_adapter
from src.teams.bot_app import (
    BOT_CREDENTIALS_CONFIGURED,
    bot_adapter,
    docwain_teams_bot,
    MICROSOFT_APP_ID,
    MICROSOFT_APP_PASSWORD,
)
from src.tools.router import tools_router
from src.training.qdrant_profile_discovery import discover_profile_ids_from_collection
from src.runtime.request_context import RequestContext

logger = logging.getLogger(__name__)


def _error(code: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details or {}}}


def _get_dw_newron():
    """
    Lazy loader for RAG pipeline functions to avoid heavy imports on module load.
    This keeps Teams/chat endpoints responsive even when optional deps are absent
    (tests, lightweight deployments) while preserving behavior for the main API.
    """
    from src.api import dw_newron  # local import to defer heavy deps
    return dw_newron


app = FastAPI(title="DocWain API")
api_router = APIRouter(prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router.include_router(screening_router)
api_router.include_router(documents_router, tags=["Documents"])
api_router.include_router(tools_router, tags=["Tools"])

session_state_store = SessionStateStore()


@app.on_event("startup")
async def _startup_checks() -> None:
    try:
        validate_containers_once()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Azure blob container validation skipped: %s", exc)
    try:
        clear_legacy_vetting_metadata()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Legacy metadata cleanup skipped: %s", exc)
    try:
        log_legacy_vetting_notice_if_missing()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Legacy config notice skipped: %s", exc)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    payload = detail if isinstance(detail, dict) and isinstance(detail.get("error"), dict) else _error(
        str(getattr(exc, "status_code", "error")), str(detail)
    )
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error for %s %s", request.method, request.url, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error("internal_error", "Unexpected error occurred", {"reason": str(exc)}),
    )


class QuestionRequest(BaseModel):
    query: constr(min_length=1)
    user_id: str = "someone@email.com"
    profile_id: constr(min_length=1) = "67ac62ddfaa3aee44d38f4a5"
    subscription_id: str = "default"
    model_name: str = "llama3.2"
    persona: str = "Document Assistant"
    session_id: Optional[str] = None
    new_session: bool = False  # Frontend sends flag here
    agent_mode: Optional[bool] = None
    stream: bool = False  # When true, /ask returns a streaming response instead of JSON.
    debug: bool = False
    tools: Optional[List[str]] = None
    tool_inputs: Optional[Dict[str, Any]] = None
    use_tools: bool = False


class AnswerPayload(BaseModel):
    response: Any
    sources: List[Any] = Field(default_factory=list)
    grounded: bool = False
    context_found: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    answer: AnswerPayload
    current_session_id: Optional[str]
    debug: Dict[str, Any] = Field(default_factory=dict)


def _resolve_session_id(request: QuestionRequest) -> Optional[str]:
    """
    Ensure we use a consistent session ID across the RAG pipeline and chat history.
    - If frontend provides session_id, keep it.
    - If a new session is requested but no ID was provided, generate one so history and
      LLM memory share the same namespace from the very first turn.
    """
    if request.session_id:
        return request.session_id
    if request.new_session:
        return str(uuid.uuid4())
    return None


def _normalize_answer(answer):
    """Backward-compatible wrapper around the shared normalizer."""
    return normalize_answer(answer)


def _safe_snippet(value: Optional[str], limit: int = 120) -> str:
    if not value:
        return ""
    if len(value) <= limit:
        return value
    return value[:limit] + "…"


def _resolve_output_root(path_str: str) -> Path:
    """Resolve user-provided output_dir to an absolute path within APP_HOME when relative."""
    root = Path(path_str)
    if not root.is_absolute():
        root = Path(Config.Path.APP_HOME) / root
    return root


def _training_runs_dir(output_dir: str) -> Path:
    run_dir = _resolve_output_root(output_dir) / "training_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _load_training_run_records(run_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not run_dir.exists():
        return records
    for path in run_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            records.append(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unable to read training run record %s: %s", path, exc)
    return records


def _persist_training_run_record(run_dir: Path, record: Dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{record['training_run_id']}.json"
    try:
        path.write_text(json.dumps(record, indent=2))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to persist training run record %s: %s", path, exc)
    return path


def _collection_config_hash(request: CollectionOnlyFinetuneRequest) -> str:
    """
    Compute a deterministic hash of the training configuration excluding volatile fields.
    Excludes: output_dir, run_name, retrain, subscription_id(s)
    """
    config = {
        "collection_name": request.collection_name,
        "base_model": request.base_model,
        "learning_rate": request.learning_rate,
        "num_epochs": request.num_epochs,
        "max_steps": request.max_steps,
        "batch_size": request.batch_size,
        "gradient_accumulation": request.gradient_accumulation,
        "lora_r": request.lora_r,
        "lora_alpha": request.lora_alpha,
        "lora_dropout": request.lora_dropout,
        "include_actual_data": request.include_actual_data,
        "dataset_path": request.dataset_path,
        "max_points": request.max_points,
        "questions_per_chunk": request.questions_per_chunk,
        "generation_model": request.generation_model,
    }
    encoded = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _map_status(status: Optional[str]) -> str:
    if status == "completed":
        return "succeeded"
    if status == "failed":
        return "failed"
    if status == "running":
        return "running"
    return "scheduled"


def _derive_overall_status(record: Dict[str, Any]) -> str:
    profiles = record.get("profiles") or []
    if not profiles:
        return "pending"
    statuses = {p.get("status") or "scheduled" for p in profiles}
    if statuses == {"succeeded"}:
        return "succeeded"
    if "failed" in statuses:
        if record.get("fail_fast"):
            return "failed"
        return "partial"
    if statuses.issubset({"scheduled", "queued", "running"}):
        return "in_progress"
    return "partial"


def _refresh_training_run_record(record: Dict[str, Any], manager) -> bool:
    """Refresh per-profile statuses from the finetune manager when possible."""
    updated = False
    profiles = record.get("profiles") or []
    for entry in profiles:
        job_id = entry.get("job_id")
        if not job_id or not manager:
            continue
        status_obj = manager.get_status(job_id)
        if not status_obj:
            continue
        mapped = _map_status(status_obj.status)
        message = getattr(status_obj, "message", "") or entry.get("message")
        if mapped != entry.get("status") or message != entry.get("message"):
            entry["status"] = mapped
            entry["message"] = message
            updated = True
        ollama_info = getattr(status_obj, "ollama", None)
        if ollama_info is not None and ollama_info != entry.get("ollama"):
            entry["ollama"] = ollama_info
            updated = True
    record["profiles"] = profiles
    ollama_map = {
        p.get("profile_id"): p.get("ollama")
        for p in profiles
        if p.get("profile_id") and p.get("ollama") is not None
    }
    if ollama_map or record.get("ollama"):
        if ollama_map != record.get("ollama"):
            record["ollama"] = ollama_map
            updated = True
    overall_before = record.get("overall_status")
    record["overall_status"] = _derive_overall_status(record)
    if record["overall_status"] in {"succeeded", "failed", "partial"} and not record.get("finished_at"):
        record["finished_at"] = time.time()
        updated = True
    if overall_before != record["overall_status"]:
        updated = True
    return updated


def _find_successful_training_run(
        run_dir: Path,
        collection_name: str,
        base_model: str,
        run_name: str,
        config_hash: str,
        manager,
) -> Optional[Dict[str, Any]]:
    for record in _load_training_run_records(run_dir):
        if record.get("collection_name") != collection_name:
            continue
        if record.get("base_model") != base_model:
            continue
        if str(record.get("run_name") or "") != run_name:
            continue
        if record.get("config_hash") != config_hash:
            continue
        if _refresh_training_run_record(record, manager):
            _persist_training_run_record(run_dir, record)
        profiles = record.get("profiles") or []
        if record.get("overall_status") == "succeeded" and profiles and all(
                (p.get("status") == "succeeded") for p in profiles
        ):
            return record
    return None


def _build_training_summary(
        record: Dict[str, Any],
        total_profiles_discovered: Optional[int] = None,
        status: Optional[str] = None,
) -> Dict[str, Any]:
    profiles = record.get("profiles") or []
    failures = [
        {"profile_id": p.get("profile_id"), "error": p.get("error") or p.get("message")}
        for p in profiles
        if p.get("status") == "failed"
    ]
    processed = len(profiles)
    failed = len(failures)
    succeeded = len([p for p in profiles if p.get("status") == "succeeded"])
    return {
        "status": status or record.get("overall_status"),
        "collection_name": record.get("collection_name"),
        "training_run_id": record.get("training_run_id"),
        "run_name": record.get("run_name") or None,
        "base_model": record.get("base_model"),
        "config_hash": record.get("config_hash"),
        "total_profiles_discovered": total_profiles_discovered or record.get("total_profiles_discovered") or len(profiles),
        "profiles_processed": processed,
        "profiles_succeeded": succeeded,
        "profiles_failed": failed,
        "fail_fast": record.get("fail_fast"),
        "profiles": profiles,
        "ollama": record.get("ollama"),
        "failures": failures,
    }


def _is_botframework_jwt(auth_header: str) -> bool:
    """Determine if the Authorization header contains a Bot Framework JWT."""
    if not auth_header:
        return False
    if not auth_header.lower().startswith("bearer "):
        return False
    token = auth_header[7:].strip()
    return token.count(".") == 2


def _build_text_fallback_activity(raw_body: bytes, headers: Dict[str, str]) -> Dict[str, Any] | None:
    """Construct a minimal Teams-like activity from a plain text payload."""
    try:
        text = raw_body.decode("utf-8", errors="ignore").strip()
    except Exception:
        return None
    if not text:
        return None

    convo_id = (
        headers.get("x-teams-conversation-id")
        or headers.get("conversation-id")
        or "teams-text-fallback"
    )
    user_id = headers.get("x-teams-user-id") or "teams_user"

    return {
        "type": "message",
        "text": text,
        "conversation": {"id": convo_id},
        "from": {"id": user_id},
    }


async def _parse_teams_activity(request: Request) -> tuple[Dict[str, Any] | None, bytes]:
    """
    Parse Teams activity payload, tolerating empty/invalid bodies.
    Returns a dict on success; None when the body cannot be parsed.
    """
    raw_body = await request.body()
    content_type = request.headers.get("content-type", "")
    if not raw_body or not raw_body.strip():
        logger.debug(
            "Teams payload missing/empty body; responding with friendly message | content_type=%s",
            content_type,
        )
        return None, raw_body

    try:
        activity = json.loads(raw_body)
    except json.JSONDecodeError:
        # Fallback: treat text/plain bodies as simple messages
        if content_type.lower().startswith("text/plain"):
            fallback = _build_text_fallback_activity(raw_body, dict(request.headers))
            if fallback:
                logger.debug(
                    "Teams payload decoded as text/plain fallback | content_type=%s | body_len=%d",
                    content_type,
                    len(raw_body or b""),
                )
                return fallback, raw_body

        logger.debug(
            "Invalid Teams payload: JSON decode failed; responding with friendly message | content_type=%s | body_len=%d",
            content_type,
            len(raw_body or b""),
        )
        return None, raw_body

    if not isinstance(activity, dict):
        logger.warning("Invalid Teams payload type: %s", type(activity))
        return None, raw_body

    return activity, raw_body


@api_router.post("/teams/messages", tags=["Teams"])
async def handle_teams_messages(request: Request):
    """Endpoint for Microsoft Teams activities (messages, attachments)."""
    activity_payload, raw_body = await _parse_teams_activity(request)
    if activity_payload is None:
        return {
            "type": "message",
            "text": "I couldn't read that Teams message. Please try sending it again.",
        }

    headers = dict(request.headers)
    auth_header = request.headers.get("Authorization", "")
    use_bot_framework = _is_botframework_jwt(auth_header)

    def _bot_credentials_valid() -> bool:
        app_id = MICROSOFT_APP_ID or ""
        pwd = MICROSOFT_APP_PASSWORD or getattr(Config.Teams, "BOT_APP_PASSWORD", None) or ""
        guid_like = bool(re.fullmatch(r"[0-9a-fA-F-]{36}", app_id))
        if not guid_like or not pwd or len(pwd) < 10:
            logger.error(
                "Teams bot credentials invalid | app_id_guid_like=%s pwd_set=%s pwd_len=%s",
                guid_like,
                bool(pwd),
                len(pwd or ""),
            )
            return False
        return True

    if use_bot_framework and not BOT_CREDENTIALS_CONFIGURED:
        logger.error(
            "Microsoft bot credentials are not configured; cannot validate Teams JWT. "
            "Set MicrosoftAppId/MicrosoftAppPassword in the environment."
        )
        raise HTTPException(status_code=401, detail="Bot credentials are not configured")
    if use_bot_framework and not _bot_credentials_valid():
        raise HTTPException(status_code=500, detail="Bot credentials are invalid or incomplete. Check MICROSOFT_APP_ID/MICROSOFT_APP_PASSWORD.")

    if use_bot_framework:
        try:
            activity_obj = Activity().deserialize(activity_payload)
        except Exception as exc:
            logger.error("Failed to deserialize Teams activity: %s", exc)
            raise HTTPException(status_code=400, detail="Invalid Teams activity payload")

        turn_state = {"incoming_headers": headers, "raw_body": raw_body}

        if getattr(Config.Teams, "DIAG_MODE", False):
            logger.info(
                "Teams BF diag: serviceUrl=%s channelId=%s conversationId=%s activityId=%s fromId=%s",
                getattr(activity_obj, "service_url", None),
                getattr(activity_obj, "channel_id", None),
                getattr(getattr(activity_obj, "conversation", None), "id", None),
                getattr(activity_obj, "id", None),
                getattr(getattr(activity_obj, "from_property", None), "id", None),
            )
            logger.info(
                "Teams BF auth header length: %s | adapter_app_id=%s",
                len(auth_header or ""),
                MICROSOFT_APP_ID,
            )

        if not bot_adapter:
            raise HTTPException(
                status_code=401,
                detail="Bot adapter is not initialized; check MICROSOFT_APP_ID and MICROSOFT_APP_PASSWORD.",
            )

        expected_app_id = MICROSOFT_APP_ID or "(unset)"
        actual_app_id = (activity_obj.recipient and getattr(activity_obj.recipient, "id", None)) or ""

        if actual_app_id and expected_app_id != actual_app_id:
            logger.warning(
                "Incoming activity AppId does not match configured bot AppId | expected=%s got=%s",
                expected_app_id,
                actual_app_id,
            )

        try:
            async def _run_bot(turn_context):
                turn_context.turn_state.update(turn_state)
                await docwain_teams_bot.on_turn(turn_context)

            response = await bot_adapter.process_activity(activity_obj, auth_header, _run_bot)
            if response:
                return JSONResponse(status_code=response.status, content=response.body)
        except PermissionError as exc:
            logger.error(
                "Failed Teams auth (likely AppId/secret mismatch). Expected AppId=%s | error=%s",
                expected_app_id,
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=401,
                detail="Unauthorized Teams request: verify MICROSOFT_APP_ID and MICROSOFT_APP_PASSWORD match the bot registration.",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to process Teams activity via Bot Framework adapter: %s", exc, exc_info=True)
            detail = "Unauthorized Teams request" if "auth" in str(exc).lower() or "unauthoriz" in str(exc).lower() else "Failed to process Teams activity"
            status_code = 401 if "auth" in detail.lower() else 500
            raise HTTPException(status_code=status_code, detail=detail)
        return {}

    try:
        return await teams_adapter.handle_teams_activity(
            activity_payload,
            headers=headers,
            raw_body=raw_body,
        )
    except teams_adapter.TeamsAuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_error("unauthorized", str(exc)))


def _prepare_execution(request: QuestionRequest, agent_mode_query: Optional[bool]):
    """
    Resolve session + mode and build a shared RequestContext for both streaming and JSON responses.
    """
    session_id = _resolve_session_id(request)
    request.session_id = session_id
    if agent_mode_query is not None:
        object.__setattr__(request, "agent_mode_query", agent_mode_query)

    session_state = session_state_store.get(session_id)
    mode = resolve_execution_mode(request, session_state)
    ctx = RequestContext.build(
        query=request.query,
        session_id=session_id,
        user_id=request.user_id,
        mode=mode.value,
        debug=bool(request.debug),
        profile_id=request.profile_id,
        subscription_id=request.subscription_id,
        model_name=request.model_name,
        persona=request.persona,
        tools=request.tools,
        use_tools=bool(getattr(request, "use_tools", False)),
        tool_inputs=request.tool_inputs,
    )
    return session_id, session_state, mode, ctx


@api_router.post("/ask", tags=["Default"], response_model=AskResponse)
def ask_question_api(
    request: QuestionRequest,
    agent_mode: Optional[bool] = Query(None),
    stream: Optional[bool] = Query(None),
):
    """
    Unified /ask handler. Toggle `stream=true` to receive a streamed response instead of JSON.
    """
    streaming = bool(stream if stream is not None else request.stream)
    session_id, session_state, mode, ctx = _prepare_execution(request, agent_mode)
    logger.info(
        "[ASK%s] user=%s session_id=%s new_session=%s agent_mode(body)=%s agent_mode(query)=%s stream(body)=%s stream(query)=%s query_snippet=%s",
        "_STREAM" if streaming else "",
        request.user_id,
        session_id,
        request.new_session,
        request.agent_mode,
        agent_mode,
        request.stream,
        stream,
        _safe_snippet(request.query),
    )

    result = execute_request(request, session_state=session_state, ctx=ctx, stream=streaming, debug=bool(request.debug))

    explicit_toggle = request.agent_mode if request.agent_mode is not None else agent_mode
    if explicit_toggle is not None:
        session_state_store.set_preferred_mode(
            session_id,
            ExecutionMode.AGENT if explicit_toggle else ExecutionMode.NORMAL,
        )

    if streaming:
        def _stream():
            try:
                normalized_answer = _normalize_answer(result.answer)
                stream_iterable = result.stream or chunk_text_stream(normalized_answer.get("response") or "")
                logger.info("[ASK_STREAM] streaming session_id=%s mode=%s", session_id, result.mode.value)
                for chunk in stream_iterable:
                    yield chunk
            except Exception as exc:
                logger.error("[ASK_STREAM] Streaming failed for session=%s: %s", session_id, exc, exc_info=True)
                yield "[error] Unable to stream response right now."

        return StreamingResponse(_stream(), media_type="text/plain")

    answer = AnswerPayload.model_validate(result.answer)

    _, active_session_id = add_message_to_history(
        request.user_id,
        request.query,
        answer.model_dump(),
        session_id=session_id,
        new_session=request.new_session,
    )

    logger.info("[ASK] completed session_id=%s mode=%s stream=%s", active_session_id, result.mode.value, streaming)

    enriched_debug = {**(result.debug or {}), "request_id": ctx.request_id}
    return AskResponse(answer=answer, current_session_id=active_session_id, debug=enriched_debug)


@api_router.post("/askStream", tags=["Default"], deprecated=True)
def ask_question_stream_api(request: QuestionRequest, agent_mode: Optional[bool] = Query(None)):
    """
    Backward-compatible alias for streaming; prefer /ask with `stream=true`.
    """
    object.__setattr__(request, "stream", True)
    return ask_question_api(request, agent_mode=agent_mode, stream=True)


@api_router.post("/extract/{doc_id}", tags=["Default"])
def trigger_single_extraction(doc_id: str, subscription_id: str = "default"):
    """API endpoint to extract a single document by its document ID."""
    try:
        logging.info(f"Received single document extraction request for: {doc_id} (subscription: {subscription_id})")
        result = train_single_document(doc_id)
        return {"status": "success", "message": result}
    except Exception as e:
        logging.error(f"Single extraction API error: {e}")
        raise HTTPException(status_code=500, detail="Single document extraction failed")


@api_router.post("/train/{doc_id}", tags=["Default"], deprecated=True)
def trigger_single_training(doc_id: str, subscription_id: str = "default"):
    """Deprecated alias for /extract/{doc_id}. Performs extraction only."""
    return trigger_single_extraction(doc_id=doc_id, subscription_id=subscription_id)


@api_router.get("/extract", tags=["Default"])
def trigger_extraction(subscription_id: str = "default"):
    """API endpoint to trigger document extraction."""
    try:
        logging.info(f"Received extraction request (subscription: {subscription_id})")
        status_response = trainData()
        logging.info(status_response)
        return {"status": "success", "message": status_response, "response": "Executed"}
    except Exception as e:
        logging.error(f"Extraction API error: {e}")
        raise HTTPException(status_code=500, detail="Extraction process failed")


@api_router.get("/train", tags=["Default"], deprecated=True)
def trigger_training(subscription_id: str = "default"):
    """Deprecated alias for /extract. Performs extraction only."""
    return trigger_extraction(subscription_id=subscription_id)


@api_router.post("/finetune/by-profile", tags=["Finetuning"])
def finetune_single_profile(request: FinetuneRequest):
    """
    Kick off a fine-tune job for a single profile using the provided dataset/training parameters.
    """
    manager = get_finetune_manager()
    try:
        status = manager.start_job(request)
        return status.dict()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))

@api_router.post("/finetune/by-collection", tags=["Finetuning"])
def finetune_from_collection(request: CollectionOnlyFinetuneRequest):
    """
    Fine-tune all profiles discovered inside a Qdrant collection (collection-only request).

    Example request:
    {
      "collection_name": "my-collection",
      "base_model": "llama3.2",
      "learning_rate": 0.0002,
      "num_epochs": 1,
      "max_steps": 200,
      "batch_size": 4,
      "gradient_accumulation": 2,
      "lora_r": 16,
      "lora_alpha": 16,
      "lora_dropout": 0.05,
      "include_actual_data": false,
      "dataset_path": "path/to/dataset.jsonl",
      "training_examples": [{"instruction": "string", "output": "string", "input": ""}],
      "output_dir": "finetune_artifacts",
      "run_name": "my-run",
      "retrain": false,
      "subscription_id": "string",
      "subscription_ids": ["string"],
      "max_points": 120,
      "questions_per_chunk": 2,
      "generation_model": "string",
      "fail_fast": false
    }
    """
    manager = get_finetune_manager()
    qdrant_client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=120)

    agentic_requested = bool(request.agentic or getattr(Config.Finetune, "AGENTIC_ENABLED", False))
    if agentic_requested and request.collection_name:
        orchestrator = AgenticFinetuneOrchestrator(client=qdrant_client)
        try:
            agentic_result = orchestrator.run(request)
            summary = _build_training_summary(
                agentic_result,
                total_profiles_discovered=agentic_result.get("total_profiles_discovered"),
                status=agentic_result.get("overall_status"),
            )
            for key in ("plan_path", "snapshot_path", "dataset_stats_path"):
                if agentic_result.get(key):
                    summary[key] = agentic_result[key]
            return summary
        except OllamaModelMissing as exc:
            if getattr(Config.Finetune, "AGENT_FALLBACK_TO_LEGACY", True):
                logger.warning("Agentic path missing model; falling back to legacy: %s", exc)
            else:
                raise HTTPException(status_code=400, detail=str(exc))
        except OllamaUnavailable as exc:
            if getattr(Config.Finetune, "AGENT_FALLBACK_TO_LEGACY", True):
                logger.warning("Agentic path Ollama unavailable; falling back to legacy: %s", exc)
            else:
                raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            if getattr(Config.Finetune, "AGENT_FALLBACK_TO_LEGACY", True):
                logger.warning("Agentic finetune failed; using legacy flow: %s", exc, exc_info=True)
            else:
                raise HTTPException(status_code=400, detail=str(exc))

    try:
        discovered_profiles = discover_profile_ids_from_collection(
            client=qdrant_client,
            collection_name=request.collection_name,
            max_points=request.max_points,
        )
    except ValueError as exc:
        status_code = 404 if "does not exist" in str(exc).lower() else 400
        raise HTTPException(status_code=status_code, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to read collection {request.collection_name}: {exc}")

    if not discovered_profiles:
        raise HTTPException(status_code=404, detail=f"No profile_ids found in collection {request.collection_name}")

    config_hash = _collection_config_hash(request)
    run_name = request.run_name or ""
    run_dir = _training_runs_dir(request.output_dir)

    existing = None
    if not request.retrain:
        existing = _find_successful_training_run(
            run_dir=run_dir,
            collection_name=request.collection_name,
            base_model=request.base_model,
            run_name=run_name,
            config_hash=config_hash,
            manager=manager,
        )
    if existing and not request.retrain:
        return _build_training_summary(existing, total_profiles_discovered=len(discovered_profiles), status="already_trained")

    training_run_id = str(uuid.uuid4())
    run_record: Dict[str, Any] = {
        "training_run_id": training_run_id,
        "collection_name": request.collection_name,
        "base_model": request.base_model,
        "run_name": run_name,
        "config_hash": config_hash,
        "fail_fast": request.fail_fast,
        "retrain": request.retrain,
        "started_at": time.time(),
        "finished_at": None,
        "overall_status": "scheduled",
        "profiles": [],
        "ollama": {},
        "total_profiles_discovered": len(discovered_profiles),
    }
    _persist_training_run_record(run_dir, run_record)

    for profile_id in discovered_profiles:
        profile_entry: Dict[str, Any] = {"profile_id": profile_id}
        try:
            dataset_result = build_dataset_from_qdrant(
                profile_id=profile_id,
                subscription_id=request.collection_name,
                collection_name=request.collection_name,
                max_points=request.max_points,
                questions_per_chunk=request.questions_per_chunk,
                generation_model=request.generation_model,
                client=qdrant_client,
                run_id=training_run_id,
            )
            profile_entry["diagnostics_path"] = str(dataset_result.diagnostics_path)
            if dataset_result.status != "success" or not dataset_result.dataset_path:
                profile_entry.update(
                    {
                        "status": "skipped_insufficient_pairs",
                        "message": f"Dataset generation {dataset_result.status}",
                        "pairs_created": dataset_result.pair_count,
                        "diagnostics": dataset_result.diagnostics,
                    }
                )
                run_record["profiles"].append(profile_entry)
                if request.fail_fast:
                    run_record["overall_status"] = "failed"
                    break
                continue
            payload = request.finetune_payload(profile_id=profile_id, dataset_path=str(dataset_result.dataset_path))
            payload["training_run_id"] = training_run_id
            status = manager.start_job(FinetuneRequest(**payload))
            profile_entry.update(
                {
                    "job_id": status.job_id,
                    "dataset_path": str(dataset_result.dataset_path),
                    "status": _map_status(status.status),
                    "message": status.message,
                    "pairs_created": dataset_result.pair_count,
                    "diagnostics": dataset_result.diagnostics,
                }
            )
        except Exception as exc:  # noqa: BLE001
            profile_entry.update({"status": "failed", "error": str(exc)})
            run_record["profiles"].append(profile_entry)
            if request.fail_fast:
                run_record["overall_status"] = "failed"
                break
            continue

        run_record["profiles"].append(profile_entry)

        if request.fail_fast and profile_entry.get("status") == "failed":
            run_record["overall_status"] = "failed"
            break

    if not run_record["profiles"]:
        run_record["overall_status"] = "failed"
        _persist_training_run_record(run_dir, run_record)
        raise HTTPException(status_code=400, detail="No finetune jobs started")

    _refresh_training_run_record(run_record, manager)
    run_record["overall_status"] = _derive_overall_status(run_record)
    if run_record["overall_status"] in {"failed", "partial", "succeeded"}:
        run_record["finished_at"] = run_record.get("finished_at") or time.time()
    _persist_training_run_record(run_dir, run_record)

    return _build_training_summary(run_record, total_profiles_discovered=len(discovered_profiles))


@api_router.get("/finetune/status/{job_id}", tags=["Finetuning"])
def finetune_status(job_id: str):
    manager = get_finetune_manager()
    status = manager.get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status.dict()


def _collect_available_models() -> List[ModelInfo]:
    models: Dict[str, ModelInfo] = {}
    try:
        for entry in list_models():
            name = entry.get("name") or entry.get("model")
            if not name:
                continue
            models[name] = ModelInfo(
                model=name,
                source=entry.get("backend") or "finetune",
                backend=entry.get("backend"),
                profile_id=entry.get("profile_id"),
                path=entry.get("path") or entry.get("model_path"),
                updated_at=entry.get("updated_at"),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to list managed models: %s", exc)

    try:
        raw = ollama.list().model_dump()
        for entry in raw.get("models", []):
            name = entry.get("model") or entry.get("name")
            if not name:
                continue
            models.setdefault(
                name,
                ModelInfo(
                    model=name,
                    source="ollama",
                    backend="ollama",
                    size=entry.get("size"),
                    digest=entry.get("digest"),
                    updated_at=entry.get("modified_at"),
                ),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to list Ollama models: %s", exc)

    models.setdefault("gemini-2.5-flash", ModelInfo(model="gemini-2.5-flash", source="gemini", backend="gemini"))
    return list(models.values())


@api_router.get("/models", tags=["Default"], response_model=ModelsResponse)
def list_available_models():
    return ModelsResponse(models=_collect_available_models())


@api_router.delete("/document/{doc_id}/embeddings", tags=["Default"])
def delete_document_embeddings_api(
        doc_id: str,
        subscription_id: str = "default",
        profile_id: Optional[str] = None
):
    """
    API endpoint to manually delete embeddings for a specific document.
    This is useful when a document is marked as DELETED in MongoDB.
    """
    try:
        from src.api.dataHandler import delete_embeddings, db, Config
        from bson.objectid import ObjectId

        # Fetch document details from MongoDB
        doc = db[Config.MongoDB.DOCUMENTS].find_one({"_id": ObjectId(doc_id)})

        if not doc:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        # Get subscription_id from document (override query param if present in doc)
        subscription_id = str(
            doc.get('subscription') or
            doc.get('subscriptionId') or
            subscription_id
        )
        profile_id = str(
            profile_id
            or doc.get('profile')
            or doc.get('profile_id')
            or ""
        )
        if not profile_id:
            raise HTTPException(status_code=400, detail="profile_id is required for deletion")

        logging.info(
            f"[API] Deleting embeddings for doc={doc_id}, "
            f"subscription={subscription_id}, "
            f"profile={profile_id}, "
            f"status={doc.get('status')}"
        )

        # Delete embeddings scoped to subscription and profile
        result = delete_embeddings(
            subscription_id=subscription_id,
            profile_id=profile_id,
            document_id=doc_id
        )

        if result["status"] == "success":
            logging.info(f"[API] Successfully deleted embeddings for document {doc_id}")
            return {
                "status": "success",
                "document_id": doc_id,
                "subscription_id": subscription_id,
                "profile_id": profile_id,
                "message": result.get("message", "Embeddings deleted"),
                "details": result
            }
        elif result["status"] == "not_found":
            # Not an error - document just has no embeddings
            return {
                "status": "success",
                "document_id": doc_id,
                "profile_id": profile_id,
                "message": result.get("message", "No embeddings found to delete"),
                "details": result
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Unknown error during deletion")
            )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[API] Failed to delete embeddings: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete embeddings: {str(e)}"
        )


@api_router.get("/metrics", tags=["Default"])
def get_metrics(
    days: int = Query(..., gt=0),
    tz: str = "UTC",
    week_start: str = "MON",
    document_id: Optional[str] = None,
    model_id: Optional[str] = None,
    agent: Optional[str] = None,
    tool: Optional[str] = None,
):
    """API endpoint to retrieve model/retrieval performance statistics."""
    try:
        tzinfo = normalize_timezone(tz)
        if tz and tzinfo == dt.timezone.utc and tz.upper() != "UTC":
            raise HTTPException(status_code=400, detail="Invalid timezone")
        week_start_norm = week_start.strip().upper()
        if week_start_norm not in {"MON", "SUN"}:
            raise HTTPException(status_code=400, detail="week_start must be MON or SUN")
        week_start_idx = normalize_week_start(week_start_norm)

        start_date, end_date, range_start, range_end = compute_date_range(days, tzinfo)
        daily = daily_boundaries(start_date, days, tzinfo)
        weekly = weekly_boundaries(daily, week_start_idx)

        start_utc = daily[0][1]
        end_utc = daily[-1][2]

        store = get_metrics_store()
        cache_ttl = int(os.getenv("METRICS_CACHE_TTL", "60"))
        cache_key = None
        if store.redis and cache_ttl > 0:
            cache_key = (
                "ai:metrics:snapshot:"
                f"days={days}:tz={tz}:week_start={week_start_norm}:"
                f"doc={document_id or 'all'}:model={model_id or 'all'}:"
                f"agent={agent or 'all'}:tool={tool or 'all'}"
            )
            cached = store.redis.get(cache_key)
            if cached:
                try:
                    return json.loads(cached)
                except Exception:
                    pass

        repo = MetricsRepository(store)
        hourly = repo.fetch_hourly(
            start_utc,
            end_utc,
            document_id=document_id,
            model_id=model_id,
            agent=agent,
            tool=tool,
        )

        daily_payload = []
        for date_str, day_start, day_end in daily:
            aggregate = aggregate_range(hourly, day_start, day_end)
            payload = build_metrics_payload(aggregate, latency_buckets=store.latency_buckets_ms)
            payload["date"] = date_str
            daily_payload.append(payload)

        weekly_payload = []
        for week_start_date, week_end_date, week_start_utc, week_end_utc in weekly:
            aggregate = aggregate_range(hourly, week_start_utc, week_end_utc)
            payload = build_metrics_payload(aggregate, latency_buckets=store.latency_buckets_ms)
            payload["week_start_date"] = week_start_date
            payload["week_end_date"] = week_end_date
            weekly_payload.append(payload)

        response = {
            "meta": {
                "days": days,
                "tz": tz,
                "week_start": week_start_norm,
                "range_start": range_start.isoformat(),
                "range_end": range_end.isoformat(),
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
            "daily": daily_payload,
            "weekly": weekly_payload,
        }
        if cache_key and store.redis and cache_ttl > 0:
            try:
                store.redis.setex(cache_key, cache_ttl, json.dumps(response))
            except Exception:
                pass
        return response
    except Exception as e:
        logging.error(f"Failed to retrieve metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

'''
@api_router.get("/chat-history/{user_id}")
def get_chat_history_api(user_id: str):
    """API endpoint to retrieve full chat history for a user."""
    try:
        history = get_chat_history(user_id)
        if history is None:
            raise HTTPException(status_code=404, detail="Chat history not found")

        logging.info(f"[CHAT HISTORY] User: {user_id}")
        return {"user_id": user_id, "chat_history": history}
    except Exception as e:
        logging.error(f"Failed to retrieve chat history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")
'''

@api_router.get("/chat-history/{user_id}", tags=["Default"])
def get_chat_history_api(user_id: str, subscription_id: str = "default"):
    """
    Returns all chat sessions for a user in the structure expected by NestJS.
    """
    try:
        history = get_chat_history(user_id)
        if not history:
            raise HTTPException(status_code=404, detail="No chat history found")

        # Ensure proper structure
        if isinstance(history, dict) and "sessions" not in history:
            # convert legacy history format to session-style
            sessions = [{
                "session_id": "default",
                "title": history[0]["query"] if isinstance(history, list) and history else "Chat Session",
                "messages": history
            }]
            formatted = {"sessions": sessions}
        elif isinstance(history, list):
            # list of sessions already
            formatted = {"sessions": history}
        else:
            formatted = history  # already correct format

        logging.info(f"[CHAT HISTORY] User: {user_id}, Sessions: {len(formatted['sessions'])}")
        return formatted
    except Exception as e:
        logging.error(f"Failed to retrieve chat history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")

@api_router.get("/sessions/{user_id}", tags=["Default"])
def get_sessions_api(user_id: str, subscription_id: str = "default"):
    """API endpoint to get list of all sessions (for sidebar)."""
    try:
        sessions = get_session_list(user_id)
        logging.info(f"[SESSIONS] User: {user_id}, Count: {len(sessions)}")
        return {"user_id": user_id, "sessions": sessions}
    except Exception as e:
        logging.error(f"Failed to retrieve sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions")


@api_router.get("/session/{user_id}/{session_id}", tags=["Default"])
def get_session_api(user_id: str, session_id: str, subscription_id: str = "default"):
    """API endpoint to get a specific session's messages."""
    try:
        session = get_session_by_id(user_id, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        logging.info(f"[SESSION] User: {user_id}, Session: {session_id}")
        return {"user_id": user_id, "session": session}
    except Exception as e:
        logging.error(f"Failed to retrieve session: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session")


@api_router.delete("/chat-history/{user_id}", tags=["Default"])
def delete_chat_history_api(user_id: str, subscription_id: str = "default"):
    """API endpoint to delete all chat history for a user."""
    try:
        result = delete_chat_history(user_id)
        return {"user_id": user_id, "message": result}
    except Exception as e:
        logging.error(f"Failed to delete chat history: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete chat history")


@api_router.delete("/session/{user_id}/{session_id}", tags=["Default"])
def delete_session_api(user_id: str, session_id: str, subscription_id: str = "default"):
    """API endpoint to delete a specific session."""
    try:
        result = delete_session(user_id, session_id)
        return {"user_id": user_id, "session_id": session_id, "result": result}
    except Exception as e:
        logging.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session")

'''
@api_router.get("/pii/{doc_id}")
def get_pii_info(doc_id: str, subscription_id: str = "default"):
    """API endpoint to retrieve PII masking stats for a document."""
    try:
        stats = get_pii_stats(doc_id)
        if not stats:
            raise HTTPException(status_code=404, detail=f"Document not found for id {doc_id}")
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to retrieve PII stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve PII stats")
'''

'-------------added by maha/maria ------------------'
'--------------for PII setting management ------------------'

class PIISettingUpdate(BaseModel):
    pii_enabled: bool


@api_router.get("/subscription/{subscription_id}/pii-setting", tags=["Subscriptions"])
def get_pii_setting(subscription_id: str):
    """
    Get current PII masking setting for a subscription
    """
    try:
        pii_enabled = get_subscription_pii_setting(subscription_id)
        return {
            "subscription_id": subscription_id,
            "pii_enabled": pii_enabled,
            "message": f"PII masking is {'ENABLED' if pii_enabled else 'DISABLED'}"
        }
    except Exception as e:
        logging.error(f"Failed to get PII setting: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve PII setting")


@api_router.put("/subscription/{subscription_id}/pii-setting", tags=["Subscriptions"])
def update_pii_setting(subscription_id: str, setting: PIISettingUpdate):
    """
    Update PII masking setting for a subscription
    Body:
    {
        "pii_enabled": true/false
    }
    """
    try:
        subscriptions_collection = getattr(Config.MongoDB, 'SUBSCRIPTIONS', 'subscriptions')
        collection = db[subscriptions_collection]
        # Find subscription
        subscription = None
        if ObjectId.is_valid(subscription_id):
            subscription = collection.find_one({"_id": ObjectId(subscription_id)})
        if not subscription:
            subscription = collection.find_one({"subscriptionId": subscription_id})
        if not subscription:
            subscription = collection.find_one({"_id": subscription_id})
        if not subscription:
            raise HTTPException(
                status_code=404,
                detail=f"Subscription {subscription_id} not found"
            )
        # Update PII setting
        filter_criteria = {"_id": subscription["_id"]}
        update_operation = {
            "$set": {
                "pii_enabled": setting.pii_enabled,
                "pii_updated_at": time.time()
            }
        }
        result = collection.update_one(filter_criteria, update_operation)
        if result.modified_count > 0:
            logging.info(f"Updated PII setting for subscription {subscription_id}: pii_enabled={setting.pii_enabled}")
            return {
                "status": "success",
                "subscription_id": subscription_id,
                "pii_enabled": setting.pii_enabled,
                "message": f"PII masking is now {'ENABLED' if setting.pii_enabled else 'DISABLED'}"
            }
        else:
            return {
                "status": "no_change",
                "subscription_id": subscription_id,
                "pii_enabled": setting.pii_enabled,
                "message": "PII setting was already at the requested value"
            }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to update PII setting: {e}")
        raise HTTPException(status_code=500, detail="Failed to update PII setting")


@api_router.post("/subscription/{subscription_id}/reprocess-documents", tags=["Subscriptions"])
def reprocess_documents_with_new_pii_setting(subscription_id: str):
    """
    Reprocess all documents in a subscription with updated PII setting
    This will re-extract documents with new PII masking rules
    """
    try:
        # Get all documents for this subscription
        documents_collection = db[Config.MongoDB.DOCUMENTS]
        # Update all documents in this subscription to UNDER_REVIEW
        # so they get reprocessed with new PII setting
        result = documents_collection.update_many(
            {
                "$or": [
                    {"subscriptionId": subscription_id},
                    {"subscription_id": subscription_id},
                    {"subscription": subscription_id}
                ],
                "status": {"$in": ["TRAINING_COMPLETED", "TRAINING_PARTIALLY_COMPLETED"]}
            },
            {
                "$set": {
                    "status": "UNDER_REVIEW",
                    "reprocess_reason": "PII setting changed",
                    "reprocess_timestamp": time.time()
                }
            }
        )
        logging.info(f"Marked {result.modified_count} documents for reprocessing")
        # Trigger extraction
        training_result = trainData()
        return {
            "status": "success",
            "subscription_id": subscription_id,
            "documents_marked_for_reprocessing": result.modified_count,
            "extraction_result": training_result
        }
    except Exception as e:
        logging.error(f"Failed to reprocess documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to reprocess documents")


app.include_router(api_router)
app.include_router(knowledge_graph_router)
app.add_api_route("/ask", ask_question_api, methods=["POST"], include_in_schema=False)
app.add_api_route("/askStream", ask_question_stream_api, methods=["POST"], include_in_schema=False)
app.add_api_route("/teams/messages", handle_teams_messages, methods=["POST"], include_in_schema=False)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
