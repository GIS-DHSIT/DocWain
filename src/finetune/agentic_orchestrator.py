import hashlib
import json
from src.utils.logging_utils import get_logger
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import ollama
from pydantic import BaseModel, Field, ValidationError
from qdrant_client import QdrantClient

from src.api.config import Config
from src.finetune.dataset_builder import _sample_chunks, build_dataset_from_qdrant
from src.finetune.models import CollectionOnlyFinetuneRequest, FinetuneRequest
from src.finetune.unsloth_trainer import get_finetune_manager
from src.training.qdrant_profile_discovery import discover_profile_ids_from_collection
from src.utils.payload_utils import get_source_name

logger = get_logger(__name__)

class OllamaModelMissing(Exception):
    """Raised when the required Ollama model is not available locally."""

class OllamaUnavailable(Exception):
    """Raised when the Ollama server is unreachable."""

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def create_nemotron_client(model_name: Optional[str] = None):
    """
    Build a minimal Ollama chat client for Nemotron orchestration with enforced JSON responses.
    Prefers the native ollama client used elsewhere in DocWain to avoid new dependencies.
    """
    target_model = model_name or Config.Finetune.ORCHESTRATOR_MODEL
    host = os.getenv("OLLAMA_HOST", Config.Finetune.OLLAMA_HOST)
    client = ollama.Client(host=host)

    class _NemotronJSONClient:
        def __init__(self, _client, _model):
            self._client = _client
            self.model = _model

        def chat(self, prompt: str) -> str:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return STRICT JSON only. No prose."},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0, "top_p": 0.9, "format": "json"},
            )
            message = response.get("message") or {}
            return (message.get("content") or "").strip()

    return _NemotronJSONClient(client, target_model)

class DatasetPlan(BaseModel):
    sampling_strategy: Optional[str] = None
    max_points: int = Field(120, ge=1)
    questions_per_chunk: int = Field(2, ge=1, le=10)
    dedup_strategy: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None

class TrainingPlan(BaseModel):
    lora_r: int = Field(16, ge=1)
    lora_alpha: int = Field(16, ge=1)
    lora_dropout: float = Field(0.05, ge=0.0, le=1.0)
    batch_size: int = Field(4, ge=1)
    max_steps: int = Field(200, ge=1)
    learning_rate: float = Field(2e-4, gt=0)
    gradient_accumulation: int = Field(2, ge=1)

class RiskFlag(BaseModel):
    code: str
    severity: str = Field("info", description="info|low|medium|high")
    mitigation: Optional[str] = None

class AcceptanceCriteria(BaseModel):
    min_examples: int = Field(5, ge=1)
    duplicate_ratio_max: float = Field(0.5, ge=0.0, le=1.0)
    require_metadata: bool = True

class FinetunePlan(BaseModel):
    dataset_plan: DatasetPlan = Field(default_factory=DatasetPlan)
    training_plan: TrainingPlan = Field(default_factory=TrainingPlan)
    risk_flags: List[RiskFlag] = Field(default_factory=list)
    acceptance_criteria: AcceptanceCriteria = Field(default_factory=AcceptanceCriteria)

class FinetuneState(BaseModel):
    stage: str
    notes: Optional[str] = None
    started_at: float = Field(default_factory=lambda: datetime.utcnow().timestamp())
    finished_at: Optional[float] = None

class AgenticFinetuneRequest(BaseModel):
    collection_name: str
    profile_ids: Optional[List[str]] = None
    agentic: bool = False
    orchestrator_model: Optional[str] = None

class NemotronOrchestratorAgent:
    """Lightweight planner using Nemotron-3-Nano via Ollama."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or Config.Finetune.ORCHESTRATOR_MODEL
        self.client = create_nemotron_client(self.model_name)

    def plan(self, collection_name: str, profiles: Sequence[str], sample_chunks: Sequence[Dict[str, Any]]) -> FinetunePlan:
        prompt = self._build_prompt(collection_name, profiles, sample_chunks)
        raw = self.client.chat(prompt)
        try:
            data = json.loads(raw)
            return FinetunePlan.model_validate(data)
        except (ValidationError, json.JSONDecodeError) as exc:
            logger.debug("Nemotron plan parse failed; using defaults: %s | raw=%s", exc, raw)
            return FinetunePlan()

    @staticmethod
    def _build_prompt(collection_name: str, profiles: Sequence[str], sample_chunks: Sequence[Dict[str, Any]]) -> str:
        examples = []
        for ch in sample_chunks[:10]:
            meta = ch.get("metadata") or {}
            examples.append(
                {
                    "source_name": get_source_name(meta),
                    "document_id": meta.get("document_id"),
                    "chunk_index": meta.get("chunk_index"),
                    "profile_id": meta.get("profile_id"),
                    "preview": (ch.get("text") or "")[:280],
                }
            )
        return json.dumps(
            {
                "instruction": "Propose a JSON-only finetune plan. Keys: dataset_plan, training_plan, risk_flags, acceptance_criteria.",
                "collection_name": collection_name,
                "profiles": list(profiles),
                "samples": examples,
                "constraints": {
                    "temperature": 0,
                    "grounded_only": True,
                    "max_steps": Config.Finetune.AGENT_MAX_STEPS,
                },
            }
        )

class AgenticFinetuneOrchestrator:
    """Agentic fine-tune runner scoped to a single Qdrant collection."""

    def __init__(self, client: Optional[QdrantClient] = None):
        self.client = client or QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=120)
        self.manager = get_finetune_manager()
        self.ollama_host = os.getenv("OLLAMA_HOST", Config.Finetune.OLLAMA_HOST)

    def is_enabled(self, request_agentic: bool) -> bool:
        return request_agentic or bool(Config.Finetune.AGENTIC_ENABLED)

    def run(self, request: CollectionOnlyFinetuneRequest, job_id: Optional[str] = None) -> Dict[str, Any]:
        collection = request.collection_name
        orchestrator_model = request.orchestrator_model or Config.Finetune.ORCHESTRATOR_MODEL
        job_id = job_id or str(uuid.uuid4())
        snapshot = self._init_snapshot(collection)
        artifact_root = Path(Config.Path.APP_HOME) / "finetune_artifacts" / collection
        artifact_root.mkdir(parents=True, exist_ok=True)

        try:
            self._ensure_model_available(orchestrator_model)
        except OllamaUnavailable as exc:
            raise
        except OllamaModelMissing as exc:
            raise

        profile_ids = request.profile_ids or self._discover_profiles(collection, snapshot)
        if not profile_ids:
            raise ValueError(f"No profile_ids found in collection {collection}")

        sample_chunks = self._plan_samples(collection, profile_ids[:3])
        agent = NemotronOrchestratorAgent(orchestrator_model)
        plan = agent.plan(collection, profile_ids, sample_chunks)

        dataset_stats: List[Dict[str, Any]] = []
        profiles: List[Dict[str, Any]] = []
        train_args_dump: List[Dict[str, Any]] = []
        for profile_id in profile_ids:
            profile_dir = artifact_root / profile_id / job_id
            profile_dir.mkdir(parents=True, exist_ok=True)
            try:
                dataset_result = build_dataset_from_qdrant(
                    profile_id=profile_id,
                    subscription_id=collection,
                    collection_name=collection,
                    max_points=plan.dataset_plan.max_points,
                    questions_per_chunk=plan.dataset_plan.questions_per_chunk,
                    generation_model=request.generation_model,
                    client=self.client,
                    run_id=job_id,
                )
                if dataset_result.status != "success" or not dataset_result.dataset_path:
                    raise ValueError(f"Dataset generation {dataset_result.status} for profile {profile_id}")
                stats = self._validate_dataset(dataset_result.dataset_path, collection, profile_id, plan.acceptance_criteria)
                dataset_stats.append(stats)

                payload = self._build_finetune_payload(
                    request=request,
                    profile_id=profile_id,
                    dataset_path=str(dataset_result.dataset_path),
                    plan=plan.training_plan,
                    snapshot=snapshot,
                    orchestrator_model=orchestrator_model,
                )
                payload["training_run_id"] = job_id
                train_args_dump.append(payload)
                status = self.manager.start_job(FinetuneRequest(**payload))
                profiles.append(
                    {
                        "profile_id": profile_id,
                        "job_id": status.job_id,
                        "dataset_path": str(dataset_result.dataset_path),
                        "status": status.status if status.status != "completed" else "succeeded",
                        "message": status.message,
                    }
                )
                self._persist_json(profile_dir / "plan.json", plan.model_dump())
                self._persist_json(profile_dir / "snapshot.json", snapshot)
                self._persist_json(profile_dir / "dataset_stats.json", stats)
                self._persist_json(profile_dir / "train_args.json", payload)
                self._persist_report(profile_dir / "run_report.md", [profiles[-1]], plan, snapshot)
            except Exception as exc:  # noqa: BLE001
                profiles.append({"profile_id": profile_id, "status": "failed", "error": str(exc)})
                self._persist_json(profile_dir / "plan.json", plan.model_dump())
                self._persist_json(profile_dir / "snapshot.json", snapshot)
                self._persist_report(profile_dir / "run_report.md", [profiles[-1]], plan, snapshot)
                if request.fail_fast:
                    break

        snapshot["profile_ids_found"] = profile_ids
        summary_dir = artifact_root / "_collection" / job_id
        summary_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = summary_dir / "snapshot.json"
        plan_path = summary_dir / "plan.json"
        dataset_stats_path = summary_dir / "dataset_stats.json"
        self._persist_json(snapshot_path, snapshot)
        self._persist_json(plan_path, plan.model_dump())
        self._persist_json(dataset_stats_path, {"profiles": dataset_stats})
        self._persist_json(summary_dir / "train_args.json", {"runs": train_args_dump})
        self._persist_report(summary_dir / "run_report.md", profiles, plan, snapshot)

        overall_status = self._derive_overall_status(profiles)
        return {
            "training_run_id": job_id,
            "collection_name": collection,
            "base_model": request.base_model,
            "run_name": request.run_name,
            "config_hash": self._config_hash(request),
            "total_profiles_discovered": len(profile_ids),
            "overall_status": overall_status,
            "fail_fast": request.fail_fast,
            "profiles": profiles,
            "plan_path": str(plan_path),
            "snapshot_path": str(snapshot_path),
            "dataset_stats_path": str(dataset_stats_path),
        }

    def _ensure_model_available(self, model_name: str) -> None:
        try:
            client = ollama.Client(host=self.ollama_host)
            tags = client.list()
            models = [m.get("name") or m.get("model") for m in tags.get("models", [])]
            normalized = {str(m).split(":")[0] for m in models if m}
            if model_name not in normalized and model_name not in models:
                raise OllamaModelMissing(f"Ollama model {model_name} not found. Run: ollama pull {model_name}")
        except OllamaModelMissing:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama check failed: %s", exc)
            raise OllamaUnavailable("Ollama host unreachable") from exc

    def _discover_profiles(self, collection_name: str, snapshot: Dict[str, Any]) -> List[str]:
        try:
            ids = discover_profile_ids_from_collection(
                client=self.client,
                collection_name=collection_name,
                max_points=512,
            )
            snapshot["points_scanned"] = len(ids)
            snapshot["scroll_checkpoints"] = snapshot.get("scroll_checkpoints") or []
            snapshot["profile_ids_found"] = ids
            return ids
        except Exception as exc:  # noqa: BLE001
            logger.warning("Profile discovery failed for %s: %s", collection_name, exc)
            return []

    def _plan_samples(self, collection_name: str, profile_ids: Sequence[str]) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for pid in profile_ids:
            try:
                samples.extend(
                    _sample_chunks(
                        profile_id=pid,
                        subscription_id=collection_name,
                        collection_name=collection_name,
                        limit=8,
                        client=self.client,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Sample collection failed for profile %s: %s", pid, exc)
        return samples

    @staticmethod
    def _init_snapshot(collection: str) -> Dict[str, Any]:
        return {
            "collection_name": collection,
            "captured_at": _now_iso(),
            "points_scanned": 0,
            "scroll_checkpoints": [],
            "profile_ids_found": [],
        }

    def _validate_dataset(
        self,
        dataset_path: Path,
        collection_name: str,
        profile_id: str,
        acceptance: AcceptanceCriteria,
    ) -> Dict[str, Any]:
        stats = {
            "dataset_path": str(dataset_path),
            "collection_name": collection_name,
            "profile_id": profile_id,
            "total_rows": 0,
            "unique_rows": 0,
            "duplicates_removed": 0,
            "missing_metadata": 0,
            "pii_removed": 0,
        }
        unique_hashes = set()
        cleaned_lines: List[str] = []
        with Path(dataset_path).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                stats["total_rows"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                metadata = row.get("metadata") or {}
                required_keys = {
                    "collection_name": collection_name,
                    "profile_id": profile_id,
                    "document_id": metadata.get("document_id"),
                    "source_name": get_source_name(metadata),
                    "chunk_id": metadata.get("chunk_id"),
                    "page": metadata.get("page"),
                    "chunk_index": metadata.get("chunk_index"),
                }
                if acceptance.require_metadata and (metadata.get("collection_name") != collection_name or metadata.get("profile_id") != profile_id):
                    stats["missing_metadata"] += 1
                    continue
                if any(v is None for v in required_keys.values()):
                    stats["missing_metadata"] += 1
                    continue
                identity = hashlib.sha256(
                    f"{row.get('instruction')}|{row.get('output')}|{row.get('input')}|{metadata.get('chunk_id')}".encode(
                        "utf-8"
                    )
                ).hexdigest()
                if identity in unique_hashes:
                    stats["duplicates_removed"] += 1
                    continue
                if self._looks_like_pii(row):
                    stats["pii_removed"] += 1
                    continue
                unique_hashes.add(identity)
                cleaned_lines.append(json.dumps(row, ensure_ascii=False))

        if cleaned_lines:
            stats["unique_rows"] = len(cleaned_lines)
            Path(dataset_path).write_text("\n".join(cleaned_lines) + "\n", encoding="utf-8")
        duplicate_ratio = 0.0
        if stats["total_rows"]:
            duplicate_ratio = stats["duplicates_removed"] / float(stats["total_rows"])
        stats["duplicate_ratio"] = duplicate_ratio
        if stats["unique_rows"] < acceptance.min_examples:
            raise ValueError(f"Dataset too small after validation ({stats['unique_rows']} rows)")
        if duplicate_ratio > acceptance.duplicate_ratio_max:
            raise ValueError(f"Dataset duplicate ratio {duplicate_ratio:.2f} exceeds threshold")
        return stats

    @staticmethod
    def _looks_like_pii(row: Dict[str, Any]) -> bool:
        text = " ".join([str(row.get("instruction") or ""), str(row.get("output") or "")]).lower()
        return any(token in text for token in ["ssn", "social security", "password:", "api_key", "apikey", "secret"])

    def _build_finetune_payload(
        self,
        request: CollectionOnlyFinetuneRequest,
        profile_id: str,
        dataset_path: str,
        plan: TrainingPlan,
        snapshot: Dict[str, Any],
        orchestrator_model: str,
    ) -> Dict[str, Any]:
        payload = request.finetune_payload(profile_id=profile_id, dataset_path=dataset_path)
        payload.update(
            {
                "learning_rate": plan.learning_rate,
                "max_steps": plan.max_steps,
                "batch_size": plan.batch_size,
                "gradient_accumulation": plan.gradient_accumulation,
                "lora_r": plan.lora_r,
                "lora_alpha": plan.lora_alpha,
                "lora_dropout": plan.lora_dropout,
                "agentic": True,
                "orchestrator_model": orchestrator_model,
                "qdrant_snapshot": snapshot,
                "collection_name": request.collection_name,
            }
        )
        return payload

    @staticmethod
    def _derive_overall_status(profiles: List[Dict[str, Any]]) -> str:
        if not profiles:
            return "failed"
        if any(p.get("status") == "failed" for p in profiles):
            return "partial" if any(p.get("status") == "succeeded" for p in profiles) else "failed"
        if all(p.get("status") in {"succeeded", "completed", "running", "queued"} for p in profiles):
            return "running" if any(p.get("status") in {"running", "queued"} for p in profiles) else "succeeded"
        return "partial"

    @staticmethod
    def _persist_json(path: Path, data: Dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2, default=str))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist %s: %s", path, exc)

    @staticmethod
    def _persist_report(path: Path, profiles: List[Dict[str, Any]], plan: FinetunePlan, snapshot: Dict[str, Any]) -> None:
        lines = [
            f"# Agentic finetune run",
            f"- Captured: {snapshot.get('captured_at')}",
            f"- Collection: {snapshot.get('collection_name')}",
            f"- Profiles: {', '.join([p.get('profile_id') for p in profiles if p.get('profile_id')])}",
            "",
            "## Plan",
            json.dumps(plan.model_dump(), indent=2),
            "",
            "## Profiles",
        ]
        for p in profiles:
            lines.append(f"- {p.get('profile_id')}: {p.get('status')} ({p.get('message') or p.get('error') or ''})")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist report %s: %s", path, exc)

    @staticmethod
    def _config_hash(request: CollectionOnlyFinetuneRequest) -> str:
        data = {
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
        encoded = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
