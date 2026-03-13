import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from qdrant_client import QdrantClient

from src.api.config import Config
from src.api.vector_store import build_collection_name
from src.finetune.dataset_builder import build_dataset_from_qdrant
from src.finetune.models import AutoFinetuneRunRequest, FinetuneRequest, FinetuneStatus
from src.finetune.qdrant_discovery import list_collections, list_profile_ids
from src.finetune.unsloth_trainer import get_finetune_manager
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_AUTO_ORCHESTRATOR = None

def get_auto_orchestrator():
    global _AUTO_ORCHESTRATOR
    if _AUTO_ORCHESTRATOR is None:
        _AUTO_ORCHESTRATOR = AutoFinetuneOrchestrator()
    return _AUTO_ORCHESTRATOR

@dataclass
class AutoFinetunePolicy:
    min_points: int = getattr(Config.Finetune, "AUTO_MIN_POINTS", 40)
    min_records: int = getattr(Config.Finetune, "AUTO_MIN_RECORDS", 20)
    max_profiles_per_run: int = getattr(Config.Finetune, "AUTO_MAX_PROFILES_PER_RUN", 10)
    interval_hours: float = getattr(Config.Finetune, "AUTO_INTERVAL_HOURS", 6)

class AutoFinetuneOrchestrator:
    def __init__(self, client: Optional[QdrantClient] = None, policy: Optional[AutoFinetunePolicy] = None):
        self.client = client or QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API, timeout=120)
        self.policy = policy or AutoFinetunePolicy()
        self.lock = threading.Lock()
        self.running = False
        self.last_runs: Dict[str, FinetuneStatus] = {}
        self.scheduler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

    def _mark_running(self, value: bool):
        with self.lock:
            self.running = value

    def status(self) -> List[FinetuneStatus]:
        with self.lock:
            return list(self.last_runs.values())

    def run_for_all_collections(self, request: AutoFinetuneRunRequest) -> List[FinetuneStatus]:
        if self.running and not request.dry_run:
            return [
                FinetuneStatus(
                    job_id="skipped",
                    profile_id="*",
                    status="skipped",
                    message="Auto finetune already running",
                    params=request.model_dump(),
                )
            ]
        self._mark_running(True)
        collections = [request.collection_name] if request.collection_name else list_collections(self.client)
        results: List[FinetuneStatus] = []
        try:
            for collection in collections:
                results.extend(self.run_for_collection(collection, request))
            return results
        finally:
            self._mark_running(False)

    def run_for_collection(self, collection_name: str, request: AutoFinetuneRunRequest) -> List[FinetuneStatus]:
        stats = list_profile_ids(collection_name, client=self.client)
        counts = stats.get("counts", {})
        profile_ids = stats.get("profile_ids", [])
        if request.profile_id:
            profile_ids = [pid for pid in profile_ids if pid == request.profile_id]
        if request.max_profiles_per_run:
            profile_ids = profile_ids[: request.max_profiles_per_run]
        elif self.policy.max_profiles_per_run:
            profile_ids = profile_ids[: self.policy.max_profiles_per_run]

        results: List[FinetuneStatus] = []
        for profile_id in profile_ids:
            profile_count = counts.get(profile_id, 0) if counts else 0
            if counts and profile_count < self.policy.min_points:
                results.append(
                    FinetuneStatus(
                        job_id="skipped",
                        profile_id=profile_id,
                        status="skipped",
                        message=f"Not enough points ({profile_count})",
                        params=request.model_dump(),
                    )
                )
                continue
            results.append(self.run_for_profile(collection_name, profile_id, request, profile_count))
        return results

    def run_for_profile(
        self,
        subscription_id: str,
        profile_id: str,
        request: AutoFinetuneRunRequest,
        profile_point_count: int,
    ) -> FinetuneStatus:
        correlation_id = f"auto-finetune-{subscription_id}-{profile_id}-{int(time.time())}"
        log = get_logger(__name__, correlation_id)
        manager = get_finetune_manager()

        if request.dry_run:
            status = FinetuneStatus(
                job_id="dry-run",
                profile_id=profile_id,
                status="planned",
                message=f"Would finetune {profile_id} in {subscription_id}",
                params=request.model_dump(),
            )
            self.last_runs[f"{subscription_id}:{profile_id}"] = status
            return status

        dataset_result = build_dataset_from_qdrant(
            profile_id=profile_id,
            subscription_id=subscription_id,
            max_points=request.max_points,
            questions_per_chunk=request.questions_per_chunk,
            generation_model=request.generation_model,
            client=self.client,
            run_id=correlation_id,
        )
        if dataset_result.status != "success" or not dataset_result.dataset_path:
            status = FinetuneStatus(
                job_id="skipped",
                profile_id=profile_id,
                status="skipped",
                message=f"Dataset generation {dataset_result.status}",
                params=request.model_dump(),
            )
            self.last_runs[f"{subscription_id}:{profile_id}"] = status
            return status

        with dataset_result.dataset_path.open("r", encoding="utf-8") as dataset_file:
            record_count = sum(1 for _ in dataset_file)
        if record_count < self.policy.min_records:
            status = FinetuneStatus(
                job_id="skipped",
                profile_id=profile_id,
                status="skipped",
                message=f"Dataset too small ({record_count} records)",
                params=request.model_dump(),
            )
            self.last_runs[f"{subscription_id}:{profile_id}"] = status
            return status

        finetune_request = FinetuneRequest(
            profile_id=profile_id,
            dataset_path=str(dataset_result.dataset_path),
            include_actual_data=False,
            qdrant_snapshot={
                "collection": build_collection_name(subscription_id),
                "points": profile_point_count,
            },
            training_run_id=correlation_id,
        )
        status = manager.start_job(finetune_request)
        log.info("Auto finetune scheduled for %s/%s", subscription_id, profile_id)
        self.last_runs[f"{subscription_id}:{profile_id}"] = status
        return status

    def start_scheduler(self):
        if not getattr(Config.Finetune, "AUTO_ENABLED", False):
            logger.info("Auto finetune scheduler disabled via configuration")
            return
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            return

        def _loop():
            while not self.stop_event.is_set():
                if not self.running:
                    try:
                        self._mark_running(True)
                        request = AutoFinetuneRunRequest(dry_run=False)
                        self.run_for_all_collections(request)
                    except Exception as exc:
                        logger.error("Auto finetune scheduler run failed: %s", exc, exc_info=True)
                    finally:
                        self._mark_running(False)
                interval = max(1.0, float(getattr(Config.Finetune, "AUTO_INTERVAL_HOURS", 6)))
                self.stop_event.wait(interval * 3600)

        self.scheduler_thread = threading.Thread(target=_loop, daemon=True)
        self.scheduler_thread.start()
