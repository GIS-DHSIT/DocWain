"""Model management API — update, fine-tune trigger, and model status endpoints."""

from __future__ import annotations

import subprocess
import time
import json
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

model_router = APIRouter(prefix="/model", tags=["Model Management"])

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ModelUpdateRequest(BaseModel):
    model_name: str = Field(default="DHS/DocWain", description="Ollama model name to update")
    base_model: str = Field(default="qwen3:14b", description="Base model to build from")


class FinetuneCheckResponse(BaseModel):
    profiles_checked: int = 0
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    healthy: List[str] = Field(default_factory=list)
    finetune_needed: bool = False


class AutoFinetuneRequest(BaseModel):
    collection_name: Optional[str] = Field(None, description="Specific collection. If empty, auto-discovers.")
    base_model: str = Field(default="qwen3", description="Base model for fine-tuning")
    max_steps: int = Field(default=200)
    learning_rate: float = Field(default=2e-4)
    lora_r: int = Field(default=16)
    force: bool = Field(default=False, description="Force fine-tune even if feedback signals are healthy")


class ModelStatusResponse(BaseModel):
    models: List[Dict[str, Any]] = Field(default_factory=list)
    active_model: str = ""


# ---------------------------------------------------------------------------
# Background job state
# ---------------------------------------------------------------------------

_finetune_jobs: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@model_router.post("/update", summary="Rebuild and push DocWain model to Ollama registry")
async def update_model(request: ModelUpdateRequest, background_tasks: BackgroundTasks):
    """Rebuild the DocWain model from Modelfile and push to Ollama registry.

    Use this after updating the system prompt in Modelfile.
    Does NOT trigger fine-tuning — only rebuilds from the current Modelfile.
    """
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    modelfile_path = os.path.join(project_dir, "Modelfile")

    if not os.path.exists(modelfile_path):
        raise HTTPException(status_code=404, detail=f"Modelfile not found at {modelfile_path}")

    job_id = f"update_{int(time.time())}"

    def _run_update():
        _finetune_jobs[job_id] = {"status": "running", "type": "model_update", "started_at": time.time()}
        try:
            # Pull base model
            logger.info("Pulling base model: %s", request.base_model)
            subprocess.run(
                ["ollama", "pull", request.base_model],
                capture_output=True, text=True, timeout=600, check=True
            )

            # Create model
            logger.info("Creating model: %s", request.model_name)
            result = subprocess.run(
                ["ollama", "create", request.model_name, "-f", modelfile_path],
                capture_output=True, text=True, timeout=300, check=True
            )

            # Push model
            logger.info("Pushing model: %s", request.model_name)
            result = subprocess.run(
                ["ollama", "push", request.model_name],
                capture_output=True, text=True, timeout=1200, check=True
            )

            _finetune_jobs[job_id].update({
                "status": "completed",
                "completed_at": time.time(),
                "model_name": request.model_name,
            })
            logger.info("Model update completed: %s", request.model_name)

        except subprocess.TimeoutExpired:
            _finetune_jobs[job_id].update({"status": "failed", "error": "Timeout"})
            logger.error("Model update timed out for %s", request.model_name)
        except subprocess.CalledProcessError as e:
            _finetune_jobs[job_id].update({"status": "failed", "error": e.stderr or str(e)})
            logger.error("Model update failed: %s", e.stderr)
        except Exception as e:
            _finetune_jobs[job_id].update({"status": "failed", "error": str(e)})
            logger.error("Model update error: %s", e)

    background_tasks.add_task(_run_update)
    return {"job_id": job_id, "status": "started", "model_name": request.model_name}


@model_router.get("/feedback-check", summary="Check which profiles need fine-tuning")
async def check_feedback_signals() -> FinetuneCheckResponse:
    """Analyze feedback signals across all profiles.

    Returns profiles that are candidates for fine-tuning based on:
    - Low confidence ratio > 30%
    - Grounded ratio < 50%
    - Minimum 20 queries recorded
    """
    from src.api.dataHandler import get_redis_client
    from pymongo import MongoClient
    from src.api.config import Config

    redis_client = get_redis_client()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis unavailable — cannot check feedback signals")

    from src.intelligence.feedback_tracker import FeedbackTracker
    tracker = FeedbackTracker(redis_client)

    client = MongoClient(Config.MongoDB.URI, serverSelectionTimeoutMS=5000)
    db = client[Config.MongoDB.DB]
    profiles = list(db["profiles"].find({}, {"profile_id": 1, "_id": 1, "profileName": 1}))

    candidates = []
    healthy = []

    for p in profiles:
        pid = str(p.get("profile_id") or p.get("_id"))
        name = p.get("profileName", pid[:12])
        try:
            result = tracker.get_tuning_candidates(pid, min_queries=20)
            entry = {
                "profile_id": pid,
                "profile_name": name,
                "total_queries": result.get("total_queries", 0),
                "avg_confidence": round(result.get("avg_confidence", 0), 3),
                "grounded_ratio": round(result.get("grounded_ratio", 0), 3),
                "low_confidence_count": result.get("low_confidence_count", 0),
                "is_candidate": result.get("is_candidate", False),
            }
            if entry["is_candidate"]:
                candidates.append(entry)
            else:
                healthy.append(pid)
        except Exception as e:
            logger.debug("Feedback check failed for %s: %s", pid, e)

    return FinetuneCheckResponse(
        profiles_checked=len(profiles),
        candidates=candidates,
        healthy=healthy,
        finetune_needed=len(candidates) > 0,
    )


@model_router.post("/auto-finetune", summary="Trigger feedback-driven fine-tune")
async def trigger_auto_finetune(request: AutoFinetuneRequest, background_tasks: BackgroundTasks):
    """Trigger fine-tuning based on feedback signals.

    1. Checks which profiles need fine-tuning (or use force=True to skip check)
    2. Builds training dataset from Qdrant + feedback
    3. Runs Unsloth LoRA fine-tune
    4. On success: rebuilds and pushes DHS/DocWain

    This is the endpoint the daily scheduler calls.
    """
    job_id = f"finetune_{int(time.time())}"

    def _run_finetune():
        _finetune_jobs[job_id] = {
            "status": "running",
            "type": "auto_finetune",
            "started_at": time.time(),
            "profiles": [],
        }
        try:
            from src.api.dataHandler import get_redis_client
            from src.intelligence.feedback_tracker import FeedbackTracker
            from pymongo import MongoClient
            from src.api.config import Config
            from qdrant_client import QdrantClient

            # Step 1: Find profiles needing fine-tune
            profiles_to_tune = []

            if request.force:
                # Force: tune all profiles with data
                qc = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API)
                collections = qc.get_collections().collections
                for col in collections:
                    info = qc.get_collection(col.name)
                    if info.points_count > 40:
                        profiles_to_tune.append({
                            "collection": col.name,
                            "reason": "forced",
                        })
                        break  # One collection at a time
            else:
                redis_client = get_redis_client()
                if redis_client:
                    tracker = FeedbackTracker(redis_client)
                    mongo_client = MongoClient(Config.MongoDB.URI, serverSelectionTimeoutMS=5000)
                    db = mongo_client[Config.MongoDB.DB]
                    profiles = list(db["profiles"].find({}, {"profile_id": 1, "_id": 1, "subscription_id": 1, "subscription": 1}))

                    for p in profiles:
                        pid = str(p.get("profile_id") or p.get("_id"))
                        sub = str(p.get("subscription_id") or p.get("subscription") or "")
                        try:
                            result = tracker.get_tuning_candidates(pid, min_queries=20)
                            if result.get("is_candidate"):
                                profiles_to_tune.append({
                                    "collection": sub,
                                    "profile_id": pid,
                                    "reason": f"confidence={result.get('avg_confidence', 0):.2f}, grounded={result.get('grounded_ratio', 0):.2f}",
                                })
                        except Exception:
                            pass

            if not profiles_to_tune:
                _finetune_jobs[job_id].update({
                    "status": "completed",
                    "completed_at": time.time(),
                    "message": "No profiles need fine-tuning",
                    "profiles": [],
                })
                logger.info("Auto fine-tune: no profiles need tuning")
                return

            # Step 2: Trigger fine-tune via existing orchestrator
            collection_name = request.collection_name or profiles_to_tune[0]["collection"]
            _finetune_jobs[job_id]["profiles"] = profiles_to_tune

            logger.info("Auto fine-tune: triggering for collection=%s profiles=%d",
                       collection_name, len(profiles_to_tune))

            from src.finetune.agentic_orchestrator import AgenticFinetuneOrchestrator
            from src.finetune.models import CollectionOnlyFinetuneRequest

            ft_request = CollectionOnlyFinetuneRequest(
                collection_name=collection_name,
                base_model=request.base_model,
                learning_rate=request.learning_rate,
                max_steps=request.max_steps,
                lora_r=request.lora_r,
                lora_alpha=request.lora_r,
                batch_size=4,
                agentic=False,
            )

            orchestrator = AgenticFinetuneOrchestrator()
            result = orchestrator.run(ft_request, job_id=job_id)
            ft_status = result.get("overall_status", "unknown")

            # Step 3: If succeeded, rebuild and push model
            if ft_status in ("succeeded", "completed"):
                logger.info("Fine-tune succeeded — rebuilding DHS/DocWain")
                project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                modelfile_path = os.path.join(project_dir, "Modelfile")

                subprocess.run(
                    ["ollama", "create", "DHS/DocWain", "-f", modelfile_path],
                    capture_output=True, text=True, timeout=300
                )
                subprocess.run(
                    ["ollama", "push", "DHS/DocWain"],
                    capture_output=True, text=True, timeout=1200
                )
                _finetune_jobs[job_id].update({
                    "status": "completed",
                    "completed_at": time.time(),
                    "message": f"Fine-tuned and pushed ({ft_status})",
                    "finetune_result": result,
                })
            else:
                _finetune_jobs[job_id].update({
                    "status": "completed",
                    "completed_at": time.time(),
                    "message": f"Fine-tune finished with status: {ft_status}",
                    "finetune_result": result,
                })

        except Exception as e:
            _finetune_jobs[job_id].update({
                "status": "failed",
                "error": str(e),
                "completed_at": time.time(),
            })
            logger.exception("Auto fine-tune failed")

    background_tasks.add_task(_run_finetune)
    return {"job_id": job_id, "status": "started", "profiles_queued": "checking..."}


@model_router.get("/jobs", summary="List model management jobs")
async def list_jobs():
    """List all model update and fine-tune jobs with their status."""
    return {"jobs": _finetune_jobs}


@model_router.get("/jobs/{job_id}", summary="Get job status")
async def get_job_status(job_id: str):
    """Get status of a specific model management job."""
    job = _finetune_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"job_id": job_id, **job}


@model_router.get("/status", summary="Get current model status")
async def get_model_status():
    """List all DocWain models available in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10
        )
        models = []
        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 3 and "docwain" in parts[0].lower():
                models.append({
                    "name": parts[0],
                    "id": parts[1] if len(parts) > 1 else "",
                    "size": parts[2] if len(parts) > 2 else "",
                })

        return ModelStatusResponse(
            models=models,
            active_model="DHS/DocWain:latest",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {e}")
