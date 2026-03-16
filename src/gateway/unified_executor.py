"""ScreeningExecutor — routes screening actions to the correct backend."""
from __future__ import annotations

import asyncio
from src.utils.logging_utils import get_logger
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Response builder
# ---------------------------------------------------------------------------

def _build_response(
    *,
    status: str,
    action: str,
    correlation_id: str,
    start_time: float,
    result: Optional[Dict[str, Any]] = None,
    documents: Optional[List[Dict]] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    grounded: bool = True,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "action": action,
        "correlation_id": correlation_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "result": result,
        "documents": documents,
        "sources": sources or [],
        "grounded": grounded,
        "warnings": warnings or [],
        "error": error,
        "duration_ms": int((time.time() - start_time) * 1000),
        "metadata": metadata,
    }

# ---------------------------------------------------------------------------
# ScreeningExecutor
# ---------------------------------------------------------------------------

class ScreeningExecutor:
    """Dedicated screening dispatcher for document analysis operations.

    The endpoint body only carries ``category`` and ``doc_ids``.
    All other context (profile, subscription, doc_type) is resolved
    automatically from the document records or the session.
    """

    def __init__(self) -> None:
        self._screening_engine = None  # lazy
        self._actions_collection = None  # lazy MongoDB collection

    def _get_screening_engine(self):
        if self._screening_engine is None:
            from src.screening.engine import ScreeningEngine
            self._screening_engine = ScreeningEngine()
        return self._screening_engine

    # -- MongoDB audit log --------------------------------------------------

    def _get_actions_collection(self):
        """Lazy access to MongoDB 'actions' collection."""
        if self._actions_collection is None:
            try:
                from src.api.dataHandler import db
                if db is not None:
                    self._actions_collection = db["actions"]
            except Exception:  # noqa: BLE001
                pass
        return self._actions_collection

    @staticmethod
    def _summarize_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Strip raw text from input for audit — store lengths only."""
        summary: Dict[str, Any] = {}
        for key, value in (input_data or {}).items():
            if isinstance(value, str) and len(value) > 100:
                summary[f"{key}_length"] = len(value)
            elif isinstance(value, (list, dict)):
                summary[f"{key}_count"] = len(value)
            else:
                summary[key] = value
        return summary

    def _persist_action(
        self,
        response: Dict[str, Any],
        *,
        doc_ids: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Fire-and-forget persistence of action to MongoDB."""
        collection = self._get_actions_collection()
        if collection is None:
            return
        try:
            doc = {
                "correlation_id": response.get("correlation_id"),
                "action": response.get("action"),
                "status": response.get("status"),
                "timestamp": datetime.now(timezone.utc),
                "duration_ms": response.get("duration_ms", 0),
                "context": context or {},
                "doc_ids": doc_ids,
                "result_summary": {
                    "documents_processed": len(response.get("documents") or []),
                    "grounded": response.get("grounded", True),
                    "has_error": response.get("error") is not None,
                },
                "error": response.get("error"),
            }
            collection.insert_one(doc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist screening action: %s", exc)

    # -- Screening persistence -----------------------------------------------

    def _persist_screening_results(
        self,
        doc_id: str,
        category_results: Dict[str, Any],
        correlation_id: str,
        categories: List[str],
    ) -> None:
        """Persist screening results to the ``screening`` MongoDB collection
        and promote document status to SCREENING_COMPLETED.
        """
        doc_entries = []
        for cat, cat_data in category_results.items():
            doc_entries.append({
                "doc_id": doc_id,
                "status": "succeeded" if cat_data.get("status") == "success" else "failed",
                "result": cat_data.get("result"),
                "errors": [cat_data.get("error")] if cat_data.get("error") else [],
            })

        try:
            from src.screening.helpers import persist_screening_reports
            persist_screening_reports(
                run_id=correlation_id,
                endpoint="gateway/screen",
                options={"categories": categories},
                doc_entries=doc_entries,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist screening results for doc_id=%s: %s", doc_id, exc)

        # Promote document status so embedding pipeline can proceed
        any_succeeded = any(
            e.get("status") == "succeeded" for e in doc_entries
        )
        if any_succeeded:
            try:
                from src.api.screening_service import promote_to_screening_completed
                promote_to_screening_completed(doc_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to promote status for doc_id=%s: %s", doc_id, exc)

    # -- Public entry point -------------------------------------------------

    async def execute_screening(
        self,
        categories: List[str],
        *,
        doc_ids: Optional[List[str]] = None,
        profile_ids: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a screening operation.

        Accepts a list of categories. Routes based on contents:
          - ["run"] → batch screening across all documents in the profile
          - anything else → category-based screening on doc_ids
            (multiple categories are run sequentially per document)
        """
        cid = correlation_id or str(uuid.uuid4())
        start = time.time()
        action_label = "screen:" + ",".join(categories)

        try:
            if "run" in categories:
                response = await self._execute_screening_batch(
                    profile_ids, doc_ids, options or {}, cid, start,
                )
            else:
                response = await self._execute_screening_categories(
                    categories, doc_ids, options or {}, cid, start,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Screening execution failed | categories=%s cid=%s", categories, cid)
            response = _build_response(
                status="error",
                action=action_label,
                correlation_id=cid,
                start_time=start,
                error={"code": "internal_error", "message": str(exc)},
            )

        self._persist_action(response, doc_ids=doc_ids)
        return response

    # -- Backend: screening_engine ------------------------------------------

    async def _execute_screening_categories(
        self,
        categories: List[str],
        doc_ids: Optional[List[str]],
        options: Dict[str, Any],
        correlation_id: str,
        start: float,
    ) -> Dict[str, Any]:
        action_label = "screen:" + ",".join(categories)

        if not doc_ids:
            return _build_response(
                status="error",
                action=action_label,
                correlation_id=correlation_id,
                start_time=start,
                error={"code": "missing_doc_ids", "message": "doc_ids required for screening operations."},
            )

        engine = self._get_screening_engine()
        loop = asyncio.get_event_loop()

        doc_type = options.get("doc_type")
        internet_enabled = options.get("internet_enabled")
        region = options.get("region")
        jurisdiction = options.get("jurisdiction")

        documents: List[Dict] = []
        overall_status = "success"

        for doc_id in doc_ids:
            doc_entry: Dict[str, Any] = {"doc_id": doc_id, "status": "success", "categories": {}}
            doc_failed = False

            # HITL gate: check extraction completed before screening
            from src.api.document_status import get_document_record
            from src.api.screening_service import SCREENING_ELIGIBLE_STATUSES
            _rec = get_document_record(doc_id)
            _status = (_rec or {}).get("status", "")
            if _status not in SCREENING_ELIGIBLE_STATUSES:
                doc_entry["status"] = "skipped"
                doc_entry["error"] = {
                    "code": "extraction_required",
                    "message": f"Document must be extracted before screening. Current status: {_status}. "
                               f"Wait for extraction to complete, then retry screening.",
                }
                documents.append(doc_entry)
                overall_status = "partial"
                continue

            for category in categories:
                try:
                    if category == "all":
                        report = await loop.run_in_executor(
                            None,
                            lambda did=doc_id: engine.run_all(
                                did, doc_type=doc_type, internet_enabled_override=internet_enabled,
                            ),
                        )
                        payload = report.to_dict()
                        payload["doc_id"] = doc_id
                        doc_entry["categories"][category] = {"status": "success", "result": payload}
                    else:
                        results = await loop.run_in_executor(
                            None,
                            lambda did=doc_id, cat=category: engine.run_category(
                                cat, did,
                                doc_type=doc_type,
                                internet_enabled_override=internet_enabled,
                                region=region if cat == "legality" else None,
                                jurisdiction=jurisdiction if cat == "legality" else None,
                            ),
                        )
                        from src.screening.helpers import format_results
                        payload = format_results(doc_id, results, engine)
                        doc_entry["categories"][category] = {"status": "success", "result": payload}
                except Exception as exc:  # noqa: BLE001
                    logger.error("Screening failed for doc_id=%s category=%s: %s", doc_id, category, exc)
                    doc_entry["categories"][category] = {"status": "failed", "error": str(exc)}
                    doc_failed = True

            # Apply security-specific results (stores security_screening in doc record)
            if "security" in doc_entry["categories"]:
                sec_data = doc_entry["categories"]["security"]
                if sec_data.get("status") == "success" and sec_data.get("result"):
                    try:
                        from src.api.screening_service import apply_security_result
                        sec_result = sec_data["result"]
                        # The security result may be nested under 'results'
                        if isinstance(sec_result, dict) and "results" in sec_result:
                            apply_security_result(doc_id, sec_result)
                        else:
                            apply_security_result(doc_id, sec_result)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Failed to apply security result for doc_id=%s: %s", doc_id, exc)

            # Persist screening results to the screening collection
            self._persist_screening_results(
                doc_id, doc_entry["categories"], correlation_id, categories,
            )

            # Flatten result for single-category requests (backward compat)
            if len(categories) == 1:
                cat_key = categories[0]
                cat_data = doc_entry["categories"][cat_key]
                if cat_data["status"] == "success":
                    documents.append({"doc_id": doc_id, "status": "success", "result": cat_data["result"]})
                else:
                    documents.append({"doc_id": doc_id, "status": "failed", "result": None, "errors": [cat_data.get("error", "")]})
                    overall_status = "partial"
            else:
                doc_entry["status"] = "partial" if doc_failed else "success"
                documents.append(doc_entry)
                if doc_failed:
                    overall_status = "partial"

        return _build_response(
            status=overall_status,
            action=action_label,
            correlation_id=correlation_id,
            start_time=start,
            documents=documents,
            metadata={"documents_processed": len(doc_ids), "categories": categories},
        )

    # -- Backend: screening_batch -------------------------------------------

    async def _execute_screening_batch(
        self,
        profile_ids: Optional[List[str]],
        doc_ids: Optional[List[str]],
        options: Dict[str, Any],
        correlation_id: str,
        start: float,
    ) -> Dict[str, Any]:
        if not profile_ids:
            return _build_response(
                status="error",
                action="screen:run",
                correlation_id=correlation_id,
                start_time=start,
                error={"code": "missing_profile_ids", "message": "profile_ids required for batch screening. Provide x-session-id header."},
            )

        from src.screening.helpers import normalize_categories
        from src.screening.api import (
            _get_documents_collection,
            _run_parallel_screening,
        )

        categories = normalize_categories(options.get("categories"))
        loop = asyncio.get_event_loop()

        def _run_batch():
            collection = _get_documents_collection()
            if collection is None:
                return {"status": "error", "error": "Document store unavailable"}

            from src.api.statuses import STATUS_EXTRACTION_COMPLETED

            profiles = []
            for profile_id in profile_ids:
                query = {
                    "$and": [
                        {"$or": [{"profile": profile_id}, {"profile_id": profile_id}, {"profileId": profile_id}]},
                        {"status": STATUS_EXTRACTION_COMPLETED},
                    ]
                }
                docs = list(collection.find(query, projection={"_id": 1, "doc_type": 1}))
                if not docs:
                    profiles.append({"profile_id": profile_id, "status": "not_found", "documents": []})
                    continue

                tasks = []
                for doc in docs:
                    doc_id = str(doc.get("_id", ""))
                    if doc_id:
                        tasks.append({
                            "doc_id": doc_id,
                            "categories": categories,
                            "doc_type": options.get("doc_type") or doc.get("doc_type"),
                            "internet_enabled": options.get("internet_enabled"),
                        })

                doc_results = _run_parallel_screening(tasks)
                profiles.append({
                    "profile_id": profile_id,
                    "status": "success",
                    "documents": doc_results,
                })

            return {"status": "success", "profiles": profiles}

        result = await loop.run_in_executor(None, _run_batch)

        return _build_response(
            status=result.get("status", "success"),
            action="screen:run",
            correlation_id=correlation_id,
            start_time=start,
            result=result,
            metadata={"profiles_count": len(profile_ids)},
        )
