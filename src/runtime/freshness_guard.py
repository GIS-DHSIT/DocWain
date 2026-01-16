from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, List, Tuple

from src.mode.session_state import SessionState
from src.runtime.request_context import RequestContext


class FreshnessGuard:
    """
    Detects stale answer reuse across requests and forces regeneration when
    fingerprints diverge. Stores only fingerprints in SessionState; never the
    answer text itself.

    Anti-patterns addressed:
    - Redis/LLM answer cache reuse: guarded via NO_ANSWER_CACHE + regeneration hook.
    - Singleton chain reuse: enforced request-scoped RequestChain per call.
    - Shared stream buffer reuse: fingerprints block returning prior response text.
    """

    def __init__(self, session_state: SessionState):
        self.session_state = session_state

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _request_fingerprint(self, ctx: RequestContext) -> str:
        normalized_query = " ".join((ctx.query or "").strip().split()).lower()
        filters_blob = json.dumps(ctx.filters or {}, sort_keys=True, default=str)
        payload = "|".join(
            [
                normalized_query,
                ctx.profile_id or "",
                ctx.subscription_id or "",
                ctx.index_version or "",
                filters_blob,
            ]
        )
        return self._hash(payload)

    def _evidence_fingerprint(self, evidence_ids: List[str]) -> str:
        if not evidence_ids:
            return "no-evidence"
        joined = "|".join(sorted(set(str(eid) for eid in evidence_ids)))
        return self._hash(joined)

    def _insufficient_response(self, ctx: RequestContext, evidence_ids: List[str]) -> Dict[str, Any]:
        suggestion = f"Try a more specific question about these documents: {', '.join(sorted(set(evidence_ids)))[:200]}" if evidence_ids else "Try naming a document or section to search."
        return {
            "response": "Not found in the selected documents. The request was re-run because the previous answer looked stale.",
            "sources": [],
            "grounded": False,
            "context_found": False,
            "metadata": {
                "request_id": ctx.request_id,
                "freshness_guard": {
                    "regenerated": True,
                    "returned_insufficient": True,
                },
                "suggested_next_step": suggestion,
            },
        }

    def enforce(
        self,
        ctx: RequestContext,
        answer: Dict[str, Any],
        evidence_ids: List[str],
        regenerate: Callable[[], Tuple[Dict[str, Any], List[str]]],
    ) -> Tuple[Dict[str, Any], List[str]]:
        request_fp = self._request_fingerprint(ctx)
        evidence_fp = self._evidence_fingerprint(evidence_ids)
        answer_text = (answer.get("response") or "").strip()
        answer_hash = self._hash(answer_text) if answer_text else ""

        previous_fp = getattr(self.session_state, "last_request_fingerprint", None)
        previous_ev = getattr(self.session_state, "last_evidence_fingerprint", None)
        previous_answer_hash = getattr(self.session_state, "last_answer_hash", None)

        is_stale = (
            bool(previous_fp)
            and request_fp != previous_fp
            and answer_hash
            and previous_answer_hash == answer_hash
            and evidence_fp != previous_ev
        )

        regenerated = False
        if is_stale:
            regenerated_answer, regenerated_evidence = regenerate()
            regenerated_text = (regenerated_answer.get("response") or "").strip()
            regenerated_hash = self._hash(regenerated_text) if regenerated_text else ""
            regenerated_fp = self._evidence_fingerprint(regenerated_evidence)
            if regenerated_hash == answer_hash or regenerated_fp == evidence_fp:
                regenerated_answer = self._insufficient_response(ctx, evidence_ids)
                regenerated_evidence = evidence_ids
                regenerated_hash = self._hash(regenerated_answer["response"])
                regenerated_fp = evidence_fp
            answer = regenerated_answer
            evidence_ids = regenerated_evidence
            answer_hash = regenerated_hash
            evidence_fp = regenerated_fp
            regenerated = True

        # Persist fingerprints only, never raw answers
        self.session_state.last_request_fingerprint = request_fp
        self.session_state.last_evidence_fingerprint = evidence_fp
        self.session_state.last_answer_hash = answer_hash
        self.session_state.last_request_id = ctx.request_id
        self.session_state.last_query = ctx.query

        metadata = answer.get("metadata") or {}
        metadata.setdefault("freshness_guard", {})
        metadata["freshness_guard"].update(
            {
                "request_fingerprint": request_fp,
                "evidence_fingerprint": evidence_fp,
                "regenerated": regenerated,
            }
        )
        metadata.setdefault("request_id", ctx.request_id)
        metadata.setdefault("session_id", ctx.session_id)
        answer["metadata"] = metadata

        return answer, evidence_ids
