from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from src.api import dw_newron
from src.execution.common import normalize_answer
from src.finetune import resolve_model_for_profile
from src.mode.execution_mode import ExecutionMode
from src.runtime.request_context import RequestContext


def _collect_source_ids(sources: Iterable[Dict[str, Any]]) -> List[str]:
    ids = []
    for source in sources or []:
        if isinstance(source, str):
            ids.append(source)
            continue
        if not isinstance(source, dict):
            continue
        for key in ("document_id", "doc_id", "docId", "chunk_id", "chunkId", "source_id"):
            value = source.get(key)
            if value:
                ids.append(str(value))
        name = source.get("source_name") or source.get("file_name")
        if name:
            ids.append(str(name))
    return sorted(set(ids))


@dataclass
class RequestChain:
    """
    Stateless request executor. A new instance is created for every call to avoid
    accidental cross-request state or cached responses.
    """

    ctx: RequestContext
    mode: ExecutionMode

    def run(self, *, stream: bool = False, debug: bool = False, force_refresh: bool = False) -> Tuple[Dict[str, Any], List[str]]:
        resolved_model = resolve_model_for_profile(self.ctx.profile_id, self.ctx.model_name or "")
        model_name = resolved_model.model_name or self.ctx.model_name
        rag_system = None
        try:
            from src.api.rag_state import get_app_state

            app_state = get_app_state()
            rag_system = app_state.rag_system if app_state else None
        except Exception:
            rag_system = None

        raw_answer = dw_newron.answer_question(
            query=self.ctx.query,
            user_id=self.ctx.user_id or "",
            profile_id=self.ctx.profile_id or "",
            subscription_id=self.ctx.subscription_id or "default",
            model_name=model_name or "DocWain-Agent",
            persona=self.ctx.persona or "Document Assistant",
            session_id=self.ctx.session_id,
            new_session=force_refresh,
            disable_answer_cache=True,
            force_refresh=force_refresh,
            request_id=self.ctx.request_id,
            index_version=self.ctx.index_version,
            tools=self.ctx.tools,
            use_tools=getattr(self.ctx, "use_tools", False),
            tool_inputs=getattr(self.ctx, "tool_inputs", None),
            enable_internet=getattr(self.ctx, "enable_internet", False),
            rag_system=rag_system,
        )

        normalized = normalize_answer(raw_answer)

        # Post-generation visualization enhancement (non-blocking)
        try:
            from src.visualization.enhancer import enhance_with_visualization
            channel = "teams" if getattr(self.ctx, "channel", "") == "teams" else "web"
            normalized = enhance_with_visualization(normalized, self.ctx.query, channel=channel)
        except Exception as _viz_exc:
            import logging as _logging
            _logging.getLogger(__name__).debug("Visualization enhancement skipped: %s", _viz_exc)

        metadata = normalized.get("metadata") or {}
        metadata.update(
            {
                "request_id": self.ctx.request_id,
                "correlation_id": self.ctx.request_id,
                "session_id": self.ctx.session_id,
                "execution_mode": self.mode.value,
            }
        )
        normalized["metadata"] = metadata

        sources = normalized.get("sources") or []
        evidence_ids = _collect_source_ids(sources)
        meta_source_ids = metadata.get("source_doc_ids") or []
        evidence_ids = sorted(set(evidence_ids + [str(i) for i in meta_source_ids if i]))

        return normalized, evidence_ids


def build_chain(ctx: RequestContext, profile_filters: Any = None, mode: ExecutionMode = ExecutionMode.NORMAL) -> RequestChain:
    """
    Build a new, stateless chain for the given request context. The returned
    chain intentionally avoids sharing mutable state across requests.
    """
    _ = profile_filters  # reserved for future per-profile tuning; kept unused intentionally
    return RequestChain(ctx=ctx, mode=mode)
