from __future__ import annotations

import json
from src.utils.logging_utils import get_logger
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .hybrid_retriever import RetrievalCandidate
from .score_utils import RerankShapeError, describe_scores, normalize_scores, to_py_scalar

logger = get_logger(__name__)

@dataclass
class RerankerConfig:
    top_k: int = 20
    llm_fallback: bool = True

class Reranker:
    """Deterministic reranking with cross-encoder preferred, LLM fallback."""

    def __init__(self, cross_encoder: Optional[Any] = None, llm_client: Optional[Any] = None, config: Optional[RerankerConfig] = None):
        self.cross_encoder = cross_encoder
        self.llm_client = llm_client
        self.config = config or RerankerConfig()

    def rerank(self, query: str, candidates: List[RetrievalCandidate], top_k: Optional[int] = None) -> List[RetrievalCandidate]:
        if not candidates:
            return []
        top_k = int(top_k or self.config.top_k)
        ordered = list(candidates)
        if self.cross_encoder:
            try:
                pairs = [[query, c.text] for c in ordered]
                scores = None
                if hasattr(self.cross_encoder, "predict"):
                    scores = self.cross_encoder.predict(pairs)
                elif callable(self.cross_encoder):
                    scores = self.cross_encoder(pairs)
                if scores is not None:
                    normalized = normalize_scores(scores, expected_k=len(ordered))
                    scored = [(idx, chunk, float(score)) for idx, (chunk, score) in enumerate(zip(ordered, normalized))]
                    scored.sort(key=lambda x: (-x[2], x[0]))
                    reranked = []
                    for _, chunk, score in scored:
                        chunk.score = float(score)
                        reranked.append(chunk)
                    logger.info("Reranker used cross-encoder", extra={"reranker": "cross_encoder", "candidates": len(reranked)})
                    return reranked[:top_k]
            except RerankShapeError as exc:
                details = {
                    "stage": "rerank",
                    "provider": "cross_encoder",
                    "expected_k": exc.expected_k,
                    "actual_len": exc.actual_len,
                    "score_type": exc.score_type,
                    "score_shape": exc.score_shape,
                }
                details.update(describe_scores(scores))
                logger.debug("Cross-encoder rerank shape mismatch; falling back: %s", exc, extra=details, exc_info=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Cross-encoder rerank failed: %s",
                    exc,
                    extra={"stage": "rerank", "provider": "cross_encoder"},
                )

        if self.llm_client and self.config.llm_fallback:
            llm_order = self._llm_rerank(query, ordered)
            if llm_order:
                reranked = [ordered[idx] for idx in llm_order if idx < len(ordered)]
                logger.info("Reranker used LLM fallback", extra={"reranker": "llm", "candidates": len(reranked)})
                return reranked[:top_k]

        # Deterministic fallback: keep original order by score then index
        scored = list(enumerate(ordered))
        scored.sort(key=lambda x: (-to_py_scalar(x[1].score), x[0]))
        reranked = [item for _, item in scored]
        logger.info("Reranker used deterministic fallback", extra={"reranker": "score", "candidates": len(reranked)})
        return reranked[:top_k]

    def _llm_rerank(self, query: str, candidates: List[RetrievalCandidate]) -> List[int]:
        try:
            prompt = self._build_llm_prompt(query, candidates)
            response = self.llm_client.generate(prompt, max_retries=2, backoff=0.4)
            payload = self._extract_json(response)
            indices = payload.get("ordered_indices") if isinstance(payload, dict) else None
            if not indices:
                return []
            return [int(idx) for idx in indices if isinstance(idx, (int, float, str)) and str(idx).isdigit()]
        except Exception as exc:  # noqa: BLE001
            logger.debug("LLM rerank failed: %s", exc)
            return []

    @staticmethod
    def _build_llm_prompt(query: str, candidates: List[RetrievalCandidate]) -> str:
        items = []
        for idx, cand in enumerate(candidates):
            snippet = " ".join((cand.text or "").split())[:200]
            items.append(f"{idx}: {snippet}")
        return (
            "You are a ranking assistant. Order the candidate passages by relevance to the question. "
            "Return strict JSON only.\n\n"
            f"QUESTION: {query}\n\n"
            "CANDIDATES:\n"
            + "\n".join(items)
            + "\n\nReturn format:\n{\"ordered_indices\": [0,1,2]}"
        )

    @staticmethod
    def _extract_json(raw: str) -> Dict[str, Any]:
        if not raw:
            return {}
        raw = raw.strip()
        if raw.startswith("{") and raw.endswith("}"):
            try:
                return json.loads(raw)
            except Exception:
                return {}
        if "{" in raw and "}" in raw:
            snippet = raw[raw.find("{"): raw.rfind("}") + 1]
            try:
                return json.loads(snippet)
            except Exception:
                return {}
        return {}
