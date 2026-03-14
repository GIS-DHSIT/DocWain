from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ModelCandidate:
    name: str
    backend: Optional[str] = None
    weight: float = 1.0


@dataclass
class CandidateResponse:
    model: str
    backend: Optional[str]
    text: str
    error: Optional[str] = None


class ModelArbitrationLayer:
    """Generate responses in parallel across multiple models and return candidates."""

    def __init__(
        self,
        client_factory: Callable[..., Any],
        *,
        max_workers: int = 3,
    ) -> None:
        self.client_factory = client_factory
        self.max_workers = max_workers

    def generate_candidates(
        self,
        prompt: str,
        *,
        primary_model: str,
        candidate_models: List[ModelCandidate],
        temperature: Optional[float] = None,
    ) -> List[CandidateResponse]:
        models = [ModelCandidate(name=primary_model, backend=None)] + [
            c for c in candidate_models if c.name and c.name != primary_model
        ]
        logger.debug("generate_candidates called with primary_model=%s, total_models=%s",
                     primary_model, len(models))

        results: List[CandidateResponse] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {
                executor.submit(self._invoke_model, prompt, model, temperature): model for model in models
            }
            for future in concurrent.futures.as_completed(future_map):
                model = future_map[future]
                try:
                    text = future.result()
                    results.append(CandidateResponse(model=model.name, backend=model.backend, text=text or ""))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("generate_candidates model=%s failed", model.name, exc_info=True)
                    results.append(
                        CandidateResponse(model=model.name, backend=model.backend, text="", error=str(exc))
                    )
        logger.debug("generate_candidates returning %s candidates, %s errors",
                     len(results), sum(1 for r in results if r.error))
        return results

    def _invoke_model(self, prompt: str, model: ModelCandidate, temperature: Optional[float]) -> str:
        logger.debug("_invoke_model model=%s, backend=%s", model.name, model.backend)
        client = self.client_factory(model.name, backend_override=model.backend)
        kwargs: Dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return client.generate(prompt, **kwargs)
