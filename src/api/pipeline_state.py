from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

_PIPELINE_STAGE: ContextVar[Optional[str]] = ContextVar("docwain_pipeline_stage", default=None)


def get_pipeline_stage() -> Optional[str]:
    return _PIPELINE_STAGE.get()


@contextmanager
def pipeline_stage(stage: str) -> Iterator[None]:
    token = _PIPELINE_STAGE.set(stage)
    try:
        yield
    finally:
        _PIPELINE_STAGE.reset(token)


def is_screening_stage() -> bool:
    return (get_pipeline_stage() or "").upper() == "SCREENING"


__all__ = ["get_pipeline_stage", "pipeline_stage", "is_screening_stage"]
