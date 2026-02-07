from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from src.screening.resume.extractor import ResumeExtractor as BaseResumeExtractor
from src.screening.resume.models import ResumeProfile

from .sanitize import FALLBACK_ANSWER
from .types import Chunk


@dataclass(frozen=True)
class ResumeExtractionContext:
    subscription_id: str
    profile_id: str
    document_id: Optional[str] = None


@dataclass(frozen=True)
class ExtractionAmbiguous:
    document_ids: List[str]
    document_names: List[str]


def _chunk_document_id(chunk: Chunk) -> Optional[str]:
    meta = chunk.meta or {}
    value = meta.get("document_id") or meta.get("doc_id") or meta.get("docId")
    return str(value) if value else None


def _chunk_document_name(chunk: Chunk) -> Optional[str]:
    meta = chunk.meta or {}
    value = meta.get("source_name") or meta.get("document_name") or getattr(chunk.source, "document_name", None)
    return str(value) if value else None


def _join_text(chunks: Iterable[Chunk]) -> str:
    parts: List[str] = []
    for chunk in chunks:
        text = (chunk.text or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


class ResumeExtractor:
    def __init__(self, base: Optional[BaseResumeExtractor] = None) -> None:
        self.base = base or BaseResumeExtractor()

    def extract(self, chunks: List[Chunk], ctx: ResumeExtractionContext):
        scoped = list(chunks or [])
        if ctx.document_id:
            scoped = [chunk for chunk in scoped if _chunk_document_id(chunk) == ctx.document_id]
        else:
            doc_ids = sorted({doc_id for doc_id in (_chunk_document_id(c) for c in scoped) if doc_id})
            if len(doc_ids) > 1:
                doc_names = sorted({name for name in (_chunk_document_name(c) for c in scoped) if name})
                return ExtractionAmbiguous(document_ids=doc_ids, document_names=doc_names)

        text = _join_text(scoped)
        if not text:
            return ResumeProfile()
        return self.base.extract(text)


class ResumeRenderer:
    AMBIGUOUS_MESSAGE = (
        "I found multiple resumes in this profile; please specify the resume or document name you want me to use."
    )

    def render(self, extraction) -> str:
        if isinstance(extraction, ExtractionAmbiguous):
            return self.AMBIGUOUS_MESSAGE
        if not isinstance(extraction, ResumeProfile):
            return FALLBACK_ANSWER

        parts: List[str] = []
        if extraction.name:
            parts.append(f"Candidate: {extraction.name}.")
        if extraction.headline:
            parts.append(str(extraction.headline).strip())
        if extraction.summary:
            parts.append(str(extraction.summary).strip())

        if extraction.experience:
            experiences = []
            for item in extraction.experience[:4]:
                title = item.title or "Role"
                company = f" at {item.company}" if item.company else ""
                dates = ""
                if item.start_date or item.end_date:
                    start = item.start_date or ""
                    end = item.end_date or ""
                    dates = f" ({start} - {end})" if (start or end) else ""
                experiences.append(f"{title}{company}{dates}".strip())
            if experiences:
                parts.append(f"Experience: {'; '.join(experiences)}.")

        if extraction.education:
            educations = []
            for item in extraction.education[:3]:
                label = item.institution or "Institution"
                if item.degree:
                    label = f"{label} ({item.degree})"
                educations.append(label)
            if educations:
                parts.append(f"Education: {', '.join(educations)}.")

        if extraction.skills:
            parts.append(f"Skills: {', '.join(extraction.skills[:12])}.")

        if extraction.certifications:
            certs = [cert.name for cert in extraction.certifications if cert.name]
            if certs:
                parts.append(f"Certifications: {', '.join(certs[:8])}.")

        if extraction.links:
            parts.append(f"Links: {', '.join(extraction.links[:5])}.")

        rendered = " ".join(part for part in parts if part).strip()
        return rendered or FALLBACK_ANSWER


__all__ = ["ResumeExtractor", "ResumeRenderer", "ResumeExtractionContext", "ExtractionAmbiguous"]
