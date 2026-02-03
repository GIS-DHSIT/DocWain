from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from src.rag.certification_extractor import extract_certifications
from src.rag.experience_extractor import ExperienceResult, extract_experience
from src.rag.resume_context_builder import build_resume_packet
from src.rag.section_classifier import classify_skills
from src.tools.base import generate_correlation_id, register_tool, standard_response
from src.tools.common.grounding import build_source_record
from src.tools.resume_router import select_resume_docs
from src.rag.doc_inventory import DocInventoryItem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/resume-analyzer-v2", tags=["Tools-ResumeAnalyzerV2"])


class ResumeAnalyzerRequest(BaseModel):
    query: str | None = Field(default=None, description="User query for resume analysis")
    documents: List[Dict[str, Any]] = Field(default_factory=list, description="Documents with text content")


@dataclass(frozen=True)
class ResumeProfile:
    candidate_name: str
    source_type: str
    source_document: str
    total_years_experience: Optional[int]
    experience_confidence: str
    experience_basis: str
    experience_details: Optional[str]
    experience_summary: str
    technical_skills: List[str]
    functional_skills: List[str]
    certifications: List[str]
    education: List[str]
    awards: List[str]


@dataclass(frozen=True)
class ResumeAnalysisResult:
    profiles: List[ResumeProfile]
    used_chunk_ids: List[str]


def _chunk_text(chunk: Any) -> str:
    return getattr(chunk, "text", "") or ""


def _chunk_meta(chunk: Any) -> dict:
    return getattr(chunk, "metadata", {}) or {}


def _chunk_id(chunk: Any) -> str:
    meta = _chunk_meta(chunk)
    return str(meta.get("chunk_id") or getattr(chunk, "id", "") or "")


def _infer_candidate_name(chunks: Sequence[Any]) -> str:
    stop_tokens = {
        "project",
        "summary",
        "experience",
        "skills",
        "certification",
        "certifications",
        "education",
        "objective",
        "profile",
        "professional",
        "career",
        "result",
        "results",
        "areas",
        "key",
        "place",
        "date",
        "personal",
        "data",
    }
    for chunk in chunks:
        meta = _chunk_meta(chunk)
        for key in ("profile_name", "candidate_name", "name"):
            value = meta.get(key)
            if value and isinstance(value, str):
                return value.strip()
    for chunk in chunks:
        for line in _chunk_text(chunk).splitlines()[:10]:
            cleaned = re.sub(r"[^A-Za-z\s.-]", "", line).strip()
            if not cleaned or any(char.isdigit() for char in line):
                continue
            if ":" in line or "page" in line.lower():
                continue
            tokens = cleaned.split()
            if not (2 <= len(tokens) <= 4):
                continue
            lowered_tokens = {tok.lower() for tok in tokens}
            if lowered_tokens & stop_tokens:
                continue
            if all(tok[0].isupper() for tok in tokens if tok):
                return cleaned
    return "Candidate"


def _looks_like_person_name(value: str) -> bool:
    if not value:
        return False
    cleaned = re.sub(r"[^A-Za-z\s.-]", "", value).strip()
    tokens = cleaned.split()
    if not (2 <= len(tokens) <= 4):
        return False
    if any(tok.lower() in {"personal", "data", "summary", "project", "result", "areas"} for tok in tokens):
        return False
    if any(len(tok) > 15 for tok in tokens):
        return False
    return all(tok[0].isupper() for tok in tokens if tok)


def _infer_name_from_doc_name(doc_name: str) -> str:
    if not doc_name:
        return "Candidate"
    cleaned = re.sub(r"\\.[A-Za-z0-9]{1,5}$", "", doc_name)
    cleaned = re.sub(r"[_\\-]+", " ", cleaned)
    cleaned = re.sub(r"\\b(resume|cv|linkedin|profile|updated|final|draft)\\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\\b\\d{2,4}\\b", "", cleaned)
    cleaned = re.sub(r"\\s+", " ", cleaned).strip()
    tokens = [tok for tok in cleaned.split() if tok]
    if 2 <= len(tokens) <= 4:
        return " ".join(tokens)
    return "Candidate"


def _infer_source_type(doc_name: str, chunks: Sequence[Any]) -> str:
    for chunk in chunks:
        meta = _chunk_meta(chunk)
        doc_type = (meta.get("doc_type") or meta.get("document_type") or meta.get("source_type") or "").lower()
        if doc_type in {"resume", "cv", "linkedin"}:
            return "LinkedIn" if doc_type == "linkedin" else "Resume"
    lowered = (doc_name or "").lower()
    if "linkedin" in lowered:
        return "LinkedIn"
    if "resume" in lowered or "cv" in lowered:
        return "Resume"
    return "Resume"


def _brief_summary(summary_text: str, experience_text: str) -> str:
    source = summary_text or experience_text
    if not source:
        return "Not Mentioned"
    lines = []
    for ln in source.splitlines():
        cleaned = ln.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered.startswith("--- page") or "page " in lowered:
            continue
        if cleaned.startswith("#") or "hrj" in lowered:
            continue
        lines.append(cleaned)
    if not lines:
        return "Not Mentioned"
    take = lines[:3]
    summary = " ".join(take)
    return summary[:280]


def _extract_list_items(section_text: str, *, limit: int = 8) -> List[str]:
    items: List[str] = []
    if not section_text:
        return items
    for line in section_text.splitlines():
        cleaned = line.strip(" \t-•*;")
        if not cleaned or len(cleaned) < 2:
            continue
        if len(cleaned.split()) > 10:
            continue
        items.append(cleaned)
        if len(items) >= limit:
            break
    deduped: List[str] = []
    seen = set()
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _build_profile(doc_name: str, chunks: Sequence[Any]) -> tuple[ResumeProfile, List[str]]:
    packet = build_resume_packet(chunks)
    sections = packet.sections
    experience_text = sections.get("experience", "")
    skills_text = sections.get("skills", "")
    cert_text = sections.get("certifications", "")
    summary_text = sections.get("summary", "")
    education_text = sections.get("education", "")
    awards_text = sections.get("awards", "")

    experience: ExperienceResult = extract_experience("\\n".join([experience_text, summary_text]))
    technical_skills, functional_skills = classify_skills(skills_text)
    fallback_text = "\n".join([summary_text, experience_text, skills_text])
    certifications = extract_certifications(cert_text, fallback_text=fallback_text)

    education = _extract_list_items(education_text)
    awards = _extract_list_items(awards_text)

    candidate_name = _infer_candidate_name(chunks)
    if not _looks_like_person_name(candidate_name):
        candidate_name = _infer_name_from_doc_name(doc_name)
    profile = ResumeProfile(
        candidate_name=candidate_name,
        source_type=_infer_source_type(doc_name, chunks),
        source_document=doc_name,
        total_years_experience=experience.total_years_experience,
        experience_confidence=experience.experience_confidence,
        experience_basis=experience.experience_basis,
        experience_details=experience.details,
        experience_summary=_brief_summary(summary_text, experience_text),
        technical_skills=technical_skills,
        functional_skills=functional_skills,
        certifications=certifications,
        education=education,
        awards=awards,
    )
    return profile, packet.used_chunk_ids


def _extract_year_range(query_text: str) -> Optional[tuple[int, int]]:
    if not query_text:
        return None
    match = re.search(r"(\\d{1,2})\\s*(?:to|-)\\s*(\\d{1,2})\\s*years", query_text.lower())
    if match:
        low = int(match.group(1))
        high = int(match.group(2))
        return (min(low, high), max(low, high))
    match = re.search(r"between\\s+(\\d{1,2})\\s+and\\s+(\\d{1,2})\\s+years", query_text.lower())
    if match:
        low = int(match.group(1))
        high = int(match.group(2))
        return (min(low, high), max(low, high))
    return None


def _requires_certifications(query_text: str) -> bool:
    lowered = (query_text or "").lower()
    return bool(
        re.search(r"\b(with|possess|have|has)\b.*\bcertifications?\b", lowered)
        or re.search(r"\bcertifications?\b.*\b(required|mandatory)\b", lowered)
    )


def _is_scm_profile(profile: ResumeProfile) -> bool:
    text = " ".join(
        [
            profile.experience_summary or "",
            " ".join(profile.technical_skills),
            " ".join(profile.functional_skills),
        ]
    ).lower()
    scm_terms = {
        "supply chain",
        "procurement",
        "logistics",
        "planning",
        "warehouse",
        "inventory",
        "demand",
        "sap ewm",
        "sap mm",
        "oracle scm",
        "s&op",
    }
    return any(term in text for term in scm_terms)


def filter_profiles_by_query(profiles: List[ResumeProfile], query_text: str) -> List[ResumeProfile]:
    filtered = list(profiles)
    year_range = _extract_year_range(query_text)
    if year_range:
        low, high = year_range
        filtered = [
            profile
            for profile in filtered
            if profile.total_years_experience is not None and low <= profile.total_years_experience <= high
        ]
    if _requires_certifications(query_text):
        filtered = [profile for profile in filtered if profile.certifications]
    if "supply chain" in (query_text or "").lower() or "scm" in (query_text or "").lower():
        filtered = [profile for profile in filtered if _is_scm_profile(profile)]
    return filtered


def analyze_resume_chunks(
    *,
    chunks_by_doc: Dict[str, List[Any]],
    doc_inventory: Sequence[DocInventoryItem] | None = None,
    force_all_docs: bool = False,
    query_text: str | None = None,
) -> ResumeAnalysisResult:
    doc_inventory = list(doc_inventory or [])
    normalized_chunks = {str(key).lower(): value for key, value in chunks_by_doc.items()}
    selected_docs = (
        select_resume_docs(doc_inventory=doc_inventory, chunks_by_doc=chunks_by_doc)
        if force_all_docs
        else list(doc_inventory)
    )
    if force_all_docs and selected_docs:
        doc_names = [doc.source_file or doc.document_name or doc.doc_id for doc in selected_docs]
    else:
        doc_names = list(chunks_by_doc.keys())

    profiles: List[ResumeProfile] = []
    used_chunk_ids: List[str] = []

    for doc_name in doc_names:
        chunks = chunks_by_doc.get(doc_name, [])
        if not chunks:
            chunks = normalized_chunks.get(str(doc_name).lower(), [])
        if not chunks:
            continue
        profile, packet_ids = _build_profile(doc_name, chunks)
        profiles.append(profile)
        if packet_ids:
            used_chunk_ids.extend(packet_ids)
        else:
            for chunk in chunks:
                cid = _chunk_id(chunk)
                if cid:
                    used_chunk_ids.append(cid)

    deduped_ids = []
    seen = set()
    for cid in used_chunk_ids:
        if cid in seen:
            continue
        seen.add(cid)
        deduped_ids.append(cid)

    if query_text:
        profiles = filter_profiles_by_query(profiles, query_text)
    return ResumeAnalysisResult(profiles=profiles, used_chunk_ids=deduped_ids)


@register_tool("resume_analyzer_v2")
async def resume_analyzer_v2_handler(payload: Dict[str, Any], correlation_id: str | None = None) -> Dict[str, Any]:
    req = ResumeAnalyzerRequest(**(payload.get("input") or payload))
    documents = req.documents or []
    chunks_by_doc: Dict[str, List[Any]] = {}
    for doc in documents:
        name = doc.get("document_name") or doc.get("source_file") or doc.get("doc_id") or "Document"
        chunks_by_doc.setdefault(name, []).append(
            type("Chunk", (), {"text": doc.get("text", ""), "metadata": doc.get("metadata", {})})()
        )

    analysis = analyze_resume_chunks(
        chunks_by_doc=chunks_by_doc,
        doc_inventory=[],
        query_text=req.query,
    )
    sources = [build_source_record("tool", correlation_id or "resume_analyzer_v2", title="resume")]

    result = {
        "profiles": [profile.__dict__ for profile in analysis.profiles],
        "used_chunk_ids": analysis.used_chunk_ids,
    }
    return {"result": result, "sources": sources, "grounded": True, "context_found": bool(analysis.profiles)}


@router.post("/analyze")
async def analyze(request: ResumeAnalyzerRequest, x_correlation_id: str | None = Header(None)):
    cid = generate_correlation_id(x_correlation_id)
    payload = {"input": request.dict()}
    result = await resume_analyzer_v2_handler(payload, correlation_id=cid)
    return standard_response(
        "resume_analyzer_v2",
        grounded=result.get("grounded", True),
        context_found=result.get("context_found", True),
        result=result.get("result"),
        sources=result.get("sources"),
        warnings=[],
        correlation_id=cid,
    )


__all__ = [
    "ResumeProfile",
    "ResumeAnalysisResult",
    "analyze_resume_chunks",
]
