from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

from qdrant_client import QdrantClient
from src.deterministic.contacts import ContactInfo, extract_contacts
from src.retrieval.profile_corpus import get_profile_points
from src.api.vector_store import build_qdrant_filter


_CONTACT_QUERY_RE = re.compile(r"(?i)\b(contact|phone|mobile|email|e-mail|linkedin)\b")
_RANK_QUERY_RE = re.compile(r"(?i)\b(rank|ranking|top\s*\d+|top|best|compare|comparison|vs\.?|versus)\b")
_TOP_N_RE = re.compile(r"(?i)\btop\s*(\d{1,3})\b")
_PROFILE_ID_KV_RE = re.compile(r"(?i)\bprofile[_\s-]?id\s*[:=]\s*([\w\-]+)\b")

_GLOBAL_SCOPE_RE = re.compile(r"(?i)\b(all|each|every)\s+(candidate|candidates|profile|profiles)\b")


def _payload_to_text(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in ("text", "chunk_text", "content", "page_text", "body"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _payload_to_profile_name(payload: Dict[str, Any]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    for key in ("profile_name", "profileName", "candidate_name", "candidateName", "name", "full_name", "fullName"):
        value = payload.get(key)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            return value_str
    return None


def _get_profile_name_hint(
    *,
    client: QdrantClient,
    subscription_id: str,
    profile_id: str,
    collection: str,
) -> Optional[str]:
    filt = build_qdrant_filter(subscription_id=str(subscription_id), profile_id=str(profile_id))
    try:
        points, _ = client.scroll(
            collection_name=collection,
            scroll_filter=filt,
            limit=8,
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        points = []

    for pt in points or []:
        payload = getattr(pt, "payload", None) or {}
        name = _payload_to_profile_name(payload)
        if name:
            return name
    return None


def _detect_contact_request(query: str) -> bool:
    query = query or ""
    if not query.strip():
        return False
    if not _CONTACT_QUERY_RE.search(query):
        return False
    q_low = query.lower()
    if "contact" in q_low:
        return True
    if any(key in q_low for key in ("phone", "mobile", "email", "e-mail", "linkedin")):
        return True
    return False


def _detect_rank_request(query: str) -> bool:
    query = query or ""
    if not query.strip():
        return False
    if not _RANK_QUERY_RE.search(query):
        return False
    q_low = query.lower()
    if any(term in q_low for term in ("profile", "profiles", "candidate", "candidates", "resume", "resumes", "cv")):
        return True
    if _PROFILE_ID_KV_RE.search(query):
        return True
    return False


def _parse_top_n(query: str, default: int = 3) -> int:
    match = _TOP_N_RE.search(query or "")
    if not match:
        return default
    try:
        n = int(match.group(1))
    except Exception:
        return default
    return max(1, min(n, 50))


def _select_scope_profile_ids(
    *,
    query: str,
    all_profile_ids: Sequence[str],
    profile_id_to_name: Dict[str, str],
) -> List[str]:
    if not all_profile_ids:
        return []

    query = query or ""
    q_low = query.lower()

    if _GLOBAL_SCOPE_RE.search(query):
        return list(all_profile_ids)

    match = _PROFILE_ID_KV_RE.search(query)
    if match:
        wanted = match.group(1).strip()
        if wanted in all_profile_ids:
            return [wanted]

    # Fast path: UUID-ish / token mention.
    for pid in all_profile_ids:
        if pid and pid in query:
            return [pid]

    # Name match (token-based).
    for pid, name in profile_id_to_name.items():
        name = (name or "").strip()
        if not name:
            continue
        tokens = [t for t in re.split(r"\s+", name.lower()) if len(t) >= 3]
        if not tokens:
            continue
        if any(t in q_low for t in tokens):
            return [pid]

    return list(all_profile_ids)


def _render_contact_info(value: List[str]) -> str:
    return ", ".join(value) if value else "Not Mentioned"


def build_contacts_answer(reports: List[Dict[str, Any]]) -> str:
    lines = ["Contact information"]
    for report in reports:
        name = report.get("profile_name") or report.get("profile_id") or "Unknown"
        pid = report.get("profile_id") or ""
        contacts = report.get("contacts") or {}
        lines.append(f"- {name} (profile_id: {pid})")
        lines.append(f"  - Phone: {_render_contact_info(contacts.get('phones') or [])}")
        lines.append(f"  - Email: {_render_contact_info(contacts.get('emails') or [])}")
        lines.append(f"  - LinkedIn: {_render_contact_info(contacts.get('linkedins') or [])}")
    return "\n".join(lines).strip()


def build_ranking_answer(ranked: List[Dict[str, Any]], top_n: int) -> str:
    lines = [f"Top {min(top_n, len(ranked))} profiles"]
    for idx, item in enumerate(ranked[:top_n], start=1):
        name = item.get("profile_name") or item.get("profile_id") or "Unknown"
        pid = item.get("profile_id") or ""
        signals = item.get("signals") or {}
        lines.append(f"{idx}) {name} (profile_id: {pid})")
        lines.append(
            "   - Signals: "
            f"skills_count={signals.get('skills_count', 0)}, "
            f"experience_markers={signals.get('experience_markers', 0)}, "
            f"date_presence={signals.get('date_presence', False)}, "
            f"total_chars={signals.get('total_chars', 0)}, "
            f"contact_presence={signals.get('contact_presence', False)}"
        )
    return "\n".join(lines).strip()


def maybe_answer_multi_profile_deterministic(
    *,
    client: QdrantClient,
    subscription_id: str,
    collection: str,
    query: str,
    profile_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Handle certain multi-profile tasks deterministically (no LLM extraction).

    - Contact info requests: scan full per-profile corpus and extract phones/emails/linkedin.
    - Rank/top/compare requests: score all profiles deterministically and return top N.
    """
    if not subscription_id or not collection:
        return None

    wants_contacts = _detect_contact_request(query)
    wants_rank = _detect_rank_request(query)
    if not (wants_contacts or wants_rank):
        return None

    if not profile_id:
        return None
    all_profile_ids = [str(profile_id)]

    # Build lightweight name hints for routing (avoid full-corpus loads unless needed).
    profile_id_to_name: Dict[str, str] = {}
    for pid in all_profile_ids:
        name = _get_profile_name_hint(client=client, subscription_id=subscription_id, profile_id=pid, collection=collection)
        if name:
            profile_id_to_name[pid] = name

    scope_profile_ids = list(all_profile_ids)

    if wants_contacts:
        reports: List[Dict[str, Any]] = []
        for pid in scope_profile_ids:
            payloads = get_profile_points(
                client=client, subscription_id=subscription_id, profile_id=pid, collection=collection
            )
            texts = [_payload_to_text(p) for p in payloads]
            contacts: ContactInfo = extract_contacts("\n".join(texts))
            profile_name = profile_id_to_name.get(pid)
            if not profile_name:
                for payload in payloads:
                    profile_name = _payload_to_profile_name(payload)
                    if profile_name:
                        break
            reports.append(
                {
                    "profile_id": pid,
                    "profile_name": profile_name or "",
                    "contacts": asdict(contacts),
                }
            )
        response = build_contacts_answer(reports)
        return {
            "response": response,
            "sources": [],
            "grounded": True,
            "deterministic": True,
            "candidate_universe": list(all_profile_ids),
            "candidate_scope": list(scope_profile_ids),
        }

    # Ranking is handled by the main retrieval pipeline within the active profile scope.
    return None


__all__ = [
    "maybe_answer_multi_profile_deterministic",
    "build_contacts_answer",
    "build_ranking_answer",
]
