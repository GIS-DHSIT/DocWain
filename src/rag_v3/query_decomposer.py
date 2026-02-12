"""Query decomposition engine: transforms complex queries into structured sub-queries."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class SubQuery:
    """A single focused retrieval query with optional scoping constraints."""
    text: str
    entity_scope: Optional[str] = None
    section_focus: Optional[str] = None
    doc_type_filter: Optional[str] = None
    importance: float = 1.0


@dataclass
class DecomposedQuery:
    """Result of query decomposition: original query + structured sub-queries."""
    original: str
    sub_queries: List[SubQuery]
    fusion_strategy: str = "rrf"
    intent: str = "general"


# ── Heuristic patterns ──────────────────────────────────────

_COMPARE_RE = re.compile(
    r"\b(?:compare|contrast|difference|versus|vs\.?)\b", re.IGNORECASE
)
_ENTITY_PAIR_RE = re.compile(
    r"(?:of|between|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
    r"\s+(?:and|vs\.?|versus|&)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
)
_DIRECT_VS_PAIR_RE = re.compile(
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
    r"\s+(?:vs\.?|versus|and)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
)
_ABOUT_ENTITY_RE = re.compile(
    r"\b(?:about|for|of|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})(?:'s?\b|\b)",
)
_POSSESSIVE_ENTITY_RE = re.compile(
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})'s?\s+",
)

_ASPECT_KEYWORDS = {
    "skills": "skills_technical",
    "technical skills": "skills_technical",
    "experience": "experience",
    "work experience": "experience",
    "education": "education",
    "certifications": "certifications",
    "certificates": "certifications",
    "projects": "projects",
    "achievements": "achievements",
    "contact": "identity_contact",
    "phone": "identity_contact",
    "email": "identity_contact",
    "address": "identity_contact",
    "summary": "summary",
}

_VS_SPLIT_RE = re.compile(r"\s+(?:vs\.?|versus)\s+", re.IGNORECASE)


def _extract_entities(query: str) -> List[str]:
    """Extract person/entity names from query text."""
    entities: List[str] = []
    m = _ENTITY_PAIR_RE.search(query)
    if m:
        entities.extend([m.group(1).strip(), m.group(2).strip()])
        return entities
    m = _DIRECT_VS_PAIR_RE.search(query)
    if m:
        entities.extend([m.group(1).strip(), m.group(2).strip()])
        return entities
    for m in _POSSESSIVE_ENTITY_RE.finditer(query):
        name = m.group(1).strip()
        if name.lower() not in ("what", "how", "who", "which", "the", "this", "that"):
            entities.append(name)
    if entities:
        return entities
    m = _ABOUT_ENTITY_RE.search(query)
    if m:
        name = m.group(1).strip()
        if name.lower() not in ("what", "how", "who", "which", "the", "this", "that", "all"):
            entities.append(name)
    return entities


def _extract_aspects(query: str) -> List[str]:
    """Extract section/aspect keywords from query."""
    q_lower = query.lower()
    found = []
    for keyword, section in _ASPECT_KEYWORDS.items():
        if keyword in q_lower and section not in found:
            found.append(section)
    return found


def _strip_entity_from_query(query: str, entity: str) -> str:
    """Remove entity name from query to get the topic part."""
    cleaned = query
    cleaned = re.sub(
        rf"\b{re.escape(entity)}(?:'s?)?\s*", "", cleaned, flags=re.IGNORECASE
    )
    cleaned = re.sub(r"\s+(?:of|for|from|between|and|vs\.?|versus)\s*$", "", cleaned.strip())
    cleaned = re.sub(r"^\s*(?:of|for|from|between|and|vs\.?|versus)\s+", "", cleaned.strip())
    return cleaned.strip() or query


def _build_entity_subquery(entity: str, topic: str, aspects: List[str]) -> SubQuery:
    """Build a focused sub-query for a specific entity."""
    if aspects:
        aspect_text = " ".join(a.replace("_", " ") for a in aspects[:2])
        text = f"{entity} {aspect_text} {topic}".strip()
    else:
        text = f"{entity} {topic}".strip()
    section = aspects[0] if len(aspects) == 1 else None
    return SubQuery(text=text, entity_scope=entity, section_focus=section)


_DECOMPOSE_PROMPT = """Decompose this user query into 2-4 focused sub-queries for document retrieval.
Each sub-query should target specific information. Return strict JSON:
{{
  "sub_queries": [
    {{"text": "focused search query", "entity_scope": "PersonName or null", "section_focus": "skills/experience/education/contact or null"}}
  ]
}}

User query: {query}

JSON response:"""


def _llm_decompose(query: str, llm_client) -> Optional[DecomposedQuery]:
    """Attempt LLM-backed query decomposition."""
    try:
        prompt = _DECOMPOSE_PROMPT.format(query=query)
        response = llm_client.generate(prompt, max_tokens=300, temperature=0.1)
        if not response:
            return None

        text = str(response).strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)

        subs = []
        for sq_data in data.get("sub_queries", [])[:4]:
            subs.append(SubQuery(
                text=sq_data.get("text", query),
                entity_scope=sq_data.get("entity_scope") or None,
                section_focus=sq_data.get("section_focus") or None,
            ))
        if subs:
            return DecomposedQuery(original=query, sub_queries=subs, intent="complex")
    except Exception:
        pass
    return None


def decompose_query(
    query: str,
    *,
    llm_client=None,
    max_sub_queries: int = 4,
) -> DecomposedQuery:
    """Decompose a user query into structured sub-queries.

    Fast heuristic path handles comparisons, multi-entity, and multi-aspect queries.
    LLM path (when llm_client provided) handles complex/ambiguous decompositions.
    """
    query = query.strip()
    if not query:
        return DecomposedQuery(original=query, sub_queries=[SubQuery(text=query)])

    entities = _extract_entities(query)
    aspects = _extract_aspects(query)
    is_comparison = bool(_COMPARE_RE.search(query))

    # ── Strategy 1: Comparison with entity pair ──
    if len(entities) >= 2 and (is_comparison or _VS_SPLIT_RE.search(query)):
        topic = _strip_entity_from_query(query, entities[0])
        topic = _strip_entity_from_query(topic, entities[1])
        topic = re.sub(r"\b(?:compare|contrast|difference)\b", "", topic, flags=re.IGNORECASE).strip()
        topic = re.sub(r"^\s*(?:the|in|for|of)\s+", "", topic).strip()

        subs = [_build_entity_subquery(e, topic, aspects) for e in entities[:max_sub_queries]]
        return DecomposedQuery(
            original=query, sub_queries=subs,
            fusion_strategy="per_entity", intent="compare",
        )

    # ── Strategy 2: Single entity + multiple aspects ──
    if len(entities) == 1 and len(aspects) >= 2:
        entity = entities[0]
        subs = []
        for aspect_section in aspects[:max_sub_queries]:
            aspect_name = aspect_section.replace("_", " ")
            subs.append(SubQuery(
                text=f"{entity} {aspect_name}",
                entity_scope=entity,
                section_focus=aspect_section,
            ))
        return DecomposedQuery(
            original=query, sub_queries=subs,
            fusion_strategy="rrf", intent="extract",
        )

    # ── Strategy 3: Single entity, single/no aspect ──
    if len(entities) == 1:
        section = aspects[0] if len(aspects) == 1 else None
        return DecomposedQuery(
            original=query,
            sub_queries=[SubQuery(text=query, entity_scope=entities[0], section_focus=section)],
            intent="extract",
        )

    # ── Strategy 4: No entities, multiple aspects (all-profile) ──
    if len(aspects) >= 2 and not entities:
        subs = []
        for aspect_section in aspects[:max_sub_queries]:
            subs.append(SubQuery(text=query, section_focus=aspect_section))
        return DecomposedQuery(
            original=query, sub_queries=subs,
            fusion_strategy="rrf", intent="extract",
        )

    # ── Strategy 5: LLM decomposition for complex queries ──
    if llm_client and len(query.split()) >= 10:
        llm_result = _llm_decompose(query, llm_client)
        if llm_result:
            return llm_result

    # ── Fallback: No decomposition needed ──
    section = aspects[0] if len(aspects) == 1 else None
    return DecomposedQuery(
        original=query,
        sub_queries=[SubQuery(text=query, section_focus=section)],
    )
