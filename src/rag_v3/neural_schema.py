"""Neural Schema — Profile-level intelligence and intent understanding.

Provides high-level context awareness for the RAG pipeline:

1. **QueryIntent**: Structured decomposition of what the user wants
   (action, top_n, criteria, entities) — no need to ask the user.
2. **CandidateDigest**: Compact per-candidate representation for ranking.
3. **ProfileIntelligence**: Profile-level understanding (document inventory,
   domain distribution, common skills, candidate digests).
4. **Intelligent ranking**: Multi-criteria scoring with profile-aware defaults.
5. **Direct response formatting**: Answers the question, not a data dump.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# QueryIntent — structured intent decomposition
# ---------------------------------------------------------------------------

@dataclass
class QueryIntent:
    """Structured decomposition of what the user wants."""
    action: str = "detail"      # rank, compare, filter, list, detail, summarize
    top_n: Optional[int] = None # "top 2" → 2, None = all
    criteria: List[str] = field(default_factory=list)  # inferred ranking criteria
    entities: List[str] = field(default_factory=list)   # mentioned names
    is_question: bool = False   # "who is the best?" vs "rank them"

# Regex patterns for intent parsing
_TOP_N_RE = re.compile(
    r"\b(?:top|best|first|leading)\s+(\d+)\b", re.IGNORECASE,
)
_BOTTOM_N_RE = re.compile(
    r"\b(?:bottom|worst|last|lowest)\s+(\d+)\b", re.IGNORECASE,
)
_RANK_RE = re.compile(
    r"\b(?:rank|ranking|rate|score|order|sort|benchmark|evaluate)\b", re.IGNORECASE,
)
_COMPARE_RE = re.compile(
    r"\b(?:compare|comparison|vs\.?|versus|differences?|similarities?|side.by.side)\b", re.IGNORECASE,
)
_FILTER_RE = re.compile(
    r"\b(?:who\s+(?:has|have|is|are|can)|which\s+(?:candidate|resume|person)|find\s+(?:candidates?|people|resumes?))\b",
    re.IGNORECASE,
)
_LIST_RE = re.compile(
    r"\b(?:list|show|display|enumerate)\s+(?:all\s+)?(?:candidates?|resumes?|profiles?|documents?)\b",
    re.IGNORECASE,
)
_SUMMARIZE_RE = re.compile(
    r"\b(?:summarize|summary|overview|brief)\b", re.IGNORECASE,
)
_QUESTION_RE = re.compile(
    r"\b(?:who|what|which|how|where|when)\b.*\??\s*$", re.IGNORECASE,
)

# Criteria extraction patterns
_CRITERIA_PATTERNS = [
    (re.compile(r"\b(?:sap|erp|s/4\s*hana|s4hana)\b", re.I), "sap"),
    (re.compile(r"\b(?:python|java|javascript|react|node|docker|kubernetes)\b", re.I), "programming"),
    (re.compile(r"\b(?:experience|years?|senior|seasoned)\b", re.I), "experience"),
    (re.compile(r"\b(?:skills?|technical|technologies|tech\s*stack)\b", re.I), "skills"),
    (re.compile(r"\b(?:certification|certified|credential)\b", re.I), "certifications"),
    (re.compile(r"\b(?:education|degree|university|academic|qualification)\b", re.I), "education"),
    (re.compile(r"\b(?:procurement|supply\s*chain|inventory|logistics|warehouse)\b", re.I), "supply_chain"),
    (re.compile(r"\b(?:data\s*science|machine\s*learning|ai|deep\s*learning|nlp)\b", re.I), "data_science"),
    (re.compile(r"\b(?:management|leadership|lead|manager)\b", re.I), "leadership"),
    (re.compile(r"\b(?:project|agile|scrum|pmp|capm)\b", re.I), "project_management"),
]

# Stop words for criteria extraction
_CRITERIA_STOP = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "from",
    "what", "how", "who", "where", "when", "which", "is", "are", "was", "were",
    "do", "does", "did", "can", "could", "would", "should", "will",
    "this", "that", "these", "those", "me", "my", "your", "his", "her",
    "tell", "give", "show", "find", "get", "list", "all", "about",
    "rank", "top", "best", "worst", "first", "last", "most", "least",
    "resume", "resumes", "candidate", "candidates", "profile", "profiles",
    "document", "documents", "them", "they", "please", "based",
})


def parse_query_intent(query: str, candidate_names: Optional[List[str]] = None) -> QueryIntent:
    """Parse a query into structured intent without asking the user.

    Extracts:
    - action: rank/compare/filter/list/detail/summarize
    - top_n: explicit count ("top 2") or None
    - criteria: inferred from query keywords + domain patterns
    - entities: mentioned candidate names
    - is_question: whether it's a question form
    """
    if not query:
        return QueryIntent(action="detail")

    lowered = query.lower().strip()

    # --- Extract top_n ---
    top_n = None
    m = _TOP_N_RE.search(query)
    if m:
        top_n = int(m.group(1))
    else:
        m = _BOTTOM_N_RE.search(query)
        if m:
            top_n = int(m.group(1))

    # Also handle "who are the top 2?" without explicit "top N" format
    if top_n is None:
        # Check for standalone numbers that imply selection
        # "rank 2 resumes" → 2, "top two" → 2
        _number_words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
        for word, num in _number_words.items():
            if re.search(rf"\b(?:top|best|first)\s+{word}\b", lowered):
                top_n = num
                break

    # --- Determine action ---
    action = "detail"
    if _RANK_RE.search(query) or top_n is not None:
        action = "rank"
    elif _COMPARE_RE.search(query):
        action = "compare"
    elif _FILTER_RE.search(query):
        action = "filter"
    elif _LIST_RE.search(query):
        action = "list"
    elif _SUMMARIZE_RE.search(query):
        action = "summarize"

    # "who are the top 2?" is a ranking question, not a filter
    if top_n is not None and action in ("filter", "detail"):
        action = "rank"

    # --- Extract criteria ---
    criteria: List[str] = []
    for pattern, criterion in _CRITERIA_PATTERNS:
        if pattern.search(query):
            criteria.append(criterion)

    # If no specific criteria found, extract meaningful keywords
    if not criteria:
        tokens = re.findall(r"[a-z]+", lowered)
        for tok in tokens:
            if tok not in _CRITERIA_STOP and len(tok) > 3:
                criteria.append(tok)
        criteria = criteria[:3]  # limit to top 3

    # --- Extract entity mentions ---
    entities: List[str] = []
    if candidate_names:
        for name in candidate_names:
            if name and name.lower() in lowered:
                entities.append(name)

    # --- Question detection ---
    is_question = bool(_QUESTION_RE.search(query))

    return QueryIntent(
        action=action,
        top_n=top_n,
        criteria=criteria,
        entities=entities,
        is_question=is_question,
    )


# ---------------------------------------------------------------------------
# CandidateDigest — compact per-candidate representation
# ---------------------------------------------------------------------------

@dataclass
class CandidateDigest:
    """Compact representation of a candidate for intelligent ranking."""
    name: str = ""
    doc_id: str = ""
    role: str = ""
    years_experience: Optional[float] = None
    skill_count: int = 0
    cert_count: int = 0
    education_level: str = ""  # "PhD", "Masters", "Bachelors", "Diploma", "N/A"
    key_skills: List[str] = field(default_factory=list)
    key_certs: List[str] = field(default_factory=list)
    domain_keywords: Set[str] = field(default_factory=set)
    summary_snippet: str = ""
    raw_candidate: Optional[Any] = field(default=None, repr=False)

    @property
    def completeness_score(self) -> float:
        """0-1 score for how complete this candidate's profile is."""
        score = 0.0
        if self.name and self.name != "Candidate":
            score += 0.15
        if self.role:
            score += 0.15
        if self.years_experience is not None:
            score += 0.15
        if self.skill_count > 0:
            score += 0.2
        if self.cert_count > 0:
            score += 0.15
        if self.education_level and self.education_level != "N/A":
            score += 0.1
        if self.summary_snippet:
            score += 0.1
        return min(score, 1.0)


def _parse_years_float(value: Any) -> Optional[float]:
    """Extract numeric years from various formats."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1))
    return None


def _infer_education_level(education: Any) -> str:
    """Infer highest education level from education field."""
    if not education:
        return "N/A"
    text = " ".join(str(e) for e in education).lower() if isinstance(education, list) else str(education).lower()
    if any(kw in text for kw in ("phd", "ph.d", "doctorate", "doctoral")):
        return "PhD"
    if any(kw in text for kw in ("m.tech", "mtech", "m.s.", "master", "mba", "m.e.", "msc")):
        return "Masters"
    if any(kw in text for kw in ("b.tech", "btech", "b.e.", "bachelor", "b.sc", "bsc", "b.s.")):
        return "Bachelors"
    if any(kw in text for kw in ("diploma", "associate", "certificate")):
        return "Diploma"
    # Check if there's any education text at all
    if len(text.strip()) > 5:
        return "Other"
    return "N/A"


def build_candidate_digest(candidate: Any) -> CandidateDigest:
    """Build a CandidateDigest from an extracted Candidate object."""
    tech_skills = candidate.technical_skills or []
    func_skills = candidate.functional_skills or []
    certs = candidate.certifications or []
    education = candidate.education if hasattr(candidate, "education") else []

    # Clean and deduplicate top skills
    all_skills = list(dict.fromkeys(s.strip() for s in tech_skills + func_skills if s and len(s.strip()) > 1))
    clean_certs = list(dict.fromkeys(c.strip() for c in certs if c and len(c.strip()) > 2))

    # Extract domain keywords from skills
    domain_kws: Set[str] = set()
    for skill in all_skills[:20]:
        for word in skill.lower().split():
            if len(word) > 3 and word not in _CRITERIA_STOP:
                domain_kws.add(word)

    role = getattr(candidate, "role", "") or getattr(candidate, "designation", "") or ""
    summary = candidate.experience_summary or ""

    return CandidateDigest(
        name=candidate.name or "Candidate",
        role=role.strip(),
        years_experience=_parse_years_float(candidate.total_years_experience),
        skill_count=len(all_skills),
        cert_count=len(clean_certs),
        education_level=_infer_education_level(education),
        key_skills=all_skills[:8],
        key_certs=clean_certs[:5],
        domain_keywords=domain_kws,
        summary_snippet=summary[:200].strip() if summary else "",
        raw_candidate=candidate,
    )


# ---------------------------------------------------------------------------
# ProfileIntelligence — profile-level context
# ---------------------------------------------------------------------------

@dataclass
class ProfileIntelligence:
    """Profile-level understanding of the document collection."""
    candidate_count: int = 0
    candidates: List[CandidateDigest] = field(default_factory=list)
    domain_distribution: Dict[str, int] = field(default_factory=dict)
    common_skills: List[str] = field(default_factory=list)
    primary_domain: str = "generic"
    avg_experience_years: Optional[float] = None

    @property
    def candidate_names(self) -> List[str]:
        return [c.name for c in self.candidates if c.name and c.name != "Candidate"]


def build_profile_intelligence(candidates: List[Any]) -> ProfileIntelligence:
    """Build profile-level intelligence from extracted candidates."""
    if not candidates:
        return ProfileIntelligence()

    digests = [build_candidate_digest(c) for c in candidates]

    # Compute common skills (appearing in 2+ candidates)
    skill_counts: Dict[str, int] = {}
    for d in digests:
        seen = set()
        for skill in d.key_skills:
            sl = skill.lower()
            if sl not in seen:
                skill_counts[sl] = skill_counts.get(sl, 0) + 1
                seen.add(sl)
    common = sorted(
        [(s, c) for s, c in skill_counts.items() if c >= 2],
        key=lambda x: x[1], reverse=True,
    )
    common_skills = [s for s, _ in common[:10]]

    # Average experience
    years_list = [d.years_experience for d in digests if d.years_experience is not None]
    avg_years = sum(years_list) / len(years_list) if years_list else None

    return ProfileIntelligence(
        candidate_count=len(digests),
        candidates=digests,
        common_skills=common_skills,
        avg_experience_years=avg_years,
    )


# ---------------------------------------------------------------------------
# Intelligent Ranking — multi-criteria scoring
# ---------------------------------------------------------------------------

# Criteria → scoring function mapping
_EDUCATION_RANK = {
    "PhD": 5, "Masters": 4, "Bachelors": 3, "Diploma": 2, "Other": 1, "N/A": 0,
}


def rank_by_criteria(
    profile: ProfileIntelligence,
    intent: QueryIntent,
) -> List[Tuple[CandidateDigest, float, str]]:
    """Rank candidates intelligently based on query intent and profile context.

    Returns list of (digest, score, reason) tuples, sorted by score descending.
    The *reason* explains why this candidate ranked where they did.
    """
    if not profile.candidates:
        return []

    criteria = intent.criteria or []
    query_keywords = set()
    for c in criteria:
        query_keywords.update(c.lower().split("_"))

    scored: List[Tuple[CandidateDigest, float, str]] = []

    for digest in profile.candidates:
        score = 0.0
        reasons: List[str] = []

        # 1. Experience depth (normalized to 0-10 scale)
        if "experience" in criteria or not criteria:
            if digest.years_experience is not None:
                exp_score = min(digest.years_experience, 20.0) / 2.0  # max 10 pts
                score += exp_score
                if digest.years_experience >= 5:
                    reasons.append(f"{digest.years_experience:.0f} years experience")

        # 2. Skills breadth
        if "skills" in criteria or "programming" in criteria or not criteria:
            skill_score = min(digest.skill_count, 15) * 0.5  # max 7.5 pts
            score += skill_score
            if digest.skill_count >= 5:
                reasons.append(f"{digest.skill_count} skills")

        # 3. Certifications
        if "certifications" in criteria or not criteria:
            cert_score = min(digest.cert_count, 5) * 1.5  # max 7.5 pts
            score += cert_score
            if digest.cert_count > 0:
                reasons.append(f"{digest.cert_count} certifications")

        # 4. Education level
        if "education" in criteria or not criteria:
            edu_score = _EDUCATION_RANK.get(digest.education_level, 0) * 1.0  # max 5 pts
            score += edu_score
            if digest.education_level not in ("N/A", "Other"):
                reasons.append(digest.education_level)

        # 5. Role relevance (if criteria mention specific roles/domains)
        if digest.role and query_keywords:
            role_lower = digest.role.lower()
            role_matches = sum(1 for kw in query_keywords if kw in role_lower)
            score += role_matches * 3.0
            if role_matches > 0:
                reasons.append(f"role: {digest.role[:50]}")

        # 6. Domain-specific keyword matching
        if query_keywords:
            domain_matches = len(query_keywords & digest.domain_keywords)
            score += domain_matches * 2.0
            if domain_matches > 0:
                matching_skills = [s for s in digest.key_skills
                                   if any(kw in s.lower() for kw in query_keywords)]
                if matching_skills:
                    reasons.append(f"relevant skills: {', '.join(matching_skills[:3])}")

        # 7. Supply chain domain boost
        if "supply_chain" in criteria:
            sc_keywords = {"procurement", "supply", "chain", "inventory", "logistics", "warehouse", "sap"}
            sc_matches = len(sc_keywords & digest.domain_keywords)
            score += sc_matches * 2.5
            if sc_matches > 0:
                reasons.append(f"supply chain expertise ({sc_matches} matches)")

        # 8. Data science domain boost
        if "data_science" in criteria:
            ds_keywords = {"python", "machine", "learning", "data", "science", "nlp", "deep"}
            ds_matches = len(ds_keywords & digest.domain_keywords)
            score += ds_matches * 2.5

        # 9. SAP domain boost
        if "sap" in criteria:
            sap_keywords = {"sap", "hana", "mm", "sd", "pp", "fi", "co", "scm"}
            sap_matches = len(sap_keywords & digest.domain_keywords)
            score += sap_matches * 3.0
            sap_certs = [c for c in digest.key_certs if "sap" in c.lower()]
            score += len(sap_certs) * 2.0
            if sap_matches > 0 or sap_certs:
                reasons.append(f"SAP expertise")

        # 10. Profile completeness bonus (tiebreaker)
        score += digest.completeness_score * 2.0

        reason_text = "; ".join(reasons[:4]) if reasons else "overall profile strength"
        scored.append((digest, score, reason_text))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Apply top_n selection
    if intent.top_n is not None and intent.top_n > 0:
        scored = scored[:intent.top_n]

    return scored


# ---------------------------------------------------------------------------
# Response formatting — direct, query-answering format
# ---------------------------------------------------------------------------

def format_ranking_response(
    ranked: List[Tuple[CandidateDigest, float, str]],
    intent: QueryIntent,
    profile: ProfileIntelligence,
) -> str:
    """Format ranking results as a direct answer to the user's query.

    No generic preamble ("I understand your question...").
    Direct answer with reasoning.
    """
    if not ranked:
        return "No candidates found to rank."

    total = profile.candidate_count
    shown = len(ranked)

    # Build header based on intent
    if intent.top_n is not None:
        if intent.criteria:
            criteria_text = _format_criteria(intent.criteria)
            header = f"**Top {shown} of {total} candidates** (ranked by {criteria_text}):\n"
        else:
            header = f"**Top {shown} of {total} candidates** (ranked by overall profile strength):\n"
    else:
        if intent.criteria:
            criteria_text = _format_criteria(intent.criteria)
            header = f"**Ranking of {shown} candidates** (by {criteria_text}):\n"
        else:
            header = f"**Ranking of {shown} candidates** (by overall profile strength):\n"

    lines = [header]

    for idx, (digest, score, reason) in enumerate(ranked, start=1):
        name = digest.name or "Candidate"
        # Build concise one-liner
        parts = []
        if digest.role:
            parts.append(digest.role[:60])
        if digest.years_experience is not None:
            parts.append(f"{digest.years_experience:.0f} yrs exp")
        if digest.key_skills:
            skills_str = ", ".join(digest.key_skills[:5])
            parts.append(f"Skills: {skills_str}")
        if digest.key_certs:
            certs_str = ", ".join(digest.key_certs[:3])
            parts.append(f"Certs: {certs_str}")
        if digest.education_level not in ("N/A", "Other", ""):
            parts.append(digest.education_level)

        detail = " | ".join(parts) if parts else "See profile for details"
        lines.append(f"{idx}. **{name}**")
        lines.append(f"   {detail}")
        lines.append(f"   *Why: {reason}*")
        lines.append("")

    # Add brief context note for partial rankings
    if intent.top_n is not None and shown < total:
        remaining = total - shown
        lines.append(f"_{remaining} other candidate{'s' if remaining != 1 else ''} not shown._")

    return "\n".join(lines).strip()


def format_comparison_response(
    digests: List[CandidateDigest],
    intent: QueryIntent,
    profile: ProfileIntelligence,
) -> str:
    """Format comparison results as a direct answer."""
    if not digests:
        return "No candidates to compare."

    if len(digests) == 2:
        return _format_two_way_comparison(digests[0], digests[1], intent)

    return _format_multi_comparison(digests, intent)


def _format_two_way_comparison(a: CandidateDigest, b: CandidateDigest, intent: QueryIntent) -> str:
    """Side-by-side comparison of two candidates."""
    lines = [f"**Candidate Comparison: {a.name} vs {b.name}**\n"]

    # Experience
    a_years = f"{a.years_experience:.0f} years" if a.years_experience else "N/A"
    b_years = f"{b.years_experience:.0f} years" if b.years_experience else "N/A"
    lines.append(f"| Aspect | {a.name} | {b.name} |")
    lines.append(f"|--------|{'---' * max(3, len(a.name)//3)}|{'---' * max(3, len(b.name)//3)}|")
    lines.append(f"| Experience | {a_years} | {b_years} |")
    lines.append(f"| Skills | {a.skill_count} | {b.skill_count} |")
    lines.append(f"| Certifications | {a.cert_count} | {b.cert_count} |")
    lines.append(f"| Education | {a.education_level} | {b.education_level} |")
    if a.role or b.role:
        lines.append(f"| Role | {a.role[:40] or 'N/A'} | {b.role[:40] or 'N/A'} |")
    lines.append("")

    # Shared skills
    shared = set(s.lower() for s in a.key_skills) & set(s.lower() for s in b.key_skills)
    if shared:
        lines.append(f"**Shared skills:** {', '.join(list(shared)[:5])}")

    # Unique strengths
    a_unique = set(s.lower() for s in a.key_skills) - set(s.lower() for s in b.key_skills)
    b_unique = set(s.lower() for s in b.key_skills) - set(s.lower() for s in a.key_skills)
    if a_unique:
        lines.append(f"**{a.name}'s unique skills:** {', '.join(list(a_unique)[:5])}")
    if b_unique:
        lines.append(f"**{b.name}'s unique skills:** {', '.join(list(b_unique)[:5])}")

    # Key skills detail per candidate
    lines.append("")
    for cand in (a, b):
        skills_str = ", ".join(cand.key_skills[:6]) if cand.key_skills else "N/A"
        cand_years = f"{cand.years_experience:.0f} years" if cand.years_experience else "N/A"
        lines.append(f"**Candidate {cand.name}:** {cand_years} experience, key skills: {skills_str}")

    return "\n".join(lines).strip()


def _format_multi_comparison(digests: List[CandidateDigest], intent: QueryIntent) -> str:
    """Tabular comparison of 3+ candidates."""
    lines = [f"**Comparison of {len(digests)} candidates**\n"]

    # Header row
    names = [d.name[:15] for d in digests]
    lines.append("| Aspect | " + " | ".join(names) + " |")
    lines.append("|--------" + "|---" * len(names) + "|")

    # Experience
    exp = [f"{d.years_experience:.0f}y" if d.years_experience else "N/A" for d in digests]
    lines.append("| Experience | " + " | ".join(exp) + " |")

    # Skills count
    sk = [str(d.skill_count) for d in digests]
    lines.append("| Skills | " + " | ".join(sk) + " |")

    # Certifications
    cr = [str(d.cert_count) for d in digests]
    lines.append("| Certifications | " + " | ".join(cr) + " |")

    # Education
    ed = [d.education_level for d in digests]
    lines.append("| Education | " + " | ".join(ed) + " |")

    lines.append("")

    # Per-candidate detail summary
    for d in digests:
        skills_str = ", ".join(d.key_skills[:5]) if d.key_skills else "N/A"
        d_years = f"{d.years_experience:.0f} years" if d.years_experience else "N/A"
        role_str = f" ({d.role})" if d.role else ""
        lines.append(f"**Candidate {d.name}**{role_str}: {d_years} experience, key skills: {skills_str}")

    # Statistical summary
    years_list = [d.years_experience for d in digests if d.years_experience]
    if years_list:
        avg_y = sum(years_list) / len(years_list)
        lines.append("")
        lines.append(
            f"Experience range: {min(years_list):.0f}-{max(years_list):.0f} years "
            f"(average {avg_y:.1f} years across {len(digests)} candidates)."
        )

    return "\n".join(lines).strip()


def _format_criteria(criteria: List[str]) -> str:
    """Format criteria list for display."""
    if not criteria:
        return "overall profile strength"
    display = [c.replace("_", " ") for c in criteria[:3]]
    if len(display) == 1:
        return display[0]
    return ", ".join(display[:-1]) + " and " + display[-1]
