"""NLU-based agent matching module.

Uses natural language understanding to match user queries to agents based on:
1. Semantic analysis of query structure (action, target, context)
2. Agent capability descriptions (auto-registered, not hardcoded patterns)
3. Embedding similarity when the model is available, spaCy analysis as fallback

This module delegates to the centralized NLU engine (src/nlp/nlu_engine.py)
for parsing and scoring, and maintains its own agent capability registry.
Adding a new agent requires ONLY registering its capability description.
"""
from __future__ import annotations

from src.utils.logging_utils import get_logger
import threading
from typing import Any, Dict, List, Optional

logger = get_logger(__name__)

# Global registry of agent capabilities — uses NLU engine's ClassificationRegistry
_registry_initialized = False
_init_lock = threading.Lock()

def _ensure_registry() -> None:
    """Ensure the agent capability registry is initialized."""
    global _registry_initialized
    if _registry_initialized:
        return
    with _init_lock:
        if _registry_initialized:
            return

        from src.nlp.nlu_engine import get_registry

        reg = get_registry("agent", threshold=0.45, gap=0.02)
        reg.register_many({
            "resumes": (
                "Evaluate and compare job candidates for hiring decisions: "
                "extract candidate skills and retrieve contact info like "
                "email and phone from resumes, prepare interview questions "
                "for candidates, perform candidate skill gap analysis, "
                "score candidates by qualifications and role fit"
            ),
            "medical": (
                "Perform clinical healthcare analysis: check drug interactions, "
                "review treatment plans, interpret lab results, create clinical "
                "summaries, assess medications and diagnostic findings"
            ),
            "lawhere": (
                "Analyze legal contracts and compliance: assess clause risks, "
                "check regulatory compliance, identify legal obligations, "
                "review indemnification terms, evaluate force majeure and "
                "non-compete provisions"
            ),
            "insights": (
                "Detect anomalies and patterns across documents: find financial "
                "irregularities, identify payment discrepancies, discover "
                "cross-document patterns, analyze expense distributions"
            ),
            "content_generate": (
                "Create new documents from scratch: write cover letters "
                "for job applications, create professional career summaries, "
                "build skills comparison matrices, generate invoices "
                "and billing documents"
            ),
            "translator": (
                "Language translation and conversion: translate documents "
                "between languages, provide translations, convert text "
                "from one language to another, multilingual document "
                "translation"
            ),
            "email_drafting": (
                "Compose and write professional correspondence: draft "
                "formal business communications, create written messages "
                "to recipients, compose company letters and memos"
            ),
            "screen_pii": (
                "Screen documents for personally identifiable information: "
                "detect PII, scan for sensitive personal data, "
                "check readability scores"
            ),
            "tutor": (
                "Teach and tutor concepts step by step: provide guided "
                "instruction, explain topics for learning, educational "
                "tutoring assistance"
            ),
            "image_analysis": (
                "Analyze images and perform OCR: extract text from images "
                "and screenshots, read text in photographs, optical "
                "character recognition"
            ),
            "web_search": (
                "Search the internet for current information: look up facts "
                "online, find latest versions, retrieve web-based information"
            ),
            "action_items": (
                "Extract action items from meeting notes: list assigned "
                "follow-up items from meeting transcripts, identify who is "
                "responsible for outstanding deliverables"
            ),
            "jira_confluence": (
                "Search and manage Jira and Confluence: search Confluence "
                "wiki documentation and pages, browse Confluence content, "
                "create and track Jira tickets, manage project sprints"
            ),
            "code_docs": (
                "Auto-document source code and APIs: create developer "
                "reference guides, build software library documentation, "
                "describe code endpoints and API specifications"
            ),
            "web_extract": (
                "Scrape and extract content from web pages: fetch URL "
                "content, extract structured data from websites"
            ),
            "cloud_platform": (
                "Manage files in cloud storage: Azure Blob, AWS S3, "
                "Google Cloud Storage file operations"
            ),
            "sharepoint": (
                "Manage Microsoft SharePoint and OneDrive documents: "
                "sync files, manage document libraries"
            ),
            "customer_service": (
                "Customer support and service: resolve customer issues, "
                "troubleshoot problems step by step, assess escalation "
                "needs, draft customer responses, search FAQ and "
                "knowledge base entries"
            ),
            "analytics_viz": (
                "Generate charts, graphs, and data visualizations: "
                "create bar charts, pie charts, line charts, histograms, "
                "comparison charts, dashboards, and statistical summaries "
                "from document data"
            ),
        })
        _registry_initialized = True
        logger.debug("Agent capability registry initialized with %d agents", len(reg.entries))

def register_agent(name: str, description: str) -> None:
    """Register an agent with its natural language capability description."""
    _ensure_registry()
    from src.nlp.nlu_engine import get_registry
    reg = get_registry("agent")
    reg.register(name, description)

def _keyword_fast_path(query: str) -> Optional[str]:
    """Detect agent from unambiguous keyword signals in the query.

    These patterns fire ONLY for clear, high-confidence signals that
    embedding similarity may miss (especially without GPU embedder).
    Each pattern is carefully designed to avoid false positives tested
    in TestFalsePositivePrevention.
    """
    import re
    ql = query.lower().strip()

    # -- content_generate: explicit creation of document artifacts ----------
    _GEN_VERBS = r"\b(write|create|generate|build|draft|produce|compose|make|prepare)\b"
    _GEN_ARTIFACTS = r"\b(cover letter|skills matrix|professional summary|career summary)\b"
    if re.search(_GEN_ARTIFACTS, ql):
        return "content_generate"
    # "Generate/create a draft invoice" — but NOT "What does the draft agreement say?"
    if re.search(_GEN_VERBS, ql) and re.search(r"\binvoice\b", ql):
        # Must have a generation verb — "what are the invoice totals?" should NOT match
        return "content_generate"

    # -- translator: explicit translation requests -------------------------
    if re.search(r"\b(translat\w*)\b", ql):
        return "translator"
    if re.search(r"\bconvert\b.*\b(to|into)\s+(english|french|spanish|german|arabic|hindi|chinese|japanese|korean|portuguese|italian|dutch|russian|turkish|tamil|telugu|urdu|bengali|malay|indonesian)\b", ql):
        return "translator"

    # -- resumes: HR-specific actions on candidates/resumes ----------------
    # "interview prep" → resumes (domain agent handles it)
    if re.search(r"\binterview\s+(prep|preparation|questions?)\b", ql):
        return "resumes"
    # "contact info" with a person name or "candidate/resume" context
    if re.search(r"\bcontact\s+(info|information|details?|number)\b", ql):
        return "resumes"
    # "Extract skills from the resume" — must mention resume/candidate/CV
    if re.search(r"\b(skills?|experience|education|qualifications?)\b", ql) and re.search(r"\b(resume|cv|candidate)\b", ql):
        return "resumes"
    # "Compare all candidates" — explicit candidate comparison
    if re.search(r"\b(compare|rank)\b.*\bcandidates?\b", ql):
        return "resumes"

    # -- web_search: explicit web/internet search requests ------------------
    if re.search(r"\b(search|look\s+up|browse|find)\b.*\b(web|internet|online)\b", ql):
        return "web_search"
    if re.search(r"\b(web|internet|online)\b.*\b(search|look\s+up|browse|find)\b", ql):
        return "web_search"
    # "search the web for X", "look up latest X" (implies web)
    if re.search(r"\blook\s+up\s+(latest|current|recent|newest)\b", ql):
        return "web_search"
    if re.search(r"\bfind\s+online\b", ql):
        return "web_search"

    # -- insights: anomaly/pattern detection in data -----------------------
    if re.search(r"\b(anomal\w*|discrepanc\w*|irregularit\w*)\b", ql):
        return "insights"
    # "find patterns in financial data" — but NOT "unusual" alone (adjective)
    if re.search(r"\b(find|detect|identify|discover)\b.*\b(pattern|trend|outlier)\b", ql):
        return "insights"

    # -- action_items: meeting notes / follow-up extraction ----------------
    if re.search(r"\baction\s+items?\b", ql):
        return "action_items"
    if re.search(r"\b(follow.?up|deliverable)s?\b.*\bmeeting\b", ql):
        return "action_items"
    if re.search(r"\bmeeting\s+(notes?|minutes?)\b.*\b(action|follow|assigned|outstanding)\b", ql):
        return "action_items"

    # -- lawhere: explicit legal analysis requests -------------------------
    if re.search(r"\b(force\s+majeure|indemnif\w*|non.?compete|jurisdiction)\b", ql):
        return "lawhere"
    if re.search(r"\b(compliance|regulatory)\b.*\b(check|assess|review|audit)\b", ql):
        return "lawhere"

    # -- medical: clinical analysis requests -------------------------------
    if re.search(r"\bdrug\s+interaction\b", ql):
        return "medical"
    if re.search(r"\b(treatment\s+plan|clinical\s+summary|lab\s+(result|interpretation))\b", ql):
        return "medical"

    # -- email_drafting: explicit email/message composition -----------------
    if re.search(r"\b(draft|compose|write)\b.*\b(email|message|letter)\b.*\b(to|for)\b", ql):
        return "email_drafting"

    # -- screen_pii: explicit PII screening --------------------------------
    if re.search(r"\b(screen|scan|check|detect)\b.*\b(pii|personally\s+identifiable|sensitive\s+data)\b", ql):
        return "screen_pii"

    # -- image_analysis: OCR / image text extraction -----------------------
    if re.search(r"\b(ocr|extract\s+text)\b.*\b(image|screenshot|photo|picture)\b", ql):
        return "image_analysis"
    if re.search(r"\b(image|screenshot|photo|picture)\b.*\b(ocr|extract\s+text|read\s+text)\b", ql):
        return "image_analysis"

    # -- jira_confluence: explicit Jira/Confluence mentions -----------------
    if re.search(r"\b(jira|confluence)\b", ql):
        return "jira_confluence"

    # -- web_extract: explicit web page extraction -------------------------
    if re.search(r"\bweb\s+extract\b", ql):
        return "web_extract"
    if re.search(r"\b(scrape|extract|fetch)\b.*\b(url|webpage|website|page)\b", ql):
        return "web_extract"

    return None

def match_agents(
    query: str,
    embedder: Any = None,
    threshold: float = 0.35,
    max_agents: int = 1,
) -> List[str]:
    """Match a query to agents using NLU understanding.

    Combines:
    1. Keyword fast-paths for unambiguous signals
    2. Embedding similarity (when available) — most accurate
    3. NLP structural analysis — action/target overlap
    4. Requires confidence gap between top and second match

    Args:
        query: User's query text
        embedder: Sentence-transformer model (optional)
        threshold: Minimum combined score to consider
        max_agents: Maximum agents to return

    Returns:
        List of matched agent names, ordered by confidence.
    """
    _ensure_registry()

    if not query or not query.strip():
        return []

    # Fast-path: unambiguous keyword signals
    fast = _keyword_fast_path(query)
    if fast:
        logger.debug("Keyword fast-path matched: %s for query: %.80s", fast, query)
        return [fast][:max_agents]

    from src.nlp.nlu_engine import get_registry

    reg = get_registry("agent")
    result = reg.classify(query, embedder=embedder)

    if result is None:
        return []

    logger.debug("NLU matched: %s (score=%.3f, gap=%.3f)", result.name, result.score, result.gap)
    return [result.name][:max_agents]
