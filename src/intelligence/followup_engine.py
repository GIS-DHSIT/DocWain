"""
Smart follow-up suggestions engine for DocWain.

After every response, generates up to 3 contextual follow-up questions.
3-tier fallback: LLM (3s timeout) -> semantic adjacency -> domain+intent templates.
Templates: 6 domains x 5 intents = 30 template sets.
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass
class FollowUpSuggestion:
    """A suggested follow-up question."""

    question: str
    source: str  # "llm", "semantic", "template"
    relevance: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "source": self.source,
            "relevance": round(self.relevance, 2),
        }


# ── Domain x Intent Template Sets ─────────────────────────────────────────

_DOMAIN_INTENT_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "hr": {
        "factual": [
            "What certifications does this candidate hold?",
            "What is this candidate's educational background?",
            "How many years of experience does this person have?",
        ],
        "contact": [
            "What other contact channels are available?",
            "Is there a LinkedIn profile for this candidate?",
            "What is the candidate's current location?",
        ],
        "rank": [
            "What specific skills differentiate the top candidates?",
            "How do the candidates compare on years of experience?",
            "Which candidate has the most relevant project experience?",
        ],
        "compare": [
            "Which candidate has stronger technical skills?",
            "How do their educational backgrounds compare?",
            "Who has more leadership experience?",
        ],
        "summary": [
            "What are this candidate's key strengths?",
            "What roles would this candidate be best suited for?",
            "Are there any skill gaps to note?",
        ],
    },
    "legal": {
        "factual": [
            "What are the key obligations in this contract?",
            "What is the governing law for this agreement?",
            "What are the termination conditions?",
        ],
        "contact": [
            "Who are the named parties in this agreement?",
            "What are the signatories' details?",
            "Where should notices be sent?",
        ],
        "rank": [
            "Which clauses carry the highest risk?",
            "What are the most critical deadlines?",
            "Which obligations have the strictest penalties?",
        ],
        "compare": [
            "How do the indemnification terms compare?",
            "Which agreement has stronger IP protections?",
            "How do the liability caps differ?",
        ],
        "summary": [
            "What are the key risks in this contract?",
            "What deadlines should be tracked?",
            "Are there any unusual or non-standard clauses?",
        ],
    },
    "medical": {
        "factual": [
            "What medications are currently prescribed?",
            "What is the patient's diagnosis?",
            "Are there any allergies noted?",
        ],
        "contact": [
            "Who is the referring physician?",
            "What is the patient's emergency contact?",
            "Which department is handling this case?",
        ],
        "rank": [
            "Which conditions are most urgent?",
            "What are the priority follow-up items?",
            "Which lab values are most concerning?",
        ],
        "compare": [
            "How have the lab values changed over time?",
            "How does the current treatment compare to the previous one?",
            "What progress has been made since the last visit?",
        ],
        "summary": [
            "What are the recommended follow-up actions?",
            "Are there any potential drug interactions?",
            "What lifestyle modifications are recommended?",
        ],
    },
    "invoice": {
        "factual": [
            "What is the total amount due?",
            "What is the payment due date?",
            "What are the individual line items?",
        ],
        "contact": [
            "Who is the vendor?",
            "What is the billing address?",
            "What is the purchase order number?",
        ],
        "rank": [
            "Which line items have the highest cost?",
            "What are the largest expenses by category?",
            "Which invoices are overdue?",
        ],
        "compare": [
            "How does this invoice compare to previous ones?",
            "Are there any price changes from the last order?",
            "How do the totals differ across vendors?",
        ],
        "summary": [
            "Are there any discrepancies in the amounts?",
            "What is the tax breakdown?",
            "Are there any applicable discounts?",
        ],
    },
    "policy": {
        "factual": [
            "What is the coverage amount?",
            "What is the policy effective date?",
            "What are the exclusions?",
        ],
        "contact": [
            "Who is the policyholder?",
            "Who is the insurance agent?",
            "What is the claims contact number?",
        ],
        "rank": [
            "Which exclusions pose the highest risk?",
            "What are the most valuable coverage areas?",
            "Which riders provide the best value?",
        ],
        "compare": [
            "How does this policy compare to industry standards?",
            "What coverage gaps exist?",
            "How do the deductibles compare?",
        ],
        "summary": [
            "What are the key coverage highlights?",
            "What are the claim filing requirements?",
            "Are there any conditions that could void coverage?",
        ],
    },
    "generic": {
        "factual": [
            "What other details are available in this document?",
            "Can you extract the key data points?",
            "What dates or numbers are mentioned?",
        ],
        "contact": [
            "Who are the people mentioned in this document?",
            "What organizations are referenced?",
            "Are there any contact details?",
        ],
        "rank": [
            "What are the most important points?",
            "Which items require immediate attention?",
            "What are the key takeaways?",
        ],
        "compare": [
            "How do these documents differ?",
            "What common themes appear across documents?",
            "Are there any contradictions between sources?",
        ],
        "summary": [
            "Can you provide a brief summary?",
            "What are the action items from this document?",
            "What are the main topics covered?",
        ],
    },
}


def _tokenize(text: str) -> Set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 2}


def _infer_intent_key(intent_type: Optional[str]) -> str:
    """Map intent type to template key."""
    if not intent_type:
        return "factual"
    lower = intent_type.lower()
    for key in ("contact", "rank", "compare", "summary"):
        if key in lower:
            return key
    return "factual"


def _infer_domain_key(domain: Optional[str]) -> str:
    """Map domain to template key."""
    if not domain:
        return "generic"
    lower = domain.lower()
    domain_map = {
        "hr": "hr",
        "resume": "hr",
        "legal": "legal",
        "contract": "legal",
        "medical": "medical",
        "invoice": "invoice",
        "policy": "policy",
        "insurance": "policy",
    }
    return domain_map.get(lower, "generic")


def _template_suggestions(
    domain: Optional[str],
    intent_type: Optional[str],
    query: str,
    max_count: int = 3,
) -> List[FollowUpSuggestion]:
    """Tier 3: Generate follow-ups from domain x intent templates."""
    domain_key = _infer_domain_key(domain)
    intent_key = _infer_intent_key(intent_type)

    templates = _DOMAIN_INTENT_TEMPLATES.get(domain_key, {}).get(intent_key, [])
    if not templates:
        templates = _DOMAIN_INTENT_TEMPLATES["generic"]["factual"]

    query_tokens = _tokenize(query)
    results = []
    for tpl in templates:
        tpl_tokens = _tokenize(tpl)
        overlap = len(query_tokens & tpl_tokens)
        if overlap < len(query_tokens) * 0.5:
            results.append(
                FollowUpSuggestion(question=tpl, source="template", relevance=0.6)
            )
    return results[:max_count]


def _semantic_suggestions(
    query: str,
    response: str,
    chunk_texts: List[str],
    max_count: int = 3,
) -> List[FollowUpSuggestion]:
    """Tier 2: Find uncovered chunk topics and generate questions about them."""
    response_tokens = _tokenize(response)
    query_tokens = _tokenize(query)
    covered = response_tokens | query_tokens

    uncovered_topics: List[tuple[str, float]] = []
    for chunk in chunk_texts:
        chunk_tokens = _tokenize(chunk)
        novel = chunk_tokens - covered
        if len(novel) >= 3:
            novelty_ratio = len(novel) / max(len(chunk_tokens), 1)
            sample_words = sorted(novel, key=len, reverse=True)[:5]
            topic_phrase = " ".join(sample_words)
            uncovered_topics.append((topic_phrase, novelty_ratio))

    uncovered_topics.sort(key=lambda x: x[1], reverse=True)

    results = []
    seen_phrases: Set[str] = set()
    for topic, relevance in uncovered_topics[:max_count * 2]:
        normalized = topic.lower()
        if normalized in seen_phrases:
            continue
        seen_phrases.add(normalized)
        question = f"Can you tell me more about {topic}?"
        results.append(
            FollowUpSuggestion(
                question=question, source="semantic", relevance=round(relevance, 2)
            )
        )
        if len(results) >= max_count:
            break

    return results


def _llm_suggestions(
    query: str,
    response: str,
    domain: Optional[str],
    llm_client: Any,
    timeout: float = 3.0,
    max_count: int = 3,
) -> List[FollowUpSuggestion]:
    """Tier 1: Use LLM to generate contextual follow-up questions."""
    if llm_client is None:
        return []

    prompt = (
        f"Based on this Q&A exchange, suggest exactly {max_count} follow-up questions "
        f"the user might want to ask next. Return ONLY the questions, one per line.\n\n"
        f"User question: {query}\n\n"
        f"Answer: {response[:1000]}\n\n"
        f"Suggested follow-ups:"
    )

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call_llm, llm_client, prompt)
            result = future.result(timeout=timeout)

        if not result:
            return []

        lines = [
            line.strip().lstrip("0123456789.-) ")
            for line in result.strip().split("\n")
            if line.strip() and len(line.strip()) > 10
        ]

        suggestions = []
        for line in lines[:max_count]:
            if line.endswith("?") or len(line) > 15:
                if not line.endswith("?"):
                    line = line + "?"
                suggestions.append(
                    FollowUpSuggestion(question=line, source="llm", relevance=0.9)
                )
        return suggestions

    except (FuturesTimeout, Exception) as exc:
        logger.debug("LLM follow-up generation failed: %s", exc)
        return []


def _call_llm(llm_client: Any, prompt: str) -> str:
    """Call LLM client, handling both gateway and raw ollama patterns."""
    if hasattr(llm_client, "generate"):
        result = llm_client.generate(prompt)
        if isinstance(result, tuple):
            return result[0] if result[0] else ""
        return result or ""
    return ""


def generate_followups(
    query: str,
    response: str,
    chunk_texts: Optional[List[str]] = None,
    domain: Optional[str] = None,
    intent_type: Optional[str] = None,
    llm_client: Any = None,
    max_count: int = 3,
    llm_timeout: float = 3.0,
) -> List[Dict[str, Any]]:
    """
    Generate follow-up suggestions using 3-tier fallback.

    Tier 1: LLM generation (if available, 3s timeout)
    Tier 2: Semantic adjacency (uncovered chunk topics)
    Tier 3: Domain x intent templates

    Returns list of suggestion dicts for metadata["suggested_followups"].
    """
    if not query or not response:
        return []

    suggestions: List[FollowUpSuggestion] = []

    # Tier 1: LLM
    if llm_client is not None and len(suggestions) < max_count:
        try:
            llm_results = _llm_suggestions(
                query, response, domain, llm_client, timeout=llm_timeout, max_count=max_count
            )
            suggestions.extend(llm_results)
        except Exception as exc:
            logger.debug("LLM follow-up tier failed: %s", exc)

    # Tier 2: Semantic adjacency
    if len(suggestions) < max_count and chunk_texts:
        remaining = max_count - len(suggestions)
        try:
            semantic_results = _semantic_suggestions(
                query, response, chunk_texts, max_count=remaining
            )
            suggestions.extend(semantic_results)
        except Exception as exc:
            logger.debug("Semantic follow-up tier failed: %s", exc)

    # Tier 3: Templates
    if len(suggestions) < max_count:
        remaining = max_count - len(suggestions)
        try:
            template_results = _template_suggestions(
                domain, intent_type, query, max_count=remaining
            )
            suggestions.extend(template_results)
        except Exception as exc:
            logger.debug("Template follow-up tier failed: %s", exc)

    # Deduplicate by lowercase question text
    seen: Set[str] = set()
    unique: List[FollowUpSuggestion] = []
    for s in suggestions:
        key = s.question.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return [s.to_dict() for s in unique[:max_count]]
