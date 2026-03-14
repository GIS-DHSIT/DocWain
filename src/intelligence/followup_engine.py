"""
Smart follow-up suggestions engine for DocWain.

After every response, generates up to 3 contextual follow-up questions.
3-tier fallback: LLM (3s timeout) -> semantic adjacency -> domain+intent templates.
Templates: 6 domains x 5 intents = 30 template sets.
"""

from __future__ import annotations

from src.utils.logging_utils import get_logger
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = get_logger(__name__)

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
        "extraction": [
            "Can you list all the technical skills from this resume?",
            "What are the job titles held by this candidate?",
            "Extract all educational qualifications and institutions.",
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
        "extraction": [
            "Extract all monetary amounts and penalties from this contract.",
            "What are the specific dates and deadlines mentioned?",
            "List all defined terms and their definitions.",
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
        "extraction": [
            "Extract all lab values and their reference ranges.",
            "What medications and dosages are listed?",
            "List all diagnoses and ICD codes mentioned.",
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
        "extraction": [
            "Extract all line items with quantities and unit prices.",
            "What are the tax amounts by category?",
            "List all payment terms and conditions.",
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
        "extraction": [
            "Extract all coverage limits and deductible amounts.",
            "What are the specific exclusions listed?",
            "List all endorsements and riders with their terms.",
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
        "extraction": [
            "Extract all names and organizations mentioned.",
            "What are the key dates and figures in this document?",
            "List all specific data points and their values.",
        ],
    },
}

# ── Analytical Intent Templates (cross-doc, timeline, risk, trend) ────────

_ANALYTICAL_INTENT_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "hr": {
        "cross_document": [
            "How do the qualifications compare across all candidates?",
            "Which candidates have overlapping skill sets?",
            "What common experience patterns emerge across resumes?",
        ],
        "timeline": [
            "What is this candidate's career progression timeline?",
            "When did the candidate transition between roles?",
            "How long has the candidate been in each position?",
        ],
        "analytics": [
            "What trends emerge in candidate qualifications?",
            "What is the average experience level across candidates?",
            "Which skills appear most frequently across all resumes?",
        ],
        "risk": [
            "Are there any employment gaps in this candidate's history?",
            "Does this candidate meet the minimum requirements?",
            "Are there any red flags in the candidate's background?",
        ],
    },
    "legal": {
        "cross_document": [
            "How do the terms compare across these agreements?",
            "Are there conflicting clauses between the contracts?",
            "What obligations overlap across documents?",
        ],
        "timeline": [
            "What are the key contractual milestones and deadlines?",
            "When do the various obligations come into effect?",
            "What is the timeline for notice periods and renewals?",
        ],
        "analytics": [
            "What patterns emerge in the liability provisions?",
            "How do indemnification caps trend across agreements?",
            "What is the distribution of risk allocation?",
        ],
        "risk": [
            "Which clauses pose the greatest legal exposure?",
            "Are there any compliance gaps in these contracts?",
            "What are the potential financial penalties across agreements?",
        ],
    },
    "medical": {
        "cross_document": [
            "How do test results compare across visits?",
            "Are there conflicting diagnoses between providers?",
            "What treatments have been tried across records?",
        ],
        "timeline": [
            "How has the patient's condition progressed over time?",
            "When were key medications started or changed?",
            "What is the timeline of diagnostic tests?",
        ],
        "analytics": [
            "What trends appear in the patient's lab values?",
            "How effective have the treatments been based on outcomes?",
            "What patterns emerge in symptom recurrence?",
        ],
        "risk": [
            "Are there potential drug interaction risks?",
            "What conditions require urgent follow-up?",
            "Are there any gaps in recommended screenings?",
        ],
    },
    "invoice": {
        "cross_document": [
            "How do costs compare across vendors?",
            "Are there duplicate charges across invoices?",
            "What items appear on multiple invoices?",
        ],
        "timeline": [
            "How have costs changed over the billing period?",
            "What is the payment history for this vendor?",
            "When are the upcoming payment deadlines?",
        ],
        "analytics": [
            "What is the spending trend by category?",
            "Which vendors represent the highest total spend?",
            "What is the average invoice value by period?",
        ],
        "risk": [
            "Are there any unusual pricing patterns?",
            "Which invoices are at risk of late payment?",
            "Are there discrepancies between PO and invoice amounts?",
        ],
    },
    "policy": {
        "cross_document": [
            "How does coverage compare across policies?",
            "Are there gaps between the policies?",
            "What exclusions overlap across documents?",
        ],
        "timeline": [
            "When do the various coverage periods begin and end?",
            "What is the renewal schedule across policies?",
            "When were the last premium adjustments made?",
        ],
        "analytics": [
            "What is the total coverage across all policies?",
            "How do premium-to-coverage ratios compare?",
            "What patterns appear in claim frequencies?",
        ],
        "risk": [
            "What scenarios could void coverage?",
            "Are there any coverage gaps that need addressing?",
            "Which exclusions pose the greatest financial risk?",
        ],
    },
    "generic": {
        "cross_document": [
            "How do the documents compare on key points?",
            "Are there any contradictions between documents?",
            "What themes appear across multiple documents?",
        ],
        "timeline": [
            "What is the chronological sequence of events described?",
            "What key dates and deadlines are mentioned?",
            "How has the situation evolved over time?",
        ],
        "analytics": [
            "What patterns emerge from the data in these documents?",
            "What are the key metrics or statistics mentioned?",
            "Can you identify any trends across the information?",
        ],
        "risk": [
            "What potential issues or risks are highlighted?",
            "Are there any areas that need immediate attention?",
            "What gaps exist in the available information?",
        ],
    },
}

def _tokenize(text: str) -> Set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 2}

_ANALYTICAL_INTENT_KEYS = frozenset({
    "cross_document", "timeline", "analytics", "risk",
})

def _infer_intent_key(intent_type: Optional[str]) -> str:
    """Map intent type to template key."""
    if not intent_type:
        return "factual"
    lower = intent_type.lower()
    # Check analytical intents first
    for key in _ANALYTICAL_INTENT_KEYS:
        if key in lower:
            return key
    # Also map common synonyms to analytical intents
    if any(w in lower for w in ("trend", "pattern", "analys")):
        return "analytics"
    if any(w in lower for w in ("cross", "multi", "across")):
        return "cross_document"
    if any(w in lower for w in ("chronolog", "progression", "history")):
        return "timeline"
    if any(w in lower for w in ("risk", "gap", "anomal", "flag")):
        return "risk"
    for key in ("contact", "rank", "compare", "summary", "extraction"):
        if key in lower:
            return key
    # Map common extraction synonyms
    if any(w in lower for w in ("extract", "list", "enumerate", "pull out")):
        return "extraction"
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

    # Check both standard and analytical template dictionaries
    templates = _DOMAIN_INTENT_TEMPLATES.get(domain_key, {}).get(intent_key, [])
    if not templates:
        templates = _ANALYTICAL_INTENT_TEMPLATES.get(domain_key, {}).get(intent_key, [])
    if not templates:
        templates = _DOMAIN_INTENT_TEMPLATES["generic"]["factual"]

    query_tokens = _tokenize(query)
    results = []
    for tpl in templates:
        tpl_tokens = _tokenize(tpl)
        overlap = len(query_tokens & tpl_tokens)
        # Include templates that have SOME topical overlap with the query
        # but are NOT just restating it (overlap between 10% and 70%)
        relevance_ratio = overlap / max(len(query_tokens), 1)
        if 0.1 <= relevance_ratio <= 0.7:
            results.append(
                FollowUpSuggestion(question=tpl, source="template", relevance=0.6)
            )

    # Mix in one analytical suggestion if we used standard templates
    if intent_key not in _ANALYTICAL_INTENT_KEYS and len(results) < max_count:
        analytical_templates = _ANALYTICAL_INTENT_TEMPLATES.get(domain_key, {})
        # Pick from cross_document or analytics as they're most generally useful
        for analytical_key in ("cross_document", "analytics"):
            pool = analytical_templates.get(analytical_key, [])
            for tpl in pool:
                tpl_tokens = _tokenize(tpl)
                overlap = len(query_tokens & tpl_tokens)
                relevance_ratio = overlap / max(len(query_tokens), 1)
                if 0.1 <= relevance_ratio <= 0.7:
                    results.append(
                        FollowUpSuggestion(
                            question=tpl, source="template", relevance=0.5,
                        )
                    )
                    break
            if len(results) >= max_count + 1:
                break

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

    # Extract named entities from chunks for more meaningful suggestions
    _entity_re = re.compile(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b')
    _generic_entities = frozenset({
        "The", "This", "That", "Based", "Summary", "Document", "Section",
        "Table", "Note", "Page", "Source", "Evidence",
    })

    uncovered_topics: List[tuple[str, float]] = []
    for chunk in chunk_texts:
        chunk_tokens = _tokenize(chunk)
        novel = chunk_tokens - covered
        if len(novel) >= 3:
            novelty_ratio = len(novel) / max(len(chunk_tokens), 1)
            # Prefer named entities as topic phrases when available
            chunk_entities = [m.group() for m in _entity_re.finditer(chunk)
                            if m.group() not in _generic_entities and len(m.group()) > 3]
            uncovered_entities = [e for e in chunk_entities
                                 if e.lower() not in covered and e.lower() not in query_tokens]
            if uncovered_entities:
                topic_phrase = uncovered_entities[0]
                uncovered_topics.append((topic_phrase, novelty_ratio + 0.1))
                continue
            # Fallback: pick the most substantive words (longer = more specific)
            sample_words = sorted(novel, key=len, reverse=True)[:4]
            # Filter out very short/generic tokens and pure numbers
            sample_words = [w for w in sample_words if len(w) > 3 and not w.isdigit()]
            if len(sample_words) < 2:
                continue
            # Limit to 3 words max to avoid unnatural concatenated phrases
            topic_phrase = " ".join(sample_words[:3])
            uncovered_topics.append((topic_phrase, novelty_ratio))

    uncovered_topics.sort(key=lambda x: x[1], reverse=True)

    # Varied question templates for more natural follow-ups
    _QUESTION_TEMPLATES = [
        "What details are available about {}?",
        "What does the document say about {}?",
        "Can you provide more information on {}?",
        "What else is mentioned about {}?",
        "Are there additional details on {}?",
    ]

    results = []
    seen_phrases: Set[str] = set()
    for idx, (topic, relevance) in enumerate(uncovered_topics[:max_count * 2]):
        normalized = topic.lower()
        if normalized in seen_phrases:
            continue
        seen_phrases.add(normalized)
        template = _QUESTION_TEMPLATES[idx % len(_QUESTION_TEMPLATES)]
        question = template.format(topic)
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

    domain_hint = f"\nDomain context: {domain}" if domain else ""
    prompt = (
        f"Based on this Q&A exchange, suggest exactly {max_count} follow-up questions "
        f"the user might want to ask next. Make them specific and actionable, "
        f"not generic. Return ONLY the questions, one per line.{domain_hint}\n\n"
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
    """Call LLM client, handling gateway, generate_with_metadata, and raw ollama patterns."""
    # Prefer generate_with_metadata (gateway client pattern)
    if hasattr(llm_client, "generate_with_metadata"):
        result = llm_client.generate_with_metadata(prompt)
        if isinstance(result, tuple):
            return result[0] if result[0] else ""
        return result or ""
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

    # Entity-aware personalization: replace generic placeholders with actual names
    _entity_name = _extract_primary_entity(response, query)
    if _entity_name:
        personalized: List[FollowUpSuggestion] = []
        for s in suggestions:
            q = s.question
            # Replace generic references with the actual entity name
            q = re.sub(r'\bthis candidate\b', _entity_name, q, flags=re.IGNORECASE)
            q = re.sub(r'\bthis person\b', _entity_name, q, flags=re.IGNORECASE)
            q = re.sub(r'\bthe patient\b', _entity_name, q, flags=re.IGNORECASE)
            q = re.sub(r'\bthis contract\b', 'the agreement', q, flags=re.IGNORECASE)
            q = re.sub(r'\bthis invoice\b', 'the invoice', q, flags=re.IGNORECASE)
            q = re.sub(r'\bthis policy\b', 'the policy', q, flags=re.IGNORECASE)
            q = re.sub(r'\bthis document\b', 'the document', q, flags=re.IGNORECASE)
            q = re.sub(r'\bthis employee\b', _entity_name, q, flags=re.IGNORECASE)
            personalized.append(FollowUpSuggestion(
                question=q, source=s.source, relevance=s.relevance,
            ))
        suggestions = personalized

    # Deduplicate by lowercase question text
    seen: Set[str] = set()
    unique: List[FollowUpSuggestion] = []
    for s in suggestions:
        key = s.question.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return [s.to_dict() for s in unique[:max_count]]

def _extract_primary_entity(response: str, query: str) -> Optional[str]:
    """Extract the primary entity name from the response or query for personalization."""
    _ENTITY_RE = re.compile(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,2}\b')
    _GENERIC = frozenset({
        "The", "This", "That", "Based", "According", "Summary", "Details",
        "Document", "Evidence", "Source", "Table", "Section", "Note",
        "Important", "However", "Therefore", "Additionally", "Furthermore",
        "Candidate", "Patient", "Client", "Vendor", "Invoice", "Policy",
        "Not Found", "Not Mentioned", "Not Available",
    })

    # Try query first (user's intent is more specific)
    for text in [query, response[:500]]:
        for m in _ENTITY_RE.finditer(text):
            name = m.group()
            if name not in _GENERIC and len(name) > 3:
                return name
    return None
