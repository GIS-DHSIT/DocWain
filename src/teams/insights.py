"""Proactive Document Intelligence for Teams.

Generates domain-specific insights and suggested follow-up questions
from document analysis results. Runs after the DI pipeline completes
during file upload.
"""
from __future__ import annotations

import json
from src.utils.logging_utils import get_logger
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Domain prompt templates
# ---------------------------------------------------------------------------

_DOMAIN_PROMPTS: Dict[str, str] = {
    "invoice": (
        "You are a financial analyst reviewing an invoice. "
        "Focus on: payment deadlines, amounts, tax accuracy, vendor details, "
        "line item breakdowns, and any unusual charges."
    ),
    "hr": (
        "You are an HR specialist reviewing a resume or employment document. "
        "Focus on: candidate qualifications, experience gaps, key skills, "
        "certifications, and career progression."
    ),
    "resume": (
        "You are an HR specialist reviewing a resume. "
        "Focus on: candidate strengths, experience highlights, skill gaps, "
        "education relevance, and career trajectory."
    ),
    "medical": (
        "You are a clinical analyst reviewing a medical document. "
        "Focus on: diagnoses, medication interactions, lab values outside "
        "normal range, treatment plans, and follow-up dates."
    ),
    "legal": (
        "You are a legal analyst reviewing a contract or legal document. "
        "Focus on: key obligations, termination clauses, liability limits, "
        "renewal terms, non-compete provisions, and important dates."
    ),
    "contract": (
        "You are a legal analyst reviewing a contract. "
        "Focus on: parties involved, effective dates, key obligations, "
        "termination conditions, penalties, and renewal terms."
    ),
    "policy": (
        "You are an insurance analyst reviewing a policy document. "
        "Focus on: coverage limits, premium amounts, deductibles, exclusions, "
        "effective period, and claims procedures."
    ),
    "tax": (
        "You are a financial analyst reviewing a tax or bank document. "
        "Focus on: total income/expenses, large transactions, tax deductions, "
        "balance trends, and anomalies."
    ),
    "bank": (
        "You are a financial analyst reviewing a bank statement. "
        "Focus on: account balance, large transactions, recurring charges, "
        "income patterns, and unusual activity."
    ),
}

_DOMAIN_ACTIONS: Dict[str, List[Dict[str, str]]] = {
    "invoice": [
        {"title": "Payment details", "action": "domain_query", "query": "What are the payment terms, due date, and bank details?"},
        {"title": "Line items", "action": "domain_query", "query": "List all line items with their amounts"},
        {"title": "Vendor info", "action": "domain_query", "query": "What are the vendor details including name and address?"},
    ],
    "hr": [
        {"title": "Skill summary", "action": "domain_query", "query": "Summarize the candidate's key skills and competencies"},
        {"title": "Experience timeline", "action": "domain_query", "query": "List the work experience chronologically with dates and roles"},
        {"title": "Role fit analysis", "action": "domain_query", "query": "Assess the candidate's strengths and potential gaps"},
    ],
    "resume": [
        {"title": "Skill summary", "action": "domain_query", "query": "Summarize the candidate's key skills and competencies"},
        {"title": "Experience timeline", "action": "domain_query", "query": "List the work experience chronologically with dates and roles"},
        {"title": "Interview questions", "action": "domain_query", "query": "Suggest 5 interview questions based on this resume"},
    ],
    "medical": [
        {"title": "Medication list", "action": "domain_query", "query": "List all medications with dosages and frequencies"},
        {"title": "Lab results", "action": "domain_query", "query": "Summarize all lab results and highlight abnormal values"},
        {"title": "Treatment plan", "action": "domain_query", "query": "What is the current treatment plan and follow-up schedule?"},
    ],
    "legal": [
        {"title": "Key clauses", "action": "domain_query", "query": "Identify and summarize the key clauses in this document"},
        {"title": "Obligations", "action": "domain_query", "query": "What are the obligations and responsibilities of each party?"},
        {"title": "Risk assessment", "action": "domain_query", "query": "Identify potential risks, liabilities, and unfavorable terms"},
    ],
    "contract": [
        {"title": "Key clauses", "action": "domain_query", "query": "Identify and summarize the key clauses in this contract"},
        {"title": "Termination terms", "action": "domain_query", "query": "What are the termination conditions and notice periods?"},
        {"title": "Obligations", "action": "domain_query", "query": "What are the obligations and responsibilities of each party?"},
    ],
    "policy": [
        {"title": "Coverage details", "action": "domain_query", "query": "What is covered and what are the coverage limits?"},
        {"title": "Premium breakdown", "action": "domain_query", "query": "What is the premium amount, payment frequency, and deductible?"},
        {"title": "Exclusions", "action": "domain_query", "query": "What exclusions and limitations apply to this policy?"},
    ],
    "tax": [
        {"title": "Income/expense", "action": "domain_query", "query": "Summarize total income and expenses with categories"},
        {"title": "Tax deductions", "action": "domain_query", "query": "List all tax deductions and credits claimed"},
        {"title": "Anomaly check", "action": "domain_query", "query": "Identify any unusual transactions or discrepancies"},
    ],
    "bank": [
        {"title": "Transaction summary", "action": "domain_query", "query": "Summarize all transactions by category"},
        {"title": "Large transactions", "action": "domain_query", "query": "List all transactions over $1,000"},
        {"title": "Balance trend", "action": "domain_query", "query": "What is the balance trend over the statement period?"},
    ],
}

# Fallback for unknown domains
_DEFAULT_ACTIONS = [
    {"title": "Summarize", "action": "domain_query", "query": "Provide a comprehensive summary of this document"},
    {"title": "Extract key fields", "action": "domain_query", "query": "Extract the most important fields, dates, and amounts from this document"},
    {"title": "Key findings", "action": "domain_query", "query": "What are the key findings and important details?"},
]

@dataclass
class ProactiveInsights:
    """Results from proactive document analysis."""
    summary: str = ""
    insights: List[str] = field(default_factory=list)
    suggested_questions: List[str] = field(default_factory=list)
    domain_actions: List[Dict[str, str]] = field(default_factory=list)
    domain: str = "general"

def _normalize_domain(doc_type: str) -> str:
    """Map document type variations to canonical domain keys."""
    if not doc_type:
        return "general"
    dt = doc_type.lower().strip()
    # Direct matches
    if dt in _DOMAIN_PROMPTS:
        return dt
    # Alias mapping
    aliases = {
        "employment": "hr", "cv": "resume", "curriculum": "resume",
        "insurance": "policy", "coverage": "policy",
        "agreement": "contract", "service agreement": "contract",
        "purchase order": "invoice", "po": "invoice", "bill": "invoice",
        "statement": "bank", "bank statement": "bank",
        "tax return": "tax", "financial": "tax",
        "prescription": "medical", "clinical": "medical",
        "report": "general", "other": "general",
    }
    for alias, domain in aliases.items():
        if alias in dt:
            return domain
    return "general"

def get_domain_actions(doc_type: str) -> List[Dict[str, str]]:
    """Get domain-specific quick actions for a document type."""
    domain = _normalize_domain(doc_type)
    return _DOMAIN_ACTIONS.get(domain, _DEFAULT_ACTIONS)

def generate_proactive_insights(
    doc_type: str,
    summary: str,
    key_entities: List[str],
    key_facts: List[str],
    filename: str,
) -> ProactiveInsights:
    """Generate domain-specific proactive insights from DI results.

    Uses LLM to analyze document summary and produce:
    - 2-3 key insights the user should know immediately
    - 3 suggested follow-up questions specific to this document
    """
    domain = _normalize_domain(doc_type)
    domain_prompt = _DOMAIN_PROMPTS.get(domain, "You are a document analyst.")
    domain_actions = _DOMAIN_ACTIONS.get(domain, _DEFAULT_ACTIONS)

    # Build context for the LLM
    entities_text = ", ".join(key_entities[:10]) if key_entities else "None detected"
    facts_text = "; ".join(key_facts[:10]) if key_facts else "None extracted"

    prompt = f"""{domain_prompt}

Document: {filename}
Type: {doc_type}
Summary: {summary or 'No summary available'}
Key entities: {entities_text}
Key facts: {facts_text}

Based on this document analysis, provide:
1. INSIGHTS: 2-3 most important things the user should know immediately (deadlines, risks, key amounts, anomalies, missing info). Be specific with numbers and dates.
2. QUESTIONS: 3 specific questions the user would likely want to ask about this document. Make them specific to the actual content, not generic.

Respond ONLY with valid JSON:
{{"insights": ["insight1", "insight2", "insight3"], "questions": ["question1", "question2", "question3"]}}"""

    result = ProactiveInsights(
        summary=summary or "",
        domain=domain,
        domain_actions=domain_actions,
    )

    try:
        from src.llm.gateway import get_llm_gateway
        client = get_llm_gateway()
        if client is None:
            logger.debug("No LLM client available for proactive insights")
            return _fallback_insights(result, doc_type, key_entities, key_facts)

        llm_response = client.generate(prompt=prompt, max_tokens=512, temperature=0.3)
        response_text = (llm_response or "").strip()

        # Parse JSON from response (handle markdown code blocks)
        if "```" in response_text:
            # Extract JSON from code block
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                response_text = response_text[start:end]

        parsed = json.loads(response_text)
        result.insights = [str(i) for i in parsed.get("insights", []) if i][:3]
        result.suggested_questions = [str(q) for q in parsed.get("questions", []) if q][:3]

    except json.JSONDecodeError:
        logger.debug("Failed to parse LLM response as JSON for insights")
        return _fallback_insights(result, doc_type, key_entities, key_facts)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Proactive insights LLM call failed: %s", exc)
        return _fallback_insights(result, doc_type, key_entities, key_facts)

    # Ensure we have at least some suggestions
    if not result.insights:
        result = _fallback_insights(result, doc_type, key_entities, key_facts)
    if not result.suggested_questions:
        result.suggested_questions = _fallback_questions(domain)

    return result

def _fallback_insights(
    result: ProactiveInsights,
    doc_type: str,
    key_entities: List[str],
    key_facts: List[str],
) -> ProactiveInsights:
    """Generate basic insights without LLM from available metadata."""
    insights = []
    if key_entities:
        insights.append(f"Key entities: {', '.join(key_entities[:3])}")
    if key_facts:
        insights.append(f"Key facts: {'; '.join(key_facts[:2])}")
    if not insights:
        insights.append(f"Document classified as: {doc_type or 'general'}")
    result.insights = insights
    if not result.suggested_questions:
        result.suggested_questions = _fallback_questions(result.domain)
    return result

def _fallback_questions(domain: str) -> List[str]:
    """Domain-specific fallback questions when LLM is unavailable."""
    fallbacks = {
        "invoice": [
            "What is the total amount due?",
            "Who is the vendor and what are the payment terms?",
            "List all line items with amounts",
        ],
        "hr": [
            "Summarize the candidate's experience",
            "What are the key skills?",
            "Are there any gaps in employment?",
        ],
        "resume": [
            "Summarize the candidate's experience",
            "What are the key skills and certifications?",
            "What is the most recent role?",
        ],
        "medical": [
            "What is the primary diagnosis?",
            "List all current medications",
            "What is the treatment plan?",
        ],
        "legal": [
            "What are the key terms of this agreement?",
            "Who are the parties involved?",
            "What are the termination conditions?",
        ],
        "contract": [
            "What are the key obligations?",
            "When does this contract expire?",
            "What are the termination conditions?",
        ],
        "policy": [
            "What is covered under this policy?",
            "What is the premium amount?",
            "What are the main exclusions?",
        ],
    }
    return fallbacks.get(domain, [
        "Summarize the key points",
        "What are the most important dates?",
        "Extract the key details",
    ])

def generate_followup_suggestions(
    query: str,
    response_text: str,
    domain: str,
    sources_count: int,
) -> List[str]:
    """Generate context-aware follow-up suggestions after an answer.

    Uses simple heuristics (no LLM) for speed.
    """
    suggestions = []
    resp_lower = (response_text or "").lower()
    query_lower = (query or "").lower()

    # Date-based follow-ups
    import re
    dates = re.findall(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b", resp_lower)
    if dates and len(suggestions) < 3:
        suggestions.append(f"What happens after {dates[0].title()}?")

    # Amount-based follow-ups
    amounts = re.findall(r"\$[\d,]+(?:\.\d{2})?", response_text or "")
    if amounts and len(suggestions) < 3:
        suggestions.append(f"Break down the {amounts[0]} in detail")

    # Person-based follow-ups
    # Simple heuristic: capitalized multi-word names
    names = re.findall(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", response_text or "")
    if names and len(suggestions) < 3:
        suggestions.append(f"What else is mentioned about {names[0]}?")

    # Domain-specific follow-ups
    if domain in ("invoice", "tax", "bank") and len(suggestions) < 3:
        if "total" in resp_lower or "amount" in resp_lower:
            suggestions.append("Are there any discrepancies or unusual amounts?")
    if domain in ("medical",) and len(suggestions) < 3:
        suggestions.append("Are there any drug interactions to watch for?")
    if domain in ("legal", "contract") and len(suggestions) < 3:
        suggestions.append("What are the potential risks in this agreement?")

    # Always have at least one suggestion
    if not suggestions:
        suggestions.append("Tell me more about this document")

    return suggestions[:3]
