"""Intelligent USAGE_HELP module for DocWain.

Provides context-aware help responses with domain-specific examples,
task-type demonstrations, and awareness of the user's loaded documents.

Imports only ``ConversationalContext`` from ``conversational_nlp`` to avoid
circular imports (``conversational_nlp`` lazy-imports *this* module).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.intelligence.conversational_nlp import ConversationalContext

# ---------------------------------------------------------------------------
# A. Sub-Intent Classifier
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HelpSubIntent:
    name: str
    confidence: float = 0.85
    domain_hint: Optional[str] = None
    task_hint: Optional[str] = None


QUICK_START = "QUICK_START"
UPLOAD_HELP = "UPLOAD_HELP"
FILE_TYPES = "FILE_TYPES"
QUERY_EXAMPLES = "QUERY_EXAMPLES"
DOMAIN_EXAMPLES = "DOMAIN_EXAMPLES"
TASK_HELP = "TASK_HELP"
SCREENING_HELP = "SCREENING_HELP"
CONTENT_GENERATION_HELP = "CONTENT_GENERATION_HELP"
ADVANCED_FEATURES = "ADVANCED_FEATURES"
CAPABILITY_OVERVIEW = "CAPABILITY_OVERVIEW"

_DOMAIN_KEYWORDS = {
    "resume": re.compile(r"\b(?:resume|cv|candidate|profile|hiring)\b", re.I),
    "invoice": re.compile(r"\b(?:invoice|billing|payment|receipt)\b", re.I),
    "legal": re.compile(r"\b(?:legal|contract|clause|agreement|nda)\b", re.I),
    "medical": re.compile(r"\b(?:medical|patient|diagnosis|prescription|health)\b", re.I),
    "report": re.compile(r"\b(?:report|analysis|findings)\b", re.I),
    "general": re.compile(r"\b(?:general|document)\b", re.I),
}

_TASK_KEYWORDS = {
    "compare": re.compile(r"\b(?:compare|comparison|vs|versus|side.by.side)\b", re.I),
    "rank": re.compile(r"\b(?:rank|ranking|top|best|sort)\b", re.I),
    "summarize": re.compile(r"\b(?:summarize|summarise|summary|overview)\b", re.I),
    "extract": re.compile(r"\b(?:extract|pull\s+out|get\s+(?:the|all))\b", re.I),
    "generate": re.compile(r"\b(?:generate|create|write|draft|compose)\b", re.I),
    "upload": re.compile(r"\b(?:upload|add|import|ingest)\b", re.I),
    "list": re.compile(r"\b(?:list|show\s+all|enumerate)\b", re.I),
    "screen": re.compile(r"\b(?:screen|check|detect|scan)\b", re.I),
}

_SUB_INTENT_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Upload help
    (re.compile(r"\bhow\s+(?:do\s+i|can\s+i|to)\s+(?:upload|add|import|ingest)\b", re.I), UPLOAD_HELP),
    (re.compile(r"\b(?:upload|add)\s+(?:a\s+)?(?:document|file|resume|invoice)\b", re.I), UPLOAD_HELP),
    (re.compile(r"\bupload\s+(?:help|guide|instructions?)\b", re.I), UPLOAD_HELP),
    # File types
    (re.compile(r"\b(?:file|document)\s+(?:types?|formats?)\s+(?:supported|accepted|allowed)\b", re.I), FILE_TYPES),
    (re.compile(r"\bwhat\s+(?:file|document)?\s*(?:types?|formats?)\b", re.I), FILE_TYPES),
    (re.compile(r"\bcan\s+i\s+upload\s+(?:images?|pdf|excel|word|csv|pptx?)\b", re.I), FILE_TYPES),
    (re.compile(r"\bsupported\s+(?:file\s+)?(?:types?|formats?)\b", re.I), FILE_TYPES),
    # Screening help
    (re.compile(r"\bscreening\s+(?:help|guide|how)\b", re.I), SCREENING_HELP),
    (re.compile(r"\bhow\s+(?:does|do)\s+screening\s+work\b", re.I), SCREENING_HELP),
    (re.compile(r"\b(?:check|detect|scan)\s+(?:for\s+)?(?:pii|plagiarism|ai)\b", re.I), SCREENING_HELP),
    (re.compile(r"\bpii\s+(?:detection|screening|check)\b", re.I), SCREENING_HELP),
    # Content generation help
    (re.compile(r"\b(?:content\s+generation|generate\s+content)\s+(?:help|guide|how)\b", re.I), CONTENT_GENERATION_HELP),
    (re.compile(r"\bhow\s+(?:do\s+i|can\s+i|to)\s+generate\b", re.I), CONTENT_GENERATION_HELP),
    (re.compile(r"\bwhat\s+(?:can\s+i|content)\s+(?:generate|create)\b", re.I), CONTENT_GENERATION_HELP),
    # Capability overview
    (re.compile(r"\bwhat\s+(?:else\s+|all\s+)?can\s+(?:you|docwain)\s+(?:do|help\s+with)\b", re.I), CAPABILITY_OVERVIEW),
    (re.compile(r"\bwhat\s+can\s+(?:docwain|i)\s+do\s+with\s+docwain\b", re.I), CAPABILITY_OVERVIEW),
    (re.compile(r"\bshow\s+(?:me\s+)?what\s+(?:you|docwain)\s+can\s+do\b", re.I), CAPABILITY_OVERVIEW),
    (re.compile(r"\bhow\s+can\s+(?:you|docwain)\s+help(?:\s+me)?\b", re.I), CAPABILITY_OVERVIEW),
    (re.compile(r"\b(?:list|show)\s+(?:your\s+)?(?:features|capabilities)\b", re.I), CAPABILITY_OVERVIEW),
    # Advanced features
    (re.compile(r"\badvanced\s+(?:features?|capabilities?|options?)\b", re.I), ADVANCED_FEATURES),
    (re.compile(r"\bfine.?tun(?:e|ing)\s+(?:help|guide|how)\b", re.I), ADVANCED_FEATURES),
    (re.compile(r"\bknowledge\s+graph\b", re.I), ADVANCED_FEATURES),
    (re.compile(r"\bteams\s+(?:integration|bot|setup)\b", re.I), ADVANCED_FEATURES),
    # Task help (how to do X)
    (re.compile(r"\bhow\s+(?:do\s+i|can\s+i|to)\s+(?:compare|rank|summarize|summarise|extract|list|find|screen)\b", re.I), TASK_HELP),
    # Domain examples
    (re.compile(r"\b(?:resume|invoice|legal|medical|contract|report)\s+(?:examples?|help|queries|questions)\b", re.I), DOMAIN_EXAMPLES),
    (re.compile(r"\bshow\s+(?:me\s+)?(?:resume|invoice|legal|medical|contract)\s+(?:examples?|queries)\b", re.I), DOMAIN_EXAMPLES),
    (re.compile(r"\bexamples?\s+(?:for|of)\s+(?:resume|invoice|legal|medical|contract)\b", re.I), DOMAIN_EXAMPLES),
    # Query examples
    (re.compile(r"\bwhat\s+(?:can\s+i|should\s+i)\s+ask\b", re.I), QUERY_EXAMPLES),
    (re.compile(r"\bshow\s+(?:me\s+)?examples?\b", re.I), QUERY_EXAMPLES),
    (re.compile(r"\bexample\s+(?:queries|questions)\b", re.I), QUERY_EXAMPLES),
    (re.compile(r"\bsample\s+(?:queries|questions)\b", re.I), QUERY_EXAMPLES),
    # Quick start (most generic — last)
    (re.compile(r"\b(?:getting|get)\s+started\b", re.I), QUICK_START),
    (re.compile(r"\bquick\s+(?:start|guide|tutorial)\b", re.I), QUICK_START),
    (re.compile(r"\b(?:help|guide|tutorial|walkthrough)\b", re.I), QUICK_START),
    (re.compile(r"\bhow\s+(?:do\s+i|can\s+i|to)\s+(?:start|begin|use)\b", re.I), QUICK_START),
]


def _detect_domain_hint(text: str) -> Optional[str]:
    for domain, pat in _DOMAIN_KEYWORDS.items():
        if domain == "general":
            continue
        if pat.search(text):
            return domain
    return None


def _detect_task_hint(text: str) -> Optional[str]:
    for task, pat in _TASK_KEYWORDS.items():
        if pat.search(text):
            return task
    return None


def classify_help_sub_intent(text: str) -> HelpSubIntent:
    """Classify a help query into a specific sub-intent."""
    text = (text or "").strip()
    if not text:
        return HelpSubIntent(name=QUICK_START)

    domain_hint = _detect_domain_hint(text)
    task_hint = _detect_task_hint(text)

    for pat, sub_intent in _SUB_INTENT_PATTERNS:
        if pat.search(text):
            return HelpSubIntent(
                name=sub_intent,
                domain_hint=domain_hint,
                task_hint=task_hint,
            )

    return HelpSubIntent(
        name=QUICK_START,
        domain_hint=domain_hint,
        task_hint=task_hint,
    )


# ---------------------------------------------------------------------------
# B. Example Query Bank
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExampleQuery:
    query: str
    description: str
    domain: str
    task_type: str


_EXAMPLE_BANK: Dict[str, Dict[str, List[ExampleQuery]]] = {
    "resume": {
        "qa": [
            ExampleQuery("What skills does the candidate have?", "Extract technical and soft skills", "resume", "qa"),
            ExampleQuery("What is the candidate's email and phone?", "Contact information retrieval", "resume", "qa"),
            ExampleQuery("Where did the candidate work previously?", "Employment history lookup", "resume", "qa"),
            ExampleQuery("What is the candidate's highest education?", "Education details extraction", "resume", "qa"),
            ExampleQuery("Does the candidate have Python experience?", "Specific skill verification", "resume", "qa"),
        ],
        "summarize": [
            ExampleQuery("Summarize this resume", "Full resume overview", "resume", "summarize"),
            ExampleQuery("Give me a brief profile of the candidate", "Candidate profile summary", "resume", "summarize"),
        ],
        "compare": [
            ExampleQuery("Compare all candidates", "Side-by-side candidate comparison", "resume", "compare"),
            ExampleQuery("Compare the two resumes", "Two-document comparison", "resume", "compare"),
        ],
        "rank": [
            ExampleQuery("Rank candidates by Python experience", "Skill-based ranking", "resume", "rank"),
            ExampleQuery("Who is the best candidate for a backend role?", "Role-fit ranking", "resume", "rank"),
            ExampleQuery("Rank all candidates by experience level", "Experience-based ranking", "resume", "rank"),
        ],
        "extract": [
            ExampleQuery("Extract all certifications", "Certification list extraction", "resume", "extract"),
            ExampleQuery("List all projects mentioned", "Project extraction", "resume", "extract"),
        ],
        "generate": [
            ExampleQuery("Generate a cover letter for this candidate", "Cover letter from resume data", "resume", "generate"),
            ExampleQuery("Create a skills matrix", "Structured skills breakdown", "resume", "generate"),
            ExampleQuery("Write a candidate summary for the hiring manager", "Professional summary generation", "resume", "generate"),
            ExampleQuery("Draft interview questions based on this resume", "Interview prep content", "resume", "generate"),
        ],
        "list": [
            ExampleQuery("List all uploaded resumes", "Document listing", "resume", "list"),
            ExampleQuery("How many candidates are in my profile?", "Candidate count", "resume", "list"),
        ],
    },
    "invoice": {
        "qa": [
            ExampleQuery("What is the invoice total?", "Total amount extraction", "invoice", "qa"),
            ExampleQuery("When is the payment due?", "Due date lookup", "invoice", "qa"),
            ExampleQuery("Who is the vendor?", "Vendor identification", "invoice", "qa"),
            ExampleQuery("What are the line items?", "Line item breakdown", "invoice", "qa"),
        ],
        "summarize": [
            ExampleQuery("Summarize this invoice", "Invoice overview", "invoice", "summarize"),
        ],
        "extract": [
            ExampleQuery("Extract all amounts from the invoice", "Amount extraction", "invoice", "extract"),
            ExampleQuery("Get the tax breakdown", "Tax detail extraction", "invoice", "extract"),
        ],
        "compare": [
            ExampleQuery("Compare the two invoices", "Invoice comparison", "invoice", "compare"),
        ],
        "generate": [
            ExampleQuery("Generate a payment summary", "Payment summary creation", "invoice", "generate"),
            ExampleQuery("Create an expense report from these invoices", "Expense report generation", "invoice", "generate"),
        ],
    },
    "legal": {
        "qa": [
            ExampleQuery("What are the termination clauses?", "Clause extraction", "legal", "qa"),
            ExampleQuery("What is the contract duration?", "Term identification", "legal", "qa"),
            ExampleQuery("Are there any liability limitations?", "Liability clause lookup", "legal", "qa"),
            ExampleQuery("What are the payment terms?", "Payment terms extraction", "legal", "qa"),
        ],
        "summarize": [
            ExampleQuery("Summarize this contract", "Contract overview", "legal", "summarize"),
            ExampleQuery("Give me a brief of the key terms", "Key terms summary", "legal", "summarize"),
        ],
        "generate": [
            ExampleQuery("Generate a contract summary for review", "Contract review document", "legal", "generate"),
            ExampleQuery("Draft key points from this agreement", "Key points extraction", "legal", "generate"),
        ],
    },
    "medical": {
        "qa": [
            ExampleQuery("What medications are prescribed?", "Medication lookup", "medical", "qa"),
            ExampleQuery("What is the diagnosis?", "Diagnosis extraction", "medical", "qa"),
            ExampleQuery("What are the lab results?", "Lab result retrieval", "medical", "qa"),
        ],
        "summarize": [
            ExampleQuery("Summarize the patient record", "Patient record overview", "medical", "summarize"),
        ],
        "generate": [
            ExampleQuery("Generate a patient summary", "Patient summary creation", "medical", "generate"),
        ],
    },
    "general": {
        "qa": [
            ExampleQuery("What is this document about?", "Document topic identification", "general", "qa"),
            ExampleQuery("Find mentions of a specific topic", "Topic search", "general", "qa"),
        ],
        "summarize": [
            ExampleQuery("Summarize this document", "Document overview", "general", "summarize"),
            ExampleQuery("What are the key points?", "Key points extraction", "general", "summarize"),
        ],
        "generate": [
            ExampleQuery("Generate key takeaways", "Takeaway generation", "general", "generate"),
            ExampleQuery("Create a bullet-point summary", "Bullet summary creation", "general", "generate"),
            ExampleQuery("Draft a brief of this document", "Brief generation", "general", "generate"),
        ],
    },
}


# ---------------------------------------------------------------------------
# C. Context-Aware Example Selector
# ---------------------------------------------------------------------------

def _flatten_examples(
    bank: Dict[str, Dict[str, List[ExampleQuery]]],
    domain_filter: Optional[str] = None,
    task_filter: Optional[str] = None,
) -> List[ExampleQuery]:
    """Flatten the example bank with optional domain/task filters."""
    result: List[ExampleQuery] = []
    for domain, tasks in bank.items():
        if domain_filter and domain != domain_filter:
            continue
        for task_type, examples in tasks.items():
            if task_filter and task_type != task_filter:
                continue
            result.extend(examples)
    return result


def select_examples(
    ctx: ConversationalContext,
    sub_intent: HelpSubIntent,
    seed: int,
    max_examples: int = 5,
) -> List[ExampleQuery]:
    """Select context-aware example queries."""
    # If domain or task hint is explicit, filter to those.
    if sub_intent.domain_hint:
        candidates = _flatten_examples(_EXAMPLE_BANK, domain_filter=sub_intent.domain_hint)
        if not candidates:
            candidates = _flatten_examples(_EXAMPLE_BANK, domain_filter="general")
    elif sub_intent.task_hint:
        candidates = _flatten_examples(_EXAMPLE_BANK, task_filter=sub_intent.task_hint)
    elif ctx.profile_is_empty:
        # Empty profile — general examples only.
        candidates = _flatten_examples(_EXAMPLE_BANK, domain_filter="general")
    elif ctx.dominant_domains:
        # Prefer examples from user's domains: 60% primary, 20% secondary, 20% general.
        primary = ctx.dominant_domains[0] if ctx.dominant_domains else "general"
        secondary = ctx.dominant_domains[1] if len(ctx.dominant_domains) > 1 else "general"
        primary_ex = _flatten_examples(_EXAMPLE_BANK, domain_filter=primary)
        secondary_ex = _flatten_examples(_EXAMPLE_BANK, domain_filter=secondary)
        general_ex = _flatten_examples(_EXAMPLE_BANK, domain_filter="general")
        # Mix: 60/20/20 ratio.
        n_primary = max(1, int(max_examples * 0.6))
        n_secondary = max(1, int(max_examples * 0.2))
        n_general = max(1, max_examples - n_primary - n_secondary)
        candidates = []
        candidates.extend(_deterministic_sample(primary_ex, n_primary, seed))
        candidates.extend(_deterministic_sample(secondary_ex, n_secondary, seed >> 4))
        candidates.extend(_deterministic_sample(general_ex, n_general, seed >> 8))
        # Deduplicate.
        seen = set()
        deduped = []
        for ex in candidates:
            if ex.query not in seen:
                seen.add(ex.query)
                deduped.append(ex)
        return deduped[:max_examples]
    else:
        candidates = _flatten_examples(_EXAMPLE_BANK)

    return _deterministic_sample(candidates, max_examples, seed)


def _deterministic_sample(
    items: List[ExampleQuery],
    n: int,
    seed: int,
) -> List[ExampleQuery]:
    """Deterministically pick *n* items using seed-based index selection."""
    if not items:
        return []
    if len(items) <= n:
        return list(items)
    selected: List[ExampleQuery] = []
    seen_indices: set = set()
    s = seed
    while len(selected) < n and len(seen_indices) < len(items):
        idx = s % len(items)
        if idx not in seen_indices:
            seen_indices.add(idx)
            selected.append(items[idx])
        s = (s * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF  # LCG step
    return selected


# ---------------------------------------------------------------------------
# D. Static Help Content
# ---------------------------------------------------------------------------

_UPLOAD_HELP_TEXT = (
    "**How to Upload Documents**\n\n"
    "1. Use the upload endpoint or UI to add documents to your profile.\n"
    "2. Supported formats: PDF, DOCX, PPTX, TXT, CSV, XLSX, JPG, PNG.\n"
    "3. Documents are automatically chunked, embedded, and indexed.\n"
    "4. Once uploaded, you can immediately ask questions about them.\n\n"
    "Each document is scoped to your profile — only you can query it."
)

_FILE_TYPES_TEXT = (
    "**Supported File Formats**\n\n"
    "| Format | Extensions |\n"
    "|--------|------------|\n"
    "| PDF | .pdf |\n"
    "| Word | .docx |\n"
    "| PowerPoint | .pptx |\n"
    "| Text | .txt |\n"
    "| CSV | .csv |\n"
    "| Excel | .xlsx |\n"
    "| Images | .jpg, .png |\n\n"
    "DocWain automatically extracts text from all formats, including OCR for images."
)

_SCREENING_HELP_TEXT = (
    "**Document Screening**\n\n"
    "DocWain can screen documents for quality and compliance:\n\n"
    "- **PII Detection** — Identify personal information (names, emails, phone numbers, SSNs)\n"
    "- **AI Authorship** — Detect AI-generated content\n"
    "- **Readability** — Score document readability and complexity\n"
    "- **Resume Screening** — Evaluate resume quality and completeness\n"
    "- **Plagiarism** — Check for duplicate or copied content\n"
    "- **Bias Detection** — Identify potential bias in language\n"
    "- **Compliance** — Check against regulatory requirements\n"
    "- **Format Validation** — Verify document structure and formatting\n\n"
    "Use the screening gateway or ask: \"Screen this document for PII.\""
)

_ADVANCED_FEATURES_TEXT = (
    "**Advanced Features**\n\n"
    "- **Fine-Tuning** — Train custom models on your document collection for improved accuracy\n"
    "- **Content Generation** — Generate cover letters, summaries, key points, and more from document data\n"
    "- **Multi-Document Analysis** — Compare, rank, and analyze across multiple documents simultaneously\n"
    "- **Microsoft Teams Integration** — Query documents directly from Teams chat\n"
    "- **Document Screening** — AI-powered screening for PII, plagiarism, readability, and more\n"
    "- **Custom Profiles** — Organize documents into separate profiles for different projects\n"
    "- **Unified Gateway** — Single API endpoint for all document operations"
)

_CONTENT_GEN_HELP_TEXT = (
    "**Content Generation**\n\n"
    "DocWain can generate content grounded in your documents:\n\n"
    "**HR / Resume**\n"
    "- Cover letters, candidate summaries, interview questions, skills matrices\n\n"
    "**Invoice / Financial**\n"
    "- Payment summaries, expense reports, financial overviews\n\n"
    "**Legal**\n"
    "- Contract summaries, key term briefs, clause analysis\n\n"
    "**General**\n"
    "- Document summaries, key takeaways, bullet-point briefs\n\n"
    "Try: \"Generate a cover letter\" or \"Create a skills matrix.\""
)

_CAPABILITY_OVERVIEW_TEXT = (
    "**What DocWain Can Do**\n\n"
    "DocWain is built for document intelligence in your profile:\n\n"
    "1. **Answer questions** from uploaded documents with grounded evidence.\n"
    "2. **Summarize** single or multiple documents.\n"
    "3. **Extract data** like skills, totals, clauses, dates, contacts, and line items.\n"
    "4. **Compare and rank** documents (for example, candidates or invoices).\n"
    "5. **Generate content** such as summaries, briefs, cover letters, and interview questions.\n"
    "6. **Run screening checks** for PII, AI-authorship signals, readability, and policy checks.\n\n"
    "**How to ask effectively**\n"
    "- Be specific: \"Rank candidates by Python and backend experience.\"\n"
    "- Name the target: \"Compare Alice and Bob resumes.\"\n"
    "- Ask for format: \"Give me a bullet summary with key risks.\"\n"
)

_TASK_HELP_TEMPLATES: Dict[str, str] = {
    "compare": (
        "**How to Compare Documents**\n\n"
        "Ask DocWain to compare two or more documents side by side:\n\n"
        "- \"Compare all candidates\" — compares all resumes in your profile\n"
        "- \"Compare the two invoices\" — side-by-side invoice comparison\n"
        "- \"Compare Alice and Bob\" — compare specific candidates by name\n\n"
        "DocWain extracts key fields and presents a structured comparison."
    ),
    "rank": (
        "**How to Rank Documents**\n\n"
        "Ask DocWain to rank documents by specific criteria:\n\n"
        "- \"Rank candidates by Python experience\" — skill-based ranking\n"
        "- \"Who is the best candidate for a backend role?\" — role-fit ranking\n"
        "- \"Rank all candidates by experience level\" — experience-based ranking\n\n"
        "DocWain scores and orders candidates using extracted document data."
    ),
    "summarize": (
        "**How to Summarize Documents**\n\n"
        "Ask DocWain for summaries at any level:\n\n"
        "- \"Summarize this document\" — full document overview\n"
        "- \"Give me a brief of the key terms\" — focused summary\n"
        "- \"What are the key points?\" — bullet-point highlights\n\n"
        "DocWain pulls the most important information from your documents."
    ),
    "generate": (
        "**How to Generate Content**\n\n"
        "Ask DocWain to create new content from your documents:\n\n"
        "- \"Generate a cover letter\" — from resume data\n"
        "- \"Create a skills matrix\" — structured breakdown\n"
        "- \"Draft interview questions\" — from candidate profile\n\n"
        "All generated content is grounded in your actual document data."
    ),
    "extract": (
        "**How to Extract Information**\n\n"
        "Ask DocWain to pull specific data from documents:\n\n"
        "- \"Extract all certifications\" — list certifications\n"
        "- \"Get the contact details\" — email, phone, address\n"
        "- \"What are the line items?\" — invoice items\n\n"
        "DocWain extracts structured data from unstructured documents."
    ),
    "upload": (
        "**How to Upload Documents**\n\n"
        "Upload documents to your profile via the API or UI:\n\n"
        "1. Select files (PDF, DOCX, TXT, images, etc.)\n"
        "2. Upload to your profile\n"
        "3. DocWain automatically processes and indexes them\n"
        "4. Start asking questions immediately\n\n"
        "Supported formats: PDF, DOCX, PPTX, TXT, CSV, XLSX, JPG, PNG."
    ),
    "screen": (
        "**How to Screen Documents**\n\n"
        "Ask DocWain to screen documents for various quality checks:\n\n"
        "- \"Screen this document for PII\" — detect personal information\n"
        "- \"Check for AI-generated content\" — AI authorship detection\n"
        "- \"Evaluate readability\" — readability scoring\n\n"
        "Screening results include detailed findings and scores."
    ),
    "list": (
        "**How to List Documents**\n\n"
        "Ask DocWain about your loaded documents:\n\n"
        "- \"List all uploaded resumes\" — see all documents\n"
        "- \"How many candidates are in my profile?\" — document count\n"
        "- \"What documents do I have?\" — profile overview\n\n"
        "DocWain tracks all documents in your profile."
    ),
}


# ---------------------------------------------------------------------------
# E. Response Composer
# ---------------------------------------------------------------------------

def _format_example(ex: ExampleQuery) -> str:
    return f"- \"{ex.query}\" \u2014 {ex.description}"


def _group_examples_by_task(examples: List[ExampleQuery]) -> Dict[str, List[ExampleQuery]]:
    """Group examples by task type for formatted display."""
    groups: Dict[str, List[ExampleQuery]] = {}
    for ex in examples:
        groups.setdefault(ex.task_type, []).append(ex)
    return groups


_TASK_TYPE_LABELS = {
    "qa": "Questions & Answers",
    "summarize": "Summarization",
    "compare": "Comparison",
    "rank": "Ranking",
    "extract": "Data Extraction",
    "generate": "Content Generation",
    "list": "Document Listing",
    "screen": "Screening",
}


def _format_grouped_examples(examples: List[ExampleQuery]) -> str:
    """Format examples grouped by task type."""
    groups = _group_examples_by_task(examples)
    parts: List[str] = []
    for task_type in ["qa", "summarize", "compare", "rank", "extract", "generate", "list", "screen"]:
        group = groups.get(task_type)
        if not group:
            continue
        label = _TASK_TYPE_LABELS.get(task_type, task_type.title())
        parts.append(f"**{label}**")
        for ex in group:
            parts.append(_format_example(ex))
        parts.append("")
    return "\n".join(parts).strip()


def _make_seed(user_key: str) -> int:
    """Deterministic seed from user key."""
    raw = f"usage_help:{user_key}"
    return int(hashlib.md5(raw.encode()).hexdigest(), 16)


def _closing_line(ctx: ConversationalContext) -> str:
    """Context-aware closing line."""
    if ctx.profile_is_empty:
        return "Upload some documents to get started!"
    if ctx.document_count == 1:
        return "With your document loaded, try asking a question about it."
    domains = ", ".join(ctx.dominant_domains[:2]) if ctx.dominant_domains else "your"
    return f"With your {ctx.document_count} document(s) loaded, try asking about {domains} data."


def compose_usage_help_response(
    user_text: str,
    ctx: ConversationalContext,
    user_key: str = "",
) -> str:
    """Compose a context-aware usage help response.

    This is the main entry point called from ``conversational_nlp.compose_response()``
    when the intent is USAGE_HELP.
    """
    sub_intent = classify_help_sub_intent(user_text)
    seed = _make_seed(user_key)

    if sub_intent.name == UPLOAD_HELP:
        return _UPLOAD_HELP_TEXT

    if sub_intent.name == FILE_TYPES:
        return _FILE_TYPES_TEXT

    if sub_intent.name == SCREENING_HELP:
        return _SCREENING_HELP_TEXT

    if sub_intent.name == CONTENT_GENERATION_HELP:
        return _CONTENT_GEN_HELP_TEXT

    if sub_intent.name == CAPABILITY_OVERVIEW:
        examples = select_examples(ctx, HelpSubIntent(name=QUERY_EXAMPLES), seed, max_examples=6)
        parts = [_CAPABILITY_OVERVIEW_TEXT]
        if examples:
            parts.append("\n**Try these queries:**\n" + _format_grouped_examples(examples))
        parts.append("\n\n" + _closing_line(ctx))
        return "".join(parts)

    if sub_intent.name == ADVANCED_FEATURES:
        return _ADVANCED_FEATURES_TEXT

    if sub_intent.name == TASK_HELP:
        task = sub_intent.task_hint or "summarize"
        template = _TASK_HELP_TEMPLATES.get(task, _TASK_HELP_TEMPLATES["summarize"])
        # Add a couple of context-aware examples.
        examples = select_examples(ctx, sub_intent, seed, max_examples=3)
        if examples:
            template += "\n\n**More examples:**\n" + "\n".join(_format_example(e) for e in examples)
        return template

    if sub_intent.name == DOMAIN_EXAMPLES:
        domain = sub_intent.domain_hint or (ctx.dominant_domains[0] if ctx.dominant_domains else "general")
        examples = select_examples(
            ctx,
            HelpSubIntent(name=DOMAIN_EXAMPLES, domain_hint=domain),
            seed,
            max_examples=7,
        )
        if not examples:
            examples = select_examples(
                ctx,
                HelpSubIntent(name=DOMAIN_EXAMPLES, domain_hint="general"),
                seed,
                max_examples=5,
            )
        header = f"Here are some {domain} queries you can try:\n\n"
        body = _format_grouped_examples(examples)
        closing = "\n\n" + _closing_line(ctx)
        return header + body + closing

    if sub_intent.name == QUERY_EXAMPLES:
        examples = select_examples(ctx, sub_intent, seed, max_examples=7)
        header = "Here are some example queries you can try:\n\n"
        body = _format_grouped_examples(examples)
        closing = "\n\n" + _closing_line(ctx)
        return header + body + closing

    # QUICK_START (default).
    if ctx.profile_is_empty:
        return (
            "Welcome to DocWain! Here's how to get started:\n\n"
            "1. **Upload documents** — PDFs, Word docs, images, spreadsheets, and more.\n"
            "2. **Ask questions** — Query your documents in natural language.\n"
            "3. **Get answers** — DocWain retrieves relevant sections and responds.\n\n"
            "**Example queries:**\n"
            "- \"Summarize this document\"\n"
            "- \"What are the key points?\"\n"
            "- \"Extract the contact details\"\n\n"
            "Upload a document to begin!"
        )

    # Quick start with documents loaded — show context-aware examples.
    examples = select_examples(ctx, sub_intent, seed, max_examples=5)
    parts: List[str] = ["Here's what you can do with DocWain:\n"]
    if examples:
        parts.append(_format_grouped_examples(examples))
    parts.append("")
    parts.append(_closing_line(ctx))
    return "\n".join(parts)


__all__ = [
    "HelpSubIntent",
    "ExampleQuery",
    "classify_help_sub_intent",
    "select_examples",
    "compose_usage_help_response",
    "QUICK_START", "UPLOAD_HELP", "FILE_TYPES", "QUERY_EXAMPLES",
    "DOMAIN_EXAMPLES", "TASK_HELP", "SCREENING_HELP",
    "CONTENT_GENERATION_HELP", "ADVANCED_FEATURES", "CAPABILITY_OVERVIEW",
]
