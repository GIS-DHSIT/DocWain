"""Dynamic Conversational NLP Engine for DocWain.

Generates varied, context-aware responses for non-retrieval intents
(greetings, farewells, meta questions, feedback, small talk).

No imports from other ``src/`` modules — all context is passed as arguments.
"""
from __future__ import annotations

import hashlib
from src.utils.logging_utils import get_logger
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# A. Intent constants
# ---------------------------------------------------------------------------
GREETING = "GREETING"
GREETING_RETURN = "GREETING_RETURN"
FAREWELL = "FAREWELL"
THANKS = "THANKS"
PRAISE = "PRAISE"
NEGATIVE_MILD = "NEGATIVE_MILD"
NEGATIVE_STRONG = "NEGATIVE_STRONG"
IDENTITY = "IDENTITY"
CAPABILITY = "CAPABILITY"
HOW_IT_WORKS = "HOW_IT_WORKS"
PRIVACY = "PRIVACY"
LIMITATIONS = "LIMITATIONS"
USAGE_HELP = "USAGE_HELP"
SMALL_TALK = "SMALL_TALK"
CLARIFICATION = "CLARIFICATION"
DOCUMENT_DISCOVERY = "DOCUMENT_DISCOVERY"

NON_RETRIEVAL_INTENTS = frozenset({
    GREETING, GREETING_RETURN, FAREWELL, THANKS, PRAISE,
    NEGATIVE_MILD, NEGATIVE_STRONG, IDENTITY, CAPABILITY,
    HOW_IT_WORKS, PRIVACY, LIMITATIONS, USAGE_HELP,
    SMALL_TALK, CLARIFICATION, DOCUMENT_DISCOVERY,
})

# ---------------------------------------------------------------------------
# B. Intent classifier
# ---------------------------------------------------------------------------

def classify_conversational_intent(
    text: str,
    turn_count: int = 0,
) -> Optional[Tuple[str, float]]:
    """Classify *text* into a conversational intent using NLU understanding.

    Uses contrastive embedding classification against document-query and
    conversational prototypes.  Returns ``(intent, confidence)`` when the
    message is conversational, or ``None`` when it should proceed to
    document retrieval.
    """
    text = (text or "").strip()
    if not text:
        return None

    try:
        from src.nlp.nlu_engine import classify_query_routing

        # Pre-check: if the query contains possessive patterns with proper nouns
        # (e.g. "Philip's skills", "Gokul's email"), it's always a document query.
        # This guard prevents embedding-based misrouting for entity-specific questions.
        import re as _re_conv
        _possessive_entity_re = _re_conv.compile(
            r"[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*'s?\s+\w+",
        )
        if _possessive_entity_re.search(text):
            return None

        routing, intent, score = classify_query_routing(text)

        # If the holistic classifier sees GREETING but the message is longer,
        # strip the greeting prefix (via spaCy INTJ detection) and
        # re-classify the substantive part.
        if routing == "conversational" and intent == GREETING:
            stripped = _strip_greeting_prefix(text)
            if stripped and stripped != text and len(stripped.split()) > 2:
                sub_routing, sub_intent, sub_score = classify_query_routing(stripped)
                if sub_routing == "document":
                    return None
                routing, intent, score = sub_routing, sub_intent, sub_score

        if routing == "document":
            return None

        # Confidence threshold gate: borderline classifications (score < 0.65)
        # should default to document retrieval.  Many document queries with
        # conversational phrasing ("Can you show me...", "I'd like to know...")
        # get low-confidence conversational scores — route these to RAG.
        # Exempt clearly conversational intents (praise, capability questions, etc.)
        _CONV_CONFIDENCE_THRESHOLD = 0.65
        _CLEARLY_CONVERSATIONAL = frozenset({
            GREETING, GREETING_RETURN, FAREWELL, THANKS, IDENTITY,
            PRAISE, NEGATIVE_MILD, NEGATIVE_STRONG, CAPABILITY,
            HOW_IT_WORKS, LIMITATIONS, SMALL_TALK,
            # NOTE: PRIVACY and USAGE_HELP removed — queries about "email",
            # "phone", "skills" were being misrouted to PRIVACY/USAGE_HELP
            # because embedding similarity to these prototypes bypassed the
            # 0.65 confidence gate. Now they must meet the threshold.
            # CLARIFICATION also removed (see prior note).
        })
        if score < _CONV_CONFIDENCE_THRESHOLD and intent not in _CLEARLY_CONVERSATIONAL:
            return None

        # Adjust greeting for returning users
        if intent == GREETING and turn_count > 0:
            return (GREETING_RETURN, score)
        return (intent, score)
    except Exception:
        # Fallback: regex pattern matching when NLU is unavailable
        return _fallback_intent_patterns(text, turn_count)

# ── Regex fallback for when NLU engine is unavailable ──────────────────

_GREETING_RE = re.compile(
    r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy|greetings)\b", re.IGNORECASE
)
_FAREWELL_RE = re.compile(
    r"\b(bye|goodbye|see\s+you|take\s+care|farewell|until\s+(next|later))\b", re.IGNORECASE
)
_THANKS_RE = re.compile(r"\b(thank|thanks|appreciate|grateful)\b", re.IGNORECASE)
_IDENTITY_RE = re.compile(r"\b(who\s+are\s+you|what\s+are\s+you|your\s+name)\b", re.IGNORECASE)
_CAPABILITY_RE = re.compile(
    r"\b(what\s+can\s+you\s+do|your\s+capabilit|help\s+me\s+with|what\s+do\s+you\s+do)\b",
    re.IGNORECASE,
)
_HELP_RE = re.compile(r"\b(how\s+do\s+i|how\s+to\s+use|usage|tutorial|guide)\b", re.IGNORECASE)

def _fallback_intent_patterns(text: str, turn_count: int) -> Optional[Tuple[str, float]]:
    """Regex-based intent detection as fallback when NLU is unavailable.

    Returns (intent, confidence) or None if text looks like a document query.
    """
    lowered = text.lower().strip()
    # Short messages are more likely conversational
    is_short = len(lowered.split()) <= 4

    if _GREETING_RE.match(lowered):
        if is_short:
            intent = GREETING_RETURN if turn_count > 0 else GREETING
            return (intent, 0.80)
        return None  # Long message starting with greeting → probably a document query

    if _FAREWELL_RE.search(lowered) and is_short:
        return (FAREWELL, 0.80)

    if _THANKS_RE.search(lowered) and is_short:
        return (THANKS, 0.75)

    if _IDENTITY_RE.search(lowered):
        return (IDENTITY, 0.85)

    if _CAPABILITY_RE.search(lowered):
        return (CAPABILITY, 0.85)

    if _HELP_RE.search(lowered) and is_short:
        return (USAGE_HELP, 0.70)

    return None

def _try_conversational_nlu(text: str) -> Optional[Tuple[str, float]]:
    """NLU-based conversational classification (embedding + spaCy)."""
    try:
        from src.nlp.nlu_engine import classify_conversational
        return classify_conversational(text)
    except Exception:
        return None

def _is_document_query(text: str) -> bool:
    """Determine whether *text* is a document query using NLU classification.

    Uses contrastive embedding similarity against document-query prototypes
    vs conversational prototypes.  Kept as a utility for any call site
    that needs a boolean document-query check.
    """
    try:
        from src.nlp.nlu_engine import classify_document_query
        return classify_document_query(text)
    except Exception:
        return False

def _strip_greeting_prefix(text: str) -> str:
    """Strip greeting prefix using spaCy part-of-speech tagging.

    Detects interjection (INTJ) tokens at the start of *text* and strips
    them along with following punctuation and whitespace.
    """
    try:
        from src.nlp.nlu_engine import _get_nlp

        nlp = _get_nlp()
        if nlp is None:
            return text
        doc = nlp(text)
        strip_to = 0
        for token in doc:
            if token.is_space:
                continue
            if token.pos_ == "INTJ":
                strip_to = token.idx + len(token.text_with_ws)
                for next_token in doc[token.i + 1 :]:
                    if next_token.is_punct or next_token.is_space:
                        strip_to = next_token.idx + len(next_token.text_with_ws)
                    else:
                        break
                break
            else:
                break
        if 0 < strip_to < len(text):
            return text[strip_to:].strip()
    except Exception:
        pass
    return text

# ---------------------------------------------------------------------------
# C. Context collector
# ---------------------------------------------------------------------------

@dataclass
class ConversationalContext:
    document_count: int = 0
    document_names: List[str] = field(default_factory=list)
    dominant_domains: List[str] = field(default_factory=list)
    profile_is_empty: bool = True
    is_first_message: bool = True
    is_returning_user: bool = False
    conversation_turn_count: int = 0
    time_of_day: str = "day"
    total_points: int = 0

def _time_of_day(hour: Optional[int] = None) -> str:
    if hour is None:
        hour = datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"

_DOC_SUMMARY_CACHE: Dict[str, Tuple[float, int, List[str], Dict[str, int]]] = {}
_DOC_SUMMARY_TTL = 30  # seconds — short enough to reflect deletions quickly

def _mongodb_doc_summary(subscription_id: str, profile_id: str):
    """Fetch document names and domains from MongoDB (authoritative source).

    Uses a 30-second TTL cache to avoid creating a new connection per request.
    """
    cache_key = f"{subscription_id}:{profile_id}"
    cached = _DOC_SUMMARY_CACHE.get(cache_key)
    if cached:
        ts, cnt, names, doms = cached
        if (time.time() - ts) < _DOC_SUMMARY_TTL:
            return cnt, names, doms

    try:
        from src.api.dataHandler import db
        from src.api.config import Config
        cursor = db[Config.MongoDB.DOCUMENTS].find(
            {"subscription_id": subscription_id, "profile_id": profile_id,
             "status": {"$nin": ["DELETED"]}},
            {"name": 1, "doc_domain": 1},
        )
        docs = list(cursor)
        names = [d.get("name", "") for d in docs if d.get("name")]
        domains_list = [d.get("doc_domain", "") for d in docs if d.get("doc_domain")]
        domain_counts: Dict[str, int] = {}
        for dom in domains_list:
            domain_counts[dom] = domain_counts.get(dom, 0) + 1
        _DOC_SUMMARY_CACHE[cache_key] = (time.time(), len(docs), names, domain_counts)
        return len(docs), names, domain_counts
    except Exception:
        return 0, [], {}

def collect_context(
    *,
    catalog: Optional[Dict[str, Any]] = None,
    session_state: Optional[Dict[str, Any]] = None,
    collection_point_count: int = 0,
    hour: Optional[int] = None,
    subscription_id: str = "",
    profile_id: str = "",
) -> ConversationalContext:
    catalog = catalog or {}
    session_state = session_state or {}
    docs = catalog.get("documents") or []
    doc_names = [str(d.get("source_name") or d.get("name") or "") for d in docs if d]
    doc_names = [n for n in doc_names if n]
    dominant = catalog.get("dominant_domains") or {}

    # Use MongoDB as authoritative source when catalog seems stale.
    # The cached _mongodb_doc_summary (30s TTL) keeps this cheap.
    doc_count = len(docs)
    if subscription_id and profile_id:
        if doc_count == 0 and collection_point_count > 0:
            # Catalog empty but Qdrant has data — check MongoDB
            mongo_count, mongo_names, mongo_domains = _mongodb_doc_summary(subscription_id, profile_id)
            if mongo_count > 0:
                doc_count = mongo_count
                doc_names = mongo_names[:5]
                dominant = mongo_domains
        elif doc_count > 0:
            # Catalog exists — verify against MongoDB (handles deletions too)
            mongo_count, mongo_names, mongo_domains = _mongodb_doc_summary(subscription_id, profile_id)
            if mongo_count != doc_count:
                doc_count = mongo_count
                if mongo_names:
                    doc_names = mongo_names[:5]
                if mongo_domains:
                    dominant = mongo_domains

    if isinstance(dominant, dict):
        domains = sorted(dominant.keys(), key=lambda k: dominant[k], reverse=True)
    elif isinstance(dominant, (list, tuple)):
        domains = list(dominant)
    else:
        domains = []
    turn_count = int(session_state.get("turn_count", 0) or 0)
    return ConversationalContext(
        document_count=doc_count,
        document_names=doc_names[:5],
        dominant_domains=domains[:3],
        profile_is_empty=doc_count == 0 and collection_point_count == 0,
        is_first_message=turn_count == 0,
        is_returning_user=turn_count > 0,
        conversation_turn_count=turn_count,
        time_of_day=_time_of_day(hour),
        total_points=collection_point_count,
    )

# ---------------------------------------------------------------------------
# D. Fragment pools
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResponseFragment:
    text: str
    requires_docs: bool = False
    requires_empty: bool = False
    requires_returning: bool = False
    requires_first: bool = False

def _frag(text: str, **kw: bool) -> ResponseFragment:
    return ResponseFragment(text=text, **kw)

_FRAGMENT_POOLS: Dict[str, Dict[str, List[ResponseFragment]]] = {
    # -- GREETING --
    GREETING: {
        "opener": [
            _frag("Good {time_of_day}!"),
            _frag("Hello!"),
            _frag("Hi there!"),
            _frag("Hey!"),
            _frag("Welcome!"),
        ],
        "core": [
            _frag("I'm DocWain, your document intelligence assistant."),
            _frag("DocWain here — ready to help with your documents."),
            _frag("I'm DocWain, built to answer questions from your uploaded documents."),
            _frag("This is DocWain, your AI-powered document assistant."),
            _frag("DocWain at your service — I analyze and answer from your documents."),
        ],
        "context_bridge": [
            _frag("You have {doc_count} document(s) loaded across {domains}.", requires_docs=True),
            _frag("I can see {doc_count} document(s) in your profile.", requires_docs=True),
            _frag("Your profile is ready with {doc_count} document(s).", requires_docs=True),
            _frag("No documents uploaded yet — start by adding some to your profile.", requires_empty=True),
            _frag("Upload a document and I'll be ready to answer questions about it.", requires_empty=True),
        ],
        "action_prompt": [
            _frag("What would you like to know?"),
            _frag("Ask me anything about your documents."),
            _frag("Go ahead — ask a question about your documents."),
            _frag("How can I help you today?"),
            _frag("What can I help you with?"),
        ],
    },

    # -- GREETING_RETURN --
    GREETING_RETURN: {
        "opener": [
            _frag("Welcome back!"),
            _frag("Hey again!"),
            _frag("Good to see you again!"),
            _frag("Hi! Back for more?"),
        ],
        "core": [
            _frag("DocWain is ready to pick up where we left off."),
            _frag("I'm still here — DocWain at your service."),
            _frag("DocWain here, ready for your next question."),
        ],
        "context_bridge": [
            _frag("Your {doc_count} document(s) are still loaded.", requires_docs=True),
            _frag("Your profile with {domains} documents is ready.", requires_docs=True),
            _frag(""),
        ],
        "action_prompt": [
            _frag("What would you like to explore next?"),
            _frag("What can I help you with this time?"),
            _frag("Ready when you are."),
        ],
    },

    # -- FAREWELL --
    FAREWELL: {
        "opener": [
            _frag("Goodbye!"),
            _frag("Take care!"),
            _frag("See you later!"),
            _frag("Until next time!"),
        ],
        "core": [
            _frag("Thanks for using DocWain."),
            _frag("It was great helping you today."),
            _frag("DocWain is always here when you need document answers."),
            _frag("Happy I could help with your documents."),
        ],
        "context_bridge": [
            _frag(""),
        ],
        "action_prompt": [
            _frag("Come back anytime you need help with your documents."),
            _frag("Your documents will be waiting for your next session."),
            _frag("Feel free to return whenever you have more questions."),
        ],
    },

    # -- THANKS --
    THANKS: {
        "opener": [
            _frag("You're welcome!"),
            _frag("Glad to help!"),
            _frag("Happy to assist!"),
            _frag("Anytime!"),
            _frag("My pleasure!"),
        ],
        "core": [
            _frag("That's what DocWain is here for."),
            _frag("DocWain is always ready to dig into your documents."),
            _frag("I'm glad the answer was useful."),
        ],
        "context_bridge": [
            _frag(""),
        ],
        "action_prompt": [
            _frag("If you have more questions, just ask."),
            _frag("Need anything else from your documents?"),
            _frag("What would you like to explore next?"),
            _frag("Let me know if there's anything else I can help with."),
        ],
    },

    # -- PRAISE --
    PRAISE: {
        "opener": [
            _frag("Thank you!"),
            _frag("That means a lot!"),
            _frag("Appreciate the kind words!"),
            _frag("Thanks for the feedback!"),
        ],
        "core": [
            _frag("DocWain strives to deliver accurate, grounded answers."),
            _frag("Glad DocWain's document intelligence hit the mark."),
            _frag("Great to hear the answer was on point."),
        ],
        "context_bridge": [
            _frag(""),
        ],
        "action_prompt": [
            _frag("Ready for your next question whenever you are."),
            _frag("What else can I help you with?"),
            _frag("Keep the questions coming!"),
        ],
    },

    # -- NEGATIVE_MILD --
    NEGATIVE_MILD: {
        "opener": [
            _frag("I understand."),
            _frag("Fair point."),
            _frag("Got it."),
            _frag("I hear you."),
        ],
        "core": [
            _frag("DocWain can try a different approach to find what you need."),
            _frag("Let me know what was off and I'll adjust my search."),
            _frag("I can refine the answer — tell me what to focus on."),
        ],
        "context_bridge": [
            _frag(""),
        ],
        "action_prompt": [
            _frag("Try rephrasing your question or specifying a document."),
            _frag("Want me to focus on a specific document or section?"),
            _frag("I can retry with deeper retrieval if you'd like."),
        ],
    },

    # -- NEGATIVE_STRONG --
    NEGATIVE_STRONG: {
        "opener": [
            _frag("I'm sorry about that."),
            _frag("I apologize for the poor response."),
            _frag("Sorry — that wasn't up to standard."),
        ],
        "core": [
            _frag("DocWain takes accuracy seriously and I want to do better."),
            _frag("Your feedback helps DocWain improve."),
            _frag("I'll try harder on the next attempt."),
        ],
        "context_bridge": [
            _frag(""),
        ],
        "action_prompt": [
            _frag("Could you tell me what specifically was wrong so I can retry?"),
            _frag("Try asking again — I'll use deeper retrieval this time."),
            _frag("Let me know what to focus on and I'll give it another shot."),
        ],
    },

    # -- IDENTITY --
    IDENTITY: {
        "opener": [
            _frag("Great question!"),
            _frag("Happy to introduce myself!"),
            _frag("Sure thing!"),
            _frag("Let me tell you."),
        ],
        "core": [
            _frag("I'm DocWain, an AI-powered document intelligence assistant. I answer questions strictly from the documents in your profile — no guessing, no browsing the web."),
            _frag("DocWain is a document intelligence platform. I retrieve, analyze, and generate answers grounded entirely in your uploaded documents."),
            _frag("I'm DocWain — a document-focused AI assistant. I read your uploaded documents and answer questions based solely on what's in them."),
            _frag("I'm DocWain, your document intelligence agent. I analyze uploaded documents across domains like resumes, invoices, contracts, and medical records to answer your questions accurately."),
        ],
        "context_bridge": [
            _frag("Right now your profile has {doc_count} document(s) covering {domains}.", requires_docs=True),
            _frag("You currently have {doc_count} document(s) loaded.", requires_docs=True),
            _frag("Your profile is empty — upload documents to get started.", requires_empty=True),
            _frag(""),
        ],
        "action_prompt": [
            _frag("Ask me anything about your documents."),
            _frag("What would you like to know?"),
            _frag("Go ahead and ask a question."),
        ],
    },

    # -- CAPABILITY --
    CAPABILITY: {
        "opener": [
            _frag("Here's what DocWain can do:"),
            _frag("DocWain has several capabilities:"),
            _frag("I'm glad you asked!"),
        ],
        "core": [
            _frag("I can summarize documents, extract key information, compare candidates, rank profiles, answer specific questions, and generate content like cover letters — all grounded in your uploaded documents."),
            _frag("DocWain supports document Q&A, summarization, comparison, ranking, data extraction, and content generation. I work across resumes, invoices, contracts, medical records, and more."),
            _frag("I analyze documents to answer questions, summarize content, compare entries, rank candidates, and extract structured data. Everything I say is backed by evidence from your files."),
        ],
        "context_bridge": [
            _frag("With your {doc_count} document(s), you can try any of these right now.", requires_docs=True),
            _frag("Upload some documents to see these features in action.", requires_empty=True),
            _frag(""),
        ],
        "action_prompt": [
            _frag("What would you like to try first?"),
            _frag("Pick a feature and let's get started."),
            _frag("Just ask a question and I'll handle the rest."),
        ],
    },

    # -- HOW_IT_WORKS --
    HOW_IT_WORKS: {
        "opener": [
            _frag("Good question!"),
            _frag("Here's the overview:"),
            _frag("Let me explain."),
        ],
        "core": [
            _frag("DocWain uses a RAG (Retrieval-Augmented Generation) pipeline. When you ask a question, I search your documents for relevant chunks, rerank them by relevance, extract key facts, and compose a grounded answer."),
            _frag("When you ask a question, DocWain retrieves the most relevant sections from your documents, verifies the information, and generates a concise answer backed by evidence from the source material."),
            _frag("DocWain works in stages: first I understand your intent, then I retrieve relevant document sections, rerank them for accuracy, extract facts, and generate a response grounded in your actual documents."),
        ],
        "context_bridge": [
            _frag(""),
        ],
        "action_prompt": [
            _frag("Try asking a question to see it in action."),
            _frag("Want to test it out? Ask me something about your documents."),
            _frag("Ready to give it a try?"),
        ],
    },

    # -- PRIVACY --
    PRIVACY: {
        "opener": [
            _frag("Great question about security."),
            _frag("Data privacy is important to us."),
            _frag("I'm glad you asked about this."),
        ],
        "core": [
            _frag("DocWain processes your documents within your profile scope. Each profile is isolated — no data crosses between profiles or subscriptions. I only access documents you've uploaded to your current profile."),
            _frag("Your documents are stored securely and scoped to your profile. DocWain never shares data across profiles or subscriptions, and I only answer from documents in your active session."),
            _frag("DocWain keeps your data private. Documents are isolated per profile, answers are generated only from your uploads, and no cross-profile access is possible."),
        ],
        "context_bridge": [
            _frag(""),
        ],
        "action_prompt": [
            _frag("Feel free to ask about your documents with confidence."),
            _frag("Your data is safe — go ahead and ask me anything."),
            _frag("Any other concerns I can address?"),
        ],
    },

    # -- LIMITATIONS --
    LIMITATIONS: {
        "opener": [
            _frag("Good to know the boundaries."),
            _frag("Fair question!"),
            _frag("Transparency matters."),
        ],
        "core": [
            _frag("DocWain only answers from documents you've uploaded — I can't browse the internet, access external databases, or generate information beyond what's in your files. I also don't retain memory between sessions."),
            _frag("I'm limited to the documents in your profile. I can't search the web, access live data, or answer from general knowledge. If information isn't in your documents, I'll let you know."),
            _frag("DocWain works strictly from uploaded documents. No internet access, no external lookups, no memory across sessions. If something isn't in your files, I'll tell you rather than guess."),
        ],
        "context_bridge": [
            _frag(""),
        ],
        "action_prompt": [
            _frag("Upload more documents if you need broader coverage."),
            _frag("Anything else you'd like to know?"),
            _frag("Want to try a question within these boundaries?"),
        ],
    },

    # -- USAGE_HELP --
    USAGE_HELP: {
        "opener": [
            _frag("Let's get you started!"),
            _frag("Here are some tips:"),
            _frag("Getting started is easy!"),
        ],
        "core": [
            _frag("Upload your documents (PDFs, text files, images) to a profile, then ask questions. DocWain will retrieve relevant sections and answer. Try questions like 'Summarize this document' or 'What skills does the candidate have?'"),
            _frag("Start by uploading documents to your profile. Then ask questions naturally — 'What's the total on the invoice?', 'Compare the two candidates', 'List the key terms in the contract'. DocWain handles the rest."),
            _frag("First, upload documents to a profile. Then simply ask questions in natural language. DocWain understands intent — whether you want summaries, comparisons, rankings, or specific data extraction."),
        ],
        "context_bridge": [
            _frag("You already have {doc_count} document(s) ready to query.", requires_docs=True),
            _frag("Your profile is empty — upload a document to get started.", requires_empty=True),
            _frag(""),
        ],
        "action_prompt": [
            _frag("Try asking a question now!"),
            _frag("What would you like to ask about your documents?"),
            _frag("Ready when you are."),
        ],
    },

    # -- SMALL_TALK --
    SMALL_TALK: {
        "opener": [
            _frag("I'm doing well, thanks for asking!"),
            _frag("All good here!"),
            _frag("Running smoothly!"),
        ],
        "core": [
            _frag("DocWain is ready and waiting for document questions."),
            _frag("I'm best at helping with your documents — that's my specialty."),
            _frag("I'm always in a good mood when there are documents to analyze!"),
        ],
        "context_bridge": [
            _frag(""),
        ],
        "action_prompt": [
            _frag("Got a document question for me?"),
            _frag("If you have a document task, I'm ready to help."),
            _frag("What can I help you with today?"),
        ],
    },

    # -- DOCUMENT_DISCOVERY --
    DOCUMENT_DISCOVERY: {
        "opener": [
            _frag("Here's a comprehensive overview of your profile:"),
            _frag("I've analyzed your profile — here's a total overview:"),
            _frag("Great question! Let me provide an overview of your documents:"),
        ],
        "core": [
            _frag("You have a total of {doc_count} document(s) across {domains}. Analyzed documents: {doc_names}. This unique collection covers a range of content types.", requires_docs=True),
            _frag("Your profile contains a total of {doc_count} document(s) in {domains}, with the most common category being the largest group. Documents: {doc_names}.", requires_docs=True),
            _frag("Your profile is currently empty. Upload documents to get started — I support resumes, invoices, contracts, insurance policies, medical records, and more.", requires_empty=True),
        ],
        "context_bridge": [
            _frag(""),
        ],
        "action_prompt": [
            _frag("You can analyze patterns across your documents, compare them, or get a total overview of common or unique details."),
            _frag("Try asking me to analyze the distribution of topics, compare documents, or find common and unique patterns across them."),
            _frag("I can provide a total overview, analyze patterns, rank items from highest to lowest, or summarize the range of information across your documents."),
        ],
    },

    # -- CLARIFICATION --
    CLARIFICATION: {
        "opener": [
            _frag("Of course!"),
            _frag("Sure thing."),
            _frag("Happy to clarify."),
            _frag("No problem."),
        ],
        "core": [
            _frag("DocWain can rephrase or dig deeper into the answer."),
            _frag("Let me know which part wasn't clear and I'll explain differently."),
            _frag("I can approach the question from a different angle."),
        ],
        "context_bridge": [
            _frag(""),
        ],
        "action_prompt": [
            _frag("Tell me what you'd like clarified."),
            _frag("Which part should I explain again?"),
            _frag("Paste the snippet you want me to review."),
        ],
    },
}

# Domain-specific suggestion fragments (injected ~30% of the time).
_DOMAIN_SUGGESTIONS: Dict[str, List[str]] = {
    "resume": [
        "Try asking about a candidate's skills, experience, or education.",
        "You can compare candidates or rank them by skill match.",
        "Ask me to summarize a resume or extract contact details.",
    ],
    "invoice": [
        "Try asking about totals, line items, or payment terms.",
        "You can compare invoices or extract specific amounts.",
        "Ask about due dates, vendor details, or tax amounts.",
    ],
    "legal": [
        "Try asking about specific clauses, terms, or obligations.",
        "You can summarize a contract or extract key provisions.",
        "Ask about termination clauses, liability terms, or warranties.",
    ],
    "medical": [
        "Try asking about diagnoses, medications, or treatment plans.",
        "You can summarize patient records or extract key findings.",
        "Ask about lab results, prescriptions, or medical history.",
    ],
    "contract": [
        "Try asking about specific clauses, terms, or obligations.",
        "You can summarize the contract or extract key provisions.",
        "Ask about termination clauses, liability terms, or warranties.",
    ],
    "policy": [
        "Try asking about coverage details, exclusions, or premium amounts.",
        "You can ask about claim procedures, deductibles, or beneficiary information.",
        "Ask about what's covered, conditions for natural calamities, or policy terms.",
    ],
    "insurance": [
        "Try asking about coverage details, exclusions, or premium amounts.",
        "You can ask about claim procedures, deductibles, or beneficiary information.",
        "Ask about what's covered, conditions for natural calamities, or policy terms.",
    ],
}

# ---------------------------------------------------------------------------
# E. Response composer
# ---------------------------------------------------------------------------

def _filter_fragments(
    fragments: List[ResponseFragment],
    ctx: ConversationalContext,
) -> List[ResponseFragment]:
    """Filter fragments by context requirements."""
    result = []
    for f in fragments:
        if f.requires_docs and ctx.profile_is_empty:
            continue
        if f.requires_empty and not ctx.profile_is_empty:
            continue
        if f.requires_returning and ctx.is_first_message:
            continue
        if f.requires_first and not ctx.is_first_message:
            continue
        result.append(f)
    return result or fragments  # Fallback to all if nothing passes.

def _make_seed(intent: str, user_key: str, extra: int = 0) -> int:
    """Deterministic-per-minute seed for weighted random selection."""
    minute = datetime.now().strftime("%Y%m%d%H%M")
    raw = f"{intent}:{user_key}:{minute}:{extra}"
    return int(hashlib.md5(raw.encode()).hexdigest(), 16)

def _pick(fragments: List[ResponseFragment], seed: int) -> ResponseFragment:
    if not fragments:
        return _frag("")
    return fragments[seed % len(fragments)]

def _format_domains(domains: List[str]) -> str:
    if not domains:
        return "various"
    if len(domains) == 1:
        return domains[0]
    return ", ".join(domains[:-1]) + " and " + domains[-1]

def _inject_vars(text: str, ctx: ConversationalContext) -> str:
    """Replace placeholders with context values. Missing vars become empty."""
    mapping: Dict[str, str] = {
        "time_of_day": ctx.time_of_day,
        "doc_count": str(ctx.document_count) if ctx.document_count else "0",
        "domains": _format_domains(ctx.dominant_domains),
        "doc_names": ", ".join(ctx.document_names[:3]) if ctx.document_names else "",
    }

    class _Default(dict):
        def __missing__(self, key: str) -> str:
            return ""

    return text.format_map(_Default(mapping))

def _domain_suggestion(ctx: ConversationalContext, seed: int) -> str:
    """Return a domain-specific suggestion ~30% of the time."""
    if seed % 10 >= 3:  # ~70% chance of skipping
        return ""
    for domain in ctx.dominant_domains:
        suggestions = _DOMAIN_SUGGESTIONS.get(domain)
        if suggestions:
            return suggestions[seed % len(suggestions)]
    return ""

_CONVERSATIONAL_SYSTEM_PROMPT = (
    "You are DocWain, a friendly and professional document intelligence assistant. "
    "You help users analyze, search, and extract insights from their uploaded documents. "
    "Keep responses warm, concise (2-3 sentences), and helpful. "
    "Mention document-related capabilities naturally when appropriate. "
    "Never use bullet points or markdown formatting in conversational responses. "
    "Do NOT start with 'I' — vary your sentence openings."
)

# Intents eligible for LLM-generated conversational responses
# Disabled: LLM conversational responses cause model-swap latency (60-200s)
# when Ollama has a different model loaded.  Template fragments are sufficient.
_LLM_CONVERSATIONAL_INTENTS: set = set()  # was {GREETING, GREETING_RETURN, FAREWELL, THANKS, PRAISE}

def _llm_conversational_response(
    intent: str,
    context: ConversationalContext,
    user_text: str,
) -> Optional[str]:
    """Generate a conversational response using the LLM.

    Returns None if LLM is unavailable or takes too long (>3s).
    """
    try:
        from src.llm.clients import OllamaClient
        import asyncio

        client = OllamaClient()
        ctx_info = (
            f"User has {context.document_count} document(s) loaded"
            + (f" covering {_format_domains(context.dominant_domains)}" if context.dominant_domains else "")
            + f". Time of day: {context.time_of_day}."
        )
        user_msg = f"[Context: {ctx_info}]\n[Intent: {intent}]\nUser says: {user_text or intent.lower()}"

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                client.chat_with_metadata,
                messages=[
                    {"role": "system", "content": _CONVERSATIONAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                thinking=False,
                options={"num_predict": 150, "num_ctx": 2048},
            )
            text, _meta = future.result(timeout=3.0)
        if text and len(text.strip()) > 20:
            return text.strip()
    except Exception as exc:
        logger.debug("LLM conversational response failed: %s", exc)
    return None

def compose_response(
    intent: str,
    context: ConversationalContext,
    user_key: str = "",
    user_text: str = "",
) -> str:
    """Compose a dynamic response from fragment pools.

    *user_key* is used to seed the randomizer (subscription_id + profile_id).
    *user_text* is the original user message (used for USAGE_HELP delegation).
    """
    # Delegate help/capability to the dedicated module when user_text is available.
    if intent in {USAGE_HELP, CAPABILITY} and user_text:
        try:
            from src.intelligence.usage_help import compose_usage_help_response
            return compose_usage_help_response(user_text, context, user_key)
        except Exception:
            pass  # Fall through to generic fragment-based response.

    # LLM-generated conversational responses for social intents
    if intent in _LLM_CONVERSATIONAL_INTENTS and user_text:
        llm_response = _llm_conversational_response(intent, context, user_text)
        if llm_response:
            return llm_response

    pools = _FRAGMENT_POOLS.get(intent)
    if pools is None:
        # Try case-insensitive lookup before falling back to GREETING
        pools = _FRAGMENT_POOLS.get(intent.upper()) if intent else None
        if pools is None:
            logger.debug("Unknown conversational intent '%s', falling back to GREETING", intent)
            pools = _FRAGMENT_POOLS[GREETING]
    seed = _make_seed(intent, user_key)

    opener = _pick(_filter_fragments(pools.get("opener", []), context), seed)
    core = _pick(_filter_fragments(pools.get("core", []), context), seed >> 4)
    bridge = _pick(_filter_fragments(pools.get("context_bridge", []), context), seed >> 8)
    action = _pick(_filter_fragments(pools.get("action_prompt", []), context), seed >> 12)

    # Inject domain suggestion if applicable.
    suggestion = _domain_suggestion(context, seed >> 16)

    parts = [
        _inject_vars(opener.text, context),
        _inject_vars(core.text, context),
        _inject_vars(bridge.text, context),
        suggestion,
        _inject_vars(action.text, context),
    ]
    text = " ".join(p for p in parts if p).strip()

    # Anti-repetition check.
    text_hash = _hash_text(text)
    if _deduplicator.is_duplicate(user_key, text_hash):
        # Retry with shifted seed (up to 3 attempts).
        for attempt in range(1, 4):
            seed2 = _make_seed(intent, user_key, extra=attempt)
            opener2 = _pick(_filter_fragments(pools.get("opener", []), context), seed2)
            core2 = _pick(_filter_fragments(pools.get("core", []), context), seed2 >> 4)
            bridge2 = _pick(_filter_fragments(pools.get("context_bridge", []), context), seed2 >> 8)
            action2 = _pick(_filter_fragments(pools.get("action_prompt", []), context), seed2 >> 12)
            suggestion2 = _domain_suggestion(context, seed2 >> 16)
            parts2 = [
                _inject_vars(opener2.text, context),
                _inject_vars(core2.text, context),
                _inject_vars(bridge2.text, context),
                suggestion2,
                _inject_vars(action2.text, context),
            ]
            text2 = " ".join(p for p in parts2 if p).strip()
            h2 = _hash_text(text2)
            if not _deduplicator.is_duplicate(user_key, h2):
                _deduplicator.record(user_key, h2)
                return text2
        # Give up — return the original text.

    _deduplicator.record(user_key, text_hash)
    return text

# ---------------------------------------------------------------------------
# F. Anti-repetition
# ---------------------------------------------------------------------------

def _hash_text(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:12], 16)

class ResponseDeduplicator:
    """Module-level ring buffer of recent response hashes per user key."""

    _MAX_PER_KEY = 8
    _MAX_KEYS = 1000

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, List[int]] = OrderedDict()

    def is_duplicate(self, user_key: str, text_hash: int) -> bool:
        with self._lock:
            ring = self._cache.get(user_key)
            if not ring:
                return False
            return text_hash in ring

    def record(self, user_key: str, text_hash: int) -> None:
        with self._lock:
            if user_key not in self._cache:
                if len(self._cache) >= self._MAX_KEYS:
                    self._cache.popitem(last=False)  # LRU eviction
                self._cache[user_key] = []
            else:
                self._cache.move_to_end(user_key)
            ring = self._cache[user_key]
            ring.append(text_hash)
            if len(ring) > self._MAX_PER_KEY:
                ring.pop(0)

_deduplicator = ResponseDeduplicator()

# ---------------------------------------------------------------------------
# G. Public API
# ---------------------------------------------------------------------------

@dataclass
class ConversationalResponse:
    text: str
    intent: str
    confidence: float
    is_conversational: bool = True

def generate_conversational_response(
    user_text: str,
    *,
    subscription_id: str = "",
    profile_id: str = "",
    redis_client: Any = None,
    conversation_history: Any = None,
    namespace: str = "",
    user_id: str = "",
    collection_point_count: int = 0,
    session_state: Optional[Dict[str, Any]] = None,
    catalog: Optional[Dict[str, Any]] = None,
) -> Optional[ConversationalResponse]:
    """Classify and generate a conversational response.

    Returns ``ConversationalResponse`` for conversational intents,
    ``None`` for document queries (proceed to retrieval).
    """
    text = (user_text or "").strip()
    if not text:
        return None

    turn_count = 0
    if session_state:
        turn_count = int(session_state.get("turn_count", 0) or 0)

    result = classify_conversational_intent(text, turn_count=turn_count)
    if result is None:
        return None

    intent, confidence = result

    ctx = collect_context(
        catalog=catalog,
        session_state=session_state,
        collection_point_count=collection_point_count,
        subscription_id=subscription_id,
        profile_id=profile_id,
    )

    user_key = f"{subscription_id}:{profile_id}"
    response_text = compose_response(intent, ctx, user_key, user_text=text)

    return ConversationalResponse(
        text=response_text,
        intent=intent,
        confidence=confidence,
    )

__all__ = [
    "NON_RETRIEVAL_INTENTS",
    "GREETING", "GREETING_RETURN", "FAREWELL", "THANKS", "PRAISE",
    "NEGATIVE_MILD", "NEGATIVE_STRONG", "IDENTITY", "CAPABILITY",
    "HOW_IT_WORKS", "PRIVACY", "LIMITATIONS", "USAGE_HELP",
    "SMALL_TALK", "CLARIFICATION", "DOCUMENT_DISCOVERY",
    "classify_conversational_intent",
    "ConversationalContext", "collect_context",
    "compose_response",
    "ConversationalResponse", "generate_conversational_response",
]
