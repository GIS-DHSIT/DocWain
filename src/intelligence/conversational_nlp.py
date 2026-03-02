"""Dynamic Conversational NLP Engine for DocWain.

Generates varied, context-aware responses for non-retrieval intents
(greetings, farewells, meta questions, feedback, small talk).

No imports from other ``src/`` modules — all context is passed as arguments.
"""
from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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

# Help-meta override — "how do I compare/rank/etc." is USAGE_HELP, not a doc query.
_HELP_META_OVERRIDE = re.compile(
    r"\bhow\s+(?:do\s+i|can\s+i|to)\s+(?:compare|rank|summarize|summarise|extract|"
    r"list|find|screen|generate|upload|add|import)\b",
    re.I,
)

# Capability-meta override — common "what else can you do/help with?" variants.
_CAPABILITY_META_OVERRIDE = re.compile(
    r"\b(?:"
    r"what\s+(?:else\s+|all\s+)?can\s+(?:you|docwain)\s+(?:do|help\s+with)|"
    r"show\s+me\s+what\s+(?:you|docwain)\s+can\s+do|"
    r"how\s+can\s+(?:you|docwain)\s+help(?:\s+me)?"
    r")\b",
    re.I,
)

# Document discovery patterns — matched BEFORE doc query overrides.
_DOCUMENT_DISCOVERY_PATTERNS = [
    re.compile(r"\bwhat\s+can\s+i\s+(?:do|perform|ask)\s+with\s+(?:these|my|the)\s+documents?\b", re.I),
    re.compile(r"\bwhat\s+(?:documents?|files?)\s+do\s+i\s+have\b", re.I),
    re.compile(r"\bshow\s+(?:me\s+)?my\s+documents?\b", re.I),
    re.compile(r"\bwhat\s+can\s+i\s+ask\s+(?:about|you)?\b", re.I),
    re.compile(r"\bwhat\s+(?:is|are)\s+(?:available|uploaded|in\s+my\s+profile)\b", re.I),
    re.compile(r"\blist\s+(?:my\s+)?(?:uploaded\s+)?documents?\b", re.I),
    # "what types/kinds of documents do I have" — inventory, not capability
    re.compile(r"\bwhat\s+(?:types?|kinds?)\s+of\s+(?:documents?|files?)\s+(?:do\s+i\s+have|are\s+(?:there|in\s+my))\b", re.I),
    # "how many documents" — document count question
    re.compile(r"\bhow\s+many\s+(?:documents?|files?|resumes?|invoices?)\b", re.I),
]

# Document-query override patterns — if these match, it's NOT conversational.
_DOC_QUERY_OVERRIDES = [
    re.compile(r"\bwho\s+is\s+[A-Z]", re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+the\s+total\b", re.IGNORECASE),
    re.compile(r"\btell\s+me\s+about\s+(?:the|this|his|her|their)\b", re.IGNORECASE),
    re.compile(r"\b(?:summarize|summarise|extract|compare|rank|list|find|analyze|analyse)\b", re.IGNORECASE),
    re.compile(r"\b(?:invoice|resume|contract|document|report|certificate|medication)\b", re.IGNORECASE),
]

# Greeting prefix pattern to strip before re-classifying.
_GREETING_PREFIX_RE = re.compile(
    r"^(?:hi|hello|hey|hiya|howdy|yo|sup|hii|good\s+(?:morning|afternoon|evening|day))"
    r"[,!.\s]+",
    re.IGNORECASE,
)

_INTENT_PATTERNS: Dict[str, List[re.Pattern]] = {
    GREETING: [
        re.compile(r"^(?:hi|hello|hey|hiya|howdy|yo|sup|hii|namaste|hola|bonjour|aloha|salutations)[\s!.,]*$", re.I),
        re.compile(r"^good\s+(?:morning|afternoon|evening|day)[\s!.,]*$", re.I),
        re.compile(r"^(?:what'?s\s+up|how\s+do\s+you\s+do)[\s!?,]*$", re.I),
        re.compile(r"^(?:nice\s+to\s+meet\s+you|greetings)[\s!.,]*$", re.I),
        re.compile(r"^(?:hi|hello|hey)\s+(?:there|team|docwain|assistant|bot|buddy|friend)[\s!.,]*$", re.I),
    ],
    FAREWELL: [
        re.compile(r"^(?:bye|goodbye|good\s*bye|see\s+you|see\s+ya|farewell|ciao|cheerio|adieu)[\s!.,]*$", re.I),
        re.compile(r"^(?:ta\s*ta|tata|catch\s+you\s+later|later|peace\s+out|au\s+revoir|sayonara)[\s!.,]*$", re.I),
        re.compile(r"^(?:take\s+care|until\s+next\s+time|signing\s+off|gotta\s+go|gtg|ttyl)[\s!.,]*$", re.I),
        re.compile(r"^(?:good\s*night|have\s+a\s+(?:good|great|nice)\s+day)[\s!.,]*$", re.I),
        re.compile(r"^(?:that'?s\s+all|i'?m\s+done|end\s+chat|quit|exit|close|finish|stop|terminate)[\s!.,]*$", re.I),
        re.compile(r"^(?:see\s+you\s+(?:soon|around|later)|talk\s+to\s+you\s+later|hasta\s+la\s+vista)[\s!.,]*$", re.I),
    ],
    THANKS: [
        re.compile(r"^(?:thanks|thank\s+you|thanks?\s+a\s+lot|thx|ty|tysm|much\s+appreciated)[\s!.,]*$", re.I),
        re.compile(r"^(?:thank\s+you\s+(?:so\s+much|very\s+much)|appreciate\s+it)[\s!.,]*$", re.I),
        re.compile(r"^(?:thanks\s+for\s+(?:your|the)\s+help|cheers)[\s!.,]*$", re.I),
    ],
    PRAISE: [
        re.compile(r"^(?:awesome|great\s+job|well\s+done|excellent|perfect|amazing|brilliant|fantastic)[\s!.,]*$", re.I),
        re.compile(r"^(?:good\s+answer|great\s+answer|very\s+good|nice|impressive|superb|outstanding)[\s!.,]*$", re.I),
        re.compile(r"^(?:love\s+it|nailed\s+it|spot\s+on|exactly\s+(?:right|what\s+i\s+needed))[\s!.,]*$", re.I),
    ],
    NEGATIVE_MILD: [
        re.compile(r"(?:not\s+quite|could\s+be\s+better|not\s+exactly|not\s+what\s+i\s+(?:meant|wanted|asked))", re.I),
        re.compile(r"(?:try\s+again|can\s+you\s+redo|not\s+(?:great|ideal)|a\s+bit\s+off|close\s+but)", re.I),
        re.compile(r"(?:incomplete|partially?\s+(?:right|correct)|needs?\s+(?:work|improvement))", re.I),
    ],
    NEGATIVE_STRONG: [
        re.compile(r"(?:terrible|useless|awful|horrible|worst|garbage|trash|pathetic|dreadful)", re.I),
        re.compile(r"(?:completely\s+wrong|totally\s+wrong|absolutely\s+wrong|100%\s+wrong)", re.I),
        re.compile(r"(?:bad\s+(?:answer|response)|wrong\s+(?:answer|response)|not\s+(?:right|correct|accurate))", re.I),
        re.compile(r"(?:this\s+is\s+(?:bad|wrong|not\s+(?:right|correct|accurate|helpful)))", re.I),
        re.compile(r"(?:doesn'?t\s+make\s+sense|does\s+not\s+make\s+sense|poor\s+(?:answer|response))", re.I),
    ],
    IDENTITY: [
        re.compile(r"\bwho\s+are\s+you\b", re.I),
        re.compile(r"\bwhat\s+are\s+you\b", re.I),
        re.compile(r"\bwhat\s+is\s+docwain\b", re.I),
        re.compile(r"\btell\s+me\s+about\s+(?:yourself|docwain)\b", re.I),
        re.compile(r"\bintroduce\s+yourself\b", re.I),
        re.compile(r"\bwhat'?s\s+your\s+name\b", re.I),
        re.compile(r"\bare\s+you\s+(?:a\s+(?:bot|ai|robot)|human)\b", re.I),
    ],
    CAPABILITY: [
        re.compile(r"\bwhat\s+can\s+you\s+do\b", re.I),
        re.compile(r"\bwhat\s+else\s+can\s+you\s+do\b", re.I),
        re.compile(r"\bwhat\s+all\s+can\s+(?:you|docwain)\s+do\b", re.I),
        re.compile(r"\bwhat\s+(?:are\s+your|do\s+you\s+have)\s+(?:features|capabilities)\b", re.I),
        re.compile(r"\blist\s+(?:your\s+)?features\b", re.I),
        re.compile(r"\blist\s+(?:your\s+)?capabilities\b", re.I),
        re.compile(r"\bwhat\s+(?:do\s+you|can\s+you)\s+(?:help\s+with|support)\b", re.I),
        re.compile(r"\bwhat\s+else\s+can\s+you\s+help\s+with\b", re.I),
        re.compile(r"\bhow\s+can\s+(?:you|docwain)\s+help(?:\s+me)?\b", re.I),
        re.compile(r"\bwhat\s+can\s+i\s+do\s+with\s+docwain\b", re.I),
        re.compile(r"\bwhat\s+can\s+docwain\s+do\b", re.I),
        re.compile(r"\bshow\s+me\s+what\s+you\s+can\s+do\b", re.I),
        re.compile(r"\bwhat\s+(?:types?|kinds?)\s+of\s+(?:documents?|files?)\s+(?:can|do)\s+you\b", re.I),
    ],
    HOW_IT_WORKS: [
        re.compile(r"\bhow\s+do\s+you\s+work\b", re.I),
        re.compile(r"\bhow\s+does\s+(?:docwain|this|it)\s+work\b", re.I),
        re.compile(r"\bhow\s+accurate\s+(?:are\s+you|is\s+(?:docwain|this|it))\b", re.I),
        re.compile(r"\bwhat\s+(?:technology|tech|model|ai)\s+do\s+you\s+use\b", re.I),
        re.compile(r"\bhow\s+do\s+you\s+(?:retrieve|find|search|process)\b", re.I),
        re.compile(r"\bexplain\s+(?:how\s+you\s+work|your\s+process)\b", re.I),
    ],
    PRIVACY: [
        re.compile(r"\b(?:is\s+)?my\s+data\s+(?:safe|secure|private|protected)\b", re.I),
        re.compile(r"\bwho\s+can\s+(?:see|access|view)\s+my\b", re.I),
        re.compile(r"\bprivacy\b", re.I),
        re.compile(r"\bdata\s+(?:security|protection|privacy|handling)\b", re.I),
        re.compile(r"\bdo\s+you\s+(?:store|keep|share|sell)\s+(?:my\s+)?data\b", re.I),
        re.compile(r"\b(?:gdpr|compliance|data\s+retention)\b", re.I),
    ],
    LIMITATIONS: [
        re.compile(r"\bwhat\s+can'?t\s+you\s+do\b", re.I),
        re.compile(r"\bwhat\s+(?:are\s+your|do\s+you\s+have)\s+limitations?\b", re.I),
        re.compile(r"\bcan\s+you\s+(?:browse|search)\s+(?:the\s+)?(?:internet|web)\b", re.I),
        re.compile(r"\bdo\s+you\s+(?:have\s+)?(?:access\s+to\s+)?(?:the\s+)?internet\b", re.I),
        re.compile(r"\bwhat\s+(?:don'?t|do\s+not)\s+you\s+(?:support|handle)\b", re.I),
    ],
    USAGE_HELP: [
        re.compile(r"\bhow\s+(?:do\s+i|can\s+i|to)\s+(?:start|begin|use)\b", re.I),
        re.compile(r"\bshow\s+(?:me\s+)?(?:examples?|a\s+demo)\b", re.I),
        re.compile(r"\bhelp\s+me\s+(?:get\s+started|begin)\b", re.I),
        re.compile(r"\bgetting\s+started\b", re.I),
        re.compile(r"\bwhat\s+(?:should|can)\s+i\s+(?:ask|try|do)\s+(?:first|next)?\b", re.I),
        re.compile(r"\bquick\s+(?:start|guide|tutorial)\b", re.I),
        # Upload queries
        re.compile(r"\bhow\s+(?:do\s+i|can\s+i|to)\s+(?:upload|add)\s+(?:a\s+)?(?:document|file)\b", re.I),
        # File type queries
        re.compile(r"\bwhat\s+(?:file|document)?\s*(?:formats?|types?)\s+(?:are\s+)?supported\b", re.I),
        re.compile(r"\bcan\s+i\s+upload\s+(?:images?|pdf|excel|word|csv|pptx?)\b", re.I),
        # Domain example queries
        re.compile(r"\b(?:resume|invoice|legal|medical|contract)\s+(?:examples?|help)\b", re.I),
        re.compile(r"\bshow\s+(?:me\s+)?(?:resume|invoice|legal|medical)\s+(?:examples?|queries)\b", re.I),
        # Screening queries
        re.compile(r"\bscreening\s+(?:help|guide)\b", re.I),
        re.compile(r"\bhow\s+(?:does|do)\s+screening\s+work\b", re.I),
        # Content generation
        re.compile(r"\b(?:content\s+generation|generate\s+content)\s+(?:help|guide)\b", re.I),
        # Advanced features
        re.compile(r"\badvanced\s+(?:features?|capabilities?)\b", re.I),
        re.compile(r"\bfine.?tun(?:e|ing)\s+(?:help|guide)\b", re.I),
        # Generic help/guide/tutorial/walkthrough
        re.compile(r"^(?:help|guide|tutorial|walkthrough)[\s!?.]*$", re.I),
        re.compile(r"\bexample\s+(?:queries|questions)\b", re.I),
        re.compile(r"\bsample\s+(?:queries|questions)\b", re.I),
    ],
    SMALL_TALK: [
        re.compile(r"^(?:how\s+are\s+you|how'?s\s+it\s+going|what'?s\s+up|how\s+do\s+you\s+do)[\s!?,]*$", re.I),
        re.compile(r"^(?:how'?s\s+your\s+day|how\s+have\s+you\s+been|you\s+good)[\s!?,]*$", re.I),
        re.compile(r"^(?:what'?s\s+new|anything\s+new|what'?s\s+happening)[\s!?,]*$", re.I),
    ],
    CLARIFICATION: [
        re.compile(r"\b(?:repeat|say)\s+that\s+(?:again|please)\b", re.I),
        re.compile(r"\bexplain\s+(?:that\s+)?again\b", re.I),
        re.compile(r"\bi\s+(?:didn'?t|did\s+not)\s+(?:understand|get\s+(?:that|it))\b", re.I),
        re.compile(r"\bcan\s+you\s+(?:clarify|rephrase|elaborate)\b", re.I),
        re.compile(r"\bwhat\s+(?:do\s+you\s+mean|did\s+you\s+mean)\b", re.I),
        re.compile(r"\bcould\s+you\s+(?:be\s+more\s+)?(?:specific|clear|explicit)\b", re.I),
    ],
}


def classify_conversational_intent(
    text: str,
    turn_count: int = 0,
) -> Optional[Tuple[str, float]]:
    """Classify *text* into a conversational intent.

    Returns ``(intent, confidence)`` or ``None`` when the message should
    proceed to document retrieval.
    """
    text = (text or "").strip()
    if not text:
        return None

    # Document discovery — "what can I do with these documents?" etc.
    for pat in _DOCUMENT_DISCOVERY_PATTERNS:
        if pat.search(text):
            return (DOCUMENT_DISCOVERY, 0.92)

    # Help-meta override — "how do I compare?" is usage help, not a doc query.
    if _HELP_META_OVERRIDE.search(text):
        return (USAGE_HELP, 0.90)

    # Capability-meta override — "what else can you do?" is conversational.
    if _CAPABILITY_META_OVERRIDE.search(text):
        return (CAPABILITY, 0.92)

    # Document query override — these are NOT conversational.
    for pat in _DOC_QUERY_OVERRIDES:
        if pat.search(text):
            return None

    # Try direct match on every intent.
    for intent, patterns in _INTENT_PATTERNS.items():
        for pat in patterns:
            if pat.search(text):
                conf = 0.95 if pat.pattern.startswith("^") else 0.85
                # Upgrade GREETING → GREETING_RETURN when conversation ongoing.
                if intent == GREETING and turn_count > 0:
                    return (GREETING_RETURN, conf)
                return (intent, conf)

    # Combo handling: strip greeting prefix and re-classify remainder.
    m = _GREETING_PREFIX_RE.match(text)
    if m:
        remainder = text[m.end():].strip()
        if remainder:
            sub = classify_conversational_intent(remainder, turn_count)
            if sub is not None:
                return sub

    return None


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


def _mongodb_doc_summary(subscription_id: str, profile_id: str):
    """Fetch document names and domains from MongoDB (authoritative source)."""
    try:
        from src.api.config import Config
        import pymongo
        client = pymongo.MongoClient(Config.MongoDB.URI, serverSelectionTimeoutMS=3000)
        db = client[Config.MongoDB.DB]
        cursor = db.documents.find(
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

    # Fallback: if catalog is incomplete but profile has data, query MongoDB
    doc_count = len(docs)
    if doc_count == 0 and collection_point_count > 0 and subscription_id and profile_id:
        mongo_count, mongo_names, mongo_domains = _mongodb_doc_summary(subscription_id, profile_id)
        if mongo_count > 0:
            doc_count = mongo_count
            doc_names = mongo_names[:5]
            dominant = mongo_domains
    elif doc_count > 0 and collection_point_count > 0 and subscription_id and profile_id:
        # Catalog exists but may be stale — check if MongoDB has more docs
        mongo_count, mongo_names, mongo_domains = _mongodb_doc_summary(subscription_id, profile_id)
        if mongo_count > doc_count:
            doc_count = mongo_count
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

    pools = _FRAGMENT_POOLS.get(intent) or _FRAGMENT_POOLS[GREETING]
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
