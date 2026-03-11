"""
Conversation state tracking for DocWain multi-turn intelligence.

Provides:
- TrackedEntity / EnrichedTurn dataclasses for conversation state
- EntityRegister for cross-turn entity tracking with alias resolution
- ConversationEntityExtractor for deterministic entity extraction (no LLM)
"""
from __future__ import annotations

import json, re, time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------

@dataclass
class TrackedEntity:
    """An entity mentioned in conversation."""
    name: str                    # "Abinaya", "Invoice_001.pdf", "technical skills"
    entity_type: str             # "person", "document", "topic"
    aliases: Set[str] = field(default_factory=set)  # {"she", "her", "abinaya"}
    first_seen_turn: int = 0
    last_seen_turn: int = 0
    mention_count: int = 1
    metadata: Dict[str, str] = field(default_factory=dict)  # {"doc_id": "doc1"}


@dataclass
class EnrichedTurn:
    """A conversation turn with entity and resolution metadata."""
    user_message: str
    assistant_response: str
    timestamp: float
    turn_number: int
    entities_mentioned: List[str] = field(default_factory=list)
    resolved_query: Optional[str] = None


# ---------------------------------------------------------------------------
# EntityRegister
# ---------------------------------------------------------------------------

class EntityRegister:
    """Tracks entities across conversation turns."""

    def __init__(self, max_entities: int = 50):
        self.entities: Dict[str, TrackedEntity] = {}  # canonical name -> entity
        self._alias_map: Dict[str, str] = {}  # lowercase alias -> canonical name
        self.max_entities = max_entities
        self._current_turn: int = 0

    def register(self, name: str, entity_type: str, aliases: Optional[Set[str]] = None,
                 metadata: Optional[Dict[str, str]] = None) -> TrackedEntity:
        """Register or update an entity. Deduplicates by case-insensitive name."""
        canonical = name.strip()
        lower = canonical.lower()
        if lower in self._alias_map:
            existing = self.entities[self._alias_map[lower]]
            existing.last_seen_turn = self._current_turn
            existing.mention_count += 1
            if metadata:
                existing.metadata.update(metadata)
            return existing
        entity = TrackedEntity(
            name=canonical, entity_type=entity_type,
            aliases=aliases or set(),
            first_seen_turn=self._current_turn, last_seen_turn=self._current_turn,
            metadata=metadata or {},
        )
        self._alias_map[lower] = canonical
        for alias in entity.aliases:
            self._alias_map[alias.lower()] = canonical
        self.entities[canonical] = entity
        if len(self.entities) > self.max_entities:
            oldest = min(self.entities.values(), key=lambda e: e.last_seen_turn)
            self._evict(oldest.name)
        return entity

    def lookup(self, reference: str) -> Optional[TrackedEntity]:
        """Look up an entity by name or alias (case-insensitive)."""
        lower = reference.lower().strip()
        if lower in self._alias_map:
            return self.entities.get(self._alias_map[lower])
        return None

    def get_by_type(self, entity_type: str) -> List[TrackedEntity]:
        """Return all entities of the given type, sorted by most recently seen."""
        return sorted(
            [e for e in self.entities.values() if e.entity_type == entity_type],
            key=lambda e: e.last_seen_turn, reverse=True,
        )

    def get_most_recent(self, entity_type: Optional[str] = None) -> Optional[TrackedEntity]:
        """Return the most recently seen entity, optionally filtered by type."""
        candidates = self.get_by_type(entity_type) if entity_type else list(self.entities.values())
        return max(candidates, key=lambda e: e.last_seen_turn) if candidates else None

    def advance_turn(self):
        """Increment the internal turn counter."""
        self._current_turn += 1

    def _evict(self, name: str):
        """Remove an entity and all its alias mappings."""
        entity = self.entities.pop(name, None)
        if entity:
            self._alias_map = {k: v for k, v in self._alias_map.items() if v != name}

    def to_dict(self) -> dict:
        """Serialize for Redis persistence."""
        return {
            "entities": {
                name: {
                    "name": e.name, "entity_type": e.entity_type,
                    "aliases": list(e.aliases),
                    "first_seen_turn": e.first_seen_turn, "last_seen_turn": e.last_seen_turn,
                    "mention_count": e.mention_count, "metadata": e.metadata,
                }
                for name, e in self.entities.items()
            },
            "current_turn": self._current_turn,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EntityRegister:
        """Deserialize from a dict (e.g. loaded from Redis)."""
        reg = cls()
        reg._current_turn = data.get("current_turn", 0)
        for name, ed in data.get("entities", {}).items():
            entity = TrackedEntity(
                name=ed["name"], entity_type=ed["entity_type"],
                aliases=set(ed.get("aliases", [])),
                first_seen_turn=ed.get("first_seen_turn", 0),
                last_seen_turn=ed.get("last_seen_turn", 0),
                mention_count=ed.get("mention_count", 1),
                metadata=ed.get("metadata", {}),
            )
            reg.entities[name] = entity
            reg._alias_map[name.lower()] = name
            for alias in entity.aliases:
                reg._alias_map[alias.lower()] = name
        return reg


# ---------------------------------------------------------------------------
# Deterministic entity extraction patterns
# ---------------------------------------------------------------------------

_PERSON_PATTERNS = [
    re.compile(r"\b(?:about|for|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"),
    re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'s\b"),
    re.compile(r"\bName:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"),
    # Comparison patterns: "X and Y", "X vs Y"
    re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:and|vs\.?|versus|&)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"),
]

_DOCUMENT_PATTERNS = [
    re.compile(r"\b([\w][\w\-]*\.(?:pdf|docx?|xlsx?|csv|txt))\b", re.I),
    re.compile(
        r"\b(?:document|file|resume|invoice|report)\s+(?:called|named|titled)\s+"
        r"[\"']?(\w[\w\s]+?)[\"']?(?:\s|$)",
        re.I,
    ),
]

_TOPIC_PATTERNS = [
    re.compile(
        r"\b(skills?|experience|education|certifications?|salary|contact|summary|qualifications?"
        r"|strengths?|weaknesses?|requirements?|responsibilities|achievements?"
        r"|projects?|languages?|references?|volunteer|objective|hobbies"
        r"|diagnosis|treatment|medication|symptoms?|allergies|vitals?"
        r"|clause|liability|obligations?|terms|indemnification|termination"
        r"|payment|total|amount|vendor|due\s+date|line\s+items?"
        r"|coverage|premium|exclusions?|deductible|beneficiary"
        r")\b",
        re.I,
    ),
]

_STOP_NAMES = {
    "the", "what", "how", "who", "which", "when", "where", "why",
    "tell", "show", "list", "compare", "rank", "find", "get",
    "all", "each", "every", "both", "other", "first", "second",
    "document", "resume", "invoice", "file", "report", "profile",
    "technical", "functional", "experience", "education", "skills",
    "name", "python", "java", "docker",
}

# Topic continuity — map topic keywords to broader topic groups
_TOPIC_GROUPS: Dict[str, str] = {
    "skills": "qualifications", "skill": "qualifications",
    "certifications": "qualifications", "certification": "qualifications",
    "qualifications": "qualifications", "competencies": "qualifications",
    "experience": "career", "work history": "career", "employment": "career",
    "career": "career", "role": "career", "position": "career",
    "education": "education", "degree": "education", "university": "education",
    "training": "education", "academic": "education",
    "salary": "compensation", "compensation": "compensation", "pay": "compensation",
    "benefits": "compensation", "package": "compensation",
    "contact": "contact", "email": "contact", "phone": "contact", "address": "contact",
    "diagnosis": "medical", "treatment": "medical", "medication": "medical",
    "symptoms": "medical", "lab": "medical", "vitals": "medical",
    "clause": "legal", "liability": "legal", "obligation": "legal",
    "terms": "legal", "agreement": "legal", "contract": "legal",
    "payment": "financial", "invoice": "financial", "amount": "financial",
    "total": "financial", "due": "financial", "vendor": "financial",
    "coverage": "policy", "premium": "policy", "exclusion": "policy",
    "deductible": "policy", "policyholder": "policy",
}

_PRONOUN_MAP = {
    "he": "person", "him": "person", "his": "person",
    "she": "person", "her": "person", "hers": "person",
    "they": "person", "them": "person", "their": "person", "theirs": "person",
    "it": None,
    "that document": "document", "this document": "document", "the document": "document",
    "that file": "document", "this file": "document", "the file": "document",
    "that resume": "document", "this resume": "document", "the resume": "document",
    "that invoice": "document", "this invoice": "document", "the invoice": "document",
}


class ConversationEntityExtractor:
    """Extracts entities from text deterministically (no LLM calls)."""

    @staticmethod
    def extract_persons(text: str) -> List[str]:
        """Extract person names from text using regex patterns."""
        names = []
        for pattern in _PERSON_PATTERNS:
            for match in pattern.finditer(text):
                # Handle patterns with multiple groups (e.g., "X and Y")
                for gi in range(1, pattern.groups + 1):
                    name = (match.group(gi) or "").strip()
                    if name and name.lower() not in _STOP_NAMES and len(name) > 1:
                        names.append(name)
        return list(dict.fromkeys(names))

    @staticmethod
    def extract_documents(text: str) -> List[str]:
        """Extract document references (filenames) from text."""
        docs = []
        for pattern in _DOCUMENT_PATTERNS:
            for match in pattern.finditer(text):
                docs.append(match.group(1).strip())
        return list(dict.fromkeys(docs))

    @staticmethod
    def extract_topics(text: str) -> List[str]:
        """Extract topic keywords from text."""
        # Words where stripping trailing 's' would corrupt them
        _NO_STRIP = frozenset({
            "diagnosis", "analysis", "basis", "emphasis", "synopsis",
            "thesis", "hypothesis", "crisis", "vitals", "details",
            "terms", "conditions", "items",
        })
        topics = []
        for pattern in _TOPIC_PATTERNS:
            for match in pattern.finditer(text):
                raw = match.group(0).lower()
                if raw not in _NO_STRIP and raw.endswith("s") and not raw.endswith("ss"):
                    raw = raw.rstrip("s")
                topics.append(raw)
        return list(dict.fromkeys(topics))

    @staticmethod
    def detect_pronouns(text: str) -> List[str]:
        """Detect pronouns and document references in text."""
        found = []
        lower = text.lower()
        for pronoun in _PRONOUN_MAP:
            if re.search(rf"\b{re.escape(pronoun)}\b", lower):
                found.append(pronoun)
        return found


# ---------------------------------------------------------------------------
# Pronoun & Reference Resolver
# ---------------------------------------------------------------------------

_ORDINAL_MAP = {"first": 0, "second": 1, "third": 2, "1st": 0, "2nd": 1, "3rd": 2}
_ORDINAL_RE = re.compile(
    r"\b(?:the\s+)?(first|second|third|1st|2nd|3rd)\s+(candidate|person|one|document|resume|invoice|file)\b", re.I,
)

# Follow-up reference patterns — "the previous answer contained X", "you mentioned X"
_FOLLOWUP_REFERENCE_RE = re.compile(
    r"\b(?:the\s+)?(?:previous|last|earlier|prior)\s+(?:answer|response|reply)\s+"
    r"(?:contained|mentioned|said|included|had|showed|stated)\s+(.+?)(?:\s+but\b|\s+however\b|[.?!]|$)",
    re.I,
)
_FOLLOWUP_MENTION_RE = re.compile(
    r"\b(?:you\s+)?(?:mentioned|said|stated|told\s+me|showed)\s+(.+?)\s+(?:earlier|before|previously|in\s+the\s+last)",
    re.I,
)

_IMPLICIT_COMPARISON_RE = re.compile(
    r"(?:what\s+about|how\s+about|how\s+(?:does|do|did)\s+\w+\s+(?:compare|differ|stack\s+up))"
    r"|(?:better\s+than|worse\s+than|compared?\s+(?:to|with))"
    r"|(?:and\s+what\s+about|versus|vs\.?\s)",
    re.IGNORECASE,
)

# "the same" / "another" / "a different" reference patterns
_ANAPHORIC_REF_RE = re.compile(
    r"\b(?:the\s+same|another|a\s+different|the\s+other|the\s+next|the\s+previous)\s+"
    r"(candidate|person|document|file|resume|invoice|patient|contract|policy|report)\b",
    re.IGNORECASE,
)


class ConversationContextResolver:
    """Resolves pronouns and references using the EntityRegister."""

    def __init__(self, register: EntityRegister, enriched_turns: Optional[List] = None):
        self.register = register
        self.enriched_turns: List[EnrichedTurn] = enriched_turns if enriched_turns is not None else []

    def resolve(self, query: str) -> str:
        resolved = query
        resolved = self._resolve_followup_references(resolved)
        resolved = self._resolve_implicit_comparison(resolved)
        resolved = self._resolve_anaphoric_refs(resolved)
        resolved = self._resolve_ordinals(resolved)
        resolved = self._resolve_pronouns(resolved)
        resolved = self._resolve_demonstratives(resolved)
        return resolved

    def needs_resolution(self, query: str) -> bool:
        pronouns = ConversationEntityExtractor.detect_pronouns(query)
        if pronouns:
            return True
        if _ORDINAL_RE.search(query):
            return True
        if _FOLLOWUP_REFERENCE_RE.search(query) or _FOLLOWUP_MENTION_RE.search(query):
            return True
        if _IMPLICIT_COMPARISON_RE.search(query):
            return True
        if _ANAPHORIC_REF_RE.search(query):
            return True
        return False

    def _resolve_followup_references(self, query: str) -> str:
        """Resolve follow-up references like 'the previous answer contained X'."""
        match = _FOLLOWUP_REFERENCE_RE.search(query)
        if not match:
            match = _FOLLOWUP_MENTION_RE.search(query)
        if not match:
            return query
        referenced_content = match.group(1).strip()
        # Find the referenced content in prior turns
        prior_context = self._find_prior_mention(referenced_content)
        if prior_context:
            # Append prior context as search context
            return f"{query} [Context from prior answer: {prior_context}]"
        return query

    def _find_prior_mention(self, content: str) -> Optional[str]:
        """Search prior enriched turns for a mention of *content*."""
        content_lower = content.lower()
        for turn in reversed(self.enriched_turns):
            response_lower = (turn.assistant_response or "").lower()
            if content_lower in response_lower:
                # Return a snippet around the mention (up to 200 chars)
                idx = response_lower.find(content_lower)
                start = max(0, idx - 50)
                end = min(len(turn.assistant_response), idx + len(content) + 100)
                return turn.assistant_response[start:end].strip()
        return None

    def _resolve_implicit_comparison(self, query: str) -> str:
        """Inject prior entity context when an implicit comparison is detected."""
        if not _IMPLICIT_COMPARISON_RE.search(query):
            return query
        prior_entities = self._get_recent_entities(max_entities=2)
        if prior_entities:
            context_note = f" [Previously discussed: {', '.join(prior_entities)}]"
            return query + context_note
        return query

    def _get_recent_entities(self, max_entities: int = 2) -> List[str]:
        """Retrieve the most recently seen entity names from the register."""
        all_entities = sorted(
            self.register.entities.values(),
            key=lambda e: e.last_seen_turn,
            reverse=True,
        )
        names: List[str] = []
        for entity in all_entities:
            if entity.name not in names:
                names.append(entity.name)
            if len(names) >= max_entities:
                break
        return names

    def _resolve_pronouns(self, query: str) -> str:
        result = query
        for pronoun, entity_type in _PRONOUN_MAP.items():
            if len(pronoun.split()) > 1:
                continue  # skip multi-word phrases (handled by _resolve_demonstratives)
            pattern = re.compile(rf"\b{re.escape(pronoun)}\b", re.I)
            if not pattern.search(result):
                continue
            entity = self.register.get_most_recent(entity_type)
            if entity:
                if pronoun in {"his", "her", "their", "its"}:
                    replacement = f"{entity.name}'s"
                else:
                    replacement = entity.name
                result = pattern.sub(replacement, result, count=1)

        # Resolve topic-continuity references: "what about that?" / "anything else?"
        if re.search(r"\b(?:anything\s+else|what\s+else|more\s+(?:about|on)\s+(?:this|that))\b", result, re.I):
            recent = self._get_recent_entities(max_entities=1)
            if recent:
                result = f"{result} [Topic continuation about: {recent[0]}]"

        return result

    def _resolve_anaphoric_refs(self, query: str) -> str:
        """Resolve 'the same candidate', 'another document', 'a different person' etc."""
        match = _ANAPHORIC_REF_RE.search(query)
        if not match:
            return query
        ref_noun = match.group(1).lower()
        full_phrase = match.group(0).lower()
        entity_type = "document" if ref_noun in {"document", "file", "resume", "invoice", "report"} else "person"

        if "same" in full_phrase:
            # "the same candidate" → most recent entity of that type
            entity = self.register.get_most_recent(entity_type)
            if entity:
                return query[:match.start()] + entity.name + query[match.end():]
        elif "another" in full_phrase or "different" in full_phrase or "other" in full_phrase:
            # "another candidate" → second-most-recent entity of that type
            entities = self.register.get_by_type(entity_type)
            if len(entities) >= 2:
                # get_by_type returns sorted by last_seen desc, so [1] is second-most-recent
                name = entities[1].name
                return query[:match.start()] + name + query[match.end():]
        elif "next" in full_phrase:
            # "the next candidate" → entity after the most recent by first_seen order
            entities = sorted(self.register.get_by_type(entity_type), key=lambda e: e.first_seen_turn)
            most_recent = self.register.get_most_recent(entity_type)
            if most_recent and entities:
                idx = next((i for i, e in enumerate(entities) if e.name == most_recent.name), -1)
                if idx >= 0 and idx + 1 < len(entities):
                    return query[:match.start()] + entities[idx + 1].name + query[match.end():]
        elif "previous" in full_phrase:
            # "the previous candidate" → entity before the most recent
            entities = sorted(self.register.get_by_type(entity_type), key=lambda e: e.first_seen_turn)
            most_recent = self.register.get_most_recent(entity_type)
            if most_recent and entities:
                idx = next((i for i, e in enumerate(entities) if e.name == most_recent.name), -1)
                if idx > 0:
                    return query[:match.start()] + entities[idx - 1].name + query[match.end():]
        return query

    def _resolve_ordinals(self, query: str) -> str:
        match = _ORDINAL_RE.search(query)
        if not match:
            return query
        ordinal = match.group(1).lower()
        ref_type = match.group(2).lower()
        idx = _ORDINAL_MAP.get(ordinal, 0)
        entity_type = "document" if ref_type in {"document", "resume", "invoice", "file"} else "person"
        entities = self.register.get_by_type(entity_type)
        entities_by_appearance = sorted(entities, key=lambda e: e.first_seen_turn)
        if idx < len(entities_by_appearance):
            name = entities_by_appearance[idx].name
            return query[:match.start()] + name + query[match.end():]
        return query

    def _resolve_demonstratives(self, query: str) -> str:
        for ref_phrase in ["that document", "this document", "the document",
                           "that file", "this file", "the file",
                           "that resume", "this resume", "the resume",
                           "that invoice", "this invoice", "the invoice"]:
            pattern = re.compile(rf"\b{re.escape(ref_phrase)}\b", re.I)
            if pattern.search(query):
                entity = self.register.get_most_recent("document")
                if entity:
                    query = pattern.sub(entity.name, query, count=1)
        return query


# ---------------------------------------------------------------------------
# Helper: person alias generation
# ---------------------------------------------------------------------------

_TITLE_STRIP_RE = re.compile(
    r"^(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Prof\.?|Sir|Rev\.?)\s+", re.I,
)

def _person_aliases(name: str) -> Set[str]:
    aliases = {name.lower()}
    parts = name.split()
    if len(parts) >= 2:
        aliases.add(parts[0].lower())   # first name
        aliases.add(parts[-1].lower())  # last name
    # Also add title-stripped version: "Dr. John Smith" → "John Smith"
    stripped = _TITLE_STRIP_RE.sub("", name).strip()
    if stripped != name and len(stripped) > 2:
        aliases.add(stripped.lower())
    return aliases


# ---------------------------------------------------------------------------
# ConversationState Manager
# ---------------------------------------------------------------------------

class ConversationState:
    """Wraps ConversationHistory with entity tracking and context resolution."""

    def __init__(self, conversation_history=None, max_enriched_turns: int = 10,
                 redis_client=None, state_ttl: int = 7 * 24 * 3600):
        self.history = conversation_history
        self.entity_register = EntityRegister()
        self.extractor = ConversationEntityExtractor()
        self.enriched_turns: List[EnrichedTurn] = []
        self.resolver = ConversationContextResolver(self.entity_register, self.enriched_turns)
        self.max_enriched_turns = max_enriched_turns
        self.current_topic: Optional[str] = None
        self.redis = redis_client
        self.state_ttl = state_ttl
        self._turn_counter = 0

    def _state_key(self, namespace: str, user_id: str) -> str:
        return f"rag:memory:conv_state:{namespace}:{user_id}"

    def resolve_query(self, query: str, namespace: str, user_id: str) -> str:
        """Resolve pronouns/references. Call BEFORE preprocess_query()."""
        if not query or len(query) > 500:
            return query
        self._load_state(namespace, user_id)
        if not self.resolver.needs_resolution(query):
            return query
        return self.resolver.resolve(query)

    def record_turn(self, namespace: str, user_id: str, user_message: str,
                    assistant_response: str, resolved_query: Optional[str] = None):
        """Record turn: extract entities, update register, persist."""
        self._load_state(namespace, user_id)
        self.entity_register.advance_turn()
        self._turn_counter += 1

        persons = self.extractor.extract_persons(user_message)
        documents = self.extractor.extract_documents(user_message)
        topics = self.extractor.extract_topics(user_message)

        # Also extract from assistant response (first 800 chars for better coverage)
        persons += self.extractor.extract_persons(assistant_response[:800])
        documents += self.extractor.extract_documents(assistant_response[:800])

        # Extract bold-formatted names from assistant responses: **John Smith**
        bold_names = re.findall(r"\*\*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\*\*", assistant_response[:800])
        for bn in bold_names:
            if bn.lower() not in _STOP_NAMES and len(bn.split()) <= 4:
                persons.append(bn)

        # Extract names from table rows: | John Smith | ... |
        table_names = re.findall(
            r"\|\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s*\|",
            assistant_response[:1200],
        )
        for tn in table_names:
            if tn.lower() not in _STOP_NAMES and len(tn.split()) <= 4:
                persons.append(tn)

        entity_names = []
        for name in dict.fromkeys(persons):
            self.entity_register.register(name, "person", aliases=_person_aliases(name))
            entity_names.append(name)
        for doc in dict.fromkeys(documents):
            self.entity_register.register(doc, "document")
            entity_names.append(doc)
        for topic in dict.fromkeys(topics):
            self.entity_register.register(topic, "topic")
            self.current_topic = topic

        enriched = EnrichedTurn(
            user_message=user_message,
            assistant_response=assistant_response[:600],
            timestamp=time.time(), turn_number=self._turn_counter,
            entities_mentioned=entity_names, resolved_query=resolved_query,
        )
        self.enriched_turns.append(enriched)
        if len(self.enriched_turns) > self.max_enriched_turns:
            self.enriched_turns = self.enriched_turns[-self.max_enriched_turns:]

        # Delegate to ConversationHistory for backward compat
        if self.history:
            self.history.add_turn(namespace, user_id, user_message, assistant_response)

        self._persist_state(namespace, user_id)

    def get_entity_context(self) -> str:
        persons = self.entity_register.get_by_type("person")
        docs = self.entity_register.get_by_type("document")
        if not persons and not docs:
            return ""
        parts = []
        if persons:
            # Sort by salience (recency + frequency)
            scored = sorted(persons, key=lambda e: self._entity_salience(e), reverse=True)
            parts.append(f"Mentioned persons: {', '.join(e.name for e in scored[:5])}")
        if docs:
            parts.append(f"Referenced documents: {', '.join(e.name for e in docs[:5])}")
        if self.current_topic:
            parts.append(f"Current topic: {self.current_topic}")
        # Add topic continuity hint
        topic_group = self._get_topic_group()
        if topic_group and topic_group != self.current_topic:
            parts.append(f"Topic area: {topic_group}")
        return "; ".join(parts)

    def _entity_salience(self, entity: TrackedEntity) -> float:
        """Score entity salience: combines recency and frequency.

        Recent + frequent entities are more salient (likely the focus of conversation).
        """
        recency = 1.0 / max(1, self._turn_counter - entity.last_seen_turn + 1)
        frequency = min(entity.mention_count / 5.0, 1.0)  # Cap at 5 mentions
        return 0.6 * recency + 0.4 * frequency

    def _get_topic_group(self) -> Optional[str]:
        """Infer the broader topic group from current topic."""
        if not self.current_topic:
            return None
        return _TOPIC_GROUPS.get(self.current_topic.lower())

    def clear(self, namespace: str, user_id: str):
        self.entity_register = EntityRegister()
        self.enriched_turns.clear()
        self.resolver = ConversationContextResolver(self.entity_register, self.enriched_turns)
        self.current_topic = None
        self._turn_counter = 0
        if self.history:
            self.history.clear_history(namespace, user_id)
        if self.redis:
            try:
                self.redis.delete(self._state_key(namespace, user_id))
            except Exception:
                pass

    def _load_state(self, namespace: str, user_id: str):
        if not self.redis:
            return
        # Only load from Redis if we haven't already loaded in this session
        if getattr(self, "_state_loaded", False):
            return
        try:
            cached = self.redis.get(self._state_key(namespace, user_id))
            if cached:
                data = json.loads(cached)
                self.entity_register = EntityRegister.from_dict(data.get("register", {}))
                self.resolver = ConversationContextResolver(self.entity_register, self.enriched_turns)
                self._turn_counter = data.get("turn_counter", 0)
                self.current_topic = data.get("current_topic")
            self._state_loaded = True
        except Exception:
            pass

    def _persist_state(self, namespace: str, user_id: str):
        if not self.redis:
            return
        try:
            data = {
                "register": self.entity_register.to_dict(),
                "turn_counter": self._turn_counter,
                "current_topic": self.current_topic,
            }
            self.redis.setex(self._state_key(namespace, user_id), self.state_ttl, json.dumps(data))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ProgressiveSummarizer
# ---------------------------------------------------------------------------

def _is_gemini_backend(llm_client) -> bool:
    return getattr(llm_client, "backend", "") == "gemini" or \
           "gemini" in getattr(llm_client, "model_name", "").lower()


class ProgressiveSummarizer:
    """Incrementally updates summary instead of regenerating from scratch."""

    def __init__(self, llm_client=None, max_bullets: int = 8):
        self.llm_client = llm_client
        self.max_bullets = max_bullets
        self._current_summary: str = ""
        self._summarized_up_to: int = 0

    def update(self, new_turn: EnrichedTurn, existing_summary: str = "") -> str:
        if new_turn.turn_number <= self._summarized_up_to:
            return self._current_summary
        if self.llm_client and not _is_gemini_backend(self.llm_client):
            try:
                return self._llm_update(new_turn, existing_summary)
            except Exception:
                pass
        return self._deterministic_update(new_turn, existing_summary)

    def _deterministic_update(self, new_turn: EnrichedTurn, existing_summary: str) -> str:
        base = existing_summary or self._current_summary
        bullets = [b.strip() for b in base.split("\n") if b.strip().startswith("-")] if base else []
        query_part = new_turn.user_message[:80].strip()
        response_part = new_turn.assistant_response[:80].strip()
        bullets.append(f"- User asked: {query_part}; answered about: {response_part}")
        if len(bullets) > self.max_bullets:
            bullets = bullets[-self.max_bullets:]
        self._current_summary = "\n".join(bullets)
        self._summarized_up_to = new_turn.turn_number
        return self._current_summary

    def _llm_update(self, new_turn: EnrichedTurn, existing_summary: str) -> str:
        prompt = f"""Update the conversation summary by incorporating the new turn.
Keep it as {self.max_bullets} concise bullets maximum. Only ADD info from the new turn.

EXISTING SUMMARY:
{existing_summary or "(empty)"}

NEW TURN:
User: {new_turn.user_message}
Assistant: {new_turn.assistant_response[:200]}

UPDATED SUMMARY:"""
        try:
            from src.llm.task_router import task_scope, TaskType
            _ctx = task_scope(TaskType.CONVERSATION_SUMMARY)
        except ImportError:
            from contextlib import nullcontext
            _ctx = nullcontext()
        with _ctx:
            result = self.llm_client.generate(prompt, max_retries=1, backoff=0.3)
        if result and result.strip():
            self._current_summary = result.strip()
            self._summarized_up_to = new_turn.turn_number
            return self._current_summary
        return self._deterministic_update(new_turn, existing_summary)

    def get_summary(self) -> str:
        return self._current_summary

    def clear(self):
        self._current_summary = ""
        self._summarized_up_to = 0
