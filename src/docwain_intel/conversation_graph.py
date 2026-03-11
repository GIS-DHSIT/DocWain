"""
In-memory conversation graph for tracking session state.

Tracks entities discussed, facts disclosed, supports pronoun resolution
and progressive disclosure across conversation turns.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """A single turn in the conversation."""

    turn_id: int
    query: str
    entities_mentioned: List[str] = Field(default_factory=list)
    entity_labels: Dict[str, str] = Field(default_factory=dict)
    facts_disclosed: List[str] = Field(default_factory=list)
    response_text: str = ""
    timestamp: float = Field(default_factory=time.time)


# Pronoun groups for resolution
_PERSON_PRONOUNS = frozenset({
    "he", "him", "his", "she", "her", "hers", "himself", "herself",
})
_NON_PERSON_PRONOUNS = frozenset({"it", "its", "itself"})
_GENERAL_PRONOUNS = frozenset({"they", "their", "them", "themselves"})
_DEMONSTRATIVE_PRONOUNS = frozenset({"this", "that"})
_SAME_REFERENCES = frozenset({"the same", "same one"})
_ALL_PRONOUNS = _PERSON_PRONOUNS | _NON_PERSON_PRONOUNS | _GENERAL_PRONOUNS | _DEMONSTRATIVE_PRONOUNS

_ANYTHING_ELSE_RE = re.compile(
    r"\b(anything\s+else|what\s+else|tell\s+me\s+more|more\s+about)\b",
    re.IGNORECASE,
)

_PRONOUN_RE = re.compile(
    r"\b("
    + "|".join(sorted(_ALL_PRONOUNS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)


class ConversationGraph:
    """
    In-memory conversation graph that tracks session state.

    Thread-safe via a reentrant lock on all mutations and reads.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._turns: List[ConversationTurn] = []
        self._next_turn_id: int = 1
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_turn(
        self,
        query: str,
        entities: List[str],
        entity_labels: Dict[str, str],
        facts_disclosed: List[str],
        response_text: str,
    ) -> ConversationTurn:
        """Add a conversation turn. Auto-increments turn_id."""
        with self._lock:
            turn = ConversationTurn(
                turn_id=self._next_turn_id,
                query=query,
                entities_mentioned=list(entities),
                entity_labels=dict(entity_labels),
                facts_disclosed=list(facts_disclosed),
                response_text=response_text,
                timestamp=time.time(),
            )
            self._turns.append(turn)
            self._next_turn_id += 1
            return turn

    def resolve_reference(self, text: str) -> Optional[str]:
        """
        Resolve a pronoun or reference phrase to an entity text.

        Returns None if no resolution is possible.
        """
        with self._lock:
            if not self._turns:
                return None

            low = text.lower().strip()

            # "the same" / "same one" → most recent entity
            if low in _SAME_REFERENCES:
                return self._most_recent_entity()

            # Demonstrative pronouns → most recent entity (topic)
            if low in _DEMONSTRATIVE_PRONOUNS:
                return self._most_recent_entity()

            # Person pronouns → most recent PERSON entity
            if low in _PERSON_PRONOUNS:
                return self._most_recent_entity_by_label("PERSON")

            # Non-person pronouns → most recent non-PERSON entity
            if low in _NON_PERSON_PRONOUNS:
                return self._most_recent_non_person_entity()

            # General pronouns → most recent entity of any type
            if low in _GENERAL_PRONOUNS:
                return self._most_recent_entity()

            return None

    def get_undisclosed_facts(
        self, entity_text: str, all_facts: List[str]
    ) -> List[str]:
        """
        Return facts about an entity that haven't been disclosed yet.

        Enables progressive disclosure — previously revealed facts are excluded.
        """
        with self._lock:
            disclosed: set[str] = set()
            for turn in self._turns:
                for fact in turn.facts_disclosed:
                    disclosed.add(fact)
            return [f for f in all_facts if f not in disclosed]

    def get_context_entities(self, max_count: int = 5) -> List[Dict[str, Any]]:
        """
        Return most salient entities across the conversation.

        Salience = 0.6 * recency_score + 0.4 * frequency_score.
        recency_score = 1.0 for most recent turn, decays by 0.15 per turn.
        frequency_score = mention_count / max_mention_count.
        """
        with self._lock:
            if not self._turns:
                return []

            # Gather frequency and most recent turn_id per entity
            entity_freq: Dict[str, int] = {}
            entity_last_turn: Dict[str, int] = {}
            entity_label: Dict[str, str] = {}

            for turn in self._turns:
                for ent in turn.entities_mentioned:
                    entity_freq[ent] = entity_freq.get(ent, 0) + 1
                    if turn.turn_id > entity_last_turn.get(ent, 0):
                        entity_last_turn[ent] = turn.turn_id
                    if ent in turn.entity_labels:
                        entity_label[ent] = turn.entity_labels[ent]

            if not entity_freq:
                return []

            max_freq = max(entity_freq.values())
            max_turn_id = self._turns[-1].turn_id

            results: List[Dict[str, Any]] = []
            for ent, freq in entity_freq.items():
                turns_ago = max_turn_id - entity_last_turn[ent]
                recency_score = max(0.0, 1.0 - 0.15 * turns_ago)
                frequency_score = freq / max_freq if max_freq > 0 else 0.0
                salience = 0.6 * recency_score + 0.4 * frequency_score
                results.append({
                    "entity": ent,
                    "label": entity_label.get(ent, "UNKNOWN"),
                    "frequency": freq,
                    "recency_score": round(recency_score, 4),
                    "frequency_score": round(frequency_score, 4),
                    "salience": round(salience, 4),
                })

            results.sort(key=lambda x: x["salience"], reverse=True)
            return results[:max_count]

    def resolve_query(self, query: str) -> str:
        """
        Full query resolution.

        - Finds pronouns and replaces them with resolved entity text.
        - Handles "anything else?" / "what else?" by appending recent entity.
        - Returns resolved query or original if no resolution needed.
        """
        with self._lock:
            if not self._turns:
                return query

            # Handle "anything else?" / "what else?" / "tell me more"
            if _ANYTHING_ELSE_RE.search(query):
                recent = self._most_recent_entity()
                if recent:
                    return f"{query} about {recent}"
                return query

            # Replace pronouns
            resolved = query

            def _replace_pronoun(match: re.Match) -> str:
                pronoun = match.group(0)
                low = pronoun.lower()

                if low in _PERSON_PRONOUNS:
                    entity = self._most_recent_entity_by_label("PERSON")
                elif low in _NON_PERSON_PRONOUNS:
                    entity = self._most_recent_non_person_entity()
                elif low in _GENERAL_PRONOUNS:
                    entity = self._most_recent_entity()
                elif low in _DEMONSTRATIVE_PRONOUNS:
                    entity = self._most_recent_entity()
                else:
                    entity = None

                return entity if entity else match.group(0)

            resolved = _PRONOUN_RE.sub(_replace_pronoun, resolved)
            return resolved

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Return a summary of the conversation state."""
        with self._lock:
            all_entities: set[str] = set()
            total_facts = 0
            most_recent_entity: Optional[str] = None

            for turn in self._turns:
                for ent in turn.entities_mentioned:
                    all_entities.add(ent)
                total_facts += len(turn.facts_disclosed)

            most_recent_entity = self._most_recent_entity()

            return {
                "session_id": self.session_id,
                "turn_count": len(self._turns),
                "entities_discussed": sorted(all_entities),
                "total_facts_disclosed": total_facts,
                "most_recent_entity": most_recent_entity,
            }

    def reset(self) -> None:
        """Clear all state."""
        with self._lock:
            self._turns.clear()
            self._next_turn_id = 1

    # ------------------------------------------------------------------
    # Internal helpers (caller must hold _lock)
    # ------------------------------------------------------------------

    def _most_recent_entity(self) -> Optional[str]:
        """Return the most recently mentioned entity across all turns."""
        for turn in reversed(self._turns):
            if turn.entities_mentioned:
                return turn.entities_mentioned[-1]
        return None

    def _most_recent_entity_by_label(self, label: str) -> Optional[str]:
        """Return the most recently mentioned entity with a given label."""
        for turn in reversed(self._turns):
            for ent in reversed(turn.entities_mentioned):
                if turn.entity_labels.get(ent) == label:
                    return ent
        return None

    def _most_recent_non_person_entity(self) -> Optional[str]:
        """Return the most recently mentioned non-PERSON entity."""
        for turn in reversed(self._turns):
            for ent in reversed(turn.entities_mentioned):
                lbl = turn.entity_labels.get(ent, "UNKNOWN")
                if lbl != "PERSON":
                    return ent
        return None
