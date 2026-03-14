"""Tests for the in-memory conversation graph."""

import pytest

from src.docwain_intel.conversation_graph import ConversationGraph, ConversationTurn


@pytest.fixture
def graph():
    return ConversationGraph(session_id="test-session-1")


# ------------------------------------------------------------------
# 1. add_turn tracks entities
# ------------------------------------------------------------------
def test_add_turn_tracks_entities(graph):
    turn = graph.add_turn(
        query="Tell me about John Smith",
        entities=["John Smith"],
        entity_labels={"John Smith": "PERSON"},
        facts_disclosed=["John is a software engineer"],
        response_text="John Smith is a software engineer.",
    )
    assert turn.turn_id == 1
    assert turn.entities_mentioned == ["John Smith"]
    assert turn.entity_labels == {"John Smith": "PERSON"}
    assert turn.facts_disclosed == ["John is a software engineer"]

    # Second turn auto-increments
    turn2 = graph.add_turn(
        query="What about Acme Corp?",
        entities=["Acme Corp"],
        entity_labels={"Acme Corp": "ORG"},
        facts_disclosed=[],
        response_text="Acme Corp is a company.",
    )
    assert turn2.turn_id == 2


# ------------------------------------------------------------------
# 2. resolve_reference "his" → most recent PERSON
# ------------------------------------------------------------------
def test_resolve_reference_his_to_person(graph):
    graph.add_turn(
        query="Tell me about John Smith",
        entities=["John Smith"],
        entity_labels={"John Smith": "PERSON"},
        facts_disclosed=[],
        response_text="John Smith is an engineer.",
    )
    assert graph.resolve_reference("his") == "John Smith"
    assert graph.resolve_reference("him") == "John Smith"
    assert graph.resolve_reference("he") == "John Smith"
    assert graph.resolve_reference("her") == "John Smith"  # gender-agnostic


# ------------------------------------------------------------------
# 3. resolve_reference "it" → most recent non-PERSON entity
# ------------------------------------------------------------------
def test_resolve_reference_it_to_non_person(graph):
    graph.add_turn(
        query="Tell me about John and Acme Corp",
        entities=["John Smith", "Acme Corp"],
        entity_labels={"John Smith": "PERSON", "Acme Corp": "ORG"},
        facts_disclosed=[],
        response_text="John works at Acme Corp.",
    )
    assert graph.resolve_reference("it") == "Acme Corp"
    assert graph.resolve_reference("its") == "Acme Corp"


# ------------------------------------------------------------------
# 4. resolve_reference with no prior context → None
# ------------------------------------------------------------------
def test_resolve_reference_no_context(graph):
    assert graph.resolve_reference("his") is None
    assert graph.resolve_reference("it") is None
    assert graph.resolve_reference("they") is None
    assert graph.resolve_reference("this") is None
    assert graph.resolve_reference("the same") is None


# ------------------------------------------------------------------
# 5. get_undisclosed_facts returns only new facts
# ------------------------------------------------------------------
def test_get_undisclosed_facts_returns_new(graph):
    graph.add_turn(
        query="Tell me about John",
        entities=["John"],
        entity_labels={"John": "PERSON"},
        facts_disclosed=["John is an engineer", "John has 5 years experience"],
        response_text="...",
    )
    all_facts = [
        "John is an engineer",
        "John has 5 years experience",
        "John knows Python",
        "John lives in NYC",
    ]
    undisclosed = graph.get_undisclosed_facts("John", all_facts)
    assert undisclosed == ["John knows Python", "John lives in NYC"]


# ------------------------------------------------------------------
# 6. Progressive: after disclosing facts, they don't appear again
# ------------------------------------------------------------------
def test_progressive_disclosure(graph):
    all_facts = ["fact_a", "fact_b", "fact_c", "fact_d"]

    graph.add_turn(
        query="q1", entities=["X"], entity_labels={"X": "PERSON"},
        facts_disclosed=["fact_a", "fact_b"], response_text="...",
    )
    assert graph.get_undisclosed_facts("X", all_facts) == ["fact_c", "fact_d"]

    graph.add_turn(
        query="q2", entities=["X"], entity_labels={"X": "PERSON"},
        facts_disclosed=["fact_c"], response_text="...",
    )
    assert graph.get_undisclosed_facts("X", all_facts) == ["fact_d"]

    graph.add_turn(
        query="q3", entities=["X"], entity_labels={"X": "PERSON"},
        facts_disclosed=["fact_d"], response_text="...",
    )
    assert graph.get_undisclosed_facts("X", all_facts) == []


# ------------------------------------------------------------------
# 7. get_context_entities respects recency
# ------------------------------------------------------------------
def test_context_entities_recency(graph):
    graph.add_turn(
        query="q1", entities=["OldEntity"], entity_labels={"OldEntity": "ORG"},
        facts_disclosed=[], response_text="...",
    )
    graph.add_turn(
        query="q2", entities=["NewEntity"], entity_labels={"NewEntity": "ORG"},
        facts_disclosed=[], response_text="...",
    )
    results = graph.get_context_entities(max_count=5)
    assert len(results) == 2
    # NewEntity should rank higher (more recent)
    assert results[0]["entity"] == "NewEntity"
    assert results[0]["salience"] > results[1]["salience"]


# ------------------------------------------------------------------
# 8. get_context_entities respects frequency
# ------------------------------------------------------------------
def test_context_entities_frequency(graph):
    # Mention "FreqEntity" 3 times vs "RareEntity" 1 time, same recency
    graph.add_turn(
        query="q1", entities=["FreqEntity", "RareEntity"],
        entity_labels={"FreqEntity": "ORG", "RareEntity": "ORG"},
        facts_disclosed=[], response_text="...",
    )
    graph.add_turn(
        query="q2", entities=["FreqEntity"],
        entity_labels={"FreqEntity": "ORG"},
        facts_disclosed=[], response_text="...",
    )
    graph.add_turn(
        query="q3", entities=["FreqEntity"],
        entity_labels={"FreqEntity": "ORG"},
        facts_disclosed=[], response_text="...",
    )
    results = graph.get_context_entities(max_count=5)
    freq_entry = next(r for r in results if r["entity"] == "FreqEntity")
    rare_entry = next(r for r in results if r["entity"] == "RareEntity")
    assert freq_entry["frequency"] == 3
    assert rare_entry["frequency"] == 1
    assert freq_entry["salience"] > rare_entry["salience"]


# ------------------------------------------------------------------
# 9. resolve_query replaces pronouns in query string
# ------------------------------------------------------------------
def test_resolve_query_replaces_pronouns(graph):
    graph.add_turn(
        query="Tell me about John Smith",
        entities=["John Smith"],
        entity_labels={"John Smith": "PERSON"},
        facts_disclosed=[], response_text="John is an engineer.",
    )
    resolved = graph.resolve_query("What is his experience?")
    assert "John Smith" in resolved
    assert "his" not in resolved.lower() or "John Smith" in resolved


# ------------------------------------------------------------------
# 10. "anything else?" → appends recent entity context
# ------------------------------------------------------------------
def test_resolve_query_anything_else(graph):
    graph.add_turn(
        query="Tell me about Acme Corp",
        entities=["Acme Corp"],
        entity_labels={"Acme Corp": "ORG"},
        facts_disclosed=[], response_text="Acme Corp is ...",
    )
    resolved = graph.resolve_query("anything else?")
    assert "Acme Corp" in resolved


def test_resolve_query_what_else(graph):
    graph.add_turn(
        query="q1", entities=["Widget Inc"],
        entity_labels={"Widget Inc": "ORG"},
        facts_disclosed=[], response_text="...",
    )
    resolved = graph.resolve_query("what else?")
    assert "Widget Inc" in resolved


# ------------------------------------------------------------------
# 11. get_conversation_summary correct counts
# ------------------------------------------------------------------
def test_conversation_summary(graph):
    graph.add_turn(
        query="q1", entities=["A", "B"],
        entity_labels={"A": "PERSON", "B": "ORG"},
        facts_disclosed=["f1", "f2"], response_text="...",
    )
    graph.add_turn(
        query="q2", entities=["B", "C"],
        entity_labels={"B": "ORG", "C": "PERSON"},
        facts_disclosed=["f3"], response_text="...",
    )
    summary = graph.get_conversation_summary()
    assert summary["session_id"] == "test-session-1"
    assert summary["turn_count"] == 2
    assert sorted(summary["entities_discussed"]) == ["A", "B", "C"]
    assert summary["total_facts_disclosed"] == 3
    assert summary["most_recent_entity"] == "C"


# ------------------------------------------------------------------
# 12. reset clears all state
# ------------------------------------------------------------------
def test_reset_clears_state(graph):
    graph.add_turn(
        query="q1", entities=["X"],
        entity_labels={"X": "PERSON"},
        facts_disclosed=["f1"], response_text="...",
    )
    assert graph.get_conversation_summary()["turn_count"] == 1

    graph.reset()

    summary = graph.get_conversation_summary()
    assert summary["turn_count"] == 0
    assert summary["entities_discussed"] == []
    assert summary["total_facts_disclosed"] == 0
    assert summary["most_recent_entity"] is None
    assert graph.resolve_reference("his") is None


# ------------------------------------------------------------------
# 13. Multi-turn: entities from turn 1 accessible in turn 3
# ------------------------------------------------------------------
def test_multi_turn_entity_access(graph):
    graph.add_turn(
        query="q1", entities=["Alice"],
        entity_labels={"Alice": "PERSON"},
        facts_disclosed=[], response_text="Alice info.",
    )
    graph.add_turn(
        query="q2", entities=["Acme Corp"],
        entity_labels={"Acme Corp": "ORG"},
        facts_disclosed=[], response_text="Acme info.",
    )
    graph.add_turn(
        query="q3", entities=["Bob"],
        entity_labels={"Bob": "PERSON"},
        facts_disclosed=[], response_text="Bob info.",
    )

    # Alice should still be in the context entities
    ctx = graph.get_context_entities(max_count=10)
    entity_names = [e["entity"] for e in ctx]
    assert "Alice" in entity_names
    assert "Acme Corp" in entity_names
    assert "Bob" in entity_names

    # "it" should resolve to Acme Corp (most recent non-PERSON)
    assert graph.resolve_reference("it") == "Acme Corp"
    # "his" should resolve to Bob (most recent PERSON)
    assert graph.resolve_reference("his") == "Bob"


# ------------------------------------------------------------------
# 14. resolve_reference "the same" / "same one"
# ------------------------------------------------------------------
def test_resolve_reference_same(graph):
    graph.add_turn(
        query="q1", entities=["Doc-123"],
        entity_labels={"Doc-123": "DOCUMENT"},
        facts_disclosed=[], response_text="...",
    )
    assert graph.resolve_reference("the same") == "Doc-123"
    assert graph.resolve_reference("same one") == "Doc-123"


# ------------------------------------------------------------------
# 15. resolve_reference "they" → most recent entity of any type
# ------------------------------------------------------------------
def test_resolve_reference_they(graph):
    graph.add_turn(
        query="q1", entities=["Team Alpha"],
        entity_labels={"Team Alpha": "ORG"},
        facts_disclosed=[], response_text="...",
    )
    assert graph.resolve_reference("they") == "Team Alpha"
    assert graph.resolve_reference("them") == "Team Alpha"
    assert graph.resolve_reference("their") == "Team Alpha"


# ------------------------------------------------------------------
# 16. ConversationTurn is a proper Pydantic model
# ------------------------------------------------------------------
def test_conversation_turn_model():
    turn = ConversationTurn(
        turn_id=1,
        query="test",
        entities_mentioned=["A"],
        entity_labels={"A": "PERSON"},
        facts_disclosed=["fact1"],
        response_text="resp",
        timestamp=1000.0,
    )
    assert turn.turn_id == 1
    d = turn.model_dump()
    assert d["query"] == "test"
    assert d["entities_mentioned"] == ["A"]
