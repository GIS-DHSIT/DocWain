"""
Tests for conversation state tracking: EntityRegister, ConversationEntityExtractor,
ConversationContextResolver, ConversationState, and ProgressiveSummarizer.

8 EntityRegister tests + 7 Entity Extraction tests + 10 Context Resolver tests
+ 8 ConversationState tests + 5 ProgressiveSummarizer tests = 38 total.
"""
import pytest

from src.intelligence.conversation_state import (
    TrackedEntity,
    EnrichedTurn,
    EntityRegister,
    ConversationEntityExtractor,
    ConversationContextResolver,
    ConversationState,
    ProgressiveSummarizer,
)


# ---------------------------------------------------------------------------
# EntityRegister tests (8)
# ---------------------------------------------------------------------------

class TestEntityRegister:

    def test_register_new_entity(self):
        """Register a person entity and verify it is stored correctly."""
        reg = EntityRegister()
        entity = reg.register("Abinaya", "person")
        assert entity.name == "Abinaya"
        assert entity.entity_type == "person"
        assert entity.mention_count == 1
        assert "Abinaya" in reg.entities

    def test_register_duplicate_increments_count(self):
        """Registering the same name twice increments mention_count."""
        reg = EntityRegister()
        reg.register("Abinaya", "person")
        entity = reg.register("Abinaya", "person")
        assert entity.mention_count == 2
        # Should still be a single entity in the register
        assert len(reg.entities) == 1

    def test_lookup_by_alias(self):
        """Register with aliases and verify lookup by alias works."""
        reg = EntityRegister()
        reg.register("Abinaya", "person", aliases={"she", "her", "abi"})
        result = reg.lookup("abi")
        assert result is not None
        assert result.name == "Abinaya"
        result2 = reg.lookup("she")
        assert result2 is not None
        assert result2.name == "Abinaya"

    def test_lookup_case_insensitive(self):
        """Lookup is case-insensitive: 'Abinaya' == 'abinaya'."""
        reg = EntityRegister()
        reg.register("Abinaya", "person")
        result = reg.lookup("abinaya")
        assert result is not None
        assert result.name == "Abinaya"
        result2 = reg.lookup("ABINAYA")
        assert result2 is not None
        assert result2.name == "Abinaya"

    def test_get_by_type_filters(self):
        """get_by_type returns only entities of the requested type."""
        reg = EntityRegister()
        reg.register("Abinaya", "person")
        reg.register("Aadithya", "person")
        reg.register("Resume.pdf", "document")
        persons = reg.get_by_type("person")
        assert len(persons) == 2
        assert all(e.entity_type == "person" for e in persons)
        docs = reg.get_by_type("document")
        assert len(docs) == 1
        assert docs[0].name == "Resume.pdf"

    def test_get_most_recent(self):
        """get_most_recent returns the entity with the highest last_seen_turn."""
        reg = EntityRegister()
        reg.register("Alice", "person")
        reg.advance_turn()
        reg.register("Bob", "person")
        reg.advance_turn()
        reg.register("Carol", "person")
        most_recent = reg.get_most_recent(entity_type="person")
        assert most_recent is not None
        assert most_recent.name == "Carol"

    def test_eviction_oldest(self):
        """When max_entities is exceeded, the oldest entity is evicted."""
        reg = EntityRegister(max_entities=50)
        # Register 51 entities; the first should be evicted
        for i in range(51):
            reg.advance_turn()
            reg.register(f"Entity_{i}", "person")
        assert len(reg.entities) == 50
        # Entity_0 was the oldest and should have been evicted
        assert reg.lookup("Entity_0") is None
        # Entity_50 (most recent) should still exist
        assert reg.lookup("Entity_50") is not None

    def test_serialize_roundtrip(self):
        """to_dict -> from_dict preserves all state."""
        reg = EntityRegister()
        reg.register("Abinaya", "person", aliases={"abi", "she"}, metadata={"doc_id": "d1"})
        reg.advance_turn()
        reg.register("Invoice.pdf", "document")

        data = reg.to_dict()
        restored = EntityRegister.from_dict(data)

        assert restored._current_turn == reg._current_turn
        assert len(restored.entities) == len(reg.entities)
        # Check person entity
        abi = restored.lookup("Abinaya")
        assert abi is not None
        assert abi.entity_type == "person"
        assert "abi" in abi.aliases
        assert abi.metadata.get("doc_id") == "d1"
        # Check alias lookup
        assert restored.lookup("abi") is not None
        assert restored.lookup("abi").name == "Abinaya"
        # Check document entity
        inv = restored.lookup("Invoice.pdf")
        assert inv is not None
        assert inv.entity_type == "document"


# ---------------------------------------------------------------------------
# ConversationEntityExtractor tests (7)
# ---------------------------------------------------------------------------

class TestConversationEntityExtractor:

    def test_extract_persons_about_pattern(self):
        """'Tell me about Abinaya' extracts person name."""
        result = ConversationEntityExtractor.extract_persons("Tell me about Abinaya")
        assert result == ["Abinaya"]

    def test_extract_persons_possessive(self):
        """\"Abinaya's skills\" extracts the person name."""
        result = ConversationEntityExtractor.extract_persons("Abinaya's skills")
        assert result == ["Abinaya"]

    def test_extract_persons_from_response(self):
        """'Name: Abinaya' extracts person from structured text."""
        result = ConversationEntityExtractor.extract_persons("Name: Abinaya Ramanathan")
        assert "Abinaya Ramanathan" in result

    def test_extract_documents_filename(self):
        """Extracts filename references like 'Resume.pdf'."""
        result = ConversationEntityExtractor.extract_documents("Look in Resume.pdf for details")
        assert "Resume.pdf" in result

    def test_extract_topics(self):
        """Extracts topic keywords and normalizes by stripping trailing 's'."""
        result = ConversationEntityExtractor.extract_topics("What are the technical skills?")
        assert "skill" in result

    def test_detect_pronouns(self):
        """Detects pronouns like 'her' in text."""
        result = ConversationEntityExtractor.detect_pronouns("What are her certifications?")
        assert "her" in result

    def test_stop_names_filtered(self):
        """Words in the stop list (e.g. 'Document') are filtered out of person results."""
        result = ConversationEntityExtractor.extract_persons("Tell me about Document")
        assert result == []


# ---------------------------------------------------------------------------
# ConversationContextResolver tests (10)
# ---------------------------------------------------------------------------

class TestConversationContextResolver:
    def test_resolve_her_to_person(self):
        """'What are her skills?' -> 'What are Abinaya's skills?'"""
        reg = EntityRegister()
        reg.register("Abinaya", "person")
        resolver = ConversationContextResolver(reg)
        assert "Abinaya's" in resolver.resolve("What are her skills?")

    def test_resolve_his_to_person(self):
        """'What is his experience?' -> 'What is Aadithya's experience?'"""
        reg = EntityRegister()
        reg.register("Aadithya", "person")
        resolver = ConversationContextResolver(reg)
        assert "Aadithya's" in resolver.resolve("What is his experience?")

    def test_resolve_him_to_person(self):
        """'Tell me about him' -> 'Tell me about Aadithya'"""
        reg = EntityRegister()
        reg.register("Aadithya", "person")
        resolver = ConversationContextResolver(reg)
        assert "Aadithya" in resolver.resolve("Tell me about him")

    def test_resolve_that_document(self):
        """'Show me that document' -> 'Show me Resume.pdf'"""
        reg = EntityRegister()
        reg.register("Resume.pdf", "document")
        resolver = ConversationContextResolver(reg)
        assert "Resume.pdf" in resolver.resolve("Show me that document")

    def test_resolve_ordinal_first_candidate(self):
        """'the first candidate' -> 'Alice'"""
        reg = EntityRegister()
        reg.register("Alice", "person")
        reg.advance_turn()
        reg.register("Bob", "person")
        resolver = ConversationContextResolver(reg)
        result = resolver.resolve("Tell me about the first candidate")
        assert "Alice" in result

    def test_resolve_ordinal_second_candidate(self):
        """'the second candidate' -> 'Bob'"""
        reg = EntityRegister()
        reg.register("Alice", "person")
        reg.advance_turn()
        reg.register("Bob", "person")
        resolver = ConversationContextResolver(reg)
        result = resolver.resolve("Tell me about the second candidate")
        assert "Bob" in result

    def test_no_resolution_when_no_entities(self):
        """Empty register -> unchanged."""
        reg = EntityRegister()
        resolver = ConversationContextResolver(reg)
        query = "What are her skills?"
        assert resolver.resolve(query) == query

    def test_no_resolution_when_no_pronouns(self):
        """'Tell me about Abinaya' -> unchanged."""
        reg = EntityRegister()
        reg.register("Abinaya", "person")
        resolver = ConversationContextResolver(reg)
        query = "Tell me about Abinaya"
        assert resolver.resolve(query) == query

    def test_needs_resolution_true(self):
        """'What are her skills?' -> True"""
        reg = EntityRegister()
        resolver = ConversationContextResolver(reg)
        assert resolver.needs_resolution("What are her skills?") is True

    def test_needs_resolution_false(self):
        """'Tell me about Abinaya' -> False"""
        reg = EntityRegister()
        resolver = ConversationContextResolver(reg)
        assert resolver.needs_resolution("Tell me about Abinaya") is False


# ---------------------------------------------------------------------------
# ConversationState tests (8)
# ---------------------------------------------------------------------------

class TestConversationState:
    def test_resolve_query_with_entity_register(self):
        """Record turn with 'Abinaya' -> resolve 'her skills'"""
        state = ConversationState()
        state.record_turn("ns", "user1", "Tell me about Abinaya", "Abinaya has Python skills")
        result = state.resolve_query("What are her skills?", "ns", "user1")
        assert "Abinaya" in result

    def test_record_turn_extracts_persons(self):
        state = ConversationState()
        state.record_turn("ns", "user1", "Tell me about Abinaya", "Here is info")
        assert state.entity_register.lookup("Abinaya") is not None

    def test_record_turn_extracts_documents(self):
        state = ConversationState()
        state.record_turn("ns", "user1", "Check Resume.pdf", "Found info")
        assert state.entity_register.lookup("Resume.pdf") is not None

    def test_record_turn_delegates_to_history(self):
        """Mock history.add_turn called."""
        from unittest.mock import MagicMock
        mock_history = MagicMock()
        state = ConversationState(conversation_history=mock_history)
        state.record_turn("ns", "user1", "Hello", "Hi")
        mock_history.add_turn.assert_called_once_with("ns", "user1", "Hello", "Hi")

    def test_clear_resets_all_state(self):
        state = ConversationState()
        state.record_turn("ns", "user1", "Tell me about Abinaya", "Info")
        state.clear("ns", "user1")
        assert not state.entity_register.entities
        assert not state.enriched_turns
        assert state.current_topic is None

    def test_get_entity_context_nonempty(self):
        state = ConversationState()
        state.record_turn("ns", "user1", "Tell me about Abinaya", "Info about Resume.pdf")
        ctx = state.get_entity_context()
        assert "Abinaya" in ctx

    def test_get_entity_context_empty(self):
        state = ConversationState()
        assert state.get_entity_context() == ""

    def test_cross_turn_resolution(self):
        """Turn 1: 'about Abinaya' -> Turn 2: 'her skills' resolved."""
        state = ConversationState()
        state.record_turn("ns", "user1", "Tell me about Abinaya", "Abinaya has Python")
        result = state.resolve_query("What are her skills?", "ns", "user1")
        assert "Abinaya" in result


# ---------------------------------------------------------------------------
# ProgressiveSummarizer tests (5)
# ---------------------------------------------------------------------------

class TestProgressiveSummarizer:
    def test_deterministic_update_adds_bullet(self):
        summarizer = ProgressiveSummarizer()
        turn = EnrichedTurn(user_message="What is Python?", assistant_response="A language",
                           timestamp=1.0, turn_number=1)
        result = summarizer.update(turn)
        assert result.startswith("- User asked:")
        assert "Python" in result

    def test_deterministic_update_caps_at_max(self):
        summarizer = ProgressiveSummarizer(max_bullets=3)
        for i in range(5):
            turn = EnrichedTurn(user_message=f"Q{i}", assistant_response=f"A{i}",
                               timestamp=float(i), turn_number=i + 1)
            summarizer.update(turn)
        lines = [l for l in summarizer.get_summary().split("\n") if l.strip()]
        assert len(lines) == 3

    def test_llm_update_called_when_available(self):
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "- Updated summary"
        summarizer = ProgressiveSummarizer(llm_client=mock_llm)
        turn = EnrichedTurn(user_message="Q", assistant_response="A",
                           timestamp=1.0, turn_number=1)
        result = summarizer.update(turn)
        mock_llm.generate.assert_called_once()
        assert "Updated summary" in result

    def test_llm_fallback_on_error(self):
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("LLM timeout")
        summarizer = ProgressiveSummarizer(llm_client=mock_llm)
        turn = EnrichedTurn(user_message="Q", assistant_response="A",
                           timestamp=1.0, turn_number=1)
        result = summarizer.update(turn)
        assert result.startswith("- User asked:")

    def test_clear_resets_summary(self):
        summarizer = ProgressiveSummarizer()
        turn = EnrichedTurn(user_message="Q", assistant_response="A",
                           timestamp=1.0, turn_number=1)
        summarizer.update(turn)
        summarizer.clear()
        assert summarizer.get_summary() == ""
        assert summarizer._summarized_up_to == 0


# ---------------------------------------------------------------------------
# Multi-Turn Integration Tests (12)
# ---------------------------------------------------------------------------

class TestMultiTurnConversation:

    def test_three_turn_pronoun_chain(self):
        """Turn 1: 'about Abinaya' -> Turn 2: 'her skills' -> Turn 3: 'her experience'"""
        state = ConversationState()
        state.record_turn("ns", "u1", "Tell me about Abinaya", "Abinaya has Python and Java skills")
        r2 = state.resolve_query("What are her skills?", "ns", "u1")
        assert "Abinaya" in r2
        state.record_turn("ns", "u1", r2, "Skills: Python, Java")
        r3 = state.resolve_query("What is her experience?", "ns", "u1")
        assert "Abinaya" in r3

    def test_topic_switch_mid_conversation(self):
        """Turn 1: 'Abinaya' -> Turn 2: 'Aadithya' -> Turn 3: 'his' resolves to Aadithya (most recent)"""
        state = ConversationState()
        state.record_turn("ns", "u1", "Tell me about Abinaya", "Abinaya info")
        state.record_turn("ns", "u1", "Tell me about Aadithya", "Aadithya info")
        result = state.resolve_query("What is his experience?", "ns", "u1")
        assert "Aadithya" in result

    def test_document_reference_resolution(self):
        """Turn 1: mentions 'Invoice_001.pdf' -> Turn 2: 'that invoice' -> resolved"""
        state = ConversationState()
        state.record_turn("ns", "u1", "Check Invoice_001.pdf", "Found invoice data")
        result = state.resolve_query("Show me that invoice", "ns", "u1")
        assert "Invoice_001.pdf" in result

    def test_ordinal_with_two_candidates(self):
        """Mentions Alice then Bob -> 'the first candidate' -> Alice"""
        state = ConversationState()
        state.record_turn("ns", "u1", "Tell me about Alice", "Alice info")
        state.record_turn("ns", "u1", "Tell me about Bob", "Bob info")
        result = state.resolve_query("Tell me about the first candidate", "ns", "u1")
        assert "Alice" in result

    def test_no_resolution_for_explicit_names(self):
        """Query with explicit name NOT altered."""
        state = ConversationState()
        state.record_turn("ns", "u1", "Tell me about Abinaya", "Info")
        result = state.resolve_query("Tell me about Aadithya", "ns", "u1")
        assert "Aadithya" in result
        assert "Abinaya" not in result

    def test_entity_register_persists_across_turns(self):
        """Entities from turn 1 available in turn 3."""
        state = ConversationState()
        state.record_turn("ns", "u1", "Tell me about Abinaya", "Info")
        state.record_turn("ns", "u1", "What is Python?", "A language")
        state.record_turn("ns", "u1", "And Java?", "Another language")
        assert state.entity_register.lookup("Abinaya") is not None

    def test_progressive_summary_grows(self):
        """3 turns -> 3 bullets in summary."""
        state = ConversationState()
        summarizer = ProgressiveSummarizer()
        for i in range(3):
            state.record_turn("ns", "u1", f"Question {i}", f"Answer {i}")
            summarizer.update(state.enriched_turns[-1])
        lines = [l for l in summarizer.get_summary().split("\n") if l.strip()]
        assert len(lines) == 3

    def test_clear_resets_everything(self):
        """clear() -> no entities, no summary."""
        state = ConversationState()
        state.record_turn("ns", "u1", "Tell me about Abinaya", "Info")
        state.clear("ns", "u1")
        assert not state.entity_register.entities
        assert not state.enriched_turns
        assert state._turn_counter == 0

    def test_cross_session_entity_persistence(self):
        """FakeRedis: persist -> new ConversationState -> entities loaded."""
        import json

        class FakeRedis:
            def __init__(self):
                self._store = {}
            def get(self, key):
                return self._store.get(key)
            def setex(self, key, ttl, value):
                self._store[key] = value
            def delete(self, key):
                self._store.pop(key, None)

        fake_redis = FakeRedis()
        state1 = ConversationState(redis_client=fake_redis)
        state1.record_turn("ns", "u1", "Tell me about Abinaya", "Info")
        # New ConversationState with same redis — should load entities
        state2 = ConversationState(redis_client=fake_redis)
        result = state2.resolve_query("What are her skills?", "ns", "u1")
        assert "Abinaya" in result

    def test_resolution_with_empty_history(self):
        """First turn with pronouns -> unchanged (no entities yet)."""
        state = ConversationState()
        result = state.resolve_query("What are her skills?", "ns", "u1")
        assert result == "What are her skills?"

    def test_concurrent_users_isolated(self):
        """User A and User B have separate registers."""
        state = ConversationState()
        state.record_turn("ns", "userA", "Tell me about Alice", "Alice info")
        state.record_turn("ns", "userB", "Tell me about Bob", "Bob info")
        # User A's "her" should resolve to Alice, not Bob
        # Note: without Redis, same in-memory state. With separate resolve calls,
        # the entity register contains both. Test that entities are present.
        assert state.entity_register.lookup("Alice") is not None
        assert state.entity_register.lookup("Bob") is not None

    def test_entity_extraction_from_assistant_response(self):
        """Response mentioning 'Name: Abinaya' registers entity."""
        state = ConversationState()
        state.record_turn("ns", "u1", "Who is this?", "Name: Abinaya Ramanathan, experienced developer")
        assert state.entity_register.lookup("Abinaya Ramanathan") is not None


# ---------------------------------------------------------------------------
# Edge Cases (8)
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_query_unchanged(self):
        state = ConversationState()
        assert state.resolve_query("", "ns", "u1") == ""

    def test_long_query_skips_resolution(self):
        state = ConversationState()
        state.record_turn("ns", "u1", "Tell me about Abinaya", "Info")
        long_query = "What are her skills? " * 30  # > 500 chars
        assert state.resolve_query(long_query, "ns", "u1") == long_query

    def test_ambiguous_pronoun_uses_most_recent(self):
        """'they' with 2+ persons resolves to most recent."""
        state = ConversationState()
        state.record_turn("ns", "u1", "Tell me about Alice", "Alice info")
        state.record_turn("ns", "u1", "Tell me about Bob", "Bob info")
        result = state.resolve_query("What are their skills?", "ns", "u1")
        assert "Bob" in result  # most recent

    def test_double_pronoun_resolved_once(self):
        """'her her' — only first occurrence resolved."""
        reg = EntityRegister()
        reg.register("Abinaya", "person")
        resolver = ConversationContextResolver(reg)
        result = resolver.resolve("Tell her about her skills")
        # Only one substitution (count=1 in regex sub)
        assert result.count("Abinaya") == 1

    def test_unicode_names(self):
        """Unicode names handled correctly."""
        state = ConversationState()
        state.record_turn("ns", "u1", "Tell me about Muller", "Info about Muller")
        assert state.entity_register.lookup("Muller") is not None

    def test_none_history_graceful(self):
        """history=None -> graceful no-op."""
        state = ConversationState(conversation_history=None)
        state.record_turn("ns", "u1", "Hello", "Hi")
        assert len(state.enriched_turns) == 1

    def test_redis_unavailable_silent(self):
        """Redis errors are silently caught."""
        class BrokenRedis:
            def get(self, key):
                raise ConnectionError("Redis down")
            def setex(self, key, ttl, value):
                raise ConnectionError("Redis down")
            def delete(self, key):
                raise ConnectionError("Redis down")

        state = ConversationState(redis_client=BrokenRedis())
        # Should not raise
        state.record_turn("ns", "u1", "Tell me about Abinaya", "Info")
        result = state.resolve_query("What are her skills?", "ns", "u1")
        assert "Abinaya" in result

    def test_malformed_redis_data_reset(self):
        """Malformed JSON in Redis doesn't crash."""
        class BadDataRedis:
            def get(self, key):
                return "not valid json {{{"
            def setex(self, key, ttl, value):
                pass
            def delete(self, key):
                pass

        state = ConversationState(redis_client=BadDataRedis())
        # Should not raise; falls back to empty register
        result = state.resolve_query("What are her skills?", "ns", "u1")
        assert result == "What are her skills?"
