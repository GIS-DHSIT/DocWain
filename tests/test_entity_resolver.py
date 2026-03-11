"""Tests for cross-document entity resolution."""
import pytest
from src.docwain_intel.models import EntitySpan
from src.docwain_intel.entity_resolver import resolve_entities


class TestEntityResolver:
    def _make_entity(self, eid, text, label, **kwargs):
        return EntitySpan(
            entity_id=eid, text=text, normalized=text.strip().lower(),
            label=label, unit_id="u1", confidence=0.9, source="spacy", **kwargs,
        )

    def test_exact_match_merges(self):
        entities = [
            self._make_entity("e1", "John Smith", "PERSON"),
            self._make_entity("e2", "John Smith", "PERSON"),
        ]
        resolved = resolve_entities(entities)
        john_ents = [e for e in resolved if "john smith" in e.normalized]
        assert len(john_ents) == 1

    def test_initial_matching(self):
        """J. Smith should match John Smith."""
        entities = [
            self._make_entity("e1", "John Smith", "PERSON"),
            self._make_entity("e2", "J. Smith", "PERSON"),
        ]
        resolved = resolve_entities(entities)
        john_ents = [e for e in resolved if e.label == "PERSON" and "smith" in e.normalized]
        assert len(john_ents) == 1
        assert "J. Smith" in john_ents[0].aliases or "John Smith" in john_ents[0].aliases

    def test_title_stripping(self):
        """Dr. John Smith should match John Smith."""
        entities = [
            self._make_entity("e1", "John Smith", "PERSON"),
            self._make_entity("e2", "Dr. John Smith", "PERSON"),
        ]
        resolved = resolve_entities(entities)
        john_ents = [e for e in resolved if e.label == "PERSON" and "smith" in e.normalized]
        assert len(john_ents) == 1

    def test_org_suffix_stripping(self):
        """Google Inc. should match Google."""
        entities = [
            self._make_entity("e1", "Google", "ORG"),
            self._make_entity("e2", "Google Inc.", "ORG"),
        ]
        resolved = resolve_entities(entities)
        google_ents = [e for e in resolved if e.label == "ORG" and "google" in e.normalized]
        assert len(google_ents) == 1

    def test_different_types_not_merged(self):
        """John (PERSON) should NOT merge with John (ORG)."""
        entities = [
            self._make_entity("e1", "John", "PERSON"),
            self._make_entity("e2", "John", "ORG"),
        ]
        resolved = resolve_entities(entities)
        assert len(resolved) == 2

    def test_jaro_winkler_threshold(self):
        """Completely different names should not merge."""
        entities = [
            self._make_entity("e1", "John Smith", "PERSON"),
            self._make_entity("e2", "Sarah Chen", "PERSON"),
        ]
        resolved = resolve_entities(entities)
        assert len(resolved) == 2

    def test_suffix_stripping(self):
        """John Smith Jr. should match John Smith."""
        entities = [
            self._make_entity("e1", "John Smith", "PERSON"),
            self._make_entity("e2", "John Smith Jr.", "PERSON"),
        ]
        resolved = resolve_entities(entities)
        john_ents = [e for e in resolved if e.label == "PERSON" and "smith" in e.normalized]
        assert len(john_ents) == 1

    def test_empty_input(self):
        assert resolve_entities([]) == []

    def test_mr_matching(self):
        """Mr. Smith should match John Smith when last names match."""
        entities = [
            self._make_entity("e1", "John Smith", "PERSON"),
            self._make_entity("e2", "Mr. Smith", "PERSON"),
        ]
        resolved = resolve_entities(entities)
        smith_ents = [e for e in resolved if e.label == "PERSON" and "smith" in e.normalized]
        assert len(smith_ents) == 1
