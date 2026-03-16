"""Tests for profile intelligence module."""

import pytest


class TestProfileIntelligenceModule:
    def test_imports(self):
        from src.intelligence.profile_intelligence import generate_profile_intelligence
        from src.intelligence.profile_intelligence import _format_briefs

    def test_format_briefs(self):
        from src.intelligence.profile_intelligence import _format_briefs
        briefs = [
            {"name": "invoice1.pdf", "brief": "An invoice for $500", "key_facts": [{"label": "Total", "value": "$500"}]},
            {"name": "invoice2.pdf", "brief": "An invoice for $700", "key_facts": [{"label": "Total", "value": "$700"}]},
        ]
        result = _format_briefs(briefs)
        assert "invoice1.pdf" in result
        assert "invoice2.pdf" in result
        assert "$500" in result

    def test_extract_text_from_dict(self):
        from src.intelligence.profile_intelligence import _extract_text
        data = {"raw": {"full_text": "Hello world content"}}
        assert _extract_text(data) == "Hello world content"

    def test_extract_text_from_facts(self):
        from src.intelligence.profile_intelligence import _extract_text
        data = {"facts": [{"statement": "Fact one", "evidence": "Evidence one"}]}
        result = _extract_text(data)
        assert "Fact one" in result
        assert "Evidence one" in result

    def test_extract_text_empty(self):
        from src.intelligence.profile_intelligence import _extract_text
        assert _extract_text({}) == ""
        assert _extract_text(None) == ""


class TestProfileIntelligenceAPI:
    def test_router_imports(self):
        from src.api.profile_intelligence_api import profile_intelligence_router
        paths = [r.path for r in profile_intelligence_router.routes]
        assert any("/{profile_id}/intelligence" in p for p in paths)
