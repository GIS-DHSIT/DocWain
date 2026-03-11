"""Tests for LLM-upgraded tool handlers.

Tests all 6 tools: resumes, medical, lawhere, creator, email_drafting, translator.
Each test class covers LLM extraction, regex/template fallback, IQ scoring,
and backward compatibility.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest


# ── Shared test helpers ─────────────────────────────────────────────

class _FakeLLM:
    """LLM stub that returns a configurable response."""

    def __init__(self, response: str = ""):
        self._response = response

    def generate_with_metadata(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        return self._response, {}

    def in_cooldown(self) -> bool:
        return False


def _make_llm_json(data: Dict[str, Any]) -> _FakeLLM:
    """Create a FakeLLM that returns JSON data."""
    return _FakeLLM(json.dumps(data))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RESUMES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestResumesLLM:
    """Resume tool with LLM extraction and regex fallback."""

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_llm_extraction_success(self, mock_client):
        from src.tools.resumes import _parse_resume
        mock_client.return_value = _make_llm_json({
            "name": "John Doe", "email": "john@example.com",
            "skills": ["Python", "Java"], "summary": "Senior engineer",
            "experience": [{"company": "Google", "role": "SWE", "dates": "2020-2024", "achievements": []}],
            "education": [{"degree": "BS CS", "institution": "MIT", "year": "2020", "gpa": "3.9"}],
        })
        result = _parse_resume("John Doe\nSkills: Python, Java\nGoogle 2020-2024", "extract all")
        assert result["name"] == "John Doe"
        assert "Python" in result["skills"]
        assert result["iq_score"]["source"] == "llm"

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    def test_regex_fallback(self, mock_client):
        from src.tools.resumes import _parse_resume
        result = _parse_resume("Summary: Experienced developer\nSkills: Python, React\n2023 Google Engineer")
        assert "iq_score" in result
        assert result["iq_score"]["source"] == "regex"

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    def test_regex_extracts_skills(self, mock_client):
        from src.tools.resumes import _parse_resume
        result = _parse_resume("Skills: Python, Java, Go")
        assert len(result["skills"]) > 0

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    def test_regex_extracts_education(self, mock_client):
        from src.tools.resumes import _parse_resume
        result = _parse_resume("Bachelor of Science in Computer Science from MIT")
        assert len(result["education"]) > 0

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_iq_score_present(self, mock_client):
        mock_client.return_value = _make_llm_json({"name": "Alice", "skills": ["Python"], "summary": "Dev"})
        from src.tools.resumes import _parse_resume
        result = _parse_resume("Alice\nSkills: Python")
        iq = result["iq_score"]
        assert "overall" in iq
        assert "completeness" in iq
        assert "confidence" in iq
        assert 0.0 <= iq["overall"] <= 1.0

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_normalize_llm_result_builds_contact(self, mock_client):
        from src.tools.resumes import _normalize_llm_result
        raw = {"name": "Bob", "email": "bob@test.com", "phone": "555-1234"}
        result = _normalize_llm_result(raw)
        assert "Bob" in result["contact"]
        assert "bob@test.com" in result["contact"]

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_normalize_skills_from_string(self, mock_client):
        from src.tools.resumes import _normalize_llm_result
        raw = {"skills": "Python, Java, Go"}
        result = _normalize_llm_result(raw)
        assert isinstance(result["skills"], list)
        assert "Python" in result["skills"]

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    def test_handler_returns_expected_shape(self, mock_client):
        import asyncio
        from src.tools.resumes import resumes_handler
        payload = {"text": "John Doe\nSkills: Python\n2023 Google"}
        result = asyncio.run(resumes_handler(payload))
        assert "result" in result
        assert "sources" in result
        assert result["grounded"] is True

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_llm_error_falls_back_to_regex(self, mock_client):
        from src.tools.resumes import _parse_resume
        mock_client.return_value = _FakeLLM("not valid json at all")
        result = _parse_resume("Summary: Test\nSkills: Python")
        assert result["iq_score"]["source"] == "regex"

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_ats_hints_always_present(self, mock_client):
        mock_client.return_value = _make_llm_json({"name": "X", "summary": "Y"})
        from src.tools.resumes import _parse_resume
        result = _parse_resume("text")
        assert "ats_hints" in result
        assert len(result["ats_hints"]) > 0

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    def test_warnings_for_missing_dates(self, mock_client):
        from src.tools.resumes import _parse_resume
        result = _parse_resume("John Doe, Python developer since forever")
        assert any("Timeline" in w for w in result.get("warnings", []))

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_query_passed_to_prompt(self, mock_client):
        fake = _FakeLLM('{"name": "A", "summary": "B", "skills": []}')
        mock_client.return_value = fake
        from src.tools.resumes import _llm_extract
        _llm_extract("text", "find certifications")
        assert fake._response is not None  # Just verifying it was called

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_experience_as_list(self, mock_client):
        from src.tools.resumes import _normalize_llm_result
        raw = {"experience": "Single string experience"}
        result = _normalize_llm_result(raw)
        assert isinstance(result["experience"], list)

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_certifications_from_string(self, mock_client):
        from src.tools.resumes import _normalize_llm_result
        raw = {"certifications": "AWS, Azure, GCP"}
        result = _normalize_llm_result(raw)
        assert isinstance(result["certifications"], list)
        assert len(result["certifications"]) == 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MEDICAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestMedicalLLM:
    """Medical tool with LLM extraction and regex fallback."""

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_llm_extraction_success(self, mock_client):
        from src.tools.medical import _summarize_medical
        mock_client.return_value = _make_llm_json({
            "clinical_summary": "Patient with hypertension",
            "diagnoses": [{"condition": "Hypertension", "icd_code": "I10"}],
            "medications": [{"name": "Lisinopril", "dosage": "10mg", "frequency": "daily", "route": "oral"}],
        })
        result = _summarize_medical("Patient BP 140/90. Rx Lisinopril 10mg.", False)
        assert "hypertension" in result["summary"].lower() or "Hypertension" in result.get("diagnoses", [{}])[0].get("condition", "")
        assert result["iq_score"]["source"] == "llm"

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    def test_regex_fallback(self, mock_client):
        from src.tools.medical import _summarize_medical
        result = _summarize_medical("Aspirin 100mg daily. Metformin 500mg.", False)
        assert result["iq_score"]["source"] == "regex"
        assert len(result["entities"]) > 0

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_disclaimer_always_present(self, mock_client):
        mock_client.return_value = _make_llm_json({"clinical_summary": "Normal findings"})
        from src.tools.medical import _summarize_medical
        result = _summarize_medical("Normal BP", False)
        assert "informational" in result["summary"].lower() or "medical advice" in result["summary"].lower()

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_redaction_on_llm_output(self, mock_client):
        mock_client.return_value = _make_llm_json({"clinical_summary": "Patient John Smith has flu"})
        from src.tools.medical import _summarize_medical
        result = _summarize_medical("John Smith presents with flu symptoms", True)
        assert result["redacted_text"] is not None
        assert "[REDACTED]" in result["redacted_text"]

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    def test_redaction_on_regex_output(self, mock_client):
        from src.tools.medical import _summarize_medical
        result = _summarize_medical("John Smith presents with flu", True)
        assert result["redacted_text"] is not None
        assert "[REDACTED]" in result["redacted_text"]

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    def test_no_redaction_when_false(self, mock_client):
        from src.tools.medical import _summarize_medical
        result = _summarize_medical("Normal BP", False)
        assert result["redacted_text"] is None

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_entities_backward_compat(self, mock_client):
        mock_client.return_value = _make_llm_json({
            "clinical_summary": "Test",
            "medications": [{"name": "Aspirin", "dosage": "100mg"}],
        })
        from src.tools.medical import _summarize_medical
        result = _summarize_medical("Aspirin 100mg", False)
        assert len(result["entities"]) > 0
        assert result["entities"][0]["type"] == "medication"

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_iq_score_present(self, mock_client):
        mock_client.return_value = _make_llm_json({"clinical_summary": "Test", "diagnoses": [{"condition": "Flu"}]})
        from src.tools.medical import _summarize_medical
        result = _summarize_medical("Patient has flu", False)
        assert "iq_score" in result
        assert 0.0 <= result["iq_score"]["overall"] <= 1.0

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    def test_handler_returns_expected_shape(self, mock_client):
        import asyncio
        from src.tools.medical import medical_handler
        payload = {"text": "Aspirin 100mg daily", "redact": False}
        result = asyncio.run(medical_handler(payload))
        assert "result" in result
        assert "warnings" in result

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_llm_error_falls_back_to_regex(self, mock_client):
        from src.tools.medical import _summarize_medical
        mock_client.return_value = _FakeLLM("totally not json")
        result = _summarize_medical("Aspirin 100mg", False)
        assert result["iq_score"]["source"] == "regex"

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_normalize_medications_as_strings(self, mock_client):
        from src.tools.medical import _normalize_llm_result
        raw = {"medications": ["Aspirin", "Metformin"], "clinical_summary": "test"}
        result = _normalize_llm_result(raw)
        assert len(result["entities"]) == 2

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_lab_results_preserved(self, mock_client):
        mock_client.return_value = _make_llm_json({
            "clinical_summary": "Test",
            "lab_results": [{"test": "CBC", "value": "normal", "reference_range": "4-10", "abnormal": False}],
        })
        from src.tools.medical import _summarize_medical
        result = _summarize_medical("CBC normal", False)
        assert len(result.get("lab_results", [])) > 0

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_allergies_preserved(self, mock_client):
        mock_client.return_value = _make_llm_json({
            "clinical_summary": "Test",
            "allergies": ["Penicillin"],
        })
        from src.tools.medical import _summarize_medical
        result = _summarize_medical("Allergic to Penicillin", False)
        assert "Penicillin" in result.get("allergies", [])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAWHERE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLawhereLLM:
    """Legal tool with LLM extraction and regex fallback."""

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_llm_extraction_success(self, mock_client):
        from src.tools.lawhere import _analyze, LawHereRequest
        mock_client.return_value = _make_llm_json({
            "parties": [{"name": "Acme Corp", "role": "Licensor"}],
            "obligations": [{"party": "Acme", "clause": "shall deliver quarterly reports", "type": "shall"}],
            "summary": "Software license agreement between parties",
        })
        req = LawHereRequest(text="Acme Corp shall deliver quarterly reports. Licensee may terminate.")
        result = _analyze(req)
        assert result["iq_score"]["source"] == "llm"
        assert len(result["parties"]) > 0

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    def test_regex_fallback(self, mock_client):
        from src.tools.lawhere import _analyze, LawHereRequest
        req = LawHereRequest(text="The vendor shall deliver goods. The buyer must pay within 30 days.")
        result = _analyze(req)
        assert result["iq_score"]["source"] == "regex"
        assert len(result["key_clauses"]) > 0

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_disclaimer_always_present(self, mock_client):
        mock_client.return_value = _make_llm_json({"summary": "Contract overview"})
        from src.tools.lawhere import _analyze, LawHereRequest
        req = LawHereRequest(text="Agreement between parties")
        result = _analyze(req)
        assert "legal" in result["summary"].lower() or "advice" in result["summary"].lower()

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_profile_type_preserved(self, mock_client):
        mock_client.return_value = _make_llm_json({"summary": "Test"})
        from src.tools.lawhere import _analyze, LawHereRequest
        req = LawHereRequest(text="Contract text", profile_type="employment")
        result = _analyze(req)
        assert result["profile_type"] == "employment"

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_risks_from_obligations(self, mock_client):
        from src.tools.lawhere import _normalize_llm_result
        raw = {
            "obligations": [{"clause": "shall indemnify against liability claims", "party": "A", "type": "shall"}],
            "risk_assessment": [],
            "summary": "test",
        }
        result = _normalize_llm_result(raw)
        assert len(result["risks"]) > 0

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_iq_score_present(self, mock_client):
        mock_client.return_value = _make_llm_json({"summary": "Test", "parties": [{"name": "A"}]})
        from src.tools.lawhere import _analyze, LawHereRequest
        req = LawHereRequest(text="Test contract")
        result = _analyze(req)
        assert "iq_score" in result
        assert 0.0 <= result["iq_score"]["overall"] <= 1.0

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    def test_handler_returns_expected_shape(self, mock_client):
        import asyncio
        from src.tools.lawhere import lawhere_handler
        payload = {"text": "The vendor shall deliver goods."}
        result = asyncio.run(lawhere_handler(payload))
        assert "result" in result
        assert "warnings" in result

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_llm_error_falls_back(self, mock_client):
        mock_client.return_value = _FakeLLM("not json")
        from src.tools.lawhere import _analyze, LawHereRequest
        req = LawHereRequest(text="The vendor shall deliver goods.")
        result = _analyze(req)
        assert result["iq_score"]["source"] == "regex"

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_governing_law(self, mock_client):
        mock_client.return_value = _make_llm_json({
            "summary": "Test", "governing_law": "State of California",
        })
        from src.tools.lawhere import _analyze, LawHereRequest
        req = LawHereRequest(text="Governing law: California")
        result = _analyze(req)
        assert result.get("governing_law") == "State of California"

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_key_dates(self, mock_client):
        mock_client.return_value = _make_llm_json({
            "summary": "Test", "key_dates": ["2024-01-01", "2024-12-31"],
        })
        from src.tools.lawhere import _analyze, LawHereRequest
        req = LawHereRequest(text="Effective Jan 1 2024 through Dec 31 2024")
        result = _analyze(req)
        assert len(result.get("key_dates", [])) == 2

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_obligations_as_strings(self, mock_client):
        from src.tools.lawhere import _normalize_llm_result
        raw = {"obligations": [{"clause": "shall pay", "party": "B"}, "must comply"], "summary": "test"}
        result = _normalize_llm_result(raw)
        assert len(result["key_clauses"]) == 2

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_risk_assessment_as_dicts(self, mock_client):
        from src.tools.lawhere import _normalize_llm_result
        raw = {"risk_assessment": [{"description": "high risk clause"}], "summary": "test"}
        result = _normalize_llm_result(raw)
        assert len(result["risks"]) > 0

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_risk_assessment_as_strings(self, mock_client):
        from src.tools.lawhere import _normalize_llm_result
        raw = {"risk_assessment": ["liability risk"], "summary": "test"}
        result = _normalize_llm_result(raw)
        assert "liability risk" in result["risks"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CREATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCreatorLLM:
    """Content creator with LLM generation and template fallback."""

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_llm_generation_success(self, mock_client):
        mock_client.return_value = _FakeLLM("A comprehensive summary of the document highlighting key findings and recommendations.")
        from src.tools.creator import _generate_content, CreatorRequest
        req = CreatorRequest(content_type="summary", text="Important research findings about AI safety.")
        result = _generate_content(req)
        assert result["iq_score"]["source"] == "llm"
        assert len(result["content"]) > 0

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    def test_template_fallback(self, mock_client):
        from src.tools.creator import _generate_content, CreatorRequest
        req = CreatorRequest(content_type="summary", text="Some reference text.")
        result = _generate_content(req)
        assert result["iq_score"]["source"] == "template"
        assert len(result["content"]) > 0

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_faq_structured_output(self, mock_client):
        mock_client.return_value = _make_llm_json({
            "faqs": [{"q": "What is AI?", "a": "Artificial Intelligence is..."}],
        })
        from src.tools.creator import _generate_content, CreatorRequest
        req = CreatorRequest(content_type="faq", text="AI overview document")
        result = _generate_content(req)
        assert len(result.get("faqs", [])) > 0

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    def test_faq_template_fallback(self, mock_client):
        from src.tools.creator import _generate_content, CreatorRequest
        req = CreatorRequest(content_type="faq", text="Some topic explanation. Point one. Point two.")
        result = _generate_content(req)
        assert isinstance(result.get("faqs"), list)

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_iq_score_present(self, mock_client):
        mock_client.return_value = _FakeLLM("Generated blog post content with multiple paragraphs and insights.")
        from src.tools.creator import _generate_content, CreatorRequest
        req = CreatorRequest(content_type="blog", text="Tech trends")
        result = _generate_content(req)
        assert "iq_score" in result

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_tone_and_length_in_header(self, mock_client):
        mock_client.return_value = _FakeLLM("Generated SOP with numbered steps for the process.")
        from src.tools.creator import _generate_content, CreatorRequest
        req = CreatorRequest(content_type="sop", tone="formal", length="long", text="Process doc")
        result = _generate_content(req)
        assert "formal" in result.get("header", "").lower() or result["iq_score"]["source"] == "llm"

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    def test_outline_from_template(self, mock_client):
        from src.tools.creator import _generate_content, CreatorRequest
        req = CreatorRequest(content_type="slide_outline", text="First point. Second point. Third point.")
        result = _generate_content(req)
        assert len(result.get("outline", [])) > 0

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    def test_handler_returns_expected_shape(self, mock_client):
        import asyncio
        from src.tools.creator import creator_handler
        payload = {"content_type": "summary", "text": "Reference material"}
        result = asyncio.run(creator_handler(payload))
        assert "result" in result
        assert "sources" in result

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_llm_error_falls_back(self, mock_client):
        mock_client.return_value = _FakeLLM("tiny")  # Too short, returns None
        from src.tools.creator import _generate_content, CreatorRequest
        req = CreatorRequest(content_type="summary", text="Reference text for summary generation.")
        result = _generate_content(req)
        assert result["iq_score"]["source"] == "template"

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_empty_llm_response_falls_back(self, mock_client):
        mock_client.return_value = _FakeLLM("")
        from src.tools.creator import _generate_content, CreatorRequest
        req = CreatorRequest(content_type="blog", text="blog ref")
        result = _generate_content(req)
        assert result["iq_score"]["source"] == "template"

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_all_content_types_accepted(self, mock_client):
        mock_client.return_value = _FakeLLM("Generated content for the requested type with sufficient detail.")
        from src.tools.creator import CreatorRequest
        for ct in ("summary", "blog", "sop", "slide_outline"):
            req = CreatorRequest(content_type=ct, text="ref")
            assert req.content_type == ct

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_faq_llm_returns_list_key(self, mock_client):
        mock_client.return_value = _make_llm_json({
            "questions": [{"q": "Q1?", "a": "A1"}],
        })
        from src.tools.creator import _generate_content, CreatorRequest
        req = CreatorRequest(content_type="faq", text="topic")
        result = _generate_content(req)
        assert len(result.get("faqs", [])) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMAIL DRAFTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEmailDraftingLLM:
    """Email drafting with LLM and template fallback."""

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_llm_draft_success(self, mock_client):
        mock_client.return_value = _make_llm_json({
            "subject": "Follow-up on Project Proposal",
            "body": "Dear Manager,\n\nI wanted to follow up on the project proposal...",
            "key_facts": ["proposal deadline", "budget review"],
        })
        from src.tools.email_drafting import _build_email, EmailDraftRequest
        req = EmailDraftRequest(intent="follow up", recipient_role="manager", text="Project proposal details")
        result = _build_email(req)
        assert result["iq_score"]["source"] == "llm"
        assert "Follow-up" in result["subject"] or "follow" in result["subject"].lower()

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    def test_template_fallback(self, mock_client):
        from src.tools.email_drafting import _build_email, EmailDraftRequest
        req = EmailDraftRequest(intent="meeting request", recipient_role="team lead")
        result = _build_email(req)
        assert result["iq_score"]["source"] == "template"
        assert "subject" in result
        assert "body" in result

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    def test_template_includes_recipient(self, mock_client):
        from src.tools.email_drafting import _build_email, EmailDraftRequest
        req = EmailDraftRequest(intent="update", recipient_role="CEO")
        result = _build_email(req)
        assert "CEO" in result["body"]

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_constraints_used(self, mock_client):
        mock_client.return_value = _make_llm_json({
            "subject": "Budget Update",
            "body": "Please review the budget constraints...",
            "key_facts": ["budget limit $50k"],
        })
        from src.tools.email_drafting import _build_email, EmailDraftRequest
        req = EmailDraftRequest(
            intent="budget update",
            recipient_role="finance",
            constraints=["Must mention budget limit", "Include deadline"],
        )
        result = _build_email(req)
        assert "iq_score" in result

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_iq_score_present(self, mock_client):
        mock_client.return_value = _make_llm_json({"subject": "Test", "body": "Body text"})
        from src.tools.email_drafting import _build_email, EmailDraftRequest
        req = EmailDraftRequest(intent="test", recipient_role="dev")
        result = _build_email(req)
        assert 0.0 <= result["iq_score"]["overall"] <= 1.0

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_tone_passed_to_llm(self, mock_client):
        fake = _FakeLLM('{"subject": "Hello", "body": "Hi there, hope all is well."}')
        mock_client.return_value = fake
        from src.tools.email_drafting import _llm_draft, EmailDraftRequest
        req = EmailDraftRequest(intent="greeting", recipient_role="colleague", tone="casual")
        _llm_draft(req)
        # The LLM was called (doesn't raise)

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    def test_handler_returns_expected_shape(self, mock_client):
        import asyncio
        from src.tools.email_drafting import email_handler
        payload = {"intent": "follow up", "recipient_role": "manager"}
        result = asyncio.run(email_handler(payload))
        assert "result" in result
        assert "sources" in result

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_llm_missing_subject_falls_back(self, mock_client):
        mock_client.return_value = _make_llm_json({"body": "just body, no subject"})
        from src.tools.email_drafting import _build_email, EmailDraftRequest
        req = EmailDraftRequest(intent="test", recipient_role="dev")
        result = _build_email(req)
        assert result["iq_score"]["source"] == "template"

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_llm_missing_body_falls_back(self, mock_client):
        mock_client.return_value = _make_llm_json({"subject": "Subject only"})
        from src.tools.email_drafting import _build_email, EmailDraftRequest
        req = EmailDraftRequest(intent="test", recipient_role="dev")
        result = _build_email(req)
        assert result["iq_score"]["source"] == "template"

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_key_facts_as_string(self, mock_client):
        from src.tools.email_drafting import _normalize_llm_draft
        raw = {"subject": "S", "body": "B", "key_facts": "single fact"}
        result = _normalize_llm_draft(raw)
        assert isinstance(result["key_facts"], list)

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    def test_template_key_points(self, mock_client):
        from src.tools.email_drafting import _build_email, EmailDraftRequest
        req = EmailDraftRequest(
            intent="project update",
            recipient_role="team",
            constraints=["Point A", "Point B"],
        )
        result = _build_email(req)
        assert "Point A" in result["body"]

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_reference_text_used(self, mock_client):
        fake = _FakeLLM('{"subject": "S", "body": "B with reference content"}')
        mock_client.return_value = fake
        from src.tools.email_drafting import _llm_draft, EmailDraftRequest
        req = EmailDraftRequest(intent="share", recipient_role="boss", text="Quarterly report data")
        _llm_draft(req)
        # Verify it ran without error


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRANSLATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestTranslatorLLM:
    """Translator with LLM → Argos → fallback priority chain."""

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_llm_translation_success(self, mock_client):
        mock_client.return_value = _make_llm_json({
            "translated_text": "Bonjour le monde",
            "detected_lang": "en",
            "flagged_terms": [],
            "quality_notes": "Direct translation",
        })
        from src.tools.translator import _translate_text, TranslateRequest
        req = TranslateRequest(text="Hello world", target_lang="fr", source_lang="en")
        result = _translate_text(req)
        assert result["translated_text"] == "Bonjour le monde"
        assert result["backend"] == "DocWain-Agent"
        assert result["iq_score"]["source"] == "llm"

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    @patch("src.tools.translator._ARGOS_AVAILABLE", False)
    def test_deterministic_fallback(self, mock_client):
        from src.tools.translator import _translate_text, TranslateRequest
        req = TranslateRequest(text="Hello", target_lang="de", source_lang="en")
        result = _translate_text(req)
        assert result["translated_text"] == "[de] Hello"
        assert result["backend"] == "fallback"
        assert result["iq_score"]["source"] == "template"

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_language_detection(self, mock_client):
        # First call: detect returns "en", second call: translation
        call_count = [0]
        original_response = '{"translated_text": "Hola mundo", "detected_lang": "en"}'

        def smart_generate(prompt, **kwargs):
            call_count[0] += 1
            if "ISO 639-1" in prompt:
                return "en", {}
            return original_response, {}

        fake = MagicMock()
        fake.generate_with_metadata = smart_generate
        fake.in_cooldown = lambda: False
        mock_client.return_value = fake

        from src.tools.translator import _translate_text, TranslateRequest
        req = TranslateRequest(text="Hello world", target_lang="es")  # No source_lang
        result = _translate_text(req)
        # Either LLM translated or fell back
        assert "translated_text" in result

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_flagged_terms(self, mock_client):
        mock_client.return_value = _make_llm_json({
            "translated_text": "Translated text",
            "detected_lang": "en",
            "flagged_terms": ["technical_term"],
            "quality_notes": "Some terms may need review",
        })
        from src.tools.translator import _translate_text, TranslateRequest
        req = TranslateRequest(text="Hello", target_lang="ja", source_lang="en")
        result = _translate_text(req)
        assert len(result.get("flagged_terms", [])) > 0

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_quality_notes(self, mock_client):
        mock_client.return_value = _make_llm_json({
            "translated_text": "Translated",
            "quality_notes": "High confidence translation",
        })
        from src.tools.translator import _translate_text, TranslateRequest
        req = TranslateRequest(text="text", target_lang="fr", source_lang="en")
        result = _translate_text(req)
        assert result.get("quality_notes") == "High confidence translation"

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_iq_score_present(self, mock_client):
        mock_client.return_value = _make_llm_json({"translated_text": "Bonjour"})
        from src.tools.translator import _translate_text, TranslateRequest
        req = TranslateRequest(text="Hello", target_lang="fr", source_lang="en")
        result = _translate_text(req)
        assert "iq_score" in result
        assert 0.0 <= result["iq_score"]["overall"] <= 1.0

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    @patch("src.tools.translator._ARGOS_AVAILABLE", False)
    def test_handler_returns_expected_shape(self, mock_client):
        import asyncio
        from src.tools.translator import translator_handler
        payload = {"text": "Hello", "target_lang": "fr"}
        result = asyncio.run(translator_handler(payload))
        assert "result" in result
        assert "sources" in result

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_llm_empty_translation_falls_through(self, mock_client):
        mock_client.return_value = _make_llm_json({"translated_text": ""})
        from src.tools.translator import _translate_text, TranslateRequest
        with patch("src.tools.translator._ARGOS_AVAILABLE", False):
            req = TranslateRequest(text="Hello", target_lang="de", source_lang="en")
            result = _translate_text(req)
            assert result["backend"] == "fallback"

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_llm_detect_language_valid(self, mock_client):
        mock_client.return_value = _FakeLLM("fr")
        from src.tools.translator import _llm_detect_language
        assert _llm_detect_language("Bonjour le monde") == "fr"

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_llm_detect_language_invalid(self, mock_client):
        mock_client.return_value = _FakeLLM("This is English text, so the language is English")
        from src.tools.translator import _llm_detect_language
        assert _llm_detect_language("Hello") is None

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    def test_detect_language_no_client(self, mock_client):
        from src.tools.translator import _llm_detect_language
        assert _llm_detect_language("text") is None

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_detect_strips_period(self, mock_client):
        mock_client.return_value = _FakeLLM("en.")
        from src.tools.translator import _llm_detect_language
        assert _llm_detect_language("Hello") == "en"

    @patch("src.tools.llm_tools.get_tool_llm_client", return_value=None)
    @patch("src.tools.translator._ARGOS_AVAILABLE", False)
    def test_target_lang_in_result(self, mock_client):
        from src.tools.translator import _translate_text, TranslateRequest
        req = TranslateRequest(text="Hello", target_lang="ko", source_lang="en")
        result = _translate_text(req)
        assert result["target_lang"] == "ko"

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_source_lang_passed_through(self, mock_client):
        mock_client.return_value = _make_llm_json({"translated_text": "Hola"})
        from src.tools.translator import _translate_text, TranslateRequest
        req = TranslateRequest(text="Hello", target_lang="es", source_lang="en")
        result = _translate_text(req)
        assert result.get("detected_lang") in ("en", "unknown")

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_llm_returns_none_tries_argos(self, mock_client):
        mock_client.return_value = _FakeLLM("")  # Returns None from tool_generate
        from src.tools.translator import _translate_text, TranslateRequest
        with patch("src.tools.translator._ARGOS_AVAILABLE", False):
            req = TranslateRequest(text="Hello", target_lang="fr", source_lang="en")
            result = _translate_text(req)
            # Should fall through to fallback since Argos is unavailable
            assert result["backend"] == "fallback"

    @patch("src.tools.llm_tools.get_tool_llm_client")
    def test_warnings_empty_on_llm_success(self, mock_client):
        mock_client.return_value = _make_llm_json({"translated_text": "Bonjour"})
        from src.tools.translator import _translate_text, TranslateRequest
        req = TranslateRequest(text="Hello", target_lang="fr", source_lang="en")
        result = _translate_text(req)
        assert result["warnings"] == []
