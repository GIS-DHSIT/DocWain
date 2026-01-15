import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.vetting.config import VettingConfig
from src.vetting.models import VettingContext
from src.vetting.search import NullSearchClient
from src.vetting.tools.ai_authorship import AIAuthorshipTool
from src.vetting.tools.pii_sensitivity import PIISensitivityTool
from src.vetting.tools.resume_vetting import ResumeVettingTool
from src.vetting.tools.template_conformance import TemplateConformanceTool


def test_ai_authorship_signals():
    text = "The system outputs consistent results. The system outputs consistent results. The system outputs consistent results."
    ctx = VettingContext(doc_id=None, doc_type=None, text=text, metadata={}, config=VettingConfig())
    result = AIAuthorshipTool().run(ctx)
    assert 0 <= result.score_0_1 <= 1
    assert "entropy" in result.raw_features
    assert result.tool_name == "ai_authorship"


def test_pii_span_detection():
    text = "Contact alice@example.com or call +1 415-555-1212 for details."
    ctx = VettingContext(doc_id="doc1", doc_type=None, text=text, metadata={}, config=VettingConfig())
    result = PIISensitivityTool().run(ctx)
    assert result.raw_features["pii_count"] >= 2
    labels = {span["label"] for span in result.evidence_spans}
    assert "email" in labels
    assert "phone" in labels


def test_resume_entity_verification_disabled():
    text = "Resume\nExperience at OpenAI Inc.\nEducation at Example University\nSkills: Python"
    cfg = VettingConfig(internet_enabled=False)
    ctx = VettingContext(
        doc_id="doc2",
        doc_type="RESUME",
        text=text,
        metadata={},
        config=cfg,
        search_client=NullSearchClient(),
    )
    result = ResumeVettingTool().run(ctx)
    statuses = {entry["status"] for entry in result.raw_features["entity_results"]}
    assert statuses == {"UNCERTAIN"}


def test_template_conformance_rules():
    text = "Scope\nThis policy covers access.\nPurpose\nExplain reason.\nPolicy\nRules only."
    ctx = VettingContext(doc_id="doc3", doc_type="POLICY", text=text, metadata={}, config=VettingConfig())
    result = TemplateConformanceTool().run(ctx)
    assert "missing_sections" in result.raw_features
    assert result.raw_features["missing_sections"]
    assert result.score_0_1 > 0
