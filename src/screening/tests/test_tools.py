import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.screening.config import ScreeningConfig
from src.screening.models import ScreeningContext
from src.screening.search import NullSearchClient
from src.screening.tools.ai_authorship import AIAuthorshipTool
from src.screening.tools.pii_sensitivity import PIISensitivityTool
from src.screening.tools.resume_screening import ResumeScreeningTool
from src.screening.tools.template_conformance import TemplateConformanceTool


def test_ai_authorship_signals():
    text = "The system outputs consistent results. The system outputs consistent results. The system outputs consistent results."
    ctx = ScreeningContext(doc_id=None, doc_type=None, text=text, metadata={}, config=ScreeningConfig())
    result = AIAuthorshipTool().run(ctx)
    assert 0 <= result.score_0_1 <= 1
    assert "entropy" in result.raw_features
    assert result.tool_name == "ai_authorship"


def test_pii_span_detection():
    text = "Contact alice@example.com or call +1 415-555-1212 for details."
    ctx = ScreeningContext(doc_id="doc1", doc_type=None, text=text, metadata={}, config=ScreeningConfig())
    result = PIISensitivityTool().run(ctx)
    assert result.raw_features["pii_count"] >= 2
    labels = {span["label"] for span in result.evidence_spans}
    assert "email" in labels
    assert "phone" in labels


def test_resume_entity_verification_disabled():
    text = "Resume\nExperience at OpenAI Inc.\nEducation at Example University\nCertifications: AWS Certified\nSkills: Python"
    cfg = ScreeningConfig(internet_enabled=False)
    ctx = ScreeningContext(
        doc_id="doc2",
        doc_type="RESUME",
        text=text,
        metadata={},
        config=cfg,
        search_client=NullSearchClient(),
    )
    result = ResumeScreeningTool().run(ctx)
    analysis = result.raw_features["analysis"]
    assert "Internet validation disabled" in " ".join(analysis.get("warnings", []))
    assert analysis["candidate_profile"]["extracted"]["certifications"]


def test_template_conformance_rules():
    text = "Scope\nThis policy covers access.\nPurpose\nExplain reason.\nPolicy\nRules only."
    ctx = ScreeningContext(doc_id="doc3", doc_type="POLICY", text=text, metadata={}, config=ScreeningConfig())
    result = TemplateConformanceTool().run(ctx)
    assert "missing_sections" in result.raw_features
    assert result.raw_features["missing_sections"]
    assert result.score_0_1 > 0
