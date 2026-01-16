import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.screening.api import screening_router
from src.screening.resume import run_resume_analysis
from src.screening.resume.authenticity import AuthenticityAnalyzer
from src.screening.resume.extractor import ResumeExtractor
from src.screening.resume.models import CertificationItem, ExperienceItem, ResumeProfile
from src.screening.search.base import SearchHit
from src.screening.search import NullSearchClient


class FakeSearchClient:
    def __init__(self):
        self.queries = []

    def search(self, query: str, k: int = 5):
        self.queries.append(query)
        return [SearchHit(title="Example Verification", snippet="Valid link", url="https://example.com/verify")]


def test_extractor_parses_certifications():
    text = "CERTIFICATIONS\nAWS Certified Solutions Architect - Amazon Web Services\nCredential ID: ABC-123\n"
    profile = ResumeExtractor().extract(text)
    assert profile.certifications
    cert = profile.certifications[0]
    assert cert.issuer == "Amazon Web Services"
    assert cert.credential_id == "ABC-123"


def test_validations_skip_when_internet_disabled_but_authenticity_runs():
    text = "Engineer at DemoCorp 2020-2021\nEducation Demo University 2015-2019\nCertifications: Demo Cert - DemoOrg"
    analysis = run_resume_analysis(
        text=text,
        doc_id="docA",
        metadata={},
        search_client=NullSearchClient(),
        internet_enabled=False,
    )
    assert any("Internet validation disabled" in warning for warning in analysis.warnings)
    assert analysis.authenticity.signals is not None


def test_certification_verifier_uses_search_results():
    text = "Certifications\nDemo Security Certificate - DemoOrg\nCredential ID: DEMO-1\n"
    search_client = FakeSearchClient()
    analysis = run_resume_analysis(
        text=text,
        doc_id="docB",
        metadata={},
        search_client=search_client,
        internet_enabled=True,
    )
    assert search_client.queries
    assert analysis.validations.certifications
    assert analysis.validations.certifications[0].sources


def test_authenticity_signals_for_timeline_and_cert_issuer():
    profile = ResumeProfile(
        experience=[
            ExperienceItem(title="Role1", company="A", start_date="2020", end_date="2022"),
            ExperienceItem(title="Role2", company="B", start_date="2021", end_date="2023"),
        ],
        certifications=[CertificationItem(name="Mystery Cert", issuer=None)],
    )
    report = AuthenticityAnalyzer().analyze(profile, full_text="Role1 2020-2022 Role2 2021-2023")
    signal_types = {sig.type for sig in report.signals}
    assert "timeline_overlap" in signal_types
    assert "missing_cert_issuer" in signal_types


def test_resume_response_contains_narrative():
    text = "Summary: Seasoned engineer\nExperience\nEngineer at DemoCorp 2018-2020\nEducation Demo University 2012-2016"
    analysis = run_resume_analysis(
        text=text,
        doc_id="docC",
        metadata={},
        search_client=NullSearchClient(),
        internet_enabled=False,
    )
    assert analysis.narrative_report
    assert analysis.candidate_profile.summary


def test_no_duplicate_resume_endpoints():
    paths = [route.path for route in screening_router.routes if "resume" in route.path]
    assert len(paths) == len(set(paths))
