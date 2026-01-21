import pytest

from src.screening.security_service import SecurityScreeningService


def _secret_categories(response):
    return {
        finding["category"]
        for finding in response.get("security_findings", [])
        if finding.get("type") == "SECRET"
    }


def test_detects_secrets_and_masks_values():
    text = (
        "Azure: DefaultEndpointsProtocol=https;AccountName=demo;"
        "AccountKey=abcd1234abcd1234abcd1234abcd1234abcd1234abcd==;EndpointSuffix=core.windows.net\n"
        "JWT: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c\n"
        "password=SuperSecret123!"
    )
    response = SecurityScreeningService().screen_text(text)
    categories = _secret_categories(response)
    assert "SECRET_CONNECTION_STRING" in categories
    assert "SECRET_JWT" in categories
    assert "SECRET_PASSWORD" in categories

    for finding in response.get("security_findings", []):
        snippet = finding.get("snippet_masked") or ""
        context = finding.get("context_masked") or ""
        assert "SuperSecret123!" not in snippet
        assert "SuperSecret123!" not in context
        assert "abcd1234abcd1234abcd1234" not in snippet
        assert "abcd1234abcd1234abcd1234" not in context
        assert "eyJhbGciOiJIUzI1Ni" not in snippet
        assert "eyJhbGciOiJIUzI1Ni" not in context


def test_detects_private_business_data(monkeypatch):
    monkeypatch.setenv("DOCWAIN_ORG_DOMAINS", "internal.company.com")
    text = (
        "Confidential rate card for internal use only. "
        "See https://internal.company.com/pricing. "
        "Server at 10.10.0.5:443 for staging VPN access."
    )
    response = SecurityScreeningService().screen_text(text)
    private_findings = [
        finding for finding in response.get("security_findings", []) if finding.get("type") == "PRIVATE_DATA"
    ]
    assert private_findings
    assert any(finding.get("severity") in {"MED", "HIGH", "CRITICAL"} for finding in private_findings)


def test_missing_classification_adds_finding():
    response = SecurityScreeningService().screen_text("Hello world", metadata={})
    assert response.get("classification") == "MINIMAL_RISK"
    assert any(
        finding.get("category") == "CLASSIFICATION_MISSING" and finding.get("severity") == "LOW"
        for finding in response.get("security_findings", [])
    )


def test_classification_boosts_overall_risk():
    baseline = SecurityScreeningService().screen_text("Hello world", metadata={})
    response = SecurityScreeningService().screen_text("Hello world", metadata={"classification": "Confidential"})
    assert response.get("classification") == "CONFIDENTIAL"
    assert response.get("overall_risk_score", 0) > baseline.get("overall_risk_score", 0)
