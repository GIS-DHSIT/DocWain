import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.screening import storage_adapter
from src.screening.api import screening_router
from src.screening.engine import ScreeningEngine


SAMPLE_PRIVACY_TEXT = """
Privacy Notice
We collect limited personal data to deliver services.
Data retention is described in this notice.
Users may contact us for any questions.
"""


def _patch_storage(monkeypatch, text=SAMPLE_PRIVACY_TEXT, doc_type="PRIVACY_NOTICE"):
    monkeypatch.setattr(storage_adapter, "get_document_text", lambda doc_id, **kwargs: text)
    monkeypatch.setattr(storage_adapter, "get_document_metadata", lambda doc_id, **kwargs: {"doc_type": doc_type})
    monkeypatch.setattr(storage_adapter, "get_document_bytes", lambda doc_id, **kwargs: None)
    monkeypatch.setattr(storage_adapter, "get_document_doc_type", lambda doc_id, **kwargs: doc_type)
    monkeypatch.setattr(storage_adapter, "get_document_subscription_id", lambda doc_id, **kwargs: "sub-test")


def test_legality_route_registered_once():
    paths = [route.path for route in screening_router.routes if "legality" in route.path]
    assert paths, "Legality endpoint not registered"
    assert len(paths) == len(set(paths)), "Duplicate legality endpoints detected"
    assert "/screening/legality" in paths


def test_legality_category_returns_structured_payload(monkeypatch):
    _patch_storage(monkeypatch)
    engine = ScreeningEngine()
    results = engine.run_category("legality", "doc-privacy", doc_type="PRIVACY_NOTICE", region="EU")
    payload = results[0].raw_features["legality"]
    assert payload["category"] == "legality"
    assert payload["region"] == "EU"
    assert payload["disclaimer"] in payload["narrative_report"]
    assert payload["scores"]["risk_score_0_1"] >= 0.0


def test_region_specific_flags_for_eu_privacy(monkeypatch):
    text = "Privacy Notice\nWe collect data for services.\nNo GDPR references here."
    _patch_storage(monkeypatch, text=text, doc_type="PRIVACY_NOTICE")
    engine = ScreeningEngine()
    results = engine.run_category("legality", "doc-eu", doc_type="PRIVACY_NOTICE", region="EU")
    payload = results[0].raw_features["legality"]
    categories = {finding["category"] for finding in payload["findings"]}
    assert "region_compliance" in categories
    assert any("GDPR" in finding["title"] or "privacy" in finding["title"].lower() for finding in payload["findings"])


def test_legality_is_deterministic(monkeypatch):
    _patch_storage(monkeypatch)
    engine = ScreeningEngine()
    first = engine.run_category("legality", "doc-stable", doc_type="PRIVACY_NOTICE", region="EU")[0]
    second = engine.run_category("legality", "doc-stable", doc_type="PRIVACY_NOTICE", region="EU")[0]
    assert json.dumps(first.raw_features["legality"], sort_keys=True) == json.dumps(
        second.raw_features["legality"], sort_keys=True
    )


def test_internet_disabled_warning(monkeypatch):
    _patch_storage(monkeypatch)
    engine = ScreeningEngine()
    result = engine.run_category(
        "legality", "doc-warning", doc_type="PRIVACY_NOTICE", region="US", internet_enabled_override=False
    )[0]
    warnings = result.raw_features["legality"]["warnings"]
    assert any("Internet validation disabled" in w for w in warnings)
