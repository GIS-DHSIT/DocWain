from src.api import screening_service
from src.api.statuses import STATUS_SCREENING_COMPLETED, STATUS_TRAINING_COMPLETED


def test_security_pass_does_not_downgrade_training_completed(monkeypatch):
    updates = []

    monkeypatch.setattr(screening_service, "update_security_screening", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        screening_service,
        "get_document_record",
        lambda *_args, **_kwargs: {"status": STATUS_TRAINING_COMPLETED},
    )

    def _capture(_document_id, fields):
        updates.append(fields)
        return None

    monkeypatch.setattr(screening_service, "update_document_fields", _capture)

    screening_service.apply_security_result(
        "doc-1",
        {"risk_level": "low", "overall_risk_level": "low"},
    )

    assert updates == []


def test_security_pass_promotes_stale_screening_status_when_embedding_done(monkeypatch):
    updates = []

    monkeypatch.setattr(screening_service, "update_security_screening", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        screening_service,
        "get_document_record",
        lambda *_args, **_kwargs: {
            "status": STATUS_SCREENING_COMPLETED,
            "embedding_status": screening_service.STATUS_EMBEDDING_COMPLETED,
            "trained_at": 123.0,
            "embedding": {"status": "COMPLETED"},
        },
    )

    def _capture(_document_id, fields):
        updates.append(fields)
        return None

    monkeypatch.setattr(screening_service, "update_document_fields", _capture)

    screening_service.apply_security_result(
        "doc-2",
        {"risk_level": "low", "overall_risk_level": "low"},
    )

    assert updates, "Expected a promotion to TRAINING_COMPLETED"
    assert updates[-1]["status"] == STATUS_TRAINING_COMPLETED
