from unittest.mock import patch

from src.finetune.auto_orchestrator import AutoFinetuneOrchestrator
from src.finetune.models import AutoFinetuneRunRequest


def test_auto_orchestrator_dry_run():
    orchestrator = AutoFinetuneOrchestrator()
    request = AutoFinetuneRunRequest(dry_run=True)
    with patch("src.finetune.auto_orchestrator.list_collections", return_value=["default"]), \
            patch(
                "src.finetune.auto_orchestrator.list_profile_ids",
                return_value={"profile_ids": ["alpha"], "counts": {"alpha": 100}},
            ):
        results = orchestrator.run_for_all_collections(request)

    assert results[0].status == "planned"
