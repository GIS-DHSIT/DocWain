"""PII Detector — mandatory security plugin for detecting personally identifiable information."""

import time

from src.screening.plugins.base import ScreeningPlugin, PluginManifest, PluginResult


class Plugin(ScreeningPlugin):
    """Detects PII entities in document content and assigns a risk level."""

    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            name="pii_detector",
            version="1.0.0",
            description="Detects personally identifiable information (PII) in documents",
            category="security",
            mandatory=True,
            supported_doc_types=["*"],
            requires=["extraction"],
            outputs=["pii_entities", "pii_risk_level"],
        )

    def can_handle(self, doc_type: str, profile_config: dict) -> bool:
        return True

    def execute(self, extraction_data: dict,
                document_meta: dict) -> PluginResult:
        # TODO: Leverage existing src/screening/security_service.py for
        #       PII detection logic (email, phone, SSN, addresses, etc.)
        start = time.time()

        pii_entities = []
        pii_risk_level = "low"

        duration_ms = int((time.time() - start) * 1000)
        return PluginResult(
            plugin_name="pii_detector",
            success=True,
            outputs={
                "pii_entities": pii_entities,
                "pii_risk_level": pii_risk_level,
            },
            duration_ms=duration_ms,
        )
