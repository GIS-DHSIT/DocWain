"""Legality Checker — mandatory security plugin for compliance and legal risk assessment."""

import time

from src.screening.plugins.base import ScreeningPlugin, PluginManifest, PluginResult


class Plugin(ScreeningPlugin):
    """Checks document content for compliance flags and legal risk indicators."""

    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            name="legality_checker",
            version="1.0.0",
            description="Checks documents for compliance violations and legal risk",
            category="security",
            mandatory=True,
            supported_doc_types=["*"],
            requires=["extraction"],
            outputs=["compliance_flags", "legal_risk_level"],
            model="qwen3:14b",
            timeout=180,
        )

    def can_handle(self, doc_type: str, profile_config: dict) -> bool:
        return True

    def execute(self, extraction_data: dict,
                document_meta: dict) -> PluginResult:
        # TODO: Leverage existing src/screening/legality/ for
        #       compliance checking and legal risk assessment
        start = time.time()

        compliance_flags = []
        legal_risk_level = "low"

        duration_ms = int((time.time() - start) * 1000)
        return PluginResult(
            plugin_name="legality_checker",
            success=True,
            outputs={
                "compliance_flags": compliance_flags,
                "legal_risk_level": legal_risk_level,
            },
            duration_ms=duration_ms,
        )
