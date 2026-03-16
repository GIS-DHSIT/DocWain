"""Domain Tagger — configurable analysis plugin for tagging document domains."""

import time

from src.screening.plugins.base import ScreeningPlugin, PluginManifest, PluginResult


class Plugin(ScreeningPlugin):
    """Tags documents with relevant domain labels (e.g. finance, legal, medical)."""

    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            name="domain_tagger",
            version="1.0.0",
            description="Tags documents with domain labels based on content analysis",
            category="analysis",
            mandatory=False,
            supported_doc_types=["*"],
            requires=["extraction"],
            outputs=["domain_tags"],
            model="qwen3:14b",
            timeout=120,
        )

    def can_handle(self, doc_type: str, profile_config: dict) -> bool:
        return True

    def execute(self, extraction_data: dict,
                document_meta: dict) -> PluginResult:
        start = time.time()

        # TODO: Use LLM to analyse extracted text and assign domain tags
        domain_tags = []

        duration_ms = int((time.time() - start) * 1000)
        return PluginResult(
            plugin_name="domain_tagger",
            success=True,
            outputs={
                "domain_tags": domain_tags,
            },
            duration_ms=duration_ms,
        )
