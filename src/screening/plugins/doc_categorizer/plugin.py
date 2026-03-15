"""Document Categorizer — configurable analysis plugin for classifying document types."""

import time

from src.screening.plugins.base import ScreeningPlugin, PluginManifest, PluginResult


class Plugin(ScreeningPlugin):
    """Categorises documents by type (e.g. contract, report, manual) with confidence."""

    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            name="doc_categorizer",
            version="1.0.0",
            description="Categorises documents by type with a confidence score",
            category="analysis",
            mandatory=False,
            supported_doc_types=["*"],
            requires=["extraction"],
            outputs=["doc_category", "category_confidence"],
            model="qwen3:14b",
            timeout=120,
        )

    def can_handle(self, doc_type: str, profile_config: dict) -> bool:
        return True

    def execute(self, extraction_data: dict,
                document_meta: dict) -> PluginResult:
        start = time.time()

        # TODO: Use LLM to determine document category
        doc_category = "uncategorised"
        category_confidence = 0.0

        duration_ms = int((time.time() - start) * 1000)
        return PluginResult(
            plugin_name="doc_categorizer",
            success=True,
            outputs={
                "doc_category": doc_category,
                "category_confidence": category_confidence,
            },
            duration_ms=duration_ms,
        )
