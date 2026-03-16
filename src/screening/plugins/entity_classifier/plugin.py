"""Entity Classifier — configurable analysis plugin for classifying named entities."""

import time

from src.screening.plugins.base import ScreeningPlugin, PluginManifest, PluginResult


class Plugin(ScreeningPlugin):
    """Classifies named entities in documents and scores their relevance."""

    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            name="entity_classifier",
            version="1.0.0",
            description="Classifies and scores named entities found in document content",
            category="analysis",
            mandatory=False,
            supported_doc_types=["*"],
            requires=["extraction"],
            outputs=["entity_scores", "entity_types"],
            model="qwen3:14b",
            timeout=120,
        )

    def can_handle(self, doc_type: str, profile_config: dict) -> bool:
        return True

    def execute(self, extraction_data: dict,
                document_meta: dict) -> PluginResult:
        start = time.time()

        # TODO: Use LLM to extract and classify named entities with scores
        entity_scores = {}
        entity_types = {}

        duration_ms = int((time.time() - start) * 1000)
        return PluginResult(
            plugin_name="entity_classifier",
            success=True,
            outputs={
                "entity_scores": entity_scores,
                "entity_types": entity_types,
            },
            duration_ms=duration_ms,
        )
