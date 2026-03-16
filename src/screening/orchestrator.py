"""Screening orchestrator — runs mandatory + configured plugins, aggregates results."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from src.screening.plugins.base import PluginResult
from src.screening.plugins.registry import get_registry

logger = logging.getLogger(__name__)


class ScreeningOrchestrator:
    """Orchestrates screening plugin execution for a document.

    1. Always runs mandatory security plugins
    2. Runs profile-configured plugins in parallel
    3. Aggregates all results into a unified screening report
    """

    def run(self, document_id: str, extraction_data: dict,
            document_meta: dict, profile_config: dict) -> dict:
        """Execute screening pipeline.

        Args:
            document_id: Document identifier
            extraction_data: Full extraction JSON from Azure Blob
            document_meta: MongoDB document record
            profile_config: Profile's screening_config

        Returns:
            Screening report dict with plugin results, aggregated metadata
        """
        registry = get_registry()
        doc_type = document_meta.get("file_type", "unknown")

        # 1. Mandatory security plugins
        mandatory_plugins = registry.get_mandatory_plugins()

        # 2. Profile-configured plugins
        enabled_names = profile_config.get("enabled_plugins", [])
        configured_plugins = registry.get_configured_plugins(
            enabled_names, doc_type, profile_config
        )

        all_plugins = mandatory_plugins + configured_plugins
        results: Dict[str, PluginResult] = {}
        plugins_run = []

        # Run all plugins in parallel
        with ThreadPoolExecutor(max_workers=min(len(all_plugins), 8)) as executor:
            futures = {}
            for plugin in all_plugins:
                manifest = plugin.get_manifest()
                overrides = profile_config.get("plugin_overrides", {}).get(
                    manifest.name, {}
                )
                meta_with_overrides = {**document_meta, **overrides}
                futures[executor.submit(
                    plugin.execute, extraction_data, meta_with_overrides
                )] = manifest.name

            for future in as_completed(futures):
                plugin_name = futures[future]
                try:
                    result = future.result(timeout=300)
                    results[plugin_name] = result
                    plugins_run.append(plugin_name)
                    logger.info(f"Plugin {plugin_name} completed for {document_id}: "
                              f"success={result.success}")
                except Exception as e:
                    logger.error(f"Plugin {plugin_name} failed for {document_id}: {e}")
                    results[plugin_name] = PluginResult(
                        plugin_name=plugin_name,
                        success=False,
                        errors=[str(e)]
                    )
                    plugins_run.append(plugin_name)

        # 3. Aggregate results
        report = self._aggregate(document_id, results, plugins_run)
        return report

    def _aggregate(self, document_id: str, results: Dict[str, PluginResult],
                   plugins_run: List[str]) -> dict:
        """Aggregate plugin results into a unified screening report."""
        domain_tags = []
        doc_category = "unknown"
        risk_level = "low"
        entity_scores = {}
        flags = []
        all_outputs = {}

        for name, result in results.items():
            all_outputs[name] = {
                "success": result.success,
                "outputs": result.outputs,
                "errors": result.errors,
                "duration_ms": result.duration_ms
            }

            if not result.success:
                flags.append(f"{name}: failed")
                continue

            outputs = result.outputs

            if "domain_tags" in outputs:
                domain_tags.extend(outputs["domain_tags"])
            if "doc_category" in outputs:
                doc_category = outputs["doc_category"]

            for key in ("pii_risk_level", "secrets_risk_level",
                       "legal_risk_level", "risk_level"):
                if key in outputs:
                    risk_level = self._max_risk(risk_level, outputs[key])

            if "entity_scores" in outputs:
                entity_scores.update(outputs["entity_scores"])

            for key in ("compliance_flags", "pii_entities", "secrets_found"):
                if key in outputs and outputs[key]:
                    flags.extend([f"{name}:{key}:{item}" for item in outputs[key]])

        return {
            "document_id": document_id,
            "plugins_run": plugins_run,
            "domain_tags": list(set(domain_tags)),
            "doc_category": doc_category,
            "risk_level": risk_level,
            "entity_scores": entity_scores,
            "flags": flags,
            "plugin_results": all_outputs
        }

    @staticmethod
    def _max_risk(current: str, new: str) -> str:
        """Return the higher risk level."""
        levels = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        if levels.get(new, 0) > levels.get(current, 0):
            return new
        return current
