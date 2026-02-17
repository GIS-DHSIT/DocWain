from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from . import storage_adapter
from .config import ScreeningConfig
from .models import ScreeningContext, ScreeningReport, ToolResult
from src.api.pipeline_models import ExtractedDocument
from .resume import run_resume_analysis
from .search import NullSearchClient, SearchClient, SimpleHttpSearchClient
from .tools import (
    AIAuthorshipTool,
    AmbiguityVaguenessTool,
    CitationSanityTool,
    IntegrityHashTool,
    MetadataConsistencyTool,
    NumericUnitConsistencyTool,
    PIISensitivityTool,
    PassiveVoiceTool,
    PolicyComplianceTool,
    ReadabilityStyleTool,
    CertificationVerifierTool,
    ResumeAuthenticityTool,
    ResumeCompanyValidatorTool,
    ResumeEntityValidationTool,
    ResumeExtractorTool,
    ResumeInstitutionValidatorTool,
    ResumeScreeningTool,
    LegalityAgentTool,
    TemplateConformanceTool,
    ScreeningTool,
)


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


CATEGORY_TOOL_MAP = {
    "integrity": ["integrity_hash", "metadata_consistency"],
    "compliance": ["template_conformance", "policy_compliance"],
    "quality": ["numeric_unit_consistency", "citation_sanity", "ambiguity_vagueness"],
    "language": ["readability_style", "passive_voice"],
    "security": ["pii_sensitivity"],
    "ai-authorship": ["ai_authorship"],
    "ai_authorship": ["ai_authorship"],
    "legality": ["legality_agent"],
    "resume": [
        "resume_extractor_tool",
        "company_validator_tool",
        "institution_validator_tool",
        "certification_verifier_tool",
        "authenticity_analyzer_tool",
        "resume_entity_validation",
        "resume_screening",
    ],
    "resume_screening": [
        "resume_extractor_tool",
        "company_validator_tool",
        "institution_validator_tool",
        "certification_verifier_tool",
        "authenticity_analyzer_tool",
        "resume_entity_validation",
        "resume_screening",
    ],
}


class ScreeningEngine:
    """Runs configured screening tools and blends their scores."""

    def __init__(
        self,
        config: Optional[ScreeningConfig] = None,
        tools: Optional[Sequence[ScreeningTool]] = None,
        search_client: Optional[SearchClient] = None,
    ) -> None:
        self.config = config or ScreeningConfig.load()
        self.search_client = search_client or self._build_search_client(self.config)
        self._tools: Dict[str, ScreeningTool] = {tool.name: tool for tool in (tools or self._default_tools())}

    def _build_search_client(self, cfg: ScreeningConfig) -> SearchClient:
        provider = (cfg.search_provider or {}).get("provider")
        api_key = (cfg.search_provider or {}).get("api_key")
        endpoint = (cfg.search_provider or {}).get("endpoint")
        if cfg.internet_enabled and provider and api_key:
            return SimpleHttpSearchClient(provider=provider, api_key=api_key, endpoint=endpoint)
        return NullSearchClient()

    def _with_config_override(self, internet_enabled_override: Optional[bool]) -> ScreeningConfig:
        if internet_enabled_override is None:
            return self.config
        cfg = ScreeningConfig(
            enabled_tools=self.config.enabled_tools,
            weights=self.config.weights,
            sigmoid_a=self.config.sigmoid_a,
            sigmoid_b=self.config.sigmoid_b,
            risk_thresholds=self.config.risk_thresholds,
            internet_enabled=internet_enabled_override,
            search_provider=self.config.search_provider,
            doc_type_templates=self.config.doc_type_templates,
            policy_rules=self.config.policy_rules,
            sensitive_keywords=self.config.sensitive_keywords,
            auto_attach_on_ingest=self.config.auto_attach_on_ingest,
            block_high_risk=self.config.block_high_risk,
            config_source=self.config.config_source,
        )
        cfg.config_hash = self.config.config_hash
        cfg._normalize_weights()
        return cfg

    def _infer_region(self, metadata: Dict[str, Any], region_override: Optional[str]) -> Optional[str]:
        if region_override:
            normalized = str(region_override).strip()
            return normalized or None
        for key in ("region", "country", "jurisdiction", "governing_law", "governingLaw"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _default_tools(self) -> List[ScreeningTool]:
        return [
            IntegrityHashTool(),
            MetadataConsistencyTool(),
            TemplateConformanceTool(),
            PolicyComplianceTool(),
            NumericUnitConsistencyTool(),
            CitationSanityTool(),
            AmbiguityVaguenessTool(),
            ReadabilityStyleTool(),
            PassiveVoiceTool(),
            PIISensitivityTool(),
            AIAuthorshipTool(),
            ResumeExtractorTool(),
            ResumeCompanyValidatorTool(),
            ResumeInstitutionValidatorTool(),
            CertificationVerifierTool(),
            ResumeAuthenticityTool(),
            ResumeEntityValidationTool(),
            ResumeScreeningTool(),
            LegalityAgentTool(),
        ]

    def _active_tools(self, doc_type: Optional[str], cfg: Optional[ScreeningConfig] = None) -> List[ScreeningTool]:
        config = cfg or self.config
        enabled = set(config.enabled_tools or [])
        active = []
        for name, tool in self._tools.items():
            if enabled and name not in enabled:
                continue
            if not tool.applies_to(doc_type):
                continue
            active.append(tool)
        return active

    def _risk_level(self, overall_score: float, cfg: Optional[ScreeningConfig] = None) -> str:
        config = cfg or self.config
        if overall_score >= config.risk_thresholds.get("high", 75):
            return "HIGH"
        if overall_score >= config.risk_thresholds.get("medium", 45):
            return "MEDIUM"
        return "LOW"

    def _blend_score(self, results: Sequence[ToolResult], cfg: Optional[ScreeningConfig] = None) -> float:
        config = cfg or self.config
        total_weight = sum(res.weight for res in results) or 1.0
        blended = sum(res.weight * res.score_0_1 for res in results) / total_weight
        calibrated = _sigmoid(config.sigmoid_a * (blended - config.sigmoid_b))
        return calibrated * 100

    def _top_findings(self, results: Sequence[ToolResult], limit: int = 5) -> List[str]:
        ranked: List[tuple[float, str]] = []
        for res in results:
            for reason in res.reasons:
                ranked.append((res.weight * res.score_0_1, f"{res.tool_name}: {reason}"))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        seen = set()
        findings: List[str] = []
        for _, text in ranked:
            if text not in seen:
                findings.append(text)
                seen.add(text)
            if len(findings) >= limit:
                break
        return findings

    def _doc_type_from_payload(self, extracted_payload: Any) -> Optional[str]:
        if isinstance(extracted_payload, ExtractedDocument):
            return extracted_payload.doc_type
        if isinstance(extracted_payload, dict):
            for value in extracted_payload.values():
                if isinstance(value, ExtractedDocument) and value.doc_type:
                    return value.doc_type
                if isinstance(value, dict):
                    doc_type = value.get("doc_type")
                    if isinstance(doc_type, str) and doc_type.strip():
                        return doc_type
        return None

    def _build_context_from_doc(
        self,
        doc_id: str,
        *,
        doc_type_override: Optional[str] = None,
        internet_enabled_override: Optional[bool] = None,
        region_override: Optional[str] = None,
        jurisdiction_override: Optional[str] = None,
        allow_region_jurisdiction: bool = False,
        extracted_payload: Optional[Any] = None,
    ) -> ScreeningContext:
        cfg = self._with_config_override(internet_enabled_override)
        metadata = storage_adapter.get_document_metadata(doc_id)
        subscription_id = storage_adapter.get_document_subscription_id(doc_id)
        text = storage_adapter.get_document_text(doc_id, extracted=extracted_payload, allow_fallback=extracted_payload is None)
        raw_bytes = storage_adapter.get_document_bytes(doc_id)
        doc_type = (
            doc_type_override
            or self._doc_type_from_payload(extracted_payload)
            or storage_adapter.get_document_doc_type(doc_id)
            or metadata.get("doc_type")
        )

        warnings: List[str] = []
        if not subscription_id:
            warnings.append("subscription_id_unavailable_for_document")

        ctx = ScreeningContext(
            doc_id=doc_id,
            doc_type=doc_type,
            text=text,
            metadata=metadata or {},
            raw_bytes=raw_bytes,
            region=self._infer_region(metadata or {}, region_override) if allow_region_jurisdiction else None,
            jurisdiction=(jurisdiction_override or (metadata or {}).get("jurisdiction"))
            if allow_region_jurisdiction
            else None,
            config=cfg,
            search_client=self._build_search_client(cfg),
        )
        ctx.metadata.setdefault("doc_id", doc_id)
        if subscription_id:
            ctx.metadata.setdefault("subscription_id", subscription_id)
        if warnings:
            ctx.metadata.setdefault("warnings", warnings)
        return ctx

    def _resolve_tools_for_category(
        self, category: str, doc_type: Optional[str], cfg: ScreeningConfig
    ) -> List[ScreeningTool]:
        normalized = category.lower()
        canonical = normalized.replace("_", "-")
        names = CATEGORY_TOOL_MAP.get(canonical) or CATEGORY_TOOL_MAP.get(canonical.replace("-", "_"))
        explicit_category = names is not None  # user explicitly requested this category
        tools: List[ScreeningTool]
        if names:
            tools = [self._tools[name] for name in names if name in self._tools]
        else:
            tools = [tool for tool in self._tools.values() if tool.category.lower() == normalized]

        enabled = set(cfg.enabled_tools or [])
        filtered = []
        for tool in tools:
            if enabled and tool.name not in enabled:
                continue
            # Skip applies_to filter when user explicitly requests a category —
            # they want those tools regardless of stored doc_type.
            # The applies_to filter still applies in run_all() via _active_tools().
            if not explicit_category and not tool.applies_to(doc_type):
                continue
            filtered.append(tool)
        return filtered

    def _run_tools(self, ctx: ScreeningContext, tools: Sequence[ScreeningTool]) -> List[ToolResult]:
        results: List[ToolResult] = []
        for tool in tools:
            if tool.requires_internet and not (ctx.config and ctx.config.internet_enabled):
                ctx.search_client = ctx.search_client or NullSearchClient()
            results.append(tool.run(ctx))
        return results

    def screen(self, ctx: ScreeningContext, cfg: Optional[ScreeningConfig] = None) -> ScreeningReport:
        active_config = cfg or ctx.config or self.config
        ctx.config = active_config
        results: List[ToolResult] = []
        active_tools = self._active_tools(ctx.doc_type, active_config)
        results.extend(self._run_tools(ctx, active_tools))

        overall = self._blend_score(results, active_config) if results else 0.0
        risk_level = self._risk_level(overall, active_config)
        top_findings = self._top_findings(results)

        provenance = {
            "tool_versions": {r.tool_name: r.tool_version for r in results},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_source": active_config.config_source,
            "config_hash": active_config.config_hash,
        }

        return ScreeningReport(
            overall_score_0_100=overall,
            risk_level=risk_level,
            results=results,
            top_findings=top_findings,
            generated_at=datetime.now(timezone.utc),
            config_version=active_config.config_hash,
            provenance=provenance,
        )

    def run_one(
        self,
        tool_name: str,
        doc_id: str,
        *,
        doc_type: Optional[str] = None,
        internet_enabled_override: Optional[bool] = None,
    ) -> ToolResult:
        ctx = self._build_context_from_doc(
            doc_id, doc_type_override=doc_type, internet_enabled_override=internet_enabled_override
        )
        cfg = ctx.config or self.config
        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown screening tool '{tool_name}'")
        if (cfg.enabled_tools and tool_name not in cfg.enabled_tools) or not tool.applies_to(ctx.doc_type):
            raise ValueError(f"Tool '{tool_name}' is not enabled for document type {ctx.doc_type}")
        return self._run_tools(ctx, [tool])[0]

    def run_category(
        self,
        category: str,
        doc_id: str,
        *,
        doc_type: Optional[str] = None,
        internet_enabled_override: Optional[bool] = None,
        region: Optional[str] = None,
        jurisdiction: Optional[str] = None,
    ) -> List[ToolResult]:
        normalized = category.lower().replace("_", "-")
        allow_region = normalized == "legality"
        ctx = self._build_context_from_doc(
            doc_id,
            doc_type_override=doc_type,
            internet_enabled_override=internet_enabled_override,
            region_override=region,
            jurisdiction_override=jurisdiction,
            allow_region_jurisdiction=allow_region,
        )
        cfg = ctx.config or self.config
        tools = self._resolve_tools_for_category(category, ctx.doc_type, cfg)
        if not tools:
            raise ValueError(f"No screening tools found for category '{category}'")
        return self._run_tools(ctx, tools)

    def run_category_from_payload(
        self,
        category: str,
        doc_id: str,
        extracted_payload: Any,
        *,
        doc_type: Optional[str] = None,
        internet_enabled_override: Optional[bool] = None,
        region: Optional[str] = None,
        jurisdiction: Optional[str] = None,
    ) -> List[ToolResult]:
        normalized = category.lower().replace("_", "-")
        allow_region = normalized == "legality"
        ctx = self._build_context_from_doc(
            doc_id,
            doc_type_override=doc_type,
            internet_enabled_override=internet_enabled_override,
            region_override=region,
            jurisdiction_override=jurisdiction,
            allow_region_jurisdiction=allow_region,
            extracted_payload=extracted_payload,
        )
        cfg = ctx.config or self.config
        tools = self._resolve_tools_for_category(category, ctx.doc_type, cfg)
        if not tools:
            raise ValueError(f"No screening tools found for category '{category}'")
        return self._run_tools(ctx, tools)

    def run_all(
        self,
        doc_id: str,
        *,
        doc_type: Optional[str] = None,
        internet_enabled_override: Optional[bool] = None,
    ) -> ScreeningReport:
        ctx = self._build_context_from_doc(
            doc_id,
            doc_type_override=doc_type,
            internet_enabled_override=internet_enabled_override,
            allow_region_jurisdiction=False,
        )
        return self.screen(ctx, cfg=ctx.config)

    def evaluate(
        self,
        text: str,
        doc_id: Optional[str] = None,
        doc_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        raw_bytes: Optional[bytes] = None,
        previous_version_text: Optional[str] = None,
        internet_enabled_override: Optional[bool] = None,
        region: Optional[str] = None,
        jurisdiction: Optional[str] = None,
    ) -> Dict[str, Any]:
        cfg = self._with_config_override(internet_enabled_override)
        ctx = ScreeningContext(
            doc_id=doc_id,
            doc_type=doc_type or (metadata or {}).get("doc_type"),
            text=text or "",
            metadata=metadata or {},
            raw_bytes=raw_bytes,
            previous_version_text=previous_version_text,
            region=self._infer_region(metadata or {}, region),
            jurisdiction=jurisdiction or (metadata or {}).get("jurisdiction"),
            config=cfg,
            search_client=self._build_search_client(cfg),
        )
        report = self.screen(ctx, cfg=cfg)
        response = report.to_dict()
        response["doc_id"] = doc_id
        if cfg.block_high_risk:
            response["block_recommended"] = report.risk_level == "HIGH"
        return response

    def resume_analysis_from_doc(
        self,
        doc_id: str,
        *,
        doc_type: Optional[str] = None,
        internet_enabled_override: Optional[bool] = None,
    ):
        ctx = self._build_context_from_doc(
            doc_id, doc_type_override=doc_type or "RESUME", internet_enabled_override=internet_enabled_override
        )
        cfg = ctx.config or self.config
        return run_resume_analysis(
            text=ctx.text or "",
            doc_id=ctx.doc_id,
            metadata=ctx.metadata,
            search_client=ctx.search_client,
            internet_enabled=cfg.internet_enabled,
        )

    def resume_analysis_from_text(
        self,
        text: str,
        *,
        doc_id: Optional[str] = None,
        doc_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        raw_bytes: Optional[bytes] = None,
        previous_version_text: Optional[str] = None,
        internet_enabled_override: Optional[bool] = None,
    ):
        cfg = self._with_config_override(internet_enabled_override)
        ctx = ScreeningContext(
            doc_id=doc_id,
            doc_type=doc_type or (metadata or {}).get("doc_type"),
            text=text or "",
            metadata=metadata or {},
            raw_bytes=raw_bytes,
            previous_version_text=previous_version_text,
            config=cfg,
            search_client=self._build_search_client(cfg),
        )
        return run_resume_analysis(
            text=ctx.text or "",
            doc_id=ctx.doc_id,
            metadata=ctx.metadata,
            search_client=ctx.search_client,
            internet_enabled=cfg.internet_enabled,
        )


_DEFAULT_ENGINE = ScreeningEngine()


def screen_and_attach_metadata(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    doc_type: Optional[str] = None,
    raw_bytes: Optional[bytes] = None,
    previous_version_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Optional hook for ingestion pipelines.

    Attaches screening results under metadata["screening_report"] and returns the metadata.
    The hook never blocks ingestion and respects the ScreeningConfig toggle.
    """
    metadata = dict(metadata or {})
    result = _DEFAULT_ENGINE.evaluate(
        text=text,
        doc_id=str(metadata.get("doc_id") or metadata.get("id") or ""),
        doc_type=doc_type or metadata.get("doc_type"),
        metadata=metadata,
        raw_bytes=raw_bytes,
        previous_version_text=previous_version_text or metadata.get("previous_version_text"),
    )
    metadata["screening_report"] = result
    # Backward-compatible alias used by earlier ingestion paths.
    metadata.setdefault("screening", result)
    return metadata
