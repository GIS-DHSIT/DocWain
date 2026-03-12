from __future__ import annotations

import json
from src.utils.logging_utils import get_logger
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.api.config import Config
from .models import compute_config_hash

logger = get_logger(__name__)
_LEGACY_VETTING_NOTICE_LOGGED = False

DEFAULT_WEIGHTS = {
    "integrity_hash": 0.07,
    "metadata_consistency": 0.07,
    "template_conformance": 0.10,
    "policy_compliance": 0.10,
    "numeric_unit_consistency": 0.07,
    "citation_sanity": 0.06,
    "ambiguity_vagueness": 0.06,
    "readability_style": 0.07,
    "passive_voice": 0.05,
    "pii_sensitivity": 0.12,
    "ai_authorship": 0.11,
    "resume_extractor_tool": 0.08,
    "company_validator_tool": 0.08,
    "institution_validator_tool": 0.06,
    "certification_verifier_tool": 0.10,
    "authenticity_analyzer_tool": 0.12,
    "resume_entity_validation": 0.08,
    "resume_screening": 0.12,
    "legality_agent": 0.14,
}

DEFAULT_TEMPLATES = {
    "RESUME": {
        "mandatory_sections": ["summary", "experience", "education", "skills"],
        "heading_order": ["summary", "experience", "education", "skills"],
    },
    "POLICY": {
        "mandatory_sections": ["scope", "purpose", "policy", "responsibilities"],
        "heading_order": ["scope", "purpose", "policy", "responsibilities", "exceptions"],
    },
}

DEFAULT_POLICY_RULES = {
    "POLICY": {
        "forbidden_phrases": ["not liable", "no responsibility"],
        "required_disclaimers": ["subject to change"],
        "required_keywords": ["scope", "compliance", "responsibilities"],
    },
    "RESUME": {"forbidden_phrases": ["lorem ipsum"], "required_keywords": ["experience", "education"]},
}

DOMAIN_SPECIFIC_TOOLS = {
    "resume_extractor_tool",
    "company_validator_tool",
    "institution_validator_tool",
    "certification_verifier_tool",
    "authenticity_analyzer_tool",
    "resume_entity_validation",
    "resume_screening",
    "legality_agent",
    "template_conformance",
    "policy_compliance",
}

def _parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

def _log_legacy_vetting_notice() -> None:
    global _LEGACY_VETTING_NOTICE_LOGGED
    if _LEGACY_VETTING_NOTICE_LOGGED:
        return
    _LEGACY_VETTING_NOTICE_LOGGED = True
    logger.info("Legacy vetting config not found — proceeding with screening defaults")

def log_legacy_vetting_notice_if_missing(path: Optional[str] = None) -> None:
    config_path = Path(path) if path else Path(
        os.getenv("SCREENING_CONFIG_PATH", "") or Path.cwd() / "screening_config.json"
    )
    if not config_path.exists():
        _log_legacy_vetting_notice()

@dataclass
class ScreeningConfig:
    """Configuration holder for the screening subsystem."""

    enabled_tools: List[str] = field(default_factory=lambda: list(DEFAULT_WEIGHTS.keys()))
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    sigmoid_a: float = 6.0
    sigmoid_b: float = 0.50
    risk_thresholds: Dict[str, float] = field(default_factory=lambda: {"high": 75.0, "medium": 45.0})
    internet_enabled: bool = False
    search_provider: Dict[str, Any] = field(default_factory=dict)
    doc_type_templates: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_TEMPLATES))
    policy_rules: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_POLICY_RULES))
    sensitive_keywords: List[str] = field(default_factory=lambda: ["confidential", "proprietary"])
    auto_attach_on_ingest: bool = True
    block_high_risk: bool = False
    config_source: str = "defaults"
    config_hash: str = ""

    def __post_init__(self) -> None:
        self._normalize_weights()

    @classmethod
    def load(cls, path: Optional[str] = None) -> "ScreeningConfig":
        """Load configuration from JSON + environment overrides."""
        default_path = Path(
            os.getenv("SCREENING_CONFIG_PATH", "")
            or Path.cwd() / "screening_config.json"
        )
        config_path = Path(path) if path else default_path
        data: Dict[str, Any] = {}
        source = "defaults"

        if config_path.exists():
            try:
                data = json.loads(config_path.read_text())
                source = str(config_path)
            except Exception:
                data = {}
        else:
            _log_legacy_vetting_notice()

        weights = dict(DEFAULT_WEIGHTS)
        weights.update(data.get("weights", {}) if isinstance(data.get("weights"), dict) else {})

        enabled_tools = data.get("enabled_tools") if isinstance(data.get("enabled_tools"), list) else list(weights.keys())
        sigmoid = data.get("sigmoid", {}) if isinstance(data.get("sigmoid"), dict) else {}
        risk_thresholds = data.get("risk_thresholds", {}) if isinstance(data.get("risk_thresholds"), dict) else {}
        templates = data.get("doc_type_templates", {}) if isinstance(data.get("doc_type_templates"), dict) else {}
        policy_rules = data.get("policy_rules", {}) if isinstance(data.get("policy_rules"), dict) else {}
        sensitive_keywords = data.get("sensitive_keywords", DEFAULT_POLICY_RULES.get("sensitive_keywords", []))
        search_provider = data.get("search_provider", {}) if isinstance(data.get("search_provider"), dict) else {}

        env_tools = os.getenv("SCREENING_ENABLED_TOOLS")
        if env_tools:
            enabled_tools = [tool.strip() for tool in env_tools.split(",") if tool.strip()]
        if "legality_agent" not in enabled_tools:
            enabled_tools.append("legality_agent")

        for tool_name in list(weights.keys()):
            env_weight = os.getenv(f"SCREENING_WEIGHT_{tool_name.upper()}")
            if env_weight:
                try:
                    weights[tool_name] = float(env_weight)
                except ValueError:
                    pass

        internet_from_file = _parse_bool(data.get("internet_enabled"), default=False)
        internet_enabled = _parse_bool(
            os.getenv("SCREENING_INTERNET_ENABLED"),
            default=internet_from_file,
        )
        auto_attach_default = _parse_bool(data.get("auto_attach_on_ingest"), default=True)
        auto_attach = _parse_bool(
            os.getenv("SCREENING_AUTO_ATTACH_ON_INGEST"),
            default=auto_attach_default,
        )
        block_default = _parse_bool(data.get("block_high_risk"), default=False)
        block_high_risk = _parse_bool(
            os.getenv("SCREENING_BLOCK_HIGH_RISK"),
            default=block_default,
        )

        if os.getenv("SCREENING_SIGMOID_A"):
            try:
                sigmoid["a"] = float(
                    os.getenv("SCREENING_SIGMOID_A") or sigmoid.get("a", 6.0)
                )
            except ValueError:
                pass
        if os.getenv("SCREENING_SIGMOID_B"):
            try:
                sigmoid["b"] = float(
                    os.getenv("SCREENING_SIGMOID_B") or sigmoid.get("b", 0.50)
                )
            except ValueError:
                pass

        if os.getenv("SCREENING_RISK_HIGH"):
            try:
                risk_thresholds["high"] = float(
                    os.getenv("SCREENING_RISK_HIGH") or "75"
                )
            except ValueError:
                pass
        if os.getenv("SCREENING_RISK_MEDIUM"):
            try:
                risk_thresholds["medium"] = float(
                    os.getenv("SCREENING_RISK_MEDIUM") or "45"
                )
            except ValueError:
                pass

        if os.getenv("SCREENING_SEARCH_PROVIDER"):
            search_provider["provider"] = os.getenv("SCREENING_SEARCH_PROVIDER")
        if os.getenv("SCREENING_SEARCH_API_KEY"):
            search_provider["api_key"] = os.getenv("SCREENING_SEARCH_API_KEY")
        if os.getenv("SCREENING_SEARCH_ENDPOINT"):
            search_provider["endpoint"] = os.getenv("SCREENING_SEARCH_ENDPOINT")

        if not Config.Features.DOMAIN_SPECIFIC_ENABLED:
            enabled_tools = [tool for tool in enabled_tools if tool not in DOMAIN_SPECIFIC_TOOLS]
            weights = {key: value for key, value in weights.items() if key not in DOMAIN_SPECIFIC_TOOLS}
            templates = {}
            policy_rules = {}

        config = cls(
            enabled_tools=enabled_tools,
            weights=weights,
            sigmoid_a=float(sigmoid.get("a", 6.0)),
            sigmoid_b=float(sigmoid.get("b", 0.50)),
            risk_thresholds={"high": float(risk_thresholds.get("high", 75.0)), "medium": float(risk_thresholds.get("medium", 45.0))},
            internet_enabled=internet_enabled or bool(search_provider.get("api_key")),
            search_provider=search_provider,
            doc_type_templates={**DEFAULT_TEMPLATES, **templates},
            policy_rules={**DEFAULT_POLICY_RULES, **policy_rules},
            sensitive_keywords=sensitive_keywords if isinstance(sensitive_keywords, list) else ["confidential", "proprietary"],
            auto_attach_on_ingest=auto_attach,
            block_high_risk=block_high_risk,
            config_source=source,
        )
        config._normalize_weights()
        config.config_hash = compute_config_hash(config.as_dict())
        return config

    def _normalize_weights(self) -> None:
        active = [t for t in self.enabled_tools if t in self.weights]
        if not active:
            active = list(self.weights.keys())
            self.enabled_tools = active
        total = sum(max(self.weights.get(t, 0.0), 0.0) for t in active)
        if total <= 0:
            even = 1.0 / len(active)
            for tool in active:
                self.weights[tool] = even
            return
        for tool in active:
            self.weights[tool] = max(self.weights.get(tool, 0.0), 0.0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "enabled_tools": self.enabled_tools,
            "weights": self.weights,
            "sigmoid_a": self.sigmoid_a,
            "sigmoid_b": self.sigmoid_b,
            "risk_thresholds": self.risk_thresholds,
            "internet_enabled": self.internet_enabled,
            "search_provider": self.search_provider,
            "doc_type_templates": self.doc_type_templates,
            "policy_rules": self.policy_rules,
            "sensitive_keywords": self.sensitive_keywords,
            "auto_attach_on_ingest": self.auto_attach_on_ingest,
            "block_high_risk": self.block_high_risk,
            "config_source": self.config_source,
        }
