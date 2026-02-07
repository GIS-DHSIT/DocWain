from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .config import ScreeningConfig
    from .search.base import SearchClient


@dataclass
class ScreeningContext:
    """Context passed to every screening tool."""

    doc_id: Optional[str]
    doc_type: Optional[str]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_bytes: Optional[bytes] = None
    region: Optional[str] = None
    jurisdiction: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)
    language: Optional[str] = None
    config: Optional["ScreeningConfig"] = None
    previous_version_text: Optional[str] = None
    search_client: Optional["SearchClient"] = None

    def text_hash(self) -> str:
        payload = self.raw_bytes if self.raw_bytes is not None else self.text.encode("utf-8", errors="ignore")
        return hashlib.sha256(payload).hexdigest()


@dataclass
class ToolResult:
    """Standardized result for each screening tool."""

    tool_name: str
    category: str
    score_0_1: float
    weight: float
    risk_level: str
    reasons: List[str] = field(default_factory=list)
    raw_features: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)
    evidence_spans: List[Dict[str, Any]] = field(default_factory=list)
    tool_version: str = "0.1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "name": self.tool_name,
            "category": self.category,
            "score_0_1": round(self.score_0_1, 4),
            "weight": round(self.weight, 4),
            "risk_level": self.risk_level,
            "reasons": self.reasons,
            "raw_features": self.raw_features,
            "actions": self.actions,
            "evidence_spans": self.evidence_spans,
            "tool_version": self.tool_version,
        }


@dataclass
class ScreeningReport:
    """Aggregated screening output."""

    overall_score_0_100: float
    risk_level: str
    results: List[ToolResult]
    top_findings: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    config_version: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score_0_100": round(self.overall_score_0_100, 2),
            "risk_level": self.risk_level,
            "results": [res.to_dict() for res in self.results],
            "top_findings": self.top_findings,
            "generated_at": self.generated_at.isoformat(),
            "config_version": self.config_version,
            "provenance": self.provenance,
        }


@dataclass
class SecurityFinding:
    """Detailed finding for security screening output."""

    finding_type: str
    category: str
    subcategory: Optional[str]
    severity: str
    confidence: float
    location: Dict[str, Any]
    snippet_masked: str
    context_masked: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.finding_type,
            "category": self.category,
            "subcategory": self.subcategory,
            "severity": self.severity,
            "confidence": round(float(self.confidence), 3),
            "location": self.location,
            "snippet_masked": self.snippet_masked,
            "context_masked": self.context_masked,
            "evidence": self.evidence,
        }


def compute_config_hash(data: Dict[str, Any]) -> str:
    """Generate a stable hash for provenance tracking."""
    packed = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()
