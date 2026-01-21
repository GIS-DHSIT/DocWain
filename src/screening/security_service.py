from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import ScreeningConfig
from .detectors.classification import resolve_classification
from .detectors.private_data import PrivateDataMatch, detect_private_data
from .detectors.secrets import SecretMatch, detect_secrets
from .engine import ScreeningEngine
from .models import ScreeningContext, SecurityFinding, ToolResult
from .tools.pii_sensitivity import PIISensitivityTool

SECRET_SEVERITY_SCORES = {"LOW": 5, "MED": 15, "HIGH": 30, "CRITICAL": 50}
CLASSIFICATION_BOOST = {
    "INTERNAL": 5,
    "CONFIDENTIAL": 15,
    "RESTRICTED": 25,
    "SECRET": 35,
}
PRIVATE_DATA_SCORE_SCALE = 0.4


def _mask_value(value: str, mask_mode: str = "value") -> str:
    if mask_mode == "block":
        return "PRIVATE_KEY_BLOCK"
    value = value or ""
    if len(value) <= 6:
        return "***"
    return f"{value[:2]}***{value[-4:]}"


def _context_excerpt(text: str, start: int, end: int, window: int = 30) -> tuple[str, int]:
    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:right], left


def _apply_extra_masks(
    text: str,
    context: str,
    context_start: int,
    spans: Optional[List[tuple[int, int, str]]] = None,
) -> str:
    if not spans:
        return context
    masked_context = context
    for span_start, span_end, span_mode in sorted(spans, key=lambda item: item[0], reverse=True):
        rel_start = span_start - context_start
        rel_end = span_end - context_start
        if rel_end <= 0 or rel_start >= len(masked_context):
            continue
        rel_start = max(rel_start, 0)
        rel_end = min(rel_end, len(masked_context))
        raw_value = text[span_start:span_end]
        masked_value = _mask_value(raw_value, span_mode)
        masked_context = masked_context[:rel_start] + masked_value + masked_context[rel_end:]
    return masked_context


def _masked_snippet_context(
    text: str,
    start: int,
    end: int,
    mask_start: Optional[int] = None,
    mask_end: Optional[int] = None,
    mask_mode: str = "value",
    window: int = 30,
    extra_spans: Optional[List[tuple[int, int, str]]] = None,
) -> tuple[str, str]:
    context_start = max(0, start - window)
    context_end = min(len(text), end + window)
    context = text[context_start:context_end]

    if mask_mode == "block":
        masked = "PRIVATE_KEY_BLOCK"
        rel_start = start - context_start
        rel_end = end - context_start
        context = context[:rel_start] + masked + context[rel_end:]
        context = _apply_extra_masks(text, context, context_start, extra_spans)
        return masked, context

    mask_start = mask_start if mask_start is not None else start
    mask_end = mask_end if mask_end is not None else end
    raw_value = text[mask_start:mask_end]
    masked_value = _mask_value(raw_value, mask_mode)
    rel_start = mask_start - context_start
    rel_end = mask_end - context_start
    if 0 <= rel_start <= rel_end <= len(context):
        context = context[:rel_start] + masked_value + context[rel_end:]
    context = _apply_extra_masks(text, context, context_start, extra_spans)
    return masked_value, context


def _location_from_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    page = metadata.get("page") or metadata.get("page_number") or metadata.get("pageIndex")
    section = metadata.get("section") or metadata.get("section_title") or metadata.get("sectionTitle")
    chunk_id = metadata.get("chunk_id") or metadata.get("chunkId") or metadata.get("chunk")
    if isinstance(page, str) and page.isdigit():
        page = int(page)
    return {
        "page": page if page is not None else "unknown",
        "section": section if section else "unknown",
        "chunk_id": chunk_id if chunk_id else "unknown",
    }


def _severity_from_score(score: int) -> str:
    if score >= 70:
        return "HIGH"
    if score >= 40:
        return "MED"
    if score >= 10:
        return "LOW"
    return "LOW"


def _overall_risk_level(score: int) -> str:
    if score >= 85:
        return "CRITICAL"
    if score >= 65:
        return "HIGH"
    if score >= 40:
        return "MED"
    if score >= 15:
        return "LOW"
    return "MINIMAL"


class SecurityScreeningService:
    def __init__(self, config: Optional[ScreeningConfig] = None) -> None:
        self.config = config or ScreeningConfig.load()
        self._pii_tool = PIISensitivityTool()

    def _pii_findings(
        self,
        text: str,
        spans: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        extra_spans: Optional[List[tuple[int, int, str]]] = None,
    ) -> List[SecurityFinding]:
        findings: List[SecurityFinding] = []
        for span in spans:
            label = str(span.get("label") or "").upper()
            category = f"PII_{label}" if label else "PII"
            start = int(span.get("start", 0))
            end = int(span.get("end", 0))
            snippet, context = _masked_snippet_context(text, start, end, extra_spans=extra_spans)
            severity = "MED" if label in {"ID", "SSN"} else "LOW"
            findings.append(
                SecurityFinding(
                    finding_type="PII",
                    category=category,
                    subcategory=None,
                    severity=severity,
                    confidence=0.8,
                    location=_location_from_metadata(metadata),
                    snippet_masked=snippet,
                    context_masked=context,
                    evidence={"pii_label": span.get("label")},
                )
            )
        return findings

    def _secret_findings(
        self,
        text: str,
        matches: List[SecretMatch],
        metadata: Dict[str, Any],
        extra_spans: Optional[List[tuple[int, int, str]]] = None,
    ) -> List[SecurityFinding]:
        findings: List[SecurityFinding] = []
        for match in matches:
            snippet, context = _masked_snippet_context(
                text,
                match.start,
                match.end,
                mask_start=match.mask_start,
                mask_end=match.mask_end,
                mask_mode=match.mask_mode,
                extra_spans=extra_spans,
            )
            findings.append(
                SecurityFinding(
                    finding_type="SECRET",
                    category=match.category,
                    subcategory=None,
                    severity=match.severity,
                    confidence=match.confidence,
                    location=_location_from_metadata(metadata),
                    snippet_masked=snippet,
                    context_masked=context,
                    evidence=match.evidence,
                )
            )
        return findings

    def _private_data_findings(
        self,
        text: str,
        matches: List[PrivateDataMatch],
        severity: str,
        metadata: Dict[str, Any],
        extra_spans: Optional[List[tuple[int, int, str]]] = None,
    ) -> List[SecurityFinding]:
        findings: List[SecurityFinding] = []
        for match in matches:
            context, context_start = _context_excerpt(text, match.start, match.end)
            context = _apply_extra_masks(text, context, context_start, extra_spans)
            snippet = text[match.start:match.end]
            findings.append(
                SecurityFinding(
                    finding_type="PRIVATE_DATA",
                    category="PRIVATE_BUSINESS_DATA",
                    subcategory=match.subcategory,
                    severity=severity,
                    confidence=0.65,
                    location=_location_from_metadata(metadata),
                    snippet_masked=snippet[:120],
                    context_masked=context,
                    evidence=match.evidence,
                )
            )
        return findings

    def _classification_findings(
        self, metadata: Dict[str, Any], missing: bool, checked_fields: List[str]
    ) -> List[SecurityFinding]:
        if not missing:
            return []
        return [
            SecurityFinding(
                finding_type="CLASSIFICATION",
                category="CLASSIFICATION_MISSING",
                subcategory=None,
                severity="LOW",
                confidence=0.4,
                location=_location_from_metadata(metadata),
                snippet_masked="classification_missing",
                context_masked="classification_missing",
                evidence={"checked_fields": checked_fields},
            )
        ]

    def _overall_risk_score(
        self,
        pii_result: ToolResult,
        secret_findings: List[SecurityFinding],
        private_data_score: int,
        classification: str,
    ) -> int:
        pii_contribution = int(round(pii_result.score_0_1 * 100 * pii_result.weight))
        secret_contribution = sum(SECRET_SEVERITY_SCORES.get(f.severity, 0) for f in secret_findings)
        private_contribution = int(round(private_data_score * PRIVATE_DATA_SCORE_SCALE))
        classification_boost = CLASSIFICATION_BOOST.get(classification, 0)
        total = pii_contribution + secret_contribution + private_contribution + classification_boost
        return max(0, min(100, total))

    def screen_context(self, ctx: ScreeningContext, include_overall_score: bool = False) -> Dict[str, Any]:
        if ctx.config is None:
            ctx.config = self.config

        pii_result = self._pii_tool.run(ctx)

        secret_matches = detect_secrets(ctx.text or "")
        secret_mask_spans = [(match.mask_start, match.mask_end, match.mask_mode) for match in secret_matches]

        pii_findings = self._pii_findings(ctx.text, pii_result.evidence_spans, ctx.metadata, secret_mask_spans)
        secret_findings = self._secret_findings(ctx.text, secret_matches, ctx.metadata, secret_mask_spans)

        private_score, private_matches = detect_private_data(ctx.text or "")
        private_severity = _severity_from_score(private_score)
        private_findings = self._private_data_findings(
            ctx.text,
            private_matches,
            private_severity,
            ctx.metadata,
            secret_mask_spans,
        )

        classification, missing, _, checked_fields = resolve_classification(ctx.metadata)
        classification_findings = self._classification_findings(ctx.metadata, missing, checked_fields)

        security_findings = [
            *[finding.to_dict() for finding in pii_findings],
            *[finding.to_dict() for finding in secret_findings],
            *[finding.to_dict() for finding in private_findings],
            *[finding.to_dict() for finding in classification_findings],
        ]

        overall_risk_score = self._overall_risk_score(pii_result, secret_findings, private_score, classification)
        overall_risk_level = _overall_risk_level(overall_risk_score)

        payload: Dict[str, Any] = {
            "doc_id": ctx.doc_id,
            "document_id": ctx.doc_id,
            "results": [pii_result.to_dict()],
            "risk_level": pii_result.risk_level,
            "classification": classification,
            "overall_risk_score": overall_risk_score,
            "overall_risk_level": overall_risk_level,
            "security_findings": security_findings,
        }

        if include_overall_score:
            engine = ScreeningEngine(config=ctx.config)
            payload["overall_score_0_100"] = round(engine._blend_score([pii_result], ctx.config), 2)

        return payload

    def screen_text(
        self,
        text: str,
        *,
        doc_id: Optional[str] = None,
        doc_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        raw_bytes: Optional[bytes] = None,
        previous_version_text: Optional[str] = None,
        include_overall_score: bool = False,
    ) -> Dict[str, Any]:
        ctx = ScreeningContext(
            doc_id=doc_id,
            doc_type=doc_type or (metadata or {}).get("doc_type"),
            text=text or "",
            metadata=metadata or {},
            raw_bytes=raw_bytes,
            previous_version_text=previous_version_text,
            config=self.config,
        )
        return self.screen_context(ctx, include_overall_score=include_overall_score)

    def screen_document(
        self,
        doc_id: str,
        *,
        doc_type: Optional[str] = None,
        internet_enabled_override: Optional[bool] = None,
        extracted_payload: Optional[Any] = None,
        include_overall_score: bool = False,
    ) -> Dict[str, Any]:
        engine = ScreeningEngine(config=self.config)
        ctx = engine._build_context_from_doc(
            doc_id,
            doc_type_override=doc_type,
            internet_enabled_override=internet_enabled_override,
            extracted_payload=extracted_payload,
        )
        return self.screen_context(ctx, include_overall_score=include_overall_score)
