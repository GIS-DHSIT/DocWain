from __future__ import annotations

from typing import Any, Dict, List, Tuple


class EvidenceVerifier:
    def verify(self, bundle: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        issues: List[str] = []
        for doc in bundle.get("documents", []):
            for obj in doc.get("objects", []):
                fields = obj.get("fields") or {}
                evidence_map = obj.get("evidence_map") or {}
                cleaned_fields = {}
                for key, value in fields.items():
                    evidence_key = _field_to_evidence_key(key)
                    if evidence_key and evidence_map.get(evidence_key):
                        cleaned_fields[key] = value
                    elif not evidence_key:
                        cleaned_fields[key] = value
                    else:
                        issues.append(f"missing_evidence:{key}")
                obj["fields"] = cleaned_fields
                if _has_artifact_strings(cleaned_fields):
                    issues.append("artifact_string")
        return bundle, {"issues": issues}


def _field_to_evidence_key(field: str) -> str:
    mapping = {
        "names": "person",
        "skills": "skill",
        "organizations": "organization",
        "dates": "date",
    }
    return mapping.get(field, "")


def _has_artifact_strings(fields: Dict[str, Any]) -> bool:
    for value in fields.values():
        if isinstance(value, list):
            for item in value:
                if _is_artifact(item):
                    return True
        elif isinstance(value, str):
            if _is_artifact(value):
                return True
    return False


def _is_artifact(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    compact = text.strip()
    if len(compact) >= 30 and " " not in compact:
        return True
    return False


__all__ = ["EvidenceVerifier"]
