from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..models import ScreeningContext
from .base import ScreeningTool, clamp


class MetadataConsistencyTool(ScreeningTool):
    name = "metadata_consistency"
    category = "Integrity & Provenance"
    default_weight = 0.07
    tool_version = "1.0"

    def _parse_dt(self, value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(value, tz=timezone.utc)
            except Exception:
                return None
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except Exception:
                try:
                    return datetime.fromtimestamp(float(value), tz=timezone.utc)
                except Exception:
                    return None
        return None

    def run(self, ctx: ScreeningContext):
        metadata: Dict[str, Any] = ctx.metadata or {}
        reasons = []
        score = 0.0

        created = self._parse_dt(metadata.get("created") or metadata.get("created_at"))
        modified = self._parse_dt(metadata.get("modified") or metadata.get("updated_at"))
        author = metadata.get("author") or metadata.get("creator")
        last_modified_by = metadata.get("last_modified_by") or metadata.get("editor")

        if created and modified and modified < created:
            reasons.append("Modified timestamp predates creation.")
            score += 0.4
        if created and created > datetime.now(timezone.utc):
            reasons.append("Creation timestamp is in the future.")
            score += 0.3
        if last_modified_by and author and last_modified_by != author:
            reasons.append("Author and last modified by differ.")
            score += 0.15

        missing_fields = [field for field in ("author", "creator", "created") if not metadata.get(field)]
        if missing_fields:
            reasons.append(f"Metadata missing fields: {', '.join(sorted(set(missing_fields)))}.")
            score += 0.15

        if not reasons:
            reasons.append("Metadata timestamps and authors look consistent.")

        raw_features = {
            "created": created.isoformat() if created else None,
            "modified": modified.isoformat() if modified else None,
            "author": author,
            "last_modified_by": last_modified_by,
        }

        return self.result(ctx, clamp(score), reasons, raw_features=raw_features, actions=["tag"])
