from __future__ import annotations

import hashlib
from difflib import SequenceMatcher
from typing import Optional

from ..models import ScreeningContext
from .base import ScreeningTool


class IntegrityHashTool(ScreeningTool):
    name = "integrity_hash"
    category = "Integrity & Provenance"
    default_weight = 0.07
    tool_version = "1.0"

    def _compute_hash(self, raw_bytes: Optional[bytes], text: str) -> tuple[str, str]:
        payload = raw_bytes if raw_bytes is not None else text.encode("utf-8", errors="ignore")
        digest = hashlib.sha256(payload).hexdigest()
        method = "raw_bytes" if raw_bytes is not None else "text_bytes"
        return digest, method

    def _similarity(self, current: str, previous: Optional[str]) -> float:
        if not previous:
            return 1.0
        return SequenceMatcher(None, previous.strip(), current.strip()).ratio()

    def run(self, ctx: ScreeningContext):
        digest, method = self._compute_hash(ctx.raw_bytes, ctx.text)
        prev_text = ctx.previous_version_text or ctx.metadata.get("previous_version_text") or ctx.metadata.get("previous_text")
        similarity = self._similarity(ctx.text, prev_text)

        reasons = []
        score = 0.0

        expected_hash = ctx.metadata.get("expected_sha256") or ctx.metadata.get("sha256")
        if expected_hash and expected_hash != digest:
            reasons.append("Content hash differs from expected metadata hash.")
            score = 0.75

        if similarity < 0.8 and prev_text:
            reasons.append(f"Document changed from previous version (similarity {similarity:.2f}).")
            score = max(score, 1 - similarity)

        if not reasons:
            reasons.append("Hash captured for provenance.")

        raw_features = {
            "sha256": digest,
            "hash_method": method,
            "text_length": len(ctx.text or ""),
            "previous_similarity": similarity,
        }

        return self.result(ctx, score, reasons, raw_features=raw_features, actions=["tag"])
