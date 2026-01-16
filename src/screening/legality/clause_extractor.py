from __future__ import annotations

import re
from typing import List, Optional

from .models import Clause

HEADING_RE = re.compile(
    r"^(?:(?:section|article)\s+\d+|[0-9]+(?:\.[0-9]+)*[.)]?|[A-Z][A-Za-z\s\-]{2,80})$",
    re.IGNORECASE,
)


class ClauseExtractor:
    """Deterministic clause segmentation with lightweight heuristics."""

    def __init__(self, max_clause_length: int = 1200) -> None:
        self.max_clause_length = max_clause_length

    def _looks_like_heading(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if len(stripped.split()) <= 12 and HEADING_RE.match(stripped):
            return True
        if stripped.isupper() and len(stripped.split()) <= 10:
            return True
        return False

    def _clause_type_from_heading(self, heading: str, text: str) -> Optional[str]:
        content = f"{heading} {text}".lower()
        mapping = {
            "confidential": "confidentiality",
            "governing law": "governing_law",
            "jurisdiction": "jurisdiction",
            "termination": "termination",
            "indemn": "indemnity",
            "liability": "liability",
            "ip": "ip",
            "intellectual property": "ip",
            "data": "data_protection",
            "privacy": "data_protection",
            "payment": "payment",
        }
        for needle, clause_type in mapping.items():
            if needle in content:
                return clause_type
        return None

    def _make_clause(self, idx: int, heading: str, body: str) -> Clause:
        text = body.strip()
        confidence = 80.0 if heading else 70.0
        clause_text = text[: self.max_clause_length].strip()
        clause_type = self._clause_type_from_heading(heading, clause_text)
        return Clause(
            id=f"clause_{idx}",
            heading=heading or "Untitled",
            text=clause_text,
            clause_type=clause_type,
            confidence_0_100=confidence,
        )

    def extract(self, text: str) -> List[Clause]:
        if not text:
            return []

        clauses: List[Clause] = []
        lines = text.splitlines()
        buffer: List[str] = []
        current_heading: Optional[str] = None
        clause_idx = 1

        for line in lines:
            if self._looks_like_heading(line):
                if buffer or current_heading:
                    clauses.append(self._make_clause(clause_idx, current_heading or line.strip(), "\n".join(buffer)))
                    clause_idx += 1
                    buffer = []
                current_heading = line.strip()
                continue
            buffer.append(line)

        if buffer or current_heading:
            clauses.append(self._make_clause(clause_idx, current_heading or "General", "\n".join(buffer)))

        if not clauses:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for idx, para in enumerate(paragraphs, start=1):
                clauses.append(self._make_clause(idx, para.split(".")[0][:40], para))

        return clauses
