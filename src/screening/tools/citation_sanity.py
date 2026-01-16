from __future__ import annotations

import re
from typing import List

from ..models import ScreeningContext
from .base import ScreeningTool, clamp


DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
URL_RE = re.compile(r"https?://[^\s)>\]]+")


class CitationSanityTool(ScreeningTool):
    name = "citation_sanity"
    category = "Information Quality"
    default_weight = 0.06
    tool_version = "1.0"

    def run(self, ctx: ScreeningContext):
        text = ctx.text
        dois = DOI_RE.findall(text)
        urls = URL_RE.findall(text)

        invalid_doi_mentions = []
        if "doi" in text.lower() and not dois:
            invalid_doi_mentions.append("Mention of DOI with no valid identifier found.")

        malformed_urls = [u for u in urls if " " in u or ".." in u]

        reasons: List[str] = []
        score = 0.0
        actions: List[str] = ["tag"]

        if invalid_doi_mentions:
            reasons.extend(invalid_doi_mentions)
            score += 0.3
        if malformed_urls:
            reasons.append("Malformed URLs detected.")
            score += min(0.7, 0.1 * len(malformed_urls))
            actions.append("warn")

        if not reasons:
            reasons.append("Citations and URLs look well-formed.")

        raw_features = {
            "doi_count": len(dois),
            "url_count": len(urls),
            "malformed_urls": malformed_urls,
        }

        return self.result(ctx, clamp(score), reasons, raw_features=raw_features, actions=actions)
