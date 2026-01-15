from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ..models import VettingContext
from ..search import NullSearchClient, SearchClient
from .base import VettingTool, clamp, tokenize_words
from .pii_sensitivity import EMAIL_RE

ORG_RE = re.compile(
    r"\b([A-Z][A-Za-z&.\-]+(?:\s+[A-Z][A-Za-z&.\-]+)*(?:\s+(?:Inc\.?|LLC|Ltd\.?|Corporation|Corp\.|University|College|School|Institute|Labs)))\b"
)
FREE_EMAIL_DOMAINS = {"gmail", "outlook", "hotmail", "yahoo", "protonmail"}
RANGE_RE = re.compile(r"\b(19\d{2}|20\d{2})\s*[-–]\s*(19\d{2}|20\d{2}|present)\b", re.IGNORECASE)


class ResumeVettingTool(VettingTool):
    name = "resume_vetting"
    category = "Resume Vetting"
    default_weight = 0.12
    tool_version = "1.0"
    requires_internet = True
    supported_doc_types = ["RESUME", "CV"]

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, object]] = {}

    def _is_resume(self, ctx: VettingContext) -> bool:
        doc_type = (ctx.doc_type or ctx.metadata.get("doc_type") or "").lower()
        if "resume" in doc_type or "cv" in doc_type:
            return True
        keywords = {"experience", "education", "skills", "objective", "summary"}
        hits = sum(1 for kw in keywords if kw in ctx.text.lower())
        return hits >= 2

    def _extract_entities(self, text: str) -> Tuple[List[str], List[str], List[str]]:
        companies: List[str] = []
        schools: List[str] = []
        titles: List[str] = []

        for match in ORG_RE.finditer(text):
            entity = match.group(1).strip()
            if any(tok in entity.lower() for tok in ["university", "college", "school", "institute"]):
                schools.append(entity)
            else:
                companies.append(entity)

        # Lightweight title detection
        titles_list = [
            "engineer",
            "manager",
            "director",
            "analyst",
            "consultant",
            "intern",
            "specialist",
            "developer",
        ]
        words = tokenize_words(text)
        for idx, word in enumerate(words):
            if word in titles_list:
                titles.append(" ".join(words[max(0, idx - 1) : idx + 1]))

        # Collapse duplicates while preserving order
        companies = list(dict.fromkeys(companies))
        schools = list(dict.fromkeys(schools))
        titles = list(dict.fromkeys(titles))
        return companies, schools, titles

    def _verify_entity(self, entity: str, search_client: SearchClient, internet_enabled: bool) -> Dict[str, object]:
        cached = self._cache.get(entity)
        if cached:
            return cached

        if not internet_enabled or isinstance(search_client, NullSearchClient):
            result = {"name": entity, "status": "UNCERTAIN", "evidence": [], "reason": "internet_disabled"}
            self._cache[entity] = result
            return result

        hits = search_client.search(f"{entity} official site", k=3) or []
        normalized_name = entity.lower().replace(",", "")
        status = "NOT_FOUND"
        evidence = []
        for hit in hits:
            if normalized_name[:40] in hit.title.lower() or normalized_name[:40] in hit.url.lower():
                status = "LIKELY_REAL"
                evidence.append({"title": hit.title, "url": hit.url})
                break
        if status == "NOT_FOUND" and hits:
            status = "UNCERTAIN"
            evidence = [{"title": hit.title, "url": hit.url} for hit in hits[:2]]

        result = {"name": entity, "status": status, "evidence": evidence}
        self._cache[entity] = result
        return result

    def _email_domains(self, text: str) -> List[str]:
        domains: List[str] = []
        for match in EMAIL_RE.finditer(text):
            parts = match.group(0).split("@")
            if len(parts) == 2:
                domain_part = parts[1].split(".")[0]
                domains.append(domain_part.lower())
        return list(dict.fromkeys(domains))

    def _date_ranges(self, text: str) -> List[Tuple[int, int]]:
        ranges: List[Tuple[int, int]] = []
        for match in RANGE_RE.finditer(text):
            try:
                start = int(match.group(1))
            except ValueError:
                continue
            end_raw = match.group(2).lower()
            if end_raw == "present":
                end = 9999
            else:
                try:
                    end = int(end_raw)
                except ValueError:
                    continue
            ranges.append((start, end))
        return ranges

    def run(self, ctx: VettingContext):
        if not self._is_resume(ctx):
            return self.result(
                ctx,
                0.0,
                ["Document does not appear to be a resume/CV; resume-specific checks skipped."],
                raw_features={"entities": [], "resume_detected": False},
                actions=["tag"],
            )

        companies, schools, titles = self._extract_entities(ctx.text)
        search_client = ctx.search_client or NullSearchClient()
        internet_enabled = bool(ctx.config and ctx.config.internet_enabled)

        entity_results = []
        for entity in companies + schools:
            entity_results.append(self._verify_entity(entity, search_client, internet_enabled))

        email_domains = self._email_domains(ctx.text)
        normalized_company_tokens = {c.lower().split()[0] for c in companies}
        suspicious_domains = [
            d for d in email_domains if d not in FREE_EMAIL_DOMAINS and d not in normalized_company_tokens
        ]

        ranges = sorted(self._date_ranges(ctx.text), key=lambda r: r[0])
        overlapping = False
        for idx in range(1, len(ranges)):
            prev_start, prev_end = ranges[idx - 1]
            start, end = ranges[idx]
            if start < prev_end and prev_end != 9999:
                overlapping = True
                break

        issues = 0
        reasons: List[str] = []
        actions: List[str] = ["tag"]

        not_found = [r for r in entity_results if r["status"] == "NOT_FOUND"]
        if not_found:
            reasons.append(f"Entities not found in web search: {', '.join(r['name'] for r in not_found)}.")
            issues += len(not_found)
            actions.append("warn")
        if suspicious_domains:
            reasons.append(f"Email domain(s) do not match entities: {', '.join(suspicious_domains)}.")
            issues += 1
            actions.append("warn")
        if overlapping:
            reasons.append("Potentially overlapping or inconsistent date ranges detected.")
            issues += 1

        if not reasons:
            reasons.append("Entities and contact signals look consistent (existence check only).")

        total_entities = max(1, len(companies) + len(schools))
        score = clamp(issues / total_entities)

        raw_features = {
            "companies": companies,
            "schools": schools,
            "titles": titles,
            "entity_results": entity_results,
            "email_domains": email_domains,
            "suspicious_domains": suspicious_domains,
            "resume_detected": True,
            "date_ranges": ranges,
        }

        return self.result(ctx, score, reasons, raw_features=raw_features, actions=actions)
