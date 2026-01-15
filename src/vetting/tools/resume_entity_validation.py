from __future__ import annotations

from typing import Dict, List, Tuple

from ..models import VettingContext
from ..search import NullSearchClient, SearchClient
from .base import VettingTool, clamp
from .resume_vetting import ORG_RE


class ResumeEntityValidationTool(VettingTool):
    name = "resume_entity_validation"
    category = "Resume Vetting"
    default_weight = 0.10
    requires_internet = True
    supported_doc_types = ["RESUME", "CV"]
    tool_version = "1.0"

    def _is_resume(self, ctx: VettingContext) -> bool:
        doc_type = (ctx.doc_type or ctx.metadata.get("doc_type") or "").lower()
        if "resume" in doc_type or "cv" in doc_type:
            return True
        text = (ctx.text or "").lower()
        keywords = {"experience", "education", "skills", "objective", "summary"}
        hits = sum(1 for kw in keywords if kw in text)
        return hits >= 2

    def _extract_entities(self, text: str) -> Tuple[List[str], List[str]]:
        companies: List[str] = []
        schools: List[str] = []

        for match in ORG_RE.finditer(text):
            entity = match.group(1).strip()
            if any(tok in entity.lower() for tok in ["university", "college", "school", "institute"]):
                schools.append(entity)
            else:
                companies.append(entity)

        companies = list(dict.fromkeys(companies))
        schools = list(dict.fromkeys(schools))
        return companies, schools

    def _verify_entity(
        self, entity: str, search_client: SearchClient, internet_enabled: bool, entity_type: str
    ) -> Dict[str, object]:
        if not internet_enabled or isinstance(search_client, NullSearchClient):
            return {
                "name": entity,
                "type": entity_type,
                "status": "UNCERTAIN",
                "evidence": [],
                "reason": "internet_disabled",
            }

        hits = search_client.search(f"{entity} official site", k=3) or []
        normalized = entity.lower().replace(",", "")
        status = "NOT_FOUND"
        evidence = []
        for hit in hits:
            title = (hit.title or "").lower()
            url = (hit.url or "").lower()
            if normalized[:48] in title or normalized[:48] in url:
                status = "LIKELY_REAL"
                evidence.append({"title": hit.title, "url": hit.url})
                break
        if status == "NOT_FOUND" and hits:
            status = "UNCERTAIN"
            evidence = [{"title": hit.title, "url": hit.url} for hit in hits[:2]]

        return {"name": entity, "type": entity_type, "status": status, "evidence": evidence}

    def run(self, ctx: VettingContext):
        if not self._is_resume(ctx):
            return self.result(
                ctx,
                0.0,
                ["Document does not appear to be a resume/CV; entity existence validation skipped."],
                raw_features={"entities": [], "resume_detected": False},
                actions=["tag"],
            )

        companies, schools = self._extract_entities(ctx.text)
        search_client = ctx.search_client or NullSearchClient()
        internet_enabled = bool(ctx.config and ctx.config.internet_enabled)

        entities = []
        for company in companies:
            entities.append(self._verify_entity(company, search_client, internet_enabled, "company"))
        for school in schools:
            entities.append(self._verify_entity(school, search_client, internet_enabled, "school"))

        total_entities = max(len(entities), 1)
        not_found = [e for e in entities if e["status"] == "NOT_FOUND"]
        uncertain = [e for e in entities if e["status"] == "UNCERTAIN"]

        score = clamp(len(not_found) / total_entities)
        reasons: List[str] = []
        if not_found:
            reasons.append(f"Entities not found in open-web existence checks: {', '.join(e['name'] for e in not_found)}.")
        if uncertain and not internet_enabled:
            reasons.append("Internet validation disabled; entity existence marked as UNCERTAIN.")
        if not reasons:
            reasons.append("Entities appear legitimate based on existence-only checks (no employment/education verification).")

        raw_features = {
            "entities": entities,
            "resume_detected": True,
            "internet_enabled": internet_enabled,
        }
        return self.result(ctx, score, reasons, raw_features=raw_features, actions=["tag"])
