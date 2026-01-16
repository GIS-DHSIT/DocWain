from __future__ import annotations

from typing import List
from urllib.parse import urlparse

from .models import CertificationEvidence, CertificationItem, EvidenceSource
from .validators import CachedSearchClient


class CertificationVerifier:
    """Lightweight certification verification using open-web evidence."""

    def __init__(self, search: CachedSearchClient) -> None:
        self.search = search

    def _source_list(self, hits) -> List[EvidenceSource]:
        sources: List[EvidenceSource] = []
        for hit in hits or []:
            sources.append(
                EvidenceSource(
                    title=hit.title or "",
                    url=hit.url or "",
                    snippet=hit.snippet or "",
                    source=getattr(hit, "source", None),
                    score=getattr(hit, "score", None),
                )
            )
        return sources

    def verify(self, item: CertificationItem) -> CertificationEvidence:
        sources: List[EvidenceSource] = []
        notes: List[str] = []
        issuer_verified = False
        cert_exists = False
        credential_verifiable = bool(item.verification_url or item.credential_id)
        credential_verified = None
        confidence = 0.0

        normalized_name = (item.name or "").strip()
        issuer_name = (item.issuer or "").strip()

        if not normalized_name:
            notes.append("Certification name missing; unable to verify.")
            return CertificationEvidence(
                certification=item,
                exists=False,
                issuer_verified=False,
                credential_verifiable=credential_verifiable,
                credential_verified=None,
                confidence_0_100=0.0,
                sources=sources,
                notes=notes,
                status="unverified",
            )

        if not self.search.internet_enabled:
            notes.append("Internet validation disabled; certification verification skipped.")

        # Stage 1: issuer existence
        if issuer_name:
            issuer_hits = self.search.search(f"{issuer_name} official certification", k=3)
            issuer_verified = bool(issuer_hits)
            sources.extend(self._source_list(issuer_hits[:2]))
            confidence += 25.0 if issuer_verified else 10.0
        else:
            notes.append("No issuer provided for certification.")

        # Stage 2: certification existence
        cert_hits = self.search.search(f"{normalized_name} {issuer_name} certification", k=4)
        if not cert_hits:
            cert_hits = self.search.search(f"{normalized_name} credential", k=3)
        cert_exists = bool(cert_hits)
        sources.extend(self._source_list(cert_hits[:2]))
        confidence += 35.0 if cert_exists else 10.0

        # Stage 3: credential verification (non-invasive)
        if item.verification_url:
            parsed = urlparse(item.verification_url)
            domain = parsed.netloc or parsed.path
            verify_hits = self.search.search(f"site:{domain} {normalized_name} verification", k=2)
            verify_hits = verify_hits or self.search.search(f"{item.verification_url} {issuer_name}", k=2)
            if verify_hits:
                credential_verified = True
                sources.extend(self._source_list(verify_hits[:2]))
                confidence += 20.0
            else:
                credential_verified = False
                notes.append("Verification URL found but no supporting evidence in search results.")
                confidence += 5.0
        elif item.credential_id:
            query = f"{issuer_name} verify credential {item.credential_id}".strip()
            id_hits = self.search.search(query, k=2)
            if not id_hits:
                id_hits = self.search.search(f'"{item.credential_id}" {normalized_name}', k=2)
            if id_hits:
                credential_verified = True
                sources.extend(self._source_list(id_hits))
                confidence += 20.0
            else:
                credential_verified = False
                notes.append("Credential ID present but no open-web verification found (treated as unverified, not fraudulent).")
                confidence += 5.0

        status = "verified"
        if not cert_exists and not issuer_verified:
            status = "unverified"
        elif not credential_verifiable:
            status = "likely_valid" if cert_exists or issuer_verified else "unverified"
        elif credential_verified is False:
            status = "suspicious" if confidence < 40 else "unverified"
        elif credential_verified is None:
            status = "unverified"
        else:
            status = "verified" if confidence >= 70 else "likely_valid"

        confidence = max(0.0, min(confidence, 100.0))

        return CertificationEvidence(
            certification=item,
            exists=cert_exists,
            issuer_verified=issuer_verified,
            credential_verifiable=credential_verifiable,
            credential_verified=credential_verified,
            confidence_0_100=confidence,
            sources=sources,
            notes=notes,
            status=status,
        )
