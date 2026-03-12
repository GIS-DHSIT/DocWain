"""Profile-level domain aggregation from document collection signals.

Aggregates per-document ``document_domain`` values across all documents
in a profile to compute a single profile-level domain tag.
"""

from src.utils.logging_utils import get_logger
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = get_logger(__name__)

def _get_mongo_db():
    """Lazy import to avoid circular imports."""
    from src.api.dataHandler import db
    return db

@dataclass
class ProfileDomainResult:
    """Result of profile-level domain aggregation."""
    domain: str = "generic"
    is_mixed: bool = False
    distribution: Dict[str, int] = field(default_factory=dict)
    confidence: float = 0.0
    total_documents: int = 0

def compute_profile_domain(
    subscription_id: str,
    profile_id: str,
) -> ProfileDomainResult:
    """Aggregate document_domain across all docs in a profile.

    Rules:
    - All same domain -> profile = that domain, confidence=1.0
    - >=80% same domain -> profile = majority domain, is_mixed=True
    - Otherwise -> "general", is_mixed=True
    """
    try:
        from src.api.config import Config
        db = _get_mongo_db()
        if db is None:
            return ProfileDomainResult()
        coll = db[Config.MongoDB.DOCUMENTS]

        # Query all documents in this profile that have a domain assigned
        query = {"profile_id": str(profile_id)}
        if subscription_id:
            query["subscription_id"] = str(subscription_id)

        cursor = coll.find(query, {"document_domain": 1, "_id": 0})

        distribution: Dict[str, int] = {}
        total = 0
        for doc in cursor:
            domain = doc.get("document_domain")
            if domain:
                distribution[domain] = distribution.get(domain, 0) + 1
                total += 1

        if total == 0:
            return ProfileDomainResult(total_documents=0)

        # Find majority domain
        majority_domain = max(distribution, key=distribution.get)
        majority_count = distribution[majority_domain]
        majority_ratio = majority_count / total

        threshold = 0.80
        try:
            threshold = Config.ProfileDomain.MAJORITY_THRESHOLD
        except Exception:
            pass

        if majority_ratio >= 1.0:
            # All documents same domain
            return ProfileDomainResult(
                domain=majority_domain,
                is_mixed=False,
                distribution=distribution,
                confidence=1.0,
                total_documents=total,
            )
        elif majority_ratio >= threshold:
            # Majority domain with some mixed
            return ProfileDomainResult(
                domain=majority_domain,
                is_mixed=True,
                distribution=distribution,
                confidence=round(majority_ratio, 3),
                total_documents=total,
            )
        else:
            # Too mixed -> general
            return ProfileDomainResult(
                domain="general",
                is_mixed=True,
                distribution=distribution,
                confidence=round(majority_ratio, 3),
                total_documents=total,
            )
    except Exception as exc:
        logger.warning("Failed to compute profile domain for %s/%s: %s", subscription_id, profile_id, exc)
        return ProfileDomainResult()

def _persist_profile_domain(
    subscription_id: str,
    profile_id: str,
    result: ProfileDomainResult,
) -> None:
    """Write profile domain to MongoDB profiles collection."""
    try:
        from src.api.config import Config
        db = _get_mongo_db()
        if db is None:
            return
        profiles_coll = db.get_collection("profiles")
        filt = {"profile_id": str(profile_id)}
        if subscription_id:
            filt["subscription_id"] = str(subscription_id)

        profiles_coll.update_one(filt, {"$set": {
            "profile_domain": result.domain,
            "domain_distribution": result.distribution,
            "is_mixed_domain": result.is_mixed,
            "domain_confidence": result.confidence,
            "domain_total_documents": result.total_documents,
        }}, upsert=True)
    except Exception as exc:
        logger.debug("Failed to persist profile domain: %s", exc)

def refresh_profile_domain_on_document_change(
    subscription_id: str,
    profile_id: str,
) -> None:
    """Called after extraction/deletion. Runs in a daemon thread."""
    def _run():
        try:
            result = compute_profile_domain(subscription_id, profile_id)
            _persist_profile_domain(subscription_id, profile_id, result)
            logger.info(
                "Profile domain updated: %s/%s -> %s (confidence=%.2f, docs=%d)",
                subscription_id, profile_id, result.domain,
                result.confidence, result.total_documents,
            )
        except Exception as exc:
            logger.debug("Profile domain refresh failed: %s", exc)

    t = threading.Thread(
        target=_run,
        daemon=True,
        name=f"profile-domain-{str(profile_id)[:8]}",
    )
    t.start()

def get_profile_domain(
    subscription_id: str,
    profile_id: str,
) -> Optional[str]:
    """Read cached profile_domain from MongoDB."""
    try:
        from src.api.config import Config
        db = _get_mongo_db()
        if db is None:
            return None
        profiles_coll = db.get_collection("profiles")
        filt = {"profile_id": str(profile_id)}
        if subscription_id:
            filt["subscription_id"] = str(subscription_id)
        doc = profiles_coll.find_one(filt, {"profile_domain": 1})
        return doc.get("profile_domain") if doc else None
    except Exception:
        return None
