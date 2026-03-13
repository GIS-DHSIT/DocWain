"""Backfill intelligence analysis for existing documents that have not yet
been processed by the intelligence v2 pipeline.

Usage:
    python -m scripts.backfill_intelligence [--batch-size 50] [--subscription-id SUB]
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Callable, Dict, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def backfill(
    mongodb: Any,
    analyzer: Any,
    load_extracted_fn: Callable[[str], Any],
    batch_size: int = 50,
    subscription_id: Optional[str] = None,
) -> Dict[str, int]:
    """Process documents that lack intelligence analysis.

    Args:
        mongodb: MongoDB collection handle (supports find/update_one).
        analyzer: ``DocumentAnalyzer`` instance with an ``analyze()`` method.
        load_extracted_fn: Callable that loads extracted content given a
            document_id.  Should raise ``ValueError`` when content is missing.
        batch_size: Maximum number of documents to process in one run.
        subscription_id: Optional filter to limit backfill to a single
            subscription.

    Returns:
        Dict with keys ``processed``, ``failed``, ``skipped``, ``total``.
    """
    query: Dict[str, Any] = {
        "$or": [
            {"intelligence_ready": {"$ne": True}},
            {"intelligence_ready": {"$exists": False}},
        ]
    }
    if subscription_id:
        query["subscription_id"] = subscription_id

    docs = list(mongodb.find(query).limit(batch_size))
    total = len(docs)

    processed = 0
    failed = 0
    skipped = 0

    for doc in docs:
        doc_id = doc.get("document_id", "unknown")
        sub_id = doc.get("subscription_id", "")
        profile_id = doc.get("profile_id", "")
        filename = doc.get("filename", "")

        # Load extracted content
        try:
            extracted = load_extracted_fn(doc_id)
        except (ValueError, FileNotFoundError, Exception) as exc:
            logger.warning(
                "Skipping doc=%s: could not load extracted content: %s",
                doc_id,
                exc,
            )
            skipped += 1
            continue

        if not extracted:
            logger.warning("Skipping doc=%s: extracted content is empty", doc_id)
            skipped += 1
            continue

        # Analyze
        try:
            analyzer.analyze(
                document_id=doc_id,
                extracted=extracted,
                subscription_id=sub_id,
                profile_id=profile_id,
                filename=filename,
            )
            processed += 1
            logger.info("Processed doc=%s (%d/%d)", doc_id, processed + failed + skipped, total)
        except Exception:
            logger.error(
                "Failed to analyze doc=%s; continuing", doc_id, exc_info=True
            )
            failed += 1

    stats = {
        "processed": processed,
        "failed": failed,
        "skipped": skipped,
        "total": total,
    }
    logger.info("Backfill complete: %s", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill intelligence analysis for existing documents"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Maximum documents to process (default: 50)",
    )
    parser.add_argument(
        "--subscription-id",
        type=str,
        default=None,
        help="Limit backfill to a specific subscription",
    )
    args = parser.parse_args()

    # --- wire real dependencies ---
    from pymongo import MongoClient

    from src.api.config import Config
    from src.api.content_store import load_extracted_pickle
    from src.intelligence_v2.analyzer import DocumentAnalyzer
    from src.kg.neo4j_store import Neo4jStore
    from src.llm.gateway import get_llm_gateway

    mongo_client = MongoClient(Config.MongoDB.URI)
    db = mongo_client[Config.MongoDB.DB]
    mongodb = db[Config.MongoDB.DOCUMENTS]

    llm_gateway = get_llm_gateway()
    neo4j_store = Neo4jStore()
    analyzer = DocumentAnalyzer(llm_gateway, neo4j_store, mongodb)

    stats = backfill(
        mongodb=mongodb,
        analyzer=analyzer,
        load_extracted_fn=load_extracted_pickle,
        batch_size=args.batch_size,
        subscription_id=args.subscription_id,
    )

    print(f"\nBackfill results: {stats}")
    if stats["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
