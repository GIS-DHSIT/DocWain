"""Collect training pairs from production feedback for fine-tuning.

Queries MongoDB for user feedback (positive and negative with corrections),
then formats them as chat-template training pairs compatible with the
DocWain fine-tuning pipeline.

Usage:
    from src.finetune.feedback_collector import collect_training_pairs
    pairs = collect_training_pairs(min_pairs=50)
    # Returns list of {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
"""
from __future__ import annotations

import json
from src.utils.logging_utils import get_logger
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = get_logger(__name__)

# System prompt used during training — must match the one in training_data_generator.py
_TRAINING_SYSTEM_PROMPT = (
    "You are DocWain, an AI document intelligence assistant. "
    "Answer questions accurately using only the provided document evidence. "
    "Be specific — include exact values, names, dates, and numbers. "
    "Use markdown formatting: **bold** for key answers, tables for comparisons, "
    "bullet points for lists."
)

def _get_feedback_collection():
    """Get the MongoDB feedback collection."""
    try:
        from src.api.dataHandler import db
        if db is not None:
            return db.feedback
    except Exception as exc:
        logger.warning("Cannot connect to MongoDB feedback collection: %s", exc)
    return None

def collect_training_pairs(
    *,
    min_pairs: int = 10,
    max_pairs: int = 5000,
    days_back: int = 90,
    include_positive: bool = True,
    include_negative_with_correction: bool = True,
) -> List[Dict[str, Any]]:
    """Collect training pairs from MongoDB feedback.

    Args:
        min_pairs: Minimum number of pairs to collect (returns empty if fewer available)
        max_pairs: Maximum number of pairs to return
        days_back: How far back to look for feedback
        include_positive: Include positive feedback as training examples
        include_negative_with_correction: Include negative feedback that has a user correction

    Returns:
        List of chat-template training pairs:
        [{"messages": [system, user, assistant], "metadata": {...}}]
    """
    collection = _get_feedback_collection()
    if collection is None:
        logger.info("Feedback collection not available — returning empty")
        return []

    cutoff = datetime.now() - timedelta(days=days_back)
    pairs: List[Dict[str, Any]] = []

    # Positive feedback → high-quality training examples
    if include_positive:
        try:
            positive_cursor = collection.find(
                {"type": "positive", "timestamp": {"$gte": cutoff}},
                {"query": 1, "answer": 1, "sources": 1, "metadata": 1, "timestamp": 1},
            ).sort("timestamp", -1).limit(max_pairs)

            for doc in positive_cursor:
                query = doc.get("query", "").strip()
                answer = doc.get("answer", "").strip()
                if not query or not answer or len(answer) < 20:
                    continue

                pairs.append({
                    "messages": [
                        {"role": "system", "content": _TRAINING_SYSTEM_PROMPT},
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": answer},
                    ],
                    "metadata": {
                        "source": "feedback_positive",
                        "timestamp": str(doc.get("timestamp", "")),
                        "has_sources": bool(doc.get("sources")),
                    },
                })
        except Exception as exc:
            logger.warning("Error collecting positive feedback: %s", exc)

    # Negative feedback with corrections → learn from user corrections
    if include_negative_with_correction:
        try:
            negative_cursor = collection.find(
                {
                    "type": "negative",
                    "timestamp": {"$gte": cutoff},
                    # Only include entries where the user provided a correction/reason
                    "reason": {"$exists": True, "$ne": None, "$ne": ""},
                },
                {"query": 1, "answer": 1, "reason": 1, "metadata": 1, "timestamp": 1},
            ).sort("timestamp", -1).limit(max_pairs)

            for doc in negative_cursor:
                query = doc.get("query", "").strip()
                reason = doc.get("reason", "").strip()
                original_answer = doc.get("answer", "").strip()

                if not query or not reason:
                    continue

                # If the reason looks like a corrected answer (long enough), use it directly
                if len(reason) > 50:
                    corrected_answer = reason
                else:
                    # Short reason = feedback note, not a full correction
                    # Create a training example that incorporates the feedback
                    corrected_answer = (
                        f"[Note: Previous response was marked as incorrect. "
                        f"User feedback: {reason}]\n\n"
                        f"Let me provide a more accurate answer based on the documents."
                    )
                    # Skip short feedback — not useful for training
                    continue

                pairs.append({
                    "messages": [
                        {"role": "system", "content": _TRAINING_SYSTEM_PROMPT},
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": corrected_answer},
                    ],
                    "metadata": {
                        "source": "feedback_negative_corrected",
                        "timestamp": str(doc.get("timestamp", "")),
                        "original_answer_length": len(original_answer),
                    },
                })
        except Exception as exc:
            logger.warning("Error collecting negative feedback: %s", exc)

    if len(pairs) < min_pairs:
        logger.info(
            "Only %d feedback pairs available (min_pairs=%d) — returning empty",
            len(pairs), min_pairs,
        )
        return []

    # Cap at max_pairs
    pairs = pairs[:max_pairs]
    logger.info("Collected %d training pairs from production feedback", len(pairs))
    return pairs

def export_training_pairs(
    output_path: str = "finetune_data/feedback_training.jsonl",
    **kwargs,
) -> int:
    """Export training pairs to JSONL file for fine-tuning pipeline.

    Returns number of pairs exported.
    """
    pairs = collect_training_pairs(**kwargs)
    if not pairs:
        logger.info("No training pairs to export")
        return 0

    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    logger.info("Exported %d training pairs to %s", len(pairs), output_path)
    return len(pairs)

def get_feedback_stats(days_back: int = 90) -> Dict[str, Any]:
    """Get statistics about available feedback data."""
    collection = _get_feedback_collection()
    if collection is None:
        return {"available": False}

    cutoff = datetime.now() - timedelta(days=days_back)
    try:
        positive = collection.count_documents({"type": "positive", "timestamp": {"$gte": cutoff}})
        negative = collection.count_documents({"type": "negative", "timestamp": {"$gte": cutoff}})
        with_reason = collection.count_documents({
            "type": "negative",
            "timestamp": {"$gte": cutoff},
            "reason": {"$exists": True, "$ne": None, "$ne": ""},
        })
        return {
            "available": True,
            "days_back": days_back,
            "positive_count": positive,
            "negative_count": negative,
            "negative_with_correction": with_reason,
            "total_usable": positive + with_reason,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}
