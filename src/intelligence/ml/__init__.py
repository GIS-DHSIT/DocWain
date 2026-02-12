"""
DocWain Pattern Intelligence Engine (DPIE).

ML-based document understanding -- zero regex.

Modules:
    line_encoder        - Line-level feature extraction (layout + char n-gram + semantic)
    doc_classifier      - Attention-weighted document type classifier
    section_detector    - Section boundary detector using transition features
    section_kind_classifier - Prototype-based section kind classifier
    entity_recognizer   - Span embedding entity recognizer
    training_bootstrap  - Auto-generate training data from Qdrant
"""

__all__ = [
    "line_encoder",
    "doc_classifier",
    "section_detector",
    "section_kind_classifier",
    "entity_recognizer",
    "training_bootstrap",
]
