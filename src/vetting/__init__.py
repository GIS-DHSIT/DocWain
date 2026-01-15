"""
Lightweight vetting framework for DocWain.

This package is optional and additive; importing it does not change existing
DocWain behavior unless explicitly wired in.
"""

from .engine import VettingEngine, vet_and_attach_metadata  # noqa: F401
