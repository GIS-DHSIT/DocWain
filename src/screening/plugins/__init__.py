"""Screening plugin system — auto-discovery, mandatory security baseline."""

from src.screening.plugins.base import ScreeningPlugin, PluginManifest, PluginResult
from src.screening.plugins.registry import get_registry

__all__ = ["ScreeningPlugin", "PluginManifest", "PluginResult", "get_registry"]
