"""Base class for all screening plugins."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class PluginManifest:
    name: str
    version: str
    description: str
    category: str  # "security" | "analysis" | "compliance"
    mandatory: bool  # True = always runs regardless of profile config
    supported_doc_types: list  # ["*"] or ["pdf", "docx", ...]
    requires: list  # ["extraction"]
    outputs: list  # what this plugin produces
    model: str = ""  # model used, if any
    timeout: int = 120  # seconds


@dataclass
class PluginResult:
    plugin_name: str
    success: bool
    outputs: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    duration_ms: int = 0


class ScreeningPlugin(ABC):
    """Base class all screening plugins must implement."""

    @abstractmethod
    def get_manifest(self) -> PluginManifest:
        """Return plugin manifest describing capabilities."""

    @abstractmethod
    def can_handle(self, doc_type: str, profile_config: dict) -> bool:
        """Check if this plugin should run for this document."""

    @abstractmethod
    def execute(self, extraction_data: dict,
                document_meta: dict) -> PluginResult:
        """Run screening analysis. Returns structured result."""
