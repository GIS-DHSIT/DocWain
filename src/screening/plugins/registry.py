"""Plugin registry — auto-discovers and manages screening plugins."""

import importlib
import logging
from pathlib import Path
from typing import Dict, List

from src.screening.plugins.base import ScreeningPlugin, PluginManifest

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for screening plugins. Auto-discovers plugins in subdirectories."""

    def __init__(self):
        self._plugins: Dict[str, ScreeningPlugin] = {}

    def register(self, plugin: ScreeningPlugin):
        """Register a plugin instance."""
        manifest = plugin.get_manifest()
        self._plugins[manifest.name] = plugin
        logger.info(f"Registered screening plugin: {manifest.name} v{manifest.version}")

    def discover(self):
        """Auto-discover plugins in src/screening/plugins/ subdirectories.

        Each plugin must be in its own subdirectory with a plugin.py file
        containing a Plugin class that extends ScreeningPlugin.
        """
        plugins_dir = Path(__file__).parent
        for item in plugins_dir.iterdir():
            if item.is_dir() and not item.name.startswith("_"):
                module_name = f"src.screening.plugins.{item.name}.plugin"
                try:
                    module = importlib.import_module(module_name)
                    if hasattr(module, "Plugin"):
                        plugin_instance = module.Plugin()
                        self.register(plugin_instance)
                except ImportError as e:
                    logger.warning(f"Failed to load plugin {item.name}: {e}")
                except Exception as e:
                    logger.error(f"Error initializing plugin {item.name}: {e}")

    def get_mandatory_plugins(self) -> List[ScreeningPlugin]:
        """Return all mandatory (security) plugins."""
        return [p for p in self._plugins.values()
                if p.get_manifest().mandatory]

    def get_configured_plugins(self, enabled_names: List[str],
                                doc_type: str,
                                profile_config: dict) -> List[ScreeningPlugin]:
        """Return plugins matching the profile's configuration."""
        result = []
        for name in enabled_names:
            plugin = self._plugins.get(name)
            if plugin and plugin.can_handle(doc_type, profile_config):
                result.append(plugin)
        return result

    def get_plugin(self, name: str) -> ScreeningPlugin:
        """Get a specific plugin by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> List[dict]:
        """List all registered plugins with their manifests."""
        return [vars(p.get_manifest()) for p in self._plugins.values()]


# Module-level singleton
_registry = None


def get_registry() -> PluginRegistry:
    """Get or create the plugin registry singleton."""
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
        _registry.discover()
    return _registry
