"""
Centralized configuration management using Pydantic Settings.

This module provides type-safe, validated configuration with:
- Environment variable loading with proper prefixes
- SecretStr for sensitive values (prevents accidental logging)
- Validation at startup
- Caching for performance

Usage:
    from src.api.settings import get_settings
    settings = get_settings()
    print(settings.qdrant.url)
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class QdrantSettings(BaseSettings):
    """Qdrant vector database configuration."""

    model_config = SettingsConfigDict(env_prefix="QDRANT_")

    url: str = Field(
        default="https://89f776c3-76fb-493f-8509-c583d9579329.europe-west3-0.gcp.cloud.qdrant.io",
        description="Qdrant server URL",
    )
    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Qdrant API key",
    )
    timeout: int = Field(default=120, description="Request timeout in seconds")

class MongoDBSettings(BaseSettings):
    """MongoDB configuration."""

    model_config = SettingsConfigDict(env_prefix="MONGODB_")

    uri: SecretStr = Field(
        default=SecretStr(""),
        description="MongoDB connection URI",
    )
    fallback_uri: str = Field(
        default="mongodb://localhost:27017",
        description="Fallback URI when primary is unavailable",
    )
    db: str = Field(default="docwain", description="Database name")
    documents_collection: str = Field(default="documents", description="Documents collection name")
    profiles_collection: str = Field(default="profiles", description="Profiles collection name")
    subscriptions_collection: str = Field(default="subscriptions", description="Subscriptions collection name")

class RedisSettings(BaseSettings):
    """Redis cache configuration."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    host: str = Field(
        default="docwain-rediscache.redis.cache.windows.net",
        description="Redis host",
    )
    port: int = Field(default=6380, description="Redis port")
    password: SecretStr = Field(
        default=SecretStr(""),
        description="Redis password",
    )
    ssl: bool = Field(default=True, description="Enable SSL/TLS")
    db: int = Field(default=0, description="Redis database number")

class OllamaSettings(BaseSettings):
    """Ollama LLM configuration."""

    model_config = SettingsConfigDict(env_prefix="OLLAMA_")

    host: str = Field(default="http://localhost:11434", description="Ollama API host")
    default_model: str = Field(default="llama3.2", description="Default model name")
    embedding_model: str = Field(default="bge-m3", description="Embedding model name")

class GeminiSettings(BaseSettings):
    """Google Gemini API configuration."""

    model_config = SettingsConfigDict(env_prefix="GEMINI_")

    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Gemini API key",
    )
    model_name: str = Field(default="gemini-2.5-flash", description="Default Gemini model")

class TeamsSettings(BaseSettings):
    """Microsoft Teams bot configuration."""

    model_config = SettingsConfigDict(env_prefix="TEAMS_")

    shared_secret: SecretStr = Field(default=SecretStr(""), description="Teams shared secret")
    signature_enabled: bool = Field(default=False, description="Enable signature verification")
    default_profile: str = Field(default="default", description="Default profile ID")
    default_subscription: str = Field(
        default="15e0c724-4de0-492e-9861-9e637b3f9076",
        description="Default subscription ID",
    )

class ObservabilitySettings(BaseSettings):
    """Logging and observability configuration."""

    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY_")

    log_level: str = Field(default="INFO", description="Root log level")
    json_logging: bool = Field(default=False, description="Enable JSON structured logging")
    include_correlation_id: bool = Field(default=True, description="Include correlation ID in logs")

class Settings(BaseSettings):
    """
    Main application settings.

    All settings can be overridden via environment variables.
    Nested settings use their respective prefixes (e.g., QDRANT_URL, MONGODB_URI).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application settings
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Enable debug mode")
    app_name: str = Field(default="DocWain", description="Application name")

    # Nested settings - initialized with defaults
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    mongodb: MongoDBSettings = Field(default_factory=MongoDBSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    teams: TeamsSettings = Field(default_factory=TeamsSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is a known value."""
        allowed = {"development", "staging", "production", "test"}
        if v.lower() not in allowed:
            warnings.warn(f"Unknown environment '{v}', expected one of {allowed}")
        return v.lower()

    def validate_required(self) -> list[str]:
        """
        Return list of missing required settings for production.

        Returns:
            List of missing configuration keys.
        """
        missing = []

        # Qdrant API key required for cloud deployment
        if not self.qdrant.api_key.get_secret_value():
            missing.append("QDRANT_API_KEY")

        # MongoDB URI required
        if not self.mongodb.uri.get_secret_value():
            missing.append("MONGODB_URI")

        # Redis password required for Azure Redis
        if not self.redis.password.get_secret_value():
            missing.append("REDIS_PASSWORD")

        return missing

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    def get_mongodb_uri(self) -> str:
        """Get MongoDB URI, falling back if primary is empty."""
        uri = self.mongodb.uri.get_secret_value()
        return uri if uri else self.mongodb.fallback_uri

@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    The settings are cached for performance. To reload settings,
    call `get_settings.cache_clear()` first.

    Returns:
        Singleton Settings instance.
    """
    return Settings()

def validate_settings_on_startup() -> None:
    """
    Validate settings at application startup.

    Logs warnings for missing configurations.
    """
    from src.utils.logging_utils import get_logger

    logger = get_logger(__name__)
    settings = get_settings()

    missing = settings.validate_required()
    if missing:
        logger.warning(
            "Missing required configuration: %s. Some features may be unavailable.",
            ", ".join(missing),
        )

    if settings.debug:
        logger.warning("Debug mode is enabled. Do not use in production.")

__all__ = [
    "Settings",
    "QdrantSettings",
    "MongoDBSettings",
    "RedisSettings",
    "OllamaSettings",
    "GeminiSettings",
    "TeamsSettings",
    "ObservabilitySettings",
    "get_settings",
    "validate_settings_on_startup",
]
