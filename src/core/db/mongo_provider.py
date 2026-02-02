import logging
import os
from typing import Iterable, List, Optional
from urllib.parse import urlparse

from pymongo import MongoClient
from pymongo.database import Database

from src.api.config import Config

logger = logging.getLogger(__name__)


class MongoProviderError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class MongoProvider:
    def __init__(self) -> None:
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
        self._db_name: Optional[str] = None

    def init(self) -> Database:
        if self._db is not None:
            return self._db

        uri_candidates = _collect_uri_candidates()
        if not uri_candidates:
            raise MongoProviderError(
                "DB_MISSING_URI",
                "Missing Mongo connection URI (MONGO_URI/MONGODB_URI/COSMOS_MONGO_URI/MONGO_CONNECTION_STRING).",
            )

        db_name = _resolve_db_name()
        last_error: Optional[Exception] = None
        for uri in uri_candidates:
            try:
                client = MongoClient(uri, serverSelectionTimeoutMS=5000)
                client.admin.command("ping")
                db = client[db_name]
                self._client = client
                self._db = db
                self._db_name = db_name
                logger.info("MongoDB connected (db=%s, uri=%s)", db_name, _mask_uri(uri))
                return db
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("MongoDB ping failed (uri=%s): %s", _mask_uri(uri), exc)

        raise MongoProviderError("DB_PING_FAILED", f"Mongo connection failed: {last_error}")

    def get_db(self) -> Database:
        if self._db is None:
            raise MongoProviderError("DB_NOT_INITIALIZED", "Mongo provider not initialized. Call init() at startup.")
        return self._db

    def get_client(self) -> MongoClient:
        if self._client is None:
            raise MongoProviderError("DB_NOT_INITIALIZED", "Mongo provider not initialized. Call init() at startup.")
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
        self._client = None
        self._db = None
        self._db_name = None


class MongoDbProxy:
    def __init__(self, provider: MongoProvider) -> None:
        self._provider = provider

    def __getattr__(self, name: str):
        db = self._provider.get_db()
        return getattr(db, name)

    def __getitem__(self, name: str):
        db = self._provider.get_db()
        return db[name]

    def __repr__(self) -> str:
        ready = self._provider._db is not None
        return f"<MongoDbProxy ready={ready}>"


mongo_provider = MongoProvider()
mongo_db_proxy = MongoDbProxy(mongo_provider)


def _collect_uri_candidates() -> List[str]:
    candidates: List[str] = []
    env_keys = [
        "MONGO_URI",
        "MONGODB_URI",
        "COSMOS_MONGO_URI",
        "MONGO_CONNECTION_STRING",
    ]
    for key in env_keys:
        value = os.getenv(key)
        if value:
            candidates.append(value)

    config_uri = getattr(Config.MongoDB, "URI", None)
    if config_uri:
        candidates.append(config_uri)

    config_fallback = getattr(Config.MongoDB, "FALLBACK_URI", None)
    if config_fallback:
        candidates.append(config_fallback)

    candidates = _dedupe(candidates)
    if _running_in_docker():
        non_srv = [uri for uri in candidates if not uri.startswith("mongodb+srv://")]
        if non_srv:
            return non_srv
        raise MongoProviderError(
            "DB_SRV_UNSUPPORTED",
            "Mongo SRV URIs are not supported in Docker. Set MONGO_URI to a non-SRV mongodb://...:10255 string.",
        )
    return candidates


def _resolve_db_name() -> str:
    name = os.getenv("MONGO_DB_NAME") or os.getenv("DB_NAME") or os.getenv("MONGODB_DB")
    if name:
        return name
    config_name = getattr(Config.MongoDB, "DB", None)
    if config_name:
        return config_name
    return "test"


def _running_in_docker() -> bool:
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "r", encoding="utf-8") as handle:
            data = handle.read()
        return "docker" in data or "containerd" in data
    except Exception:
        return False


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _mask_uri(uri: str) -> str:
    if not uri:
        return "<empty>"
    parsed = urlparse(uri)
    if parsed.scheme and parsed.hostname:
        host = parsed.hostname
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return f"{parsed.scheme}://{host}"
    if len(uri) <= 6:
        return "******"
    return f"...{uri[-6:]}"
