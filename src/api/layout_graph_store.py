from __future__ import annotations

import json
from src.utils.logging_utils import get_logger
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from azure.storage.blob import ContentSettings

from src.api.config import Config
from src.storage.azure_blob_client import get_document_container_client, normalize_blob_name

logger = get_logger(__name__)

_DEFAULT_PREFIX = "layout-graphs"

def _layout_prefix() -> str:
    prefix = os.getenv("DOCWAIN_LAYOUT_PREFIX", _DEFAULT_PREFIX).strip().strip("/")
    return prefix or _DEFAULT_PREFIX

def _blob_paths(document_id: str, run_id: str) -> Dict[str, str]:
    prefix = _layout_prefix()
    base = f"{prefix}/{document_id}"
    return {
        "latest": f"{base}/latest.json",
        "versioned": f"{base}/{run_id}.json",
    }

def _write_blob(blob_name: str, payload: bytes, metadata: Dict[str, str]) -> None:
    container = get_document_container_client()
    blob_client = container.get_blob_client(normalize_blob_name(blob_name))
    blob_client.upload_blob(
        payload,
        overwrite=True,
        metadata=metadata,
        content_settings=ContentSettings(content_type="application/json"),
    )

def save_layout_graph(
    *,
    document_id: str,
    layout_graph: Dict[str, Any],
    subscription_id: Optional[str] = None,
    profile_id: Optional[str] = None,
) -> Dict[str, Any]:
    run_id = uuid.uuid4().hex
    paths = _blob_paths(document_id, run_id)
    payload = json.dumps(layout_graph, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    sha256 = __import__("hashlib").sha256(payload).hexdigest()
    metadata = {
        "docwain_artifact": "true",
        "document_id": str(document_id),
        "type": "layout_graph",
        "version": "v1",
        "sha256": sha256,
    }
    if subscription_id:
        metadata["subscription_id"] = str(subscription_id)
    if profile_id:
        metadata["profile_id"] = str(profile_id)

    try:
        _write_blob(paths["latest"], payload, metadata)
        _write_blob(paths["versioned"], payload, metadata)
        return {
            "latest_path": paths["latest"],
            "versioned_path": paths["versioned"],
            "sha256": sha256,
            "bytes": len(payload),
            "updated_at": time.time(),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("LayoutGraph blob save failed for %s: %s", document_id, exc)
        return {
            "latest_path": None,
            "versioned_path": None,
            "sha256": sha256,
            "bytes": len(payload),
            "updated_at": time.time(),
            "error": str(exc),
        }

def _local_layout_dir(document_id: str) -> Path:
    base = Path(getattr(Config.Path, "DOCUMENTS_DIR", Path.cwd()))
    return base / "layout-graphs" / str(document_id)

def save_layout_graph_local(
    *,
    document_id: str,
    layout_graph: Dict[str, Any],
) -> Dict[str, Any]:
    run_id = uuid.uuid4().hex
    base = _local_layout_dir(document_id)
    base.mkdir(parents=True, exist_ok=True)
    latest_path = base / "latest.json"
    versioned_path = base / f"{run_id}.json"
    payload = json.dumps(layout_graph, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    sha256 = __import__("hashlib").sha256(payload).hexdigest()
    latest_path.write_bytes(payload)
    versioned_path.write_bytes(payload)
    return {
        "latest_path": str(latest_path),
        "versioned_path": str(versioned_path),
        "sha256": sha256,
        "bytes": len(payload),
        "updated_at": time.time(),
    }

def load_layout_graph(document_id: str) -> Optional[Dict[str, Any]]:
    """Attempt to load latest layout graph from local disk first, then blob storage."""
    local_latest = _local_layout_dir(document_id) / "latest.json"
    if local_latest.exists():
        try:
            return json.loads(local_latest.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
    # Fallback to blob storage
    paths = _blob_paths(document_id, run_id="latest")
    blob_name = paths["latest"]
    try:
        container = get_document_container_client()
        blob_client = container.get_blob_client(normalize_blob_name(blob_name))
        payload = blob_client.download_blob().readall()
        return json.loads(payload)
    except Exception:
        return None

__all__ = ["save_layout_graph", "save_layout_graph_local", "load_layout_graph"]
