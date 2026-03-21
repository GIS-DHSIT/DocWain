"""Deployer — GGUF export and Ollama model registration."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class Deployer:
    def __init__(self, ollama_host="http://localhost:11434"):
        self._ollama_host = ollama_host

    def _build_tag(self, model_name, is_latest=False):
        if is_latest:
            return "DocWain:latest"
        return f"DocWain:{model_name}"

    def _build_previous_tag(self):
        return "DocWain:previous"

    def plan_deployment(self, winner, all_models, keep_previous=True):
        plan = []
        if keep_previous:
            plan.append({"action": "backup", "tag": self._build_previous_tag()})
        plan.append({"action": "deploy", "tag": self._build_tag(winner, is_latest=True), "model": winner})
        for m in all_models:
            if m != winner:
                plan.append({"action": "deploy", "tag": self._build_tag(m, is_latest=False), "model": m})
        return plan

    def export_gguf(self, merged_dir):
        from src.finetune.docwain_finetune import _export_gguf
        return _export_gguf(merged_dir)

    def register_ollama(self, gguf_path, tag):
        sha256 = hashlib.sha256()
        with open(gguf_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        digest = f"sha256:{sha256.hexdigest()}"
        resp = httpx.put(f"{self._ollama_host}/api/blobs/{digest}", content=open(gguf_path, "rb"), timeout=600.0)
        resp.raise_for_status()
        modelfile = f"FROM @{digest}\n"
        resp = httpx.post(f"{self._ollama_host}/api/create", json={"model": tag, "modelfile": modelfile, "stream": False}, timeout=300.0)
        resp.raise_for_status()
        return {"tag": tag, "digest": digest, "status": "registered"}

    def deploy_all(self, plan, artifact_dirs):
        results = []
        for step in plan:
            if step["action"] == "backup":
                try:
                    httpx.post(f"{self._ollama_host}/api/copy", json={"source": "DocWain:latest", "destination": step["tag"]}, timeout=120.0)
                    results.append({"tag": step["tag"], "status": "backed_up"})
                except Exception as e:
                    logger.warning("Backup failed: %s", e)
                    results.append({"tag": step["tag"], "status": "skip_no_previous"})
            elif step["action"] == "deploy":
                model = step["model"]
                merged_dir = artifact_dirs.get(model)
                if not merged_dir:
                    results.append({"tag": step["tag"], "status": "skip_no_artifact"})
                    continue
                try:
                    gguf_path = self.export_gguf(merged_dir)
                    reg = self.register_ollama(gguf_path, step["tag"])
                    results.append(reg)
                except Exception as e:
                    logger.error("Deploy failed for %s: %s", step["tag"], e)
                    results.append({"tag": step["tag"], "status": "failed", "error": str(e)})
        return results
