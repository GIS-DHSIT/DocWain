import json
import logging
import os
import platform
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are DocWain (Document Wise AI Node), a document-based AI assistant.

CRITICAL BEHAVIOR RULE (HIGHEST PRIORITY):

You MUST NOT introduce yourself or describe your role unless the user's query is explicitly about:
- your identity
- your role
- your purpose
- how you work
- privacy or data handling

Persona or introduction text MUST be conditionally generated, NOT included by default.

INTENT CLASSIFICATION (MANDATORY)

Before generating a response, analyze the user query and classify it into ONE category:

A) META / PERSONA INTENT
   Examples:
   - "Who are you?"
   - "What is DocWain?"
   - "About you"
   - "What can you do?"
   - "How do you work?"

B) DOCUMENT / INFORMATION INTENT
   Examples:
   - "What is the total invoice on Lenovo laptop?"
   - "Summarize this contract"
   - "Compare invoices"
   - "List candidates"

RESPONSE RULES BY INTENT

IF intent == META / PERSONA:
- Respond ONLY with persona/system information
- DO NOT retrieve from documents
- DO NOT cite sources
- DO NOT mix document data into persona
- Use a concise, neutral introduction

Allowed persona response template:
"I’m DocWain — a document-based AI assistant. I help you understand and analyze
information strictly from the documents you provide."

IF intent == DOCUMENT / INFORMATION:
- DO NOT include:
  - self-introduction
  - persona description
  - privacy statements
  - product marketing language
- Start response DIRECTLY with the answer
- Use ONLY retrieved document context; do not hallucinate values
- Do not expose internal IDs, chunk references, hashes, or system metadata
- Cite sources in user-safe format
- If information is missing, say so clearly

AGGREGATION / SUMMARY / CALCULATION RULES (MANDATORY)

If the intent involves totals, aggregation, summaries, or calculations:
- Retrieve all relevant chunks from the same document or multiple documents in the same profile
- Do not rely on semantic similarity alone; use lexical/numeric matches too
- Normalize the evidence into structured data (lists, tables, numeric fields) before answering
- Perform calculations explicitly and show the math when numbers are present

ABSOLUTE PROHIBITION:
- Never prepend persona text to document answers
- Never mix META content with DOCUMENT answers

DOCUMENT ANSWER STRUCTURE (STRICT)

Document-based answers MUST follow this structure ONLY:

1) Direct answer (1–2 lines)
2) Supporting points (optional bullets; may include compact structured data)
3) Citations (file_name, section, page)

NO additional sections allowed.

FAIL-SAFE BEHAVIOR

If the exact answer is missing in the documents:
- Provide any partial or computable information you do have
- State what is missing and what was searched
- Suggest what document or section might be needed
- Do NOT respond with "Not found" when partial/computable information exists

DO NOT:
- Speculate
- Generalize
- Add persona explanations

VIOLATION POLICY

If persona text appears in a DOCUMENT / INFORMATION response,
this is a critical failure.

The system must treat persona generation as a gated capability,
not a default behavior."""

DEFAULT_PARAMS = {"temperature": 0.2, "top_p": 0.9, "repeat_penalty": 1.1}


class OllamaPublisher:
    """
    Package a fine-tuned adapter into an Ollama model via Modelfile and register it locally.

    The publisher is intentionally defensive: if Ollama is unavailable or the adapter folder
    is missing, it returns a structured status without failing the training pipeline.
    """

    def __init__(
        self,
        base_model: str,
        run_dir: str,
        model_name: str,
        adapter_dir: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        params: Optional[Dict[str, float]] = None,
        latest_alias: Optional[str] = None,
        config_hash: Optional[str] = None,
        dataset_manifest_path: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        self.base_model = base_model
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.latest_alias = latest_alias
        self.adapter_dir = Path(adapter_dir) if adapter_dir else None
        self.system_prompt = textwrap.dedent(system_prompt or DEFAULT_SYSTEM_PROMPT).strip()
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.config_hash = config_hash
        self.dataset_manifest_path = Path(dataset_manifest_path) if dataset_manifest_path else None
        self.run_id = run_id or self.run_dir.name
        self.modelfile_path = self.run_dir / "Modelfile"
        self.env = os.environ.copy()
        self.publish_record: Dict[str, object] = {}
        self._artifact: Optional[Dict[str, object]] = None

    @staticmethod
    def _run_command(cmd: list[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
        try:
            proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
            return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
        except FileNotFoundError as exc:
            return 127, "", str(exc)

    def check_ollama_installed(self) -> Tuple[bool, str]:
        code, out, err = self._run_command(["ollama", "--version"], env=self.env)
        if code == 0:
            logger.info("Ollama detected: %s", out or err)
            return True, out or err
        logger.warning("Ollama CLI not available: %s %s", out, err)
        return False, err or out or "ollama CLI not found"

    def check_ollama_running(self) -> Tuple[bool, str]:
        code, out, err = self._run_command(["ollama", "list"], env=self.env)
        if code == 0:
            return True, out
        msg = err or out or "ollama list failed (is ollama serve running?)"
        logger.warning("Ollama daemon check failed: %s", msg)
        return False, msg

    def _resolve_artifact(self) -> Optional[Dict[str, object]]:
        if self._artifact:
            return self._artifact

        if self.adapter_dir and self.adapter_dir.exists():
            safetensors = list(self.adapter_dir.rglob("*.safetensors"))
            if safetensors:
                self._artifact = {"type": "adapter", "path": self.adapter_dir.resolve()}
                return self._artifact

        gguf = next(self.run_dir.rglob("*.gguf"), None)
        if gguf:
            self._artifact = {"type": "gguf", "path": gguf.resolve()}
            return self._artifact

        return None

    def write_modelfile(self) -> Path:
        artifact = self._resolve_artifact()
        if not artifact:
            raise FileNotFoundError("No adapter (.safetensors) directory or GGUF artifact found for Ollama packaging.")

        lines = []
        if artifact["type"] == "adapter":
            lines.append(f"FROM {self.base_model}")
            lines.append(f"ADAPTER {artifact['path']}")
        else:
            lines.append(f"FROM {artifact['path']}")

        lines.append(f"SYSTEM {json.dumps(self.system_prompt)}")
        for key, value in self.params.items():
            lines.append(f"PARAMETER {key} {value}")

        content = "\n".join(lines) + "\n"
        self.modelfile_path.write_text(content)
        logger.info("Wrote Modelfile for %s at %s", self.model_name, self.modelfile_path)
        return self.modelfile_path

    def ollama_create(self, name: str, modelfile: Optional[Path] = None) -> Tuple[bool, str]:
        path = modelfile or self.modelfile_path
        code, out, err = self._run_command(["ollama", "create", name, "-f", str(path)], cwd=self.run_dir, env=self.env)
        ok = code == 0
        if not ok:
            logger.warning("ollama create %s failed: %s | %s", name, out, err)
        return ok, out or err

    def ollama_show_verify(self, name: Optional[str] = None) -> Tuple[bool, str]:
        target = name or self.model_name
        code, out, err = self._run_command(["ollama", "show", target], env=self.env)
        ok = code == 0
        if not ok:
            logger.warning("ollama show %s failed: %s | %s", target, out, err)
        return ok, out or err

    def ollama_run_smoke(self, name: Optional[str] = None) -> Tuple[bool, str]:
        target = name or self.model_name
        prompt = 'Say "ready" and summarize what you can do.'
        code, out, err = self._run_command(
            ["ollama", "run", target, prompt],
            env=self.env,
        )
        ok = code == 0
        if not ok:
            logger.warning("ollama run %s smoke test failed: %s | %s", target, out, err)
        return ok, out or err

    def optional_push_private(self, name: str) -> Dict[str, object]:
        push_enabled = os.getenv("OLLAMA_PUSH", "").lower() in {"1", "true", "yes"}
        if not push_enabled:
            return {"skipped": True, "reason": "OLLAMA_PUSH not set"}

        target_repo = os.getenv("OLLAMA_REPO") or name
        private_flag = os.getenv("OLLAMA_PRIVATE", "").lower() in {"1", "true", "yes"}
        cmd = ["ollama", "push", target_repo]
        if private_flag:
            cmd.append("--private")

        code, out, err = self._run_command(cmd, env=self.env)
        ok = code == 0
        if not ok and private_flag:
            fallback = self._run_command(["ollama", "push", target_repo], env=self.env)
            code, out, err = fallback
            ok = code == 0

        if not ok:
            logger.warning("ollama push %s failed: %s | %s", target_repo, out, err)
            return {
                "skipped": False,
                "ok": False,
                "target": target_repo,
                "error": err or out,
                "private_requested": private_flag,
            }
        return {"skipped": False, "ok": True, "target": target_repo, "output": out, "private_requested": private_flag}

    def write_publish_artifacts(self, status: str, notes: Optional[list[str]] = None):
        record = {
            "status": status,
            "ollama_model_name": self.model_name,
            "model_name": self.model_name,
            "latest_alias": self.latest_alias,
            "base_model": self.base_model,
            "adapter_path": str(self.adapter_dir) if self.adapter_dir else None,
            "artifact_type": self._artifact["type"] if self._artifact else None,
            "artifact_path": str(self._artifact["path"]) if self._artifact else None,
            "Modelfile_path": str(self.modelfile_path),
            "modelfile": str(self.modelfile_path),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "run_dir": str(self.run_dir),
            "config_hash": self.config_hash,
            "run_id": self.run_id,
            "dataset_manifest_path": str(self.dataset_manifest_path) if self.dataset_manifest_path else None,
            "platform": platform.system(),
            "notes": notes or [],
        }
        self.publish_record = record
        path = self.run_dir / "ollama_publish.json"
        path.write_text(json.dumps(record, indent=2))
        self._write_replay_scripts()
        logger.info("Recorded Ollama publish metadata at %s", path)
        return record

    def _write_replay_scripts(self):
        scripts = {
            "publish_to_ollama.sh": self._shell_script(),
            "publish_to_ollama.ps1": self._powershell_script(),
        }
        for name, content in scripts.items():
            path = self.run_dir / name
            path.write_text(content)
            try:
                path.chmod(0o755)
            except Exception:
                pass

    def _shell_script(self) -> str:
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f'MODEL_NAME="{self.model_name}"',
            f'MODELFILE="{self.modelfile_path}"',
            "ollama create \"$MODEL_NAME\" -f \"$MODELFILE\"",
            "ollama show \"$MODEL_NAME\"",
        ]
        if self.latest_alias:
            lines.append(f'LATEST_NAME="{self.latest_alias}"')
            lines.append("ollama create \"$LATEST_NAME\" -f \"$MODELFILE\"")
        lines.extend(
            [
                'PUSH_FLAG="${OLLAMA_PUSH,,}"',
                'if [[ "$PUSH_FLAG" == "1" || "$PUSH_FLAG" == "true" || "$PUSH_FLAG" == "yes" ]]; then',
                '  TARGET="${OLLAMA_REPO:-$MODEL_NAME}"',
                '  PRIVATE_FLAG="${OLLAMA_PRIVATE,,}"',
                '  if [[ "$PRIVATE_FLAG" == "1" || "$PRIVATE_FLAG" == "true" || "$PRIVATE_FLAG" == "yes" ]]; then',
                '    ollama push "$TARGET" --private',
                "  else",
                '    ollama push "$TARGET"',
                "  fi",
                "fi",
                "",
            ]
        )
        return "\n".join(lines)

    def _powershell_script(self) -> str:
        lines = [
            "#!/usr/bin/env pwsh",
            "$ErrorActionPreference = \"Stop\"",
            f'$ModelName = "{self.model_name}"',
            f'$Modelfile = "{self.modelfile_path}"',
            'ollama create $ModelName -f $Modelfile',
            'ollama show $ModelName',
        ]
        if self.latest_alias:
            lines.append(f'$LatestName = "{self.latest_alias}"')
            lines.append('ollama create $LatestName -f $Modelfile')
        lines.extend(
            [
                'if ($env:OLLAMA_PUSH -and $env:OLLAMA_PUSH.ToLower() -in @("1","true","yes")) {',
                '  $target = if ($env:OLLAMA_REPO) { $env:OLLAMA_REPO } else { $ModelName }',
                '  if ($env:OLLAMA_PRIVATE -and $env:OLLAMA_PRIVATE.ToLower() -in @("1","true","yes")) {',
                '    ollama push $target --private',
                "  } else {",
                '    ollama push $target',
                "  }",
                "}",
                "",
            ]
        )
        return "\n".join(lines)

    def publish(self, smoke_test: bool = False) -> Dict[str, object]:
        notes: list[str] = []
        ok, msg = self.check_ollama_installed()
        if not ok:
            notes.append(msg)
            notes.append("Install Ollama CLI from https://ollama.com and ensure it is on PATH.")
            return self.write_publish_artifacts("skipped", notes)

        running, run_msg = self.check_ollama_running()
        if not running:
            notes.append(run_msg)
            notes.append("Ensure `ollama serve` is running and accessible.")
            return self.write_publish_artifacts("skipped", notes)

        try:
            self.write_modelfile()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to write Modelfile: %s", exc)
            notes.append(str(exc))
            return self.write_publish_artifacts("failed", notes)

        create_ok, create_out = self.ollama_create(self.model_name)
        notes.append(create_out)
        if not create_ok:
            return self.write_publish_artifacts("failed", notes)

        if self.latest_alias:
            alias_ok, alias_out = self.ollama_create(self.latest_alias)
            notes.append(alias_out)
            if not alias_ok:
                notes.append("Latest alias creation failed.")

        show_ok, show_out = self.ollama_show_verify(self.model_name)
        notes.append(show_out)

        if smoke_test and show_ok:
            run_ok, run_out = self.ollama_run_smoke(self.model_name)
            notes.append(run_out)
            show_ok = show_ok and run_ok

        push_result = self.optional_push_private(self.model_name)
        if not push_result.get("skipped"):
            notes.append(json.dumps(push_result, default=str))

        status = "success" if create_ok and show_ok else "partial" if create_ok else "failed"
        if not push_result.get("skipped") and push_result.get("ok") is False:
            status = "partial"
        return self.write_publish_artifacts(status, notes)
