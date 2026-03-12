from __future__ import annotations

import io
from src.utils.logging_utils import get_logger
import re
import zipfile
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Header, UploadFile
from pydantic import BaseModel, Field

from src.tools.base import ToolError, generate_correlation_id, register_tool, standard_response
from src.tools.common.grounding import build_source_record
from src.tools.common.io_limits import ALLOWED_ARCHIVES, MAX_BINARY_BYTES, enforce_limit

logger = get_logger(__name__)

router = APIRouter(prefix="/code", tags=["Tools-Code"])

class CodeFile(BaseModel):
    path: str
    content: str

class CodeDocsRequest(BaseModel):
    files: List[CodeFile] = Field(default_factory=list, description="Inline code files")

def _extract_zip(upload: UploadFile) -> List[CodeFile]:
    if upload.content_type not in ALLOWED_ARCHIVES:
        raise ToolError("Zip file required", code="unsupported_media_type")
    raw = upload.file.read(MAX_BINARY_BYTES + 1)
    enforce_limit(len(raw), MAX_BINARY_BYTES, "archive")
    buffer = io.BytesIO(raw)
    files: List[CodeFile] = []
    with zipfile.ZipFile(buffer) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            try:
                data = zf.read(name)
            except Exception:
                continue
            text = data.decode("utf-8", errors="ignore")
            files.append(CodeFile(path=name, content=text))
    return files

def _summarize_file(code_file: CodeFile) -> Dict[str, Any]:
    lines = code_file.content.splitlines()
    functions = len(re.findall(r"def\s+\w+\(", code_file.content))
    classes = len(re.findall(r"class\s+\w+\(", code_file.content))
    return {
        "path": code_file.path,
        "lines": len(lines),
        "functions": functions,
        "classes": classes,
    }

def _generate_docs(files: List[CodeFile]) -> Dict[str, Any]:
    summaries = [_summarize_file(f) for f in files]
    total_lines = sum(item["lines"] for item in summaries)
    module_docs = []
    for file_obj, summary in zip(files, summaries):
        module_docs.append(
            {
                "path": summary["path"],
                "summary": f"{summary['lines']} lines, {summary['functions']} functions, {summary['classes']} classes.",
                "preview": file_obj.content[:400],
            }
        )
    suggested_readme = f"Project with {len(files)} files and {total_lines} lines."
    return {
        "project_summary": suggested_readme,
        "modules": module_docs,
        "total_lines": total_lines,
    }

@register_tool("code_docs")
async def code_docs_handler(payload: Dict[str, Any], correlation_id: Optional[str] = None) -> Dict[str, Any]:
    req = CodeDocsRequest(**(payload.get("input") or {}))
    docs = _generate_docs(req.files)
    sources = [build_source_record("tool", correlation_id or "code_docs", title="code_docs")]
    return {"result": docs, "sources": sources, "grounded": True, "context_found": bool(req.files)}

@router.post("/docs")
async def docs(
    request: CodeDocsRequest,
    zip_file: UploadFile | None = File(None),
    x_correlation_id: str | None = Header(None),
):
    cid = generate_correlation_id(x_correlation_id)
    try:
        files = list(request.files)
        if zip_file:
            files.extend(_extract_zip(zip_file))
        docs = _generate_docs(files)
        sources = [build_source_record("tool", cid, title="code_docs")]
        return standard_response(
            "code_docs",
            grounded=True,
            context_found=bool(files),
            result=docs,
            sources=sources,
            correlation_id=cid,
        )
    except ToolError as exc:
        return standard_response(
            "code_docs",
            status="error",
            grounded=False,
            context_found=False,
            error=exc.as_dict(),
            correlation_id=cid,
        )
