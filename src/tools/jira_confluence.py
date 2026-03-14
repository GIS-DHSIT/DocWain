"""Cloud Platform Agent — Jira, Confluence, and SharePoint integration."""
from __future__ import annotations

from src.utils.logging_utils import get_logger
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.tools.base import generate_correlation_id, register_tool, registry, standard_response
from src.tools.common.grounding import build_source_record
from src.tools.common.http_client import fetch_text
from src.tools.common.text_extract import sanitize_text

logger = get_logger(__name__)

# Use an empty prefix so paths map directly to /api/tools/<...> without duplication.
router = APIRouter(prefix="", tags=["Tools-JiraConfluence"])

class JiraIssuesRequest(BaseModel):
    issues: Optional[List[Dict[str, Any]]] = Field(default=None, description="Inline issues payload")
    url: Optional[str] = Field(default=None, description="Optional JIRA REST URL to fetch issues (GET)")
    project_key: Optional[str] = None

class ConfluencePagesRequest(BaseModel):
    pages: Optional[List[Dict[str, Any]]] = Field(default=None, description="Inline pages payload")
    url: Optional[str] = Field(default=None, description="Optional Confluence REST URL to fetch pages")

class JiraConfluenceSummaryRequest(BaseModel):
    issues: Optional[List[Dict[str, Any]]] = None
    pages: Optional[List[Dict[str, Any]]] = None
    url: Optional[str] = None

class SharePointRequest(BaseModel):
    """SharePoint document/site query."""
    items: Optional[List[Dict[str, Any]]] = Field(default=None, description="Inline SharePoint items payload")
    url: Optional[str] = Field(default=None, description="SharePoint site or document library URL")
    site_name: Optional[str] = None

async def _fetch_items(url: Optional[str]) -> List[Dict[str, Any]]:
    if not url:
        return []
    fetched = await fetch_text(url, max_bytes=250_000)
    text = sanitize_text(fetched.get("content", ""), max_chars=2000)
    return [{"id": "fetched", "title": url, "content": text}]

def _summarize_items(items: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    titles = [item.get("title") or item.get("id") or f"{label}-{idx}" for idx, item in enumerate(items)]
    summary = f"{len(items)} {label} item(s) processed."
    action_items = [f"Review {title}" for title in titles[:5]]
    return {"count": len(items), "titles": titles, "action_items": action_items, "summary": summary}

async def _fetch_sharepoint_items(url: Optional[str]) -> List[Dict[str, Any]]:
    """Fetch items from a SharePoint URL (URL-based, no Graph SDK)."""
    from src.api.config import Config
    if not getattr(Config, "CloudPlatform", None) or not Config.CloudPlatform.SHAREPOINT_ENABLED:
        return []
    if not url:
        return []
    fetched = await fetch_text(url, max_bytes=250_000)
    text = sanitize_text(fetched.get("content", ""), max_chars=2000)
    return [{"id": "sharepoint-fetched", "title": url, "content": text}]

def _summarize_sharepoint(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize SharePoint items."""
    titles = [item.get("title") or item.get("name") or f"item-{idx}" for idx, item in enumerate(items)]
    summary = f"{len(items)} SharePoint item(s) processed."
    return {"count": len(items), "titles": titles, "summary": summary}

@register_tool("jira_confluence")
async def jira_confluence_handler(payload: Dict[str, Any], correlation_id: Optional[str] = None) -> Dict[str, Any]:
    input_payload = payload.get("input") or payload
    issues = input_payload.get("issues") or []
    pages = input_payload.get("pages") or []
    result = {
        "issues": _summarize_items(issues, "issue"),
        "pages": _summarize_items(pages, "page"),
    }
    sources = [build_source_record("tool", correlation_id or "jira_confluence", title="jira_confluence")]
    return {"result": result, "sources": sources, "grounded": True, "context_found": bool(issues or pages)}

@router.post("/jira/issues")
async def jira_issues(request: JiraIssuesRequest, x_correlation_id: str | None = Header(None)):
    cid = generate_correlation_id(x_correlation_id)
    issues = request.issues or []
    if request.url:
        issues.extend(await _fetch_items(request.url))
    result = _summarize_items(issues, "issue")
    sources = [build_source_record("jira", cid, title=request.project_key or "jira")]
    return standard_response(
        "jira_confluence",
        grounded=True,
        context_found=bool(issues),
        result={"issues": result},
        sources=sources,
        warnings=[],
        correlation_id=cid,
    )

@router.post("/confluence/pages")
async def confluence_pages(request: ConfluencePagesRequest, x_correlation_id: str | None = Header(None)):
    cid = generate_correlation_id(x_correlation_id)
    pages = request.pages or []
    if request.url:
        pages.extend(await _fetch_items(request.url))
    result = _summarize_items(pages, "page")
    sources = [build_source_record("confluence", cid, title="confluence")]
    return standard_response(
        "jira_confluence",
        grounded=True,
        context_found=bool(pages),
        result={"pages": result},
        sources=sources,
        correlation_id=cid,
    )

@router.post("/jira_confluence/summarize")
async def summarize(request: JiraConfluenceSummaryRequest, x_correlation_id: str | None = Header(None)):
    cid = generate_correlation_id(x_correlation_id)
    issues = request.issues or []
    pages = request.pages or []
    if request.url:
        fetched = await _fetch_items(request.url)
        issues.extend(fetched)
    result = {
        "issues": _summarize_items(issues, "issue"),
        "pages": _summarize_items(pages, "page"),
    }
    sources = [build_source_record("tool", cid, title="jira_confluence")]
    return standard_response(
        "jira_confluence",
        grounded=True,
        context_found=bool(issues or pages),
        result=result,
        sources=sources,
        warnings=[],
        correlation_id=cid,
    )

# Dual registration: also register as "cloud_platform"
registry.register("cloud_platform", jira_confluence_handler)

@router.post("/sharepoint/documents")
async def sharepoint_documents(request: SharePointRequest, x_correlation_id: str | None = Header(None)):
    """Fetch and summarize SharePoint documents."""
    from src.api.config import Config
    if not getattr(Config, "CloudPlatform", None) or not Config.CloudPlatform.SHAREPOINT_ENABLED:
        return JSONResponse(
            status_code=400,
            content={"error": "SharePoint integration is not enabled. Set DOCWAIN_SHAREPOINT_ENABLED=true"},
        )
    cid = generate_correlation_id(x_correlation_id)
    items = request.items or []
    if request.url:
        items.extend(await _fetch_sharepoint_items(request.url))
    result = _summarize_sharepoint(items)
    sources = [build_source_record("sharepoint", cid, title=request.site_name or "sharepoint")]
    return standard_response(
        "cloud_platform",
        grounded=True,
        context_found=bool(items),
        result={"sharepoint": result},
        sources=sources,
        correlation_id=cid,
    )

@router.post("/sharepoint/sites")
async def sharepoint_sites(request: SharePointRequest, x_correlation_id: str | None = Header(None)):
    """Query SharePoint sites."""
    from src.api.config import Config
    if not getattr(Config, "CloudPlatform", None) or not Config.CloudPlatform.SHAREPOINT_ENABLED:
        return JSONResponse(
            status_code=400,
            content={"error": "SharePoint integration is not enabled. Set DOCWAIN_SHAREPOINT_ENABLED=true"},
        )
    cid = generate_correlation_id(x_correlation_id)
    items = request.items or []
    if request.url:
        items.extend(await _fetch_sharepoint_items(request.url))
    result = _summarize_sharepoint(items)
    sources = [build_source_record("sharepoint", cid, title=request.site_name or "sharepoint")]
    return standard_response(
        "cloud_platform",
        grounded=True,
        context_found=bool(items),
        result={"sites": result},
        sources=sources,
        correlation_id=cid,
    )
