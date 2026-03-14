"""Web search and URL fetching tool for DocWain.

Provides internet access as an optional fallback when local documents
cannot answer the query.  Two capabilities:

1. **Web search** — DuckDuckGo (default, free) with Tavily as paid alternative.
2. **URL fetching** — fetches user-provided URLs, strips HTML, returns plain text.

Security: SSRF denylist prevents access to cloud metadata endpoints and
private IP ranges.
"""
from __future__ import annotations

import ipaddress
from src.utils.logging_utils import get_logger
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from src.tools.base import register_tool, standard_response

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# URL detection
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://[^\s<>\"{}|\\^`\[\]]+")

def detect_urls_in_query(query: str) -> Tuple[List[str], str]:
    """Extract URLs from query text.

    Returns ``(urls, cleaned_query)`` where *cleaned_query* has the URLs
    removed and excess whitespace collapsed.
    """
    urls = _URL_RE.findall(query)
    if not urls:
        return [], query
    cleaned = query
    for url in urls:
        cleaned = cleaned.replace(url, "")
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return urls, cleaned

# ---------------------------------------------------------------------------
# SSRF protection
# ---------------------------------------------------------------------------

_SSRF_DENY_HOSTNAMES = frozenset({
    "169.254.169.254",
    "metadata.google.internal",
    "metadata.azure.internal",
})

def _is_ssrf_target(url: str) -> bool:
    """Return True if *url* points to a denied host (cloud metadata, private IP)."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
    except Exception:
        return True  # reject unparseable URLs

    if hostname in _SSRF_DENY_HOSTNAMES:
        return True

    try:
        addr = ipaddress.ip_address(hostname)
        if addr.is_private or addr.is_loopback or addr.is_link_local:
            return True
    except ValueError:
        pass  # hostname is not an IP literal — allowed

    return False

# ---------------------------------------------------------------------------
# Web search result dataclass
# ---------------------------------------------------------------------------

@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str
    source: str = "web"

# ---------------------------------------------------------------------------
# DuckDuckGo search
# ---------------------------------------------------------------------------

def _search_duckduckgo(query: str, max_results: int, timeout: float) -> List[WebSearchResult]:
    """Search via ddgs package (formerly duckduckgo-search)."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # legacy fallback
        except ImportError:
            logger.warning("ddgs (or duckduckgo-search) not installed")
            return []

    results: List[WebSearchResult] = []

    def _run_ddgs() -> list:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    try:
        # Wrap DDGS in ThreadPoolExecutor with hard timeout to prevent hangs
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run_ddgs)
            hits = future.result(timeout=timeout)
            for hit in hits:
                results.append(WebSearchResult(
                    title=hit.get("title", ""),
                    url=hit.get("href", hit.get("link", "")),
                    snippet=hit.get("body", hit.get("snippet", "")),
                    source="duckduckgo",
                ))
    except FuturesTimeoutError:
        logger.warning("DuckDuckGo search timed out after %.1fs for query: %s", timeout, query[:80])
    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)
    return results

# ---------------------------------------------------------------------------
# Tavily search
# ---------------------------------------------------------------------------

def _search_tavily(query: str, max_results: int, timeout: float, api_key: str) -> List[WebSearchResult]:
    """Search via Tavily API (paid, higher quality)."""
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed — Tavily search unavailable")
        return []

    results: List[WebSearchResult] = []
    try:
        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        with httpx.Client(timeout=timeout) as client:
            resp = client.post("https://api.tavily.com/search", json=payload)
            resp.raise_for_status()
            data = resp.json()
        for item in data.get("results", []):
            results.append(WebSearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                source="tavily",
            ))
    except Exception as exc:
        logger.warning("Tavily search failed: %s", exc)
    return results

# ---------------------------------------------------------------------------
# Unified search entry point
# ---------------------------------------------------------------------------

def search_web(
    query: str,
    *,
    max_results: int = 0,
    engine: str = "",
    timeout: float = 0.0,
) -> List[WebSearchResult]:
    """Search the web for *query* using the configured engine.

    Falls back through engines on failure:
      - ``duckduckgo`` (default, free)
      - ``tavily`` (if API key configured)
    """
    from src.api.config import Config

    ws = Config.WebSearch
    if not ws.ENABLED:
        return []

    max_results = max_results or ws.MAX_RESULTS
    timeout = timeout or ws.TIMEOUT
    engine = engine or ws.ENGINE

    if engine == "tavily" and ws.TAVILY_API_KEY:
        results = _search_tavily(query, max_results, timeout, ws.TAVILY_API_KEY)
        if results:
            return results
        # fall through to DuckDuckGo
        logger.info("Tavily returned no results; falling back to DuckDuckGo")

    results = _search_duckduckgo(query, max_results, timeout)
    if results:
        return results

    # Last resort: try Tavily if we haven't already and key is configured
    if engine != "tavily" and ws.TAVILY_API_KEY:
        results = _search_tavily(query, max_results, timeout, ws.TAVILY_API_KEY)

    return results

# ---------------------------------------------------------------------------
# URL fetching
# ---------------------------------------------------------------------------

def fetch_url_content(url: str, *, max_chars: int = 0) -> Dict[str, Any]:
    """Fetch *url* and return extracted plain text.

    Returns a dict with ``url``, ``title``, ``text``, and ``error`` keys.
    SSRF-denied URLs return an error without making a request.
    """
    from src.api.config import Config

    max_chars = max_chars or Config.WebSearch.MAX_URL_FETCH_CHARS

    if _is_ssrf_target(url):
        return {"url": url, "title": "", "text": "", "error": "SSRF denied: private or metadata endpoint"}

    try:
        import httpx
    except ImportError:
        return {"url": url, "title": "", "text": "", "error": "httpx not installed"}

    try:
        headers = {
            "User-Agent": "DocWain/1.0 (document intelligence; +https://www.docwain.ai)",
        }
        with httpx.Client(timeout=Config.WebSearch.TIMEOUT, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            raw = resp.text
    except Exception as exc:
        return {"url": url, "title": "", "text": "", "error": str(exc)[:300]}

    # Extract text from HTML
    title = ""
    text = raw
    if "html" in content_type.lower():
        title, text = _extract_text_from_html(raw)

    # Truncate
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Content truncated]"

    return {"url": url, "title": title, "text": text, "error": ""}

def _extract_text_from_html(html: str) -> Tuple[str, str]:
    """Strip HTML tags and return (title, plain_text)."""
    title = ""
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if title_match:
        title = re.sub(r"<[^>]+>", "", title_match.group(1)).strip()

    # Remove script and style blocks
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    # Remove all remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return title, text

# ---------------------------------------------------------------------------
# Format search results for display
# ---------------------------------------------------------------------------

_WEB_DISCLAIMER = (
    "\n\n---\n*Note: This information is from web search results, "
    "not from your uploaded documents. Verify important details independently.*"
)

def format_web_results(results: List[WebSearchResult], query: str = "") -> str:
    """Render search results as readable markdown text."""
    if not results:
        return "No web results found."

    parts: List[str] = []
    for i, r in enumerate(results, 1):
        entry = f"**{i}. [{r.title}]({r.url})**\n{r.snippet}"
        parts.append(entry)

    body = "\n\n".join(parts)
    return f"{body}{_WEB_DISCLAIMER}"

def build_web_sources(results: List[WebSearchResult]) -> List[Dict[str, Any]]:
    """Build source records from web search results for the answer payload."""
    sources = []
    for r in results:
        sources.append({
            "source_name": r.title or r.url,
            "url": r.url,
            "type": "web",
            "snippet": r.snippet[:200] if r.snippet else "",
            "engine": r.source,
        })
    return sources

# ---------------------------------------------------------------------------
# Tool handler (registered in tool registry)
# ---------------------------------------------------------------------------

@register_tool("web_search")
def handle_web_search(
    inputs: Dict[str, Any],
    *,
    chunks: Any = None,
    llm_client: Any = None,
    correlation_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Tool handler for explicit web search requests."""
    # Extract from nested "input" dict (pipeline passes {"input": {...}, "context": {...}})
    input_data = inputs.get("input") or inputs
    query = input_data.get("query", input_data.get("text", ""))
    if not query:
        return standard_response(
            "web_search",
            status="error",
            error={"code": "MISSING_QUERY", "message": "No query provided for web search"},
            correlation_id=correlation_id,
        )

    results = search_web(query)
    if not results:
        return standard_response(
            "web_search",
            status="no_results",
            result={"query": query, "results": [], "formatted": "No web results found."},
            grounded=False,
            context_found=False,
            correlation_id=correlation_id,
        )

    formatted = format_web_results(results, query=query)
    sources = build_web_sources(results)

    return standard_response(
        "web_search",
        result={
            "query": query,
            "results": [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in results],
            "formatted": formatted,
            "rendered": formatted,
            "result_count": len(results),
        },
        sources=sources,
        grounded=False,
        context_found=True,
        correlation_id=correlation_id,
    )
