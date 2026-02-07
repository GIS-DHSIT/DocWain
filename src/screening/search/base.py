from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

import requests


@dataclass
class SearchHit:
    title: str
    snippet: str
    url: str
    source: str = ""
    score: float | None = None


class SearchClient(Protocol):
    def search(self, query: str, k: int = 5) -> List[SearchHit]:
        ...


class NullSearchClient:
    """Search client that never performs network calls."""

    def search(self, query: str, k: int = 5) -> List[SearchHit]:
        return []


class SimpleHttpSearchClient:
    """
    Minimal HTTP-based search client.

    Supports SerpAPI and Bing Web Search. Only active when an API key is
    provided and internet checks are explicitly enabled.
    """

    def __init__(self, provider: str, api_key: str, endpoint: str | None = None, timeout: int = 8) -> None:
        self.provider = provider.lower()
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout

    def search(self, query: str, k: int = 5) -> List[SearchHit]:
        try:
            if self.provider == "serpapi":
                return self._serpapi(query, k)
            if self.provider == "bing":
                return self._bing(query, k)
        except Exception:
            return []
        return []

    def _serpapi(self, query: str, k: int) -> List[SearchHit]:
        url = self.endpoint or "https://serpapi.com/search"
        params = {"engine": "google", "api_key": self.api_key, "q": query, "num": k}
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        hits = []
        for item in data.get("organic_results", [])[:k]:
            hits.append(
                SearchHit(
                    title=item.get("title", ""),
                    snippet=item.get("snippet", "") or item.get("snippet_highlighted_words", ""),
                    url=item.get("link", ""),
                    source="serpapi",
                )
            )
        return hits

    def _bing(self, query: str, k: int) -> List[SearchHit]:
        url = self.endpoint or "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {"q": query, "count": k}
        resp = requests.get(url, headers=headers, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        web_pages = data.get("webPages", {}) if isinstance(data, dict) else {}
        value = web_pages.get("value", []) if isinstance(web_pages, dict) else []
        hits = []
        for item in value[:k]:
            hits.append(
                SearchHit(
                    title=item.get("name", ""),
                    snippet=item.get("snippet", ""),
                    url=item.get("url", ""),
                    source="bing",
                    score=item.get("rankingScore"),
                )
            )
        return hits
