from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

from tavily import TavilyClient


@dataclass
class SearchDocument:
    """检索得到的单条候选资料（包含可供落盘的文本内容）。"""

    title: str
    url: str
    snippet: str
    raw_content: str
    query: str


class ResearchSearcher:
    """轻量检索器：当前固定使用 Tavily。"""

    def __init__(self, top_k: int = 8) -> None:
        self.top_k = top_k
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("Missing TAVILY_API_KEY for Tavily search.")
        self._client = TavilyClient(api_key=api_key)

    def search(self, queries: list[str]) -> list[SearchDocument]:
        raw_items = self._tavily_search(queries)
        return self._deduplicate(raw_items)[: self.top_k]

    def _tavily_search(self, queries: list[str]) -> list[SearchDocument]:
        items: list[SearchDocument] = []
        errors: list[str] = []
        per_query = max(5, self.top_k)
        for query in queries:
            try:
                payload: dict[str, Any] = self._client.search(
                    query=query,
                    max_results=per_query,
                    topic="general",
                    include_raw_content=True,
                )
                for row in payload.get("results", []):
                    title = str(row.get("title") or "").strip()
                    url = str(row.get("url") or "").strip()
                    snippet = str(row.get("content") or row.get("snippet") or "").strip()
                    raw_content = str(row.get("raw_content") or "").strip()
                    if not title or not url:
                        continue
                    items.append(
                        SearchDocument(
                            title=title,
                            url=url,
                            snippet=snippet,
                            raw_content=raw_content,
                            query=query,
                        )
                    )
            except Exception as exc:
                errors.append(f"{query}: {exc}")
                continue
        if not items and errors:
            raise RuntimeError(
                "Tavily search failed for all queries: " + " | ".join(errors[:3])
            ) from None
        return items

    @staticmethod
    def _deduplicate(items: Iterable[SearchDocument]) -> list[SearchDocument]:
        seen: set[str] = set()
        deduped: list[SearchDocument] = []
        for item in items:
            if item.url in seen:
                continue
            seen.add(item.url)
            deduped.append(item)
        return deduped
