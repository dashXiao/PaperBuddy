from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

warnings.filterwarnings(
    "ignore",
    message=r"This package \(`duckduckgo_search`\) has been renamed to `ddgs`!.*",
    category=RuntimeWarning,
)


@dataclass
class SearchSnippet:
    """检索得到的单条候选资料。"""

    title: str
    url: str
    snippet: str


class ResearchSearcher:
    """轻量检索器：当前支持 duckduckgo 和 none。"""

    def __init__(self, provider: str = "duckduckgo", top_k: int = 8) -> None:
        self.provider = provider
        self.top_k = top_k

    def search(self, queries: list[str]) -> list[SearchSnippet]:
        if self.provider == "none":
            return []
        if self.provider != "duckduckgo":
            raise ValueError(f"Unsupported search provider: {self.provider}")

        raw_items = self._duckduckgo_search(queries)
        return self._deduplicate(raw_items)[: self.top_k]

    def _duckduckgo_search(self, queries: list[str]) -> list[SearchSnippet]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                from duckduckgo_search import DDGS
        except Exception:
            return []

        items: list[SearchSnippet] = []
        per_query = max(2, self.top_k // max(1, len(queries)))
        for query in queries:
            try:
                with warnings.catch_warnings():
                    # duckduckgo_search emits a rename RuntimeWarning in recent versions.
                    warnings.simplefilter("ignore", RuntimeWarning)
                    with DDGS() as ddgs:
                        results = ddgs.text(query, max_results=per_query)
                        for row in results:
                            title = (row.get("title") or "").strip()
                            url = (row.get("href") or "").strip()
                            snippet = (row.get("body") or "").strip()
                            if not title or not url:
                                continue
                            items.append(SearchSnippet(title=title, url=url, snippet=snippet))
            except Exception:
                continue
        return items

    @staticmethod
    def _deduplicate(items: Iterable[SearchSnippet]) -> list[SearchSnippet]:
        seen: set[str] = set()
        deduped: list[SearchSnippet] = []
        for item in items:
            if item.url in seen:
                continue
            seen.add(item.url)
            deduped.append(item)
        return deduped
