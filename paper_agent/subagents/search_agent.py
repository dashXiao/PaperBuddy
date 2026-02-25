from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tavily import TavilyClient

from ..models import DirectionResult, SourceArtifact
from .common import CollectedSource, _to_json

SEARCH_TOOL_NAME = "tavily_search_tool"
PERSIST_TOOL_NAME = "persist_sources_tool"
_RUN_CANDIDATES_CACHE: dict[str, dict[str, dict[str, Any]]] = {}


class TavilySearchToolInput(BaseModel):
    queries: list[str] = Field(default_factory=list)
    top_k: int = Field(default=8, ge=1)


class PersistSourcesToolInput(BaseModel):
    run_dir: str
    selected_ids: list[str] = Field(default_factory=list)
    discarded: list[dict[str, Any]] = Field(default_factory=list)


def _safe_json(raw: str, fallback: Any) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return fallback


def _next_run_dir(artifacts_root: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = artifacts_root / run_id
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = artifacts_root / f"{run_id}_{suffix}"
    return candidate


@tool(SEARCH_TOOL_NAME, args_schema=TavilySearchToolInput)
def tavily_search_tool(queries: list[str], top_k: int = 8) -> str:
    """Use Tavily to search candidates and return JSON list."""
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    pool_size = max(1, top_k)
    rows: list[dict[str, Any]] = []

    for query in queries:
        payload: dict[str, Any] = client.search(
            query=query,
            max_results=pool_size,
            topic="general",
            include_raw_content=True,
        )
        for item in payload.get("results", []):
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            snippet = str(item.get("content") or item.get("snippet") or "").strip()
            if title and url and snippet:
                rows.append(
                    {
                        "title": title,
                        "url": url,
                        "query": query,
                        "snippet": snippet,
                        "text_chars": len(snippet),
                    }
                )

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if row["url"] in seen:
            continue
        seen.add(row["url"])
        deduped.append(row)

    selected = deduped[:pool_size]
    for idx, item in enumerate(selected, start=1):
        item["candidate_id"] = f"C{idx:03d}"
    return json.dumps(selected, ensure_ascii=False)


def _persist_one(run_path: Path, source_id: str, source: dict[str, Any]) -> dict[str, Any] | None:
    snippet = str(source.get("snippet") or "").strip()
    if not snippet:
        return None

    title = str(source.get("title") or "").strip()
    url = str(source.get("url") or "").strip()
    query = str(source.get("query") or "").strip()
    text_chars = int(source.get("text_chars") or len(snippet))
    json_path = run_path / f"{source_id}.json"
    md_path = run_path / f"{source_id}.md"

    json_path.write_text(
        json.dumps(
            {
                "source_id": source_id,
                "query": query,
                "title": title,
                "url": url,
                "snippet": snippet,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    md_path.write_text(
        f"# {title}\n\n- source_id: {source_id}\n- query: {query}\n- url: {url}\n\n## Snippet\n{snippet}\n",
        encoding="utf-8",
    )
    return {
        "source_id": source_id,
        "title": title,
        "url": url,
        "query": query,
        "snippet": snippet,
        "text_chars": text_chars,
        "json_path": str(json_path),
        "md_path": str(md_path),
    }


@tool(PERSIST_TOOL_NAME, args_schema=PersistSourcesToolInput)
def persist_sources_tool(
    run_dir: str,
    selected_ids: list[str],
    discarded: list[dict[str, Any]] | None = None,
) -> str:
    """Persist cleaned sources and return metadata JSON."""
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    candidate_map = _RUN_CANDIDATES_CACHE.get(run_dir, {})
    persisted: list[dict[str, Any]] = []

    for idx, cid in enumerate(selected_ids or [], start=1):
        source = candidate_map.get(str(cid))
        if not source:
            continue
        item = _persist_one(run_path, f"S{idx:03d}", source)
        if item:
            persisted.append(item)
    return json.dumps({"sources": persisted, "discarded": discarded or []}, ensure_ascii=False)


class SearchAgent:
    """Two-stage search agent: search first, then filter and persist."""

    def __init__(
        self,
        llm: BaseChatModel,
        search_top_k: int = 8,
        min_source_chars: int = 400,
    ) -> None:
        self.target_top_k = max(1, search_top_k)
        self.min_source_chars = max(100, min_source_chars)
        self.search_pool_size = max(self.target_top_k * 2, self.target_top_k + 2)
        self._search_llm = llm.bind_tools([tavily_search_tool], tool_choice=[SEARCH_TOOL_NAME])
        self._persist_llm = llm.bind_tools([persist_sources_tool], tool_choice=[PERSIST_TOOL_NAME])
        self._search_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"你是 Search subagent。必须调用 `{SEARCH_TOOL_NAME}` 一次，且只能调用这个工具。",
                ),
                (
                    "human",
                    "论文方向：\n{direction}\n\n"
                    "请直接使用 keywords 作为 queries：\n{queries}\n\n"
                    "检索上限：top_k={search_pool_size}",
                ),
            ]
        )
        self._persist_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"你是 Search subagent。先过滤候选来源，再调用 `{PERSIST_TOOL_NAME}` 一次并只调用这个工具。"
                    "规则：保留相关、可读、去重来源；数量 <= target_top_k；selected_ids 不得为空。",
                ),
                (
                    "human",
                    "论文方向：\n{direction}\n\n"
                    "候选来源：\n{candidates}\n\n"
                    "约束：target_top_k={target_top_k}, min_source_chars={min_source_chars}\n"
                    "落盘目录：{run_dir}",
                ),
            ]
        )

    def run(
        self,
        direction: DirectionResult,
        artifacts_root: Path,
    ) -> tuple[list[CollectedSource], list[SourceArtifact]]:
        queries = [k.strip() for k in direction.keywords if k.strip()]
        if not queries:
            raise RuntimeError("SearchAgent failed: direction.keywords is empty.")

        run_dir = _next_run_dir(artifacts_root)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_key = str(run_dir)

        candidates = self._search(direction, queries)
        _RUN_CANDIDATES_CACHE[run_key] = {
            str(item.get("candidate_id")): item for item in candidates if item.get("candidate_id")
        }
        try:
            persisted = self._persist(direction, candidates, run_dir)
        finally:
            _RUN_CANDIDATES_CACHE.pop(run_key, None)

        sources = persisted.get("sources", [])
        if not sources:
            raise RuntimeError("SearchAgent failed: no sources were persisted.")

        collected, artifacts = self._build_outputs(sources)
        self._write_manifest(
            run_dir=run_dir,
            direction=direction,
            artifacts=artifacts,
            discarded=persisted.get("discarded", []),
        )
        return collected, artifacts

    def _search(self, direction: DirectionResult, queries: list[str]) -> list[dict[str, Any]]:
        raw = self._call_tool(
            llm_with_tool=self._search_llm,
            messages=self._search_prompt.format_messages(
                direction=_to_json(direction),
                queries=_to_json(queries),
                search_pool_size=self.search_pool_size,
            ),
            tool_name=SEARCH_TOOL_NAME,
            reminder=f"请立即调用 `{SEARCH_TOOL_NAME}`。",
        )
        parsed = _safe_json(raw, [])
        if not isinstance(parsed, list) or not parsed:
            raise RuntimeError("SearchAgent failed: search tool returned empty results.")
        return parsed

    def _persist(
        self,
        direction: DirectionResult,
        candidates: list[dict[str, Any]],
        run_dir: Path,
    ) -> dict[str, Any]:
        base_messages = self._persist_prompt.format_messages(
            direction=_to_json(direction),
            candidates=_to_json(
                [
                    {
                        "candidate_id": item.get("candidate_id"),
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "query": item.get("query"),
                        "snippet": item.get("snippet"),
                        "text_chars": item.get("text_chars"),
                    }
                    for item in candidates
                ]
            ),
            target_top_k=self.target_top_k,
            min_source_chars=self.min_source_chars,
            run_dir=str(run_dir),
        )
        raw = self._call_tool(
            llm_with_tool=self._persist_llm,
            messages=base_messages,
            tool_name=PERSIST_TOOL_NAME,
            reminder=f"请立即调用 `{PERSIST_TOOL_NAME}`，并提供 run_dir/selected_ids/discarded。",
        )
        parsed = _safe_json(raw, {})
        if isinstance(parsed, dict) and parsed.get("sources"):
            return parsed

        retry = self._call_tool(
            llm_with_tool=self._persist_llm,
            messages=[*base_messages, HumanMessage(content="你刚才传入了空 selected_ids，请至少选 1 条。")],
            tool_name=PERSIST_TOOL_NAME,
            reminder=f"请立即调用 `{PERSIST_TOOL_NAME}`，selected_ids 不能为空。",
        )
        parsed_retry = _safe_json(retry, {})
        if not isinstance(parsed_retry, dict) or not parsed_retry.get("sources"):
            raise RuntimeError("SearchAgent failed: persist tool returned empty sources.")
        return parsed_retry

    def _call_tool(
        self,
        llm_with_tool: Any,
        messages: list[Any],
        tool_name: str,
        reminder: str,
    ) -> str:
        ai = llm_with_tool.invoke(messages)
        call = self._pick_call(ai, tool_name)
        if call is None:
            ai = llm_with_tool.invoke([*messages, ai, HumanMessage(content=reminder)])
            call = self._pick_call(ai, tool_name)
        if call is None:
            raise RuntimeError(f"SearchAgent failed: model did not call `{tool_name}`.")

        args = call.get("args", {})
        if isinstance(args, str):
            args = _safe_json(args, {})
        if not isinstance(args, dict):
            args = {}
        tool_fn = tavily_search_tool if tool_name == SEARCH_TOOL_NAME else persist_sources_tool
        return str(tool_fn.invoke(args))

    @staticmethod
    def _pick_call(ai_message: Any, tool_name: str) -> dict[str, Any] | None:
        for item in (getattr(ai_message, "tool_calls", None) or []):
            if str(item.get("name") or "") == tool_name:
                return item
        return None

    @staticmethod
    def _build_outputs(sources: list[dict[str, Any]]) -> tuple[list[CollectedSource], list[SourceArtifact]]:
        collected: list[CollectedSource] = []
        artifacts: list[SourceArtifact] = []
        for item in sources:
            json_path = Path(item["json_path"])
            md_path = Path(item["md_path"])
            collected.append(
                CollectedSource(
                    source_id=item["source_id"],
                    query=item["query"],
                    title=item["title"],
                    url=item["url"],
                    snippet=item.get("snippet", ""),
                    json_path=json_path,
                    md_path=md_path,
                )
            )
            artifacts.append(
                SourceArtifact(
                    source_id=item["source_id"],
                    title=item["title"],
                    url=item["url"],
                    query=item["query"],
                    json_path=str(json_path),
                    md_path=str(md_path),
                    text_chars=int(item.get("text_chars") or len(item.get("snippet") or "")),
                )
            )
        return collected, artifacts

    def _write_manifest(
        self,
        run_dir: Path,
        direction: DirectionResult,
        artifacts: list[SourceArtifact],
        discarded: list[dict[str, Any]],
    ) -> None:
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "topic": direction.refined_topic,
                    "research_question": direction.research_question,
                    "target_top_k": self.target_top_k,
                    "min_source_chars": self.min_source_chars,
                    "collected_count": len(artifacts),
                    "discarded_count": len(discarded),
                    "sources": [item.model_dump() for item in artifacts],
                    "discarded": discarded,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
