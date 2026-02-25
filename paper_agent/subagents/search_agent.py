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

# 搜索工具入参。
class TavilySearchToolInput(BaseModel):
    queries: list[str] = Field(default_factory=list)
    top_k: int = Field(default=8, ge=1)


# 落盘工具入参。
class PersistSourcesToolInput(BaseModel):
    run_dir: str
    selected_ids: list[str] = Field(default_factory=list)
    discarded: list[dict[str, Any]] = Field(default_factory=list)


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
            if not title or not url or not snippet:
                continue
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
    seen_urls: set[str] = set()
    for row in rows:
        if row["url"] in seen_urls:
            continue
        seen_urls.add(row["url"])
        deduped.append(row)

    selected = deduped[:pool_size]
    for index, item in enumerate(selected, start=1):
        item["candidate_id"] = f"C{index:03d}"
    return json.dumps(selected, ensure_ascii=False)


@tool(PERSIST_TOOL_NAME, args_schema=PersistSourcesToolInput)
def persist_sources_tool(
    run_dir: str,
    selected_ids: list[str],
    discarded: list[dict[str, Any]] | None = None,
) -> str:
    """Persist cleaned sources and return JSON metadata."""

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    persisted: list[dict[str, Any]] = []
    candidate_map = _RUN_CANDIDATES_CACHE.get(run_dir, {})

    for index, candidate_id in enumerate(selected_ids or [], start=1):
        source = candidate_map.get(str(candidate_id))
        if not source:
            continue
        source_id = f"S{index:03d}"
        title = str(source.get("title") or "").strip()
        url = str(source.get("url") or "").strip()
        query = str(source.get("query") or "").strip()
        snippet = str(source.get("snippet") or "").strip()
        if not snippet:
            continue
        text_chars = int(source.get("text_chars") or len(snippet))

        json_path = run_path / f"{source_id}.json"
        md_path = run_path / f"{source_id}.md"

        payload = {
            "source_id": source_id,
            "query": query,
            "title": title,
            "url": url,
            "snippet": snippet,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        md_content = (
            f"# {title}\n\n"
            f"- source_id: {source_id}\n"
            f"- query: {query}\n"
            f"- url: {url}\n\n"
            f"## Snippet\n{snippet or 'N/A'}\n"
        )
        md_path.write_text(md_content, encoding="utf-8")

        persisted.append(
            {
                "source_id": source_id,
                "title": title,
                "url": url,
                "query": query,
                "snippet": snippet,
                "text_chars": text_chars,
                "json_path": str(json_path),
                "md_path": str(md_path),
            }
        )

    return json.dumps({"sources": persisted, "discarded": discarded or []}, ensure_ascii=False)


class SearchAgent:
    """两阶段 Search agent：先检索，再清洗并落盘。"""

    def __init__(
        self,
        llm: BaseChatModel,
        search_top_k: int = 8,
        min_source_chars: int = 400,
    ) -> None:
        self.target_top_k = max(1, search_top_k)
        self.min_source_chars = max(100, min_source_chars)
        self.search_pool_size = max(self.target_top_k * 2, self.target_top_k + 2)

        # 阶段 A：只允许调用搜索工具。
        self._search_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是 Search subagent。"
                    f"必须调用 `{SEARCH_TOOL_NAME}` 一次来搜索候选来源。"
                    "只允许调用这个工具。",
                ),
                (
                    "human",
                    "论文方向：\n{direction}\n\n"
                    "请直接使用以下 keywords 作为 queries：\n{queries}\n\n"
                    "检索上限：top_k={search_pool_size}",
                ),
            ]
        )
        # 阶段 B：只允许调用落盘工具。
        self._persist_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是 Search subagent。"
                    "你会收到候选来源，请先清洗过滤再落盘。"
                    f"必须调用 `{PERSIST_TOOL_NAME}` 一次。"
                    "只允许调用这个工具。"
                    "过滤规则：仅保留与研究问题相关、可读、非登录/验证码/错误页的来源；按 URL 去重。"
                    "保留数量不超过 target_top_k。"
                    "如果候选里有可用来源，selected_ids 至少保留 1 条，不要传空数组。"
                    "你只需传 selected_ids（candidate_id 列表）和 discarded。",
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

        self._tool_map = {
            SEARCH_TOOL_NAME: tavily_search_tool,
            PERSIST_TOOL_NAME: persist_sources_tool,
        }
        self._search_llm = llm.bind_tools([tavily_search_tool], tool_choice=[SEARCH_TOOL_NAME])
        self._persist_llm = llm.bind_tools([persist_sources_tool], tool_choice=[PERSIST_TOOL_NAME])

    def run(
        self,
        direction: DirectionResult,
        artifacts_root: Path,
    ) -> tuple[list[CollectedSource], list[SourceArtifact]]:
        # 直接使用 DirectionAgent 产出的关键词。
        queries = [item.strip() for item in direction.keywords if item.strip()]
        if not queries:
            raise RuntimeError("SearchAgent failed: direction.keywords is empty.")

        run_dir = self._build_run_dir(artifacts_root)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dir_str = str(run_dir)

        # 阶段 A：执行检索工具。
        candidates = self._run_search_step(direction=direction, queries=queries)
        _RUN_CANDIDATES_CACHE[run_dir_str] = {
            str(item.get("candidate_id")): item
            for item in candidates
            if item.get("candidate_id")
        }
        # 阶段 B：让 agent 清洗过滤后调用落盘工具。
        try:
            persisted_payload = self._run_persist_step(
                direction=direction,
                candidates=candidates,
                run_dir=run_dir,
            )
        finally:
            _RUN_CANDIDATES_CACHE.pop(run_dir_str, None)

        sources = persisted_payload.get("sources", [])
        discarded = persisted_payload.get("discarded", [])
        if not sources:
            raise RuntimeError("SearchAgent failed: no sources were persisted.")

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

        manifest_path = run_dir / "manifest.json"
        manifest_payload = {
            "topic": direction.refined_topic,
            "research_question": direction.research_question,
            "target_top_k": self.target_top_k,
            "min_source_chars": self.min_source_chars,
            "collected_count": len(artifacts),
            "discarded_count": len(discarded),
            "sources": [item.model_dump() for item in artifacts],
            "discarded": discarded,
        }
        manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return collected, artifacts

    def _run_search_step(self, direction: DirectionResult, queries: list[str]) -> list[dict[str, Any]]:
        messages = self._search_prompt.format_messages(
            direction=_to_json(direction),
            queries=_to_json(queries),
            search_pool_size=self.search_pool_size,
        )
        raw = self._invoke_single_tool(
            llm_with_tool=self._search_llm,
            messages=messages,
            tool_name=SEARCH_TOOL_NAME,
            reminder=f"请立即调用 `{SEARCH_TOOL_NAME}`，不要输出解释文本。",
        )
        parsed = self._safe_json_loads(raw)
        if not isinstance(parsed, list) or not parsed:
            raise RuntimeError("SearchAgent failed: search tool returned empty results.")
        return parsed

    def _run_persist_step(
        self,
        direction: DirectionResult,
        candidates: list[dict[str, Any]],
        run_dir: Path,
    ) -> dict[str, Any]:
        messages = self._persist_prompt.format_messages(
            direction=_to_json(direction),
            candidates=_to_json(self._compact_candidates(candidates)),
            target_top_k=self.target_top_k,
            min_source_chars=self.min_source_chars,
            run_dir=str(run_dir),
        )
        raw = self._invoke_single_tool(
            llm_with_tool=self._persist_llm,
            messages=messages,
            tool_name=PERSIST_TOOL_NAME,
            reminder=(
                f"请先完成清洗过滤，再立即调用 `{PERSIST_TOOL_NAME}`，"
                "参数中必须包含 run_dir/selected_ids/discarded。"
            ),
        )
        parsed = self._safe_json_loads(raw)
        if isinstance(parsed, dict) and parsed.get("sources"):
            return parsed

        retry_messages = [
            *messages,
            HumanMessage(
                content=(
                    "你刚才传入了空 selected_ids。"
                    "请基于候选来源至少选择 1 条 candidate_id，再调用 persist_sources_tool。"
                )
            ),
        ]
        retry_raw = self._invoke_single_tool(
            llm_with_tool=self._persist_llm,
            messages=retry_messages,
            tool_name=PERSIST_TOOL_NAME,
            reminder=(
                f"请立即调用 `{PERSIST_TOOL_NAME}`，并确保 selected_ids 至少包含 1 条有效 candidate_id。"
            ),
        )
        retry_parsed = self._safe_json_loads(retry_raw)
        if not isinstance(retry_parsed, dict):
            raise RuntimeError("SearchAgent failed: persist tool returned invalid payload.")
        if not retry_parsed.get("sources"):
            preview = json.dumps(retry_parsed, ensure_ascii=False)[:600]
            raise RuntimeError(f"SearchAgent failed: persist tool returned empty sources. payload={preview}")
        return retry_parsed

    def _invoke_single_tool(
        self,
        llm_with_tool: Any,
        messages: list[Any],
        tool_name: str,
        reminder: str,
    ) -> str:
        ai_message = llm_with_tool.invoke(messages)
        tool_call = self._pick_tool_call(getattr(ai_message, "tool_calls", None), tool_name)
        if tool_call is None:
            retry_messages = [*messages, ai_message, HumanMessage(content=reminder)]
            ai_message = llm_with_tool.invoke(retry_messages)
            tool_call = self._pick_tool_call(getattr(ai_message, "tool_calls", None), tool_name)
        if tool_call is None:
            raise RuntimeError(f"SearchAgent failed: model did not call `{tool_name}`.")
        return self._execute_tool_call(tool_name=tool_name, tool_call=tool_call)

    @staticmethod
    def _pick_tool_call(tool_calls: Any, tool_name: str) -> dict[str, Any] | None:
        if not isinstance(tool_calls, list):
            return None
        for item in tool_calls:
            if str(item.get("name") or "") == tool_name:
                return item
        return None

    def _execute_tool_call(self, tool_name: str, tool_call: dict[str, Any]) -> str:
        args = tool_call.get("args", {})
        if isinstance(args, str):
            parsed = self._safe_json_loads(args)
            args = parsed if isinstance(parsed, dict) else {}
        tool_item = self._tool_map[tool_name]
        return str(tool_item.invoke(args))

    @staticmethod
    def _compact_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for item in candidates:
            compact.append(
                {
                    "candidate_id": item.get("candidate_id"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "query": item.get("query"),
                    "snippet": item.get("snippet"),
                    "text_chars": item.get("text_chars"),
                }
            )
        return compact

    @staticmethod
    def _safe_json_loads(raw: str) -> Any:
        try:
            return json.loads(raw)
        except Exception:
            return {}

    @staticmethod
    def _build_run_dir(artifacts_root: Path) -> Path:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = artifacts_root / run_id
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = artifacts_root / f"{run_id}_{suffix}"
        return candidate
