from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tavily import TavilyClient

from ..models import DirectionResult, SourceArtifact
from .common import CollectedSource, _to_json

SEARCH_TOOL_NAME = "tavily_search_tool"
PERSIST_TOOL_NAME = "persist_sources_tool"


# 搜索工具的入参结构。
class TavilySearchToolInput(BaseModel):
    queries: list[str] = Field(default_factory=list)
    top_k: int = Field(default=8, ge=1)


# 落盘工具的入参结构。
class PersistSourcesToolInput(BaseModel):
    run_dir: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    discarded: list[dict[str, Any]] = Field(default_factory=list)


@tool(SEARCH_TOOL_NAME, args_schema=TavilySearchToolInput)
def tavily_search_tool(queries: list[str], top_k: int = 8) -> str:
    """Use Tavily to search documents and return JSON string results."""

    # 按 query 批量检索并聚合候选结果。
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    bounded_top_k = max(1, top_k)
    per_query = max(5, bounded_top_k)
    items: list[dict[str, str]] = []
    errors: list[str] = []

    for query in queries:
        try:
            payload: dict[str, Any] = client.search(
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
                raw_text = (raw_content or snippet)[:3500] # 避免字段过长
                items.append(
                    {
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "raw_content": raw_text,
                        "query": query,
                    }
                )
        except Exception as exc:
            errors.append(f"{query}: {exc}")

    if not items and errors:
        raise RuntimeError("Tavily search failed for all queries: " + " | ".join(errors[:3]))

    # 对 URL 去重后截断到 top_k。
    deduped: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for item in items:
        url = item["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(item)

    return json.dumps(deduped[:bounded_top_k], ensure_ascii=False)


@tool(PERSIST_TOOL_NAME, args_schema=PersistSourcesToolInput)
def persist_sources_tool(
    run_dir: str,
    sources: list[dict[str, Any]],
    discarded: list[dict[str, Any]] | None = None,
) -> str:
    """Persist cleaned sources to JSON/Markdown and return written metadata."""

    # 将清洗后的来源写成与现有流程一致的 JSON/Markdown 格式。
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    persisted: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        source_id = f"S{index:03d}"
        title = str(source.get("title") or "").strip()
        url = str(source.get("url") or "").strip()
        query = str(source.get("query") or "").strip()
        snippet = str(source.get("snippet") or "").strip()
        raw_text = str(source.get("raw_text") or "").strip()
        if not raw_text:
            continue
        text_chars = int(source.get("text_chars") or len(raw_text))

        json_path = run_path / f"{source_id}.json"
        md_path = run_path / f"{source_id}.md"

        payload = {
            "source_id": source_id,
            "query": query,
            "title": title,
            "url": url,
            "snippet": snippet,
            "raw_text": raw_text,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        md_content = (
            f"# {title}\n\n"
            f"- source_id: {source_id}\n"
            f"- query: {query}\n"
            f"- url: {url}\n\n"
            f"## Snippet\n{snippet or 'N/A'}\n\n"
            f"## Raw Text\n{raw_text}\n"
        )
        md_path.write_text(md_content, encoding="utf-8")

        persisted.append(
            {
                "source_id": source_id,
                "title": title,
                "url": url,
                "query": query,
                "snippet": snippet,
                "raw_text": raw_text,
                "text_chars": text_chars,
                "json_path": str(json_path),
                "md_path": str(md_path),
            }
        )

    return json.dumps(
        {
            "sources": persisted,
            "discarded": discarded or [],
        },
        ensure_ascii=False,
    )


class SearchAgent:
    """Search subagent: tool-calls search, filters results, and persists artifacts."""

    def __init__(
        self,
        llm: BaseChatModel,
        search_top_k: int = 8,
        min_source_chars: int = 400,
        oversample_factor: int = 4,
        max_tool_rounds: int = 6,
    ) -> None:
        # 运行参数：目标保留数、最小文本长度、检索池大小、工具调用轮数。
        self.target_top_k = max(1, search_top_k)
        self.min_source_chars = max(100, min_source_chars)
        self.search_pool_size = max(self.target_top_k * max(2, oversample_factor), self.target_top_k + 12)
        self.max_tool_rounds = max(3, max_tool_rounds)

        # 提示词约束模型按“先搜后存”的工具调用顺序执行。
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是多代理论文系统中的 Search subagent。"
                    "你的任务是：调用工具检索、清洗过滤、再落盘。"
                    f"必须先调用 `{SEARCH_TOOL_NAME}`，最后调用 `{PERSIST_TOOL_NAME}`。"
                    "过滤要求：删除无关内容、验证码/403/登录页、过短文本；按 URL 去重。"
                    "只保留与研究问题相关的来源，数量不超过 target_top_k。"
                    "请在传给落盘工具的 sources 中提供字段：title/url/query/snippet/raw_text/text_chars。",
                ),
                (
                    "human",
                    "论文方向：\n{direction}\n\n"
                    "主查询建议：\n{primary_queries}\n\n"
                    "约束：target_top_k={target_top_k}, min_source_chars={min_source_chars}, search_pool_size={search_pool_size}\n"
                    "落盘目录：{run_dir}",
                ),
            ]
        )
        # 绑定可调用工具并建立 name -> tool 映射。
        self._tools = [tavily_search_tool, persist_sources_tool]
        self._tool_map = {tool_item.name: tool_item for tool_item in self._tools}
        self._llm_with_tools = llm.bind_tools(self._tools)

    def run(
        self,
        direction: DirectionResult,
        artifacts_root: Path,
    ) -> tuple[list[CollectedSource], list[SourceArtifact]]:
        # 直接使用 DirectionAgent 产出的关键词作为检索词。
        primary_queries = direction.keywords

        run_dir = self._build_run_dir(artifacts_root)
        run_dir.mkdir(parents=True, exist_ok=True)

        messages = self._prompt.format_messages(
            direction=_to_json(direction),
            primary_queries=_to_json(primary_queries),
            target_top_k=self.target_top_k,
            min_source_chars=self.min_source_chars,
            search_pool_size=self.search_pool_size,
            run_dir=str(run_dir),
        )
        # 交给 LLM 执行工具调用循环，拿到落盘结果。
        persisted_payload = self._run_tool_agent(messages)
        sources = persisted_payload.get("sources", [])
        discarded = persisted_payload.get("discarded", [])
        if not sources:
            raise RuntimeError("SearchAgent failed: no sources were persisted after filtering.")

        # 把工具输出转成工作流下游所需的结构化对象。
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
                    raw_text=item["raw_text"],
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
                    text_chars=int(item.get("text_chars") or len(item["raw_text"])),
                )
            )

        # 生成 manifest，记录保留与丢弃来源。
        manifest_path = run_dir / "manifest.json"
        manifest_payload = {
            "topic": direction.refined_topic,
            "research_question": direction.research_question,
            "target_top_k": self.target_top_k,
            "min_source_chars": self.min_source_chars,
            "collected_count": len(artifacts),
            "discarded_count": len(discarded),
            "sources": [a.model_dump() for a in artifacts],
            "discarded": discarded,
        }
        manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return collected, artifacts

    def _run_tool_agent(self, messages: list[Any]) -> dict[str, Any]:
        # 通用 tool-calling 循环：模型出工具调用 -> 本地执行 -> 回灌 ToolMessage。
        state = list(messages)
        persisted_payload: dict[str, Any] | None = None

        for step in range(1, self.max_tool_rounds + 1):
            ai_message = self._llm_with_tools.invoke(state)
            state.append(ai_message)
            tool_calls = getattr(ai_message, "tool_calls", None) or []
            if not tool_calls:
                break

            for idx, tool_call in enumerate(tool_calls, start=1):
                tool_name = str(tool_call.get("name") or "")
                tool_call_id = str(tool_call.get("id") or f"call_{step}_{idx}")
                result = self._execute_tool_call(tool_call)
                if tool_name == PERSIST_TOOL_NAME:
                    parsed = self._safe_json_loads(result)
                    if isinstance(parsed, dict):
                        persisted_payload = parsed
                state.append(ToolMessage(content=result, tool_call_id=tool_call_id, name=tool_name))

        if not persisted_payload:
            raise RuntimeError(f"SearchAgent failed: model did not call `{PERSIST_TOOL_NAME}` successfully.")
        return persisted_payload

    def _execute_tool_call(self, tool_call: dict[str, Any]) -> str:
        # 执行单个工具调用并兜底返回错误信息。
        tool_name = str(tool_call.get("name") or "")
        args = tool_call.get("args", {})
        if isinstance(args, str):
            parsed = self._safe_json_loads(args)
            args = parsed if isinstance(parsed, dict) else {}

        tool_item = self._tool_map.get(tool_name)
        if tool_item is None:
            return json.dumps({"error": f"unknown_tool({tool_name})"}, ensure_ascii=False)

        try:
            return str(tool_item.invoke(args))
        except Exception as exc:
            return json.dumps({"error": f"{tool_name} failed: {exc}"}, ensure_ascii=False)

    @staticmethod
    def _safe_json_loads(raw: str) -> Any:
        # 容错 JSON 解析，避免异常中断流程。
        try:
            return json.loads(raw)
        except Exception:
            return {}

    @staticmethod
    def _build_run_dir(artifacts_root: Path) -> Path:
        # 用 UTC 时间戳隔离每次搜索落盘目录。
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = artifacts_root / run_id
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = artifacts_root / f"{run_id}_{suffix}"
        return candidate
