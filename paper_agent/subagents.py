from __future__ import annotations

import json
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from .models import (
    DirectionResult,
    DraftResult,
    OutlineResult,
    ResearchResult,
    ReviewResult,
)
from .research_tools import ResearchSearcher, SearchSnippet


def _feedback_block(feedback: Optional[str]) -> str:
    if not feedback:
        return "无额外反馈。"
    return f"上轮评审反馈如下，请优先修正：\n{feedback}"


def _to_json(payload: object) -> str:
    if hasattr(payload, "model_dump"):
        return json.dumps(payload.model_dump(), ensure_ascii=False, indent=2)
    return json.dumps(payload, ensure_ascii=False, indent=2)


class DirectionAgent:
    def __init__(self, llm: BaseChatModel) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你正在一个多代理论文系统中工作，整体目标是产出一篇完整论文草稿。"
                    "你是 Direction subagent，负责第一阶段：把宽泛题目收敛为可执行研究方向。"
                    "你的输出会交给 ResearchAgent 做资料搜集，因此必须可检索、可落地。"
                    "成功标准：研究问题明确、中心论点清晰、范围边界具体、关键词可用于后续检索。",
                ),
                (
                    "human",
                    "原始题目：\n{topic}\n\n{feedback}",
                ),
            ]
        )
        self._chain = prompt | llm.with_structured_output(DirectionResult)

    def run(self, topic: str, feedback: Optional[str] = None) -> DirectionResult:
        return self._chain.invoke({"topic": topic, "feedback": _feedback_block(feedback)})


class ResearchAgent:
    def __init__(
        self,
        llm: BaseChatModel,
        search_top_k: int = 8,
    ) -> None:
        self.searcher = ResearchSearcher(top_k=search_top_k)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你正在一个多代理论文系统中工作，整体目标是产出一篇完整论文草稿。"
                    "你是 Research subagent，负责基于论文方向生成结构化信息卡片。"
                    "你的输出会交给 OutlineAgent 和 WriterAgent 使用，必须服务研究问题和论点。"
                    "你会收到查询词与外部检索片段，优先使用这些可追溯来源。"
                    "每条信息要有来源描述和可信度备注。"
                    "若提供了来源链接，请填入 url 字段。"
                    "不要编造不存在的来源链接。检索结果不足时，在 gaps 明确写缺口。",
                ),
                (
                    "human",
                    "论文方向：\n{direction}\n\n检索查询词：\n{queries}\n\n外部检索片段：\n{search_context}\n\n{feedback}",
                ),
            ]
        )
        self._chain = prompt | llm.with_structured_output(ResearchResult)

    def run(self, direction: DirectionResult, feedback: Optional[str] = None) -> ResearchResult:
        queries = self._build_queries(direction)
        snippets = self.searcher.search(queries)
        if not snippets:
            raise RuntimeError(
                "ResearchAgent failed: no external search results were retrieved; aborting workflow."
            )
        context = self._format_search_context(snippets)
        return self._chain.invoke(
            {
                "direction": _to_json(direction),
                "queries": "\n".join(f"- {q}" for q in queries),
                "search_context": context,
                "feedback": _feedback_block(feedback),
            }
        )

    @staticmethod
    def _build_queries(direction: DirectionResult) -> list[str]:
        keywords = [k.strip() for k in direction.keywords if k.strip()]
        queries = [
            direction.refined_topic.strip(),
            direction.research_question.strip(),
        ]
        if keywords:
            queries.append(" ".join(keywords[:4]))
            queries.append(f"{direction.research_question.strip()} {' '.join(keywords[:2])}")

        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            if not query or query in seen:
                continue
            seen.add(query)
            deduped.append(query)
        return deduped[:5]

    @staticmethod
    def _format_search_context(snippets: list[SearchSnippet]) -> str:
        if not snippets:
            return "无外部检索结果。请在输出中明确资料缺口，并避免虚构来源。"
        lines: list[str] = []
        for idx, item in enumerate(snippets, start=1):
            lines.append(f"[{idx}] {item.title}")
            lines.append(f"URL: {item.url}")
            if item.snippet:
                lines.append(f"摘要片段: {item.snippet}")
            lines.append("")
        return "\n".join(lines).strip()


class OutlineAgent:
    def __init__(self, llm: BaseChatModel) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你正在一个多代理论文系统中工作，整体目标是产出一篇完整论文草稿。"
                    "你是 Outline subagent，负责根据方向和资料卡设计论文结构。"
                    "你的输出会直接交给 WriterAgent，因此必须具备可写作性和证据映射。"
                    "成功标准：结构完整、章节目标明确、论证顺序合理。"
                    "每一章都要映射至少一个证据索引。",
                ),
                (
                    "human",
                    "论文方向：\n{direction}\n\n研究资料：\n{research}\n\n{feedback}",
                ),
            ]
        )
        self._chain = prompt | llm.with_structured_output(OutlineResult)

    def run(
        self,
        direction: DirectionResult,
        research: ResearchResult,
        feedback: Optional[str] = None,
    ) -> OutlineResult:
        return self._chain.invoke(
            {
                "direction": _to_json(direction),
                "research": _to_json(research),
                "feedback": _feedback_block(feedback),
            }
        )


class WriterAgent:
    def __init__(self, llm: BaseChatModel) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你正在一个多代理论文系统中工作，整体目标是产出一篇完整论文草稿。"
                    "你是 Writer subagent，负责将方向、资料和大纲整合为完整论文。"
                    "你的输出会交给 ReviewerAgent 验收，因此要确保结构、一致性和可追溯性。"
                    "成功标准：论证连贯、与大纲一致、明确标注局限与证据边界。"
                    "请写出一版完整论文草稿，要求："
                    "结构清晰、观点一致、段落之间有过渡。"
                    "如果资料不足，需在文中注明假设或局限，不要伪造具体文献细节。",
                ),
                (
                    "human",
                    "论文方向：\n{direction}\n\n研究资料：\n{research}\n\n论文大纲：\n{outline}\n\n{feedback}",
                ),
            ]
        )
        self._chain = prompt | llm.with_structured_output(DraftResult)

    def run(
        self,
        direction: DirectionResult,
        research: ResearchResult,
        outline: OutlineResult,
        feedback: Optional[str] = None,
    ) -> DraftResult:
        return self._chain.invoke(
            {
                "direction": _to_json(direction),
                "research": _to_json(research),
                "outline": _to_json(outline),
                "feedback": _feedback_block(feedback),
            }
        )


class ReviewerAgent:
    def __init__(self, llm: BaseChatModel) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你正在一个多代理论文系统中工作，整体目标是产出一篇完整论文草稿。"
                    "你是 Supervisor 的评审子代理，负责阶段质量门控。"
                    "你的评审结论会决定是否重试该阶段，因此必须严格对照验收标准。"
                    "成功标准：指出关键问题、给出可执行修正建议、避免空泛评价。"
                    "你要根据验收标准判定阶段结果是否通过，并给出可执行修正建议。",
                ),
                (
                    "human",
                    "阶段名：{stage}\n\n验收标准：\n{criteria}\n\n阶段输出：\n{output}",
                ),
            ]
        )
        self._chain = prompt | llm.with_structured_output(ReviewResult)

    def review(self, stage: str, criteria: str, output_payload: object) -> ReviewResult:
        return self._chain.invoke(
            {"stage": stage, "criteria": criteria, "output": _to_json(output_payload)}
        )
