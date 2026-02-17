from __future__ import annotations

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from ..models import DirectionResult, DraftResult, OutlineResult, ResearchResult
from .common import _feedback_block, _to_json


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
                    "写作时优先依据 point_evidences 的 source_passages 展开论证，不要脱离证据。"
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
