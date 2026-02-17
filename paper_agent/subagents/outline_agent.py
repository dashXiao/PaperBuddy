from __future__ import annotations

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from ..models import DirectionResult, OutlineResult, ResearchResult
from .common import _feedback_block, _to_json


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
                    "优先依据 point_evidences 中的 source_passages 组织证据映射。"
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
