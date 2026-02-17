from __future__ import annotations

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from ..models import DirectionResult
from .common import _feedback_block


class DirectionAgent:
    def __init__(self, llm: BaseChatModel) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你正在一个多代理论文系统中工作，整体目标是产出一篇完整论文草稿。"
                    "你是 Direction subagent，负责第一阶段：把宽泛题目收敛为可执行研究方向。"
                    "你的输出会交给 Research stage 做资料搜集，因此必须可检索、可落地。"
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
