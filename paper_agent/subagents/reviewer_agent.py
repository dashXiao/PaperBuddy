from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from ..models import ReviewResult
from .common import _to_json


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
