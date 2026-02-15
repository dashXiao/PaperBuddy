from __future__ import annotations

from typing import Callable, TypeVar

from langchain_openai import ChatOpenAI

from .models import StageAttempt, WorkflowState
from .subagents import DirectionAgent, OutlineAgent, ResearchAgent, ReviewerAgent, WriterAgent

T = TypeVar("T")


class PaperSupervisor:
    def __init__(
        self,
        llm: ChatOpenAI,
        max_retries_per_stage: int = 1,
        search_provider: str = "duckduckgo",
        research_top_k: int = 8,
    ) -> None:
        self.max_retries_per_stage = max_retries_per_stage
        self.direction_agent = DirectionAgent(llm)
        self.research_agent = ResearchAgent(
            llm,
            search_provider=search_provider,
            search_top_k=research_top_k,
        )
        self.outline_agent = OutlineAgent(llm)
        self.writer_agent = WriterAgent(llm)
        self.reviewer_agent = ReviewerAgent(llm)

    def generate(self, topic: str) -> WorkflowState:
        state = WorkflowState(topic=topic)

        state.direction = self._run_stage(
            state=state,
            stage_name="direction",
            criteria=(
                "1) 题目已收敛。2) 有明确研究问题与中心论点。"
                "3) 范围边界清晰，不泛化。4) 关键词可用于后续资料搜集。"
            ),
            producer=lambda feedback: self.direction_agent.run(topic=topic, feedback=feedback),
        )

        state.research = self._run_stage(
            state=state,
            stage_name="research",
            criteria=(
                "1) 至少提供若干条结构化信息卡。2) 每条有来源描述。"
                "3) 关键点支撑研究问题。4) 标注可信度与缺口。"
                "5) 来源应可追溯（若有链接则提供）。"
            ),
            producer=lambda feedback: self.research_agent.run(
                direction=state.direction, feedback=feedback
            ),
        )

        state.outline = self._run_stage(
            state=state,
            stage_name="outline",
            criteria=(
                "1) 结构完整（引言、主体、结论）。2) 章节目标与论点清晰。"
                "3) 每章映射证据索引。4) 与研究问题保持一致。"
            ),
            producer=lambda feedback: self.outline_agent.run(
                direction=state.direction,
                research=state.research,
                feedback=feedback,
            ),
        )

        state.draft = self._run_stage(
            state=state,
            stage_name="draft",
            criteria=(
                "1) 正文完整。2) 段落组织连贯。3) 论点与大纲一致。"
                "4) 明确参考资料与局限，不伪造细节。"
            ),
            producer=lambda feedback: self.writer_agent.run(
                direction=state.direction,
                research=state.research,
                outline=state.outline,
                feedback=feedback,
            ),
        )

        return state

    def _run_stage(
        self,
        state: WorkflowState,
        stage_name: str,
        criteria: str,
        producer: Callable[[str | None], T],
    ) -> T:
        feedback: str | None = None
        result: T | None = None
        total_attempts = self.max_retries_per_stage + 1

        for attempt in range(1, total_attempts + 1):
            result = producer(feedback)
            review = self.reviewer_agent.review(stage=stage_name, criteria=criteria, output_payload=result)
            state.review_log.append(
                StageAttempt(
                    stage=stage_name,
                    attempt=attempt,
                    passed=review.passed,
                    score=review.score,
                    issues=review.issues,
                    suggestions=review.suggestions,
                )
            )
            if review.passed:
                return result

            feedback = "\n".join(review.suggestions or review.issues)

        assert result is not None
        return result
