from __future__ import annotations

from pathlib import Path
from typing import Callable, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel

from .models import DirectionResult, ResearchResult, StageAttempt, WorkflowState
from .subagents import (
    CollectedSource,
    DirectionAgent,
    ExtractorAgent,
    OutlineAgent,
    ReviewerAgent,
    SearchAgent,
    WriterAgent,
)

T = TypeVar("T")


class PaperSupervisor:
    def __init__(
        self,
        llm: BaseChatModel,
        max_retries_per_stage: int = 1,
        research_top_k: int = 8,
    ) -> None:
        self.max_retries_per_stage = max_retries_per_stage
        self.direction_agent = DirectionAgent(llm)
        self.search_agent = SearchAgent(search_top_k=research_top_k)
        self.extractor_agent = ExtractorAgent(llm)
        self.outline_agent = OutlineAgent(llm)
        self.writer_agent = WriterAgent(llm)
        self.reviewer_agent = ReviewerAgent(llm)

    @staticmethod
    def _log(stage: str, message: str) -> None:
        print(f"[{stage}] {message}")

    def generate(self, topic: str, artifacts_root: Path) -> WorkflowState:
        state = WorkflowState(topic=topic)
        self._log("workflow", "start")
        self._log("workflow", f"topic: {topic}")

        state.direction = self._run_stage(
            state=state,
            stage_name="direction",
            criteria=(
                "1) 题目已收敛。2) 有明确研究问题与中心论点。"
                "3) 范围边界清晰，不泛化。4) 关键词可用于后续资料搜集。"
            ),
            producer=lambda feedback: self.direction_agent.run(topic=topic, feedback=feedback),
        )

        latest_artifacts = state.source_artifacts

        self._log("search", "SearchAgent running")
        collected_sources, source_artifacts = self.search_agent.run(
            direction=state.direction,
            artifacts_root=artifacts_root,
        )
        latest_artifacts.clear()
        latest_artifacts.extend(source_artifacts)
        state.source_artifacts = list(latest_artifacts)
        self._log("search", f"done: {len(source_artifacts)} sources")

        state.research = self._run_stage(
            state=state,
            stage_name="extract",
            criteria=(
                "1) 至少提供若干条结构化信息卡。2) 每条有来源描述。"
                "3) 关键点支撑研究问题。4) 标注可信度与缺口。"
                "5) 来源应可追溯（若有链接则提供）。"
                "6) point_evidences 中每个 key_point 必须有 1+ 条 source_passages 原文段落支持。"
                "7) 同一信息卡内不同 key_point 的 source_passages 不重复，且不跨来源混用。"
            ),
            producer=lambda feedback: self._run_extract(
                direction=state.direction,
                collected_sources=collected_sources,
                feedback=feedback,
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
        self._log("workflow", "done")

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
        self._log(stage_name, "start")

        for attempt in range(1, total_attempts + 1):
            self._log(stage_name, f"attempt {attempt}/{total_attempts}")
            result = producer(feedback)
            self._log(stage_name, "ReviewerAgent reviewing")
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
            self._log(stage_name, f"review score={review.score} passed={review.passed}")
            if review.passed:
                self._log(stage_name, "passed")
                return result

            feedback = "\n".join(review.suggestions or review.issues)
            self._log(stage_name, "retrying")

        assert result is not None
        self._log(stage_name, "max retries reached; continue with last result")
        return result

    def _run_extract(
        self,
        direction: DirectionResult,
        collected_sources: list[CollectedSource],
        feedback: str | None,
    ) -> ResearchResult:
        findings = []
        gaps: list[str] = []
        total = len(collected_sources)

        for index, source in enumerate(collected_sources, start=1):
            self._log("extract", f"ExtractorAgent source {index}/{total} ({source.source_id})")
            try:
                item = self.extractor_agent.run(direction=direction, source=source, feedback=feedback)
            except Exception as exc:
                gaps.append(f"{source.source_id} 提取失败：{exc}")
                continue

            if not item.point_evidences:
                gaps.append(f"{source.source_id} 未提取到有效观点。")
                continue
            findings.append(item)

        if not findings:
            raise RuntimeError(
                "ExtractorAgent failed: sources were collected but no usable evidence cards were extracted."
            )

        self._log("extract", f"done: findings={len(findings)} gaps={len(gaps)}")
        return ResearchResult(findings=findings, gaps=gaps)
