from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DirectionResult(BaseModel):
    """方向阶段输出：把宽泛题目收敛为可执行的研究定义。"""

    refined_topic: str = Field(description="收敛后的论文方向")
    research_question: str = Field(description="核心研究问题")
    thesis_statement: str = Field(description="一句话中心论点")
    scope: str = Field(description="研究范围和边界")
    keywords: list[str] = Field(default_factory=list, description="用于检索的关键词")


class EvidenceItem(BaseModel):
    """单条证据信息卡：记录来源、要点和可信度说明。"""

    source: str = Field(description="来源名称或出处描述")
    url: Optional[str] = Field(default=None, description="来源链接（若可用）")
    source_type: str = Field(description="来源类型，例如 paper/report/book/web")
    key_points: list[str] = Field(default_factory=list, description="关键信息点")
    reliability_note: str = Field(description="对可信度的简要说明")


class ResearchResult(BaseModel):
    """研究阶段输出：证据卡集合与当前信息缺口。"""

    findings: list[EvidenceItem] = Field(default_factory=list, description="信息卡片集合")
    gaps: list[str] = Field(default_factory=list, description="尚未覆盖但应关注的缺口")


class SectionPlan(BaseModel):
    """大纲中的单个章节计划：目标、论点及证据映射。"""

    section_title: str = Field(description="章节标题")
    objective: str = Field(description="本章目标")
    key_argument: str = Field(description="本章核心论点")
    evidence_indices: list[int] = Field(
        default_factory=list,
        description="支持本章的证据索引，对应 findings 的下标",
    )


class OutlineResult(BaseModel):
    """大纲阶段输出：论文标题与章节结构。"""

    title: str = Field(description="论文标题")
    sections: list[SectionPlan] = Field(default_factory=list, description="论文章节规划")


class DraftResult(BaseModel):
    """写作阶段输出：标题、摘要、正文与参考文献。"""

    title: str = Field(description="论文标题")
    abstract: str = Field(description="摘要")
    body_markdown: str = Field(description="正文，Markdown 格式")
    references_markdown: str = Field(description="参考文献，Markdown 列表")


class ReviewResult(BaseModel):
    """评审结果：是否通过、评分、问题与修正建议。"""

    passed: bool = Field(description="是否通过本阶段验收")
    score: int = Field(ge=0, le=10, description="0-10 分")
    issues: list[str] = Field(default_factory=list, description="主要问题")
    suggestions: list[str] = Field(default_factory=list, description="修正建议")


class StageAttempt(BaseModel):
    """单次阶段执行记录：用于追踪重试和验收过程。"""

    stage: str
    attempt: int
    passed: bool
    score: int
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class WorkflowState(BaseModel):
    """全局工作流状态：聚合各阶段产物与评审日志。"""

    topic: str
    direction: Optional[DirectionResult] = None
    research: Optional[ResearchResult] = None
    outline: Optional[OutlineResult] = None
    draft: Optional[DraftResult] = None
    review_log: list[StageAttempt] = Field(default_factory=list)
