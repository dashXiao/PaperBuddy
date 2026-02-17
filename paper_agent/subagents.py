from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from .models import (
    DirectionResult,
    DraftResult,
    EvidenceItem,
    OutlineResult,
    ResearchResult,
    ReviewResult,
    SourceArtifact,
)
from .research_tools import ResearchSearcher, SearchDocument


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


@dataclass
class CollectedSource:
    source_id: str
    query: str
    title: str
    url: str
    snippet: str
    raw_text: str
    json_path: Path
    md_path: Path


class SourceCollectorAgent:
    """Research 子代理 1：检索并落盘原始资料文件。"""

    _INVALID_TEXT_PATTERNS = [
        "access denied",
        "403 forbidden",
        "404 not found",
        "captcha",
        "cloudflare",
        "enable javascript",
        "javascript required",
        "please enable javascript",
        "just a moment",
        "验证失败",
        "请完成验证",
        "暂无权限",
        "页面不存在",
        "内容不存在",
        "无法访问",
        "无法读取",
    ]

    def __init__(
        self,
        search_top_k: int = 8,
        min_source_chars: int = 400,
        oversample_factor: int = 4,
    ) -> None:
        self.target_top_k = max(1, search_top_k)
        self.min_source_chars = max(100, min_source_chars)
        search_pool_size = max(self.target_top_k * max(2, oversample_factor), self.target_top_k + 12)
        self.searcher = ResearchSearcher(top_k=search_pool_size)

    def run(
        self,
        direction: DirectionResult,
        artifacts_root: Path,
    ) -> tuple[list[CollectedSource], list[SourceArtifact]]:
        primary_queries = self._build_queries(direction)
        backup_queries = self._build_backup_queries(direction, primary_queries)
        docs = self._collect_candidates(primary_queries, backup_queries)
        if not docs:
            raise RuntimeError("SourceCollectorAgent failed: no search results were retrieved.")

        run_dir = self._build_run_dir(artifacts_root)
        run_dir.mkdir(parents=True, exist_ok=True)

        collected: list[CollectedSource] = []
        artifacts: list[SourceArtifact] = []
        discarded: list[dict[str, object]] = []

        for doc in docs:
            source_id = f"S{len(collected) + 1:03d}"
            normalized_text = self._normalize_source_text(doc.raw_content or doc.snippet)
            body_text = self._extract_body_text(normalized_text)
            invalid_reason = self._detect_invalid_text(body_text)
            if invalid_reason:
                discarded.append(
                    {
                        "title": doc.title,
                        "url": doc.url,
                        "query": doc.query,
                        "text_chars": len(body_text),
                        "reason": invalid_reason,
                    }
                )
                continue
            text_chars = len(body_text)
            if text_chars < self.min_source_chars:
                discarded.append(
                    {
                        "title": doc.title,
                        "url": doc.url,
                        "query": doc.query,
                        "text_chars": text_chars,
                        "reason": f"text_too_short(<{self.min_source_chars})",
                    }
                )
                continue

            base = f"{source_id}"
            json_path = run_dir / f"{base}.json"
            md_path = run_dir / f"{base}.md"

            payload = {
                "source_id": source_id,
                "query": doc.query,
                "title": doc.title,
                "url": doc.url,
                "snippet": doc.snippet,
                "raw_text": body_text,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            md_content = (
                f"# {doc.title}\n\n"
                f"- source_id: {source_id}\n"
                f"- query: {doc.query}\n"
                f"- url: {doc.url}\n\n"
                f"## Snippet\n{doc.snippet or 'N/A'}\n\n"
                f"## Raw Text\n{body_text}\n"
            )
            md_path.write_text(md_content, encoding="utf-8")

            collected.append(
                CollectedSource(
                    source_id=source_id,
                    query=doc.query,
                    title=doc.title,
                    url=doc.url,
                    snippet=doc.snippet,
                    raw_text=body_text,
                    json_path=json_path,
                    md_path=md_path,
                )
            )
            artifacts.append(
                SourceArtifact(
                    source_id=source_id,
                    title=doc.title,
                    url=doc.url,
                    query=doc.query,
                    json_path=str(json_path),
                    md_path=str(md_path),
                    text_chars=text_chars,
                )
            )
            if len(collected) >= self.target_top_k:
                break

        if not collected:
            raise RuntimeError(
                "SourceCollectorAgent failed: search returned items but none passed the minimum text length filter."
            )

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
        manifest_path.write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return collected, artifacts

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
    def _build_backup_queries(direction: DirectionResult, primary_queries: list[str]) -> list[str]:
        keywords = [k.strip() for k in direction.keywords if k.strip()]
        candidates = [
            f"{direction.research_question.strip()} 综述",
            f"{direction.refined_topic.strip()} 系统综述",
            f"{direction.refined_topic.strip()} meta analysis",
            f"{direction.refined_topic.strip()} guideline report",
        ]
        if keywords:
            candidates.append(" ".join(keywords[:3]) + " review")
            candidates.append(" ".join(keywords[:3]) + " guideline")
        deduped: list[str] = []
        seen = set(primary_queries)
        for query in candidates:
            q = query.strip()
            if not q or q in seen:
                continue
            seen.add(q)
            deduped.append(q)
        return deduped[:6]

    def _collect_candidates(
        self,
        primary_queries: list[str],
        backup_queries: list[str],
    ) -> list[SearchDocument]:
        collected: list[SearchDocument] = []
        seen_urls: set[str] = set()

        for query_group in [primary_queries, backup_queries]:
            if not query_group:
                continue
            docs = self.searcher.search(query_group)
            for doc in docs:
                url = (doc.url or "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                collected.append(doc)
            if len(collected) >= self.target_top_k * 2:
                break
        return collected

    @staticmethod
    def _normalize_source_text(text: str) -> str:
        compact = "\n".join(line.strip() for line in (text or "").splitlines() if line.strip())
        compact = re.sub(r"\n{3,}", "\n\n", compact)
        return compact.strip()

    @staticmethod
    def _extract_body_text(text: str) -> str:
        if not text:
            return ""
        cleaned = re.sub(r"<[^>]+>", " ", text)
        lines: list[str] = []
        for raw_line in cleaned.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if re.fullmatch(r"!\[[^\]]*\]\([^)]+\)", line):
                continue
            if "javascript:;" in line.lower():
                continue
            if re.fullmatch(r"[\*\-\+]\s*\[[^\]]+\]\([^)]+\)", line):
                continue
            if re.fullmatch(r"\[[^\]]+\]\([^)]+\)", line):
                continue
            if re.search(r"(二维码登录|手机登录|邮箱登录|验证成功|请重试|登录后继续)", line):
                continue
            line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line).strip()
            if not line:
                continue
            lines.append(line)
        body = "\n".join(lines)
        body = re.sub(r"\n{3,}", "\n\n", body)
        return body.strip()

    @classmethod
    def _detect_invalid_text(cls, text: str) -> str:
        lowered = (text or "").lower()
        if not lowered:
            return "empty_text_after_cleanup"
        for token in cls._INVALID_TEXT_PATTERNS:
            if token in lowered:
                return f"invalid_text_pattern({token})"
        lines = [line for line in text.splitlines() if line.strip()]
        if lines:
            short_line_count = sum(1 for line in lines if len(line.strip()) <= 10)
            if short_line_count / len(lines) >= 0.75 and len(text) < 1200:
                return "invalid_text_mostly_short_lines"
        return ""

    @staticmethod
    def _build_run_dir(artifacts_root: Path) -> Path:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = artifacts_root / run_id
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = artifacts_root / f"{run_id}_{suffix}"
        return candidate


class EvidenceExtractorAgent:
    """Research 子代理 2：逐文件提取“观点-来源段落”卡。"""

    def __init__(self, llm: BaseChatModel) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你正在一个多代理论文系统中工作，整体目标是产出一篇完整论文草稿。"
                    "你是 EvidenceExtractor subagent，负责分析单个来源文件并提取证据卡。"
                    "你只处理一个来源，不要混入其他来源内容。"
                    "输出格式是 EvidenceItem，必须使用 point_evidences。"
                    "每个 point_evidences 包含 key_point 与 source_passages。"
                    "每个 key_point 至少 1 条 source_passages，推荐 1-3 条。"
                    "同一信息卡内，不同 key_point 的 source_passages 不要重复。"
                    "source_passages 应直接引用来源文本中的原文段落，不要改写。"
                    "key_point 应是对 source_passages 的概括，不要抄句子。"
                    "若资料不足，可减少观点数量，不要编造来源内容。",
                ),
                (
                    "human",
                    "论文方向：\n{direction}\n\n"
                    "来源文件元数据：\n{source_meta}\n\n"
                    "来源文本：\n{source_text}\n\n"
                    "{feedback}",
                ),
            ]
        )
        self._chain = prompt | llm.with_structured_output(EvidenceItem)

    def run(
        self,
        direction: DirectionResult,
        source: CollectedSource,
        feedback: Optional[str] = None,
    ) -> EvidenceItem:
        source_text = self._prepare_text_for_llm(source.raw_text)
        result = self._chain.invoke(
            {
                "direction": _to_json(direction),
                "source_meta": _to_json(
                    {
                        "source_id": source.source_id,
                        "title": source.title,
                        "url": source.url,
                        "query": source.query,
                        "snippet": source.snippet,
                        "json_path": str(source.json_path),
                        "md_path": str(source.md_path),
                    }
                ),
                "source_text": source_text,
                "feedback": _feedback_block(feedback),
            }
        )
        return self._normalize_evidence_item(result, source)

    @staticmethod
    def _prepare_text_for_llm(text: str, max_chars: int = 12000) -> str:
        raw = (text or "").strip()
        if len(raw) <= max_chars:
            return raw
        head = raw[:7000]
        tail = raw[-4500:]
        return f"{head}\n\n...[内容过长，已截断，中间省略]...\n\n{tail}"

    @staticmethod
    def _normalize_fragment(text: str, limit: int = 340) -> str:
        compact = " ".join((text or "").split())
        if len(compact) <= limit:
            return compact
        return compact[:limit].rstrip() + "..."

    @classmethod
    def _canonical(cls, text: str) -> str:
        lowered = (text or "").lower()
        return re.sub(r"[\s,，。.!！？?、;；:：\"'“”‘’（）()\[\]【】\-_/]+", "", lowered)

    @staticmethod
    def _strip_boilerplate(text: str) -> str:
        cleaned = text or ""
        patterns = [
            r"来源[:：][^\n。]*",
            r"发布时间[:：][^\n。]*",
            r"浏览次数[:：][^\n。]*",
            r"【字体[：:].*?】",
            r"【?大中小】?",
        ]
        for pattern in patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        return " ".join(cleaned.split())

    @classmethod
    def _dedupe_sentences(cls, passage: str) -> str:
        text = (passage or "").strip()
        if not text:
            return ""
        sentences = [s.strip() for s in re.split(r"(?<=[。！？.!?])\s*", text) if s.strip()]
        if not sentences:
            return text
        seen: list[str] = []
        keep: list[str] = []
        for sentence in sentences:
            canon = cls._canonical(sentence)
            if not canon:
                continue
            if any(canon == s or canon in s or s in canon for s in seen):
                continue
            seen.append(canon)
            keep.append(sentence)
        return " ".join(keep) if keep else text

    @staticmethod
    def _is_near_duplicate_canon(
        candidate: str,
        existing_canons: list[str],
        ratio_threshold: float = 0.9,
    ) -> bool:
        if not candidate:
            return True
        for item in existing_canons:
            if candidate == item or candidate in item or item in candidate:
                return True
            ratio = difflib.SequenceMatcher(None, candidate, item).ratio()
            if ratio >= ratio_threshold:
                return True
        return False

    @classmethod
    def _dedupe_passages(cls, passages: list[str]) -> list[str]:
        deduped: list[str] = []
        seen_canons: list[str] = []
        for passage in passages:
            norm = cls._normalize_fragment(cls._dedupe_sentences(cls._strip_boilerplate(passage)))
            canon = cls._canonical(norm)
            if not canon:
                continue
            if cls._is_near_duplicate_canon(canon, seen_canons):
                continue
            deduped.append(norm)
            seen_canons.append(canon)
        return deduped

    @classmethod
    def _split_paragraphs(cls, text: str) -> list[str]:
        normalized = cls._strip_boilerplate((text or "").replace("\r", "\n"))
        if not normalized:
            return []
        raw_parts: list[str] = []
        for block in normalized.split("\n"):
            block = block.strip()
            if not block:
                continue
            sentences = [s.strip() for s in re.split(r"(?<=[。！？.!?])\s+", block) if s.strip()]
            if len(sentences) <= 1:
                raw_parts.append(block)
                continue
            chunk = ""
            for sentence in sentences:
                if not chunk:
                    chunk = sentence
                    continue
                if len(chunk) + len(sentence) <= 280:
                    chunk = f"{chunk} {sentence}"
                else:
                    raw_parts.append(chunk)
                    chunk = sentence
            if chunk:
                raw_parts.append(chunk)
        return cls._dedupe_passages(raw_parts)

    @classmethod
    def _best_matching_paragraph(
        cls,
        passage: str,
        source_paragraphs: list[str],
        min_ratio: float = 0.55,
    ) -> str:
        if not source_paragraphs:
            return ""
        target = cls._canonical(passage)
        if not target:
            return ""
        best = ""
        best_ratio = 0.0
        for paragraph in source_paragraphs:
            pcanon = cls._canonical(paragraph)
            if not pcanon:
                continue
            ratio = difflib.SequenceMatcher(None, target, pcanon).ratio()
            if target in pcanon or pcanon in target:
                ratio = max(ratio, 0.99)
            if ratio > best_ratio:
                best_ratio = ratio
                best = paragraph
        if best_ratio < min_ratio:
            return ""
        return best

    @classmethod
    def _pick_support_passages(
        cls,
        key_point: str,
        source_paragraphs: list[str],
        excluded_canons: list[str],
        limit: int = 2,
    ) -> list[str]:
        if not source_paragraphs:
            return []
        key_chars = {c for c in key_point.lower() if not c.isspace()}
        scored: list[tuple[int, int, str]] = []
        for paragraph in source_paragraphs:
            canon = cls._canonical(paragraph)
            if cls._is_near_duplicate_canon(canon, excluded_canons):
                continue
            para_lower = paragraph.lower()
            score = sum(1 for c in key_chars if c in para_lower)
            scored.append((score, len(paragraph), paragraph))
        if not scored:
            return []
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        selected: list[str] = []
        for _, _, paragraph in scored:
            selected.append(paragraph)
            if len(selected) >= limit:
                break
        return cls._dedupe_passages(selected)

    @classmethod
    def _normalize_evidence_item(
        cls,
        item: EvidenceItem,
        source: CollectedSource,
    ) -> EvidenceItem:
        source_paragraphs = cls._split_paragraphs(source.raw_text)

        item.source = source.title
        item.url = source.url
        if not (item.source_type or "").strip():
            item.source_type = "web"
        if not (item.reliability_note or "").strip():
            item.reliability_note = "基于外部来源文本提取，建议结合原文复核。"

        used_canons: list[str] = []
        normalized_points: list[type(item).PointEvidence] = []
        point_cls = type(item).PointEvidence

        for point in item.point_evidences:
            key_point = (point.key_point or "").strip()
            if not key_point:
                continue

            passages = cls._dedupe_passages([p for p in point.source_passages if p and p.strip()])
            key_canon = cls._canonical(key_point)
            passages = [
                p
                for p in passages
                if cls._canonical(p) != key_canon
                and not cls._is_near_duplicate_canon(cls._canonical(p), [key_canon])
            ]

            anchored: list[str] = []
            for passage in passages:
                matched = cls._best_matching_paragraph(passage, source_paragraphs)
                if matched:
                    anchored.append(matched)
            passages = cls._dedupe_passages(anchored)

            unique_passages: list[str] = []
            for passage in passages:
                canon = cls._canonical(passage)
                if cls._is_near_duplicate_canon(canon, used_canons):
                    continue
                unique_passages.append(passage)
                used_canons.append(canon)
            passages = unique_passages

            if not passages:
                passages = cls._pick_support_passages(
                    key_point,
                    source_paragraphs,
                    excluded_canons=used_canons,
                    limit=2,
                )
                for passage in passages:
                    canon = cls._canonical(passage)
                    if not cls._is_near_duplicate_canon(canon, used_canons):
                        used_canons.append(canon)

            if len(passages) == 1 and len(passages[0]) < max(90, len(key_point) + 25):
                supplements = cls._pick_support_passages(
                    key_point,
                    source_paragraphs,
                    excluded_canons=used_canons,
                    limit=1,
                )
                passages = cls._dedupe_passages(passages + supplements)
                passages = [
                    p
                    for p in passages
                    if not cls._is_near_duplicate_canon(cls._canonical(p), [key_canon])
                ]
                unique_passages = []
                for passage in passages:
                    canon = cls._canonical(passage)
                    if cls._is_near_duplicate_canon(canon, used_canons):
                        continue
                    unique_passages.append(passage)
                    used_canons.append(canon)
                passages = unique_passages

            if not passages:
                passages = ["（该来源可用原文段落不足，无法为该观点分配不重复段落）"]

            normalized_points.append(point_cls(key_point=key_point, source_passages=passages[:3]))

        if not normalized_points:
            fallback = cls._pick_support_passages(
                "该来源支持与研究问题相关的核心结论。",
                source_paragraphs,
                excluded_canons=used_canons,
                limit=2,
            )
            if not fallback:
                fallback = ["（该来源可用原文段落不足，无法生成观点证据对）"]
            normalized_points = [
                point_cls(
                    key_point="该来源支持与研究问题相关的核心结论。",
                    source_passages=fallback,
                )
            ]

        item.point_evidences = normalized_points
        return item


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
