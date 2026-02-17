from __future__ import annotations

import difflib
import re
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from ..models import DirectionResult, EvidenceItem
from .common import _feedback_block, _to_json
from .types import CollectedSource


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
