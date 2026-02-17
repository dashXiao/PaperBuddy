from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from ..models import DirectionResult, SourceArtifact
from ..research_tools import ResearchSearcher, SearchDocument
from .common import CollectedSource


class SearchAgent:
    """检索并落盘原始资料文件。"""

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
            raise RuntimeError("SearchAgent failed: no search results were retrieved.")

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
                "SearchAgent failed: search returned items but none passed the minimum text length filter."
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
