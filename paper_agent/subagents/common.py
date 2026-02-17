from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _feedback_block(feedback: Optional[str]) -> str:
    if not feedback:
        return "无额外反馈。"
    return f"上轮评审反馈如下，请优先修正：\n{feedback}"


def _to_json(payload: object) -> str:
    if hasattr(payload, "model_dump"):
        return json.dumps(payload.model_dump(), ensure_ascii=False, indent=2)
    return json.dumps(payload, ensure_ascii=False, indent=2)


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
