from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
