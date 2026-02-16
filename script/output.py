from __future__ import annotations

from pathlib import Path
from typing import Any


def prepare_output_paths(output: str | Path) -> tuple[Path, Path]:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_root = output_path.parent / f"{output_path.stem}.sources"
    return output_path, artifacts_root


def render_markdown(state: Any) -> str:
    draft = state.draft
    assert draft is not None

    return (
        f"# {draft.title}\n\n"
        f"## Abstract\n{draft.abstract}\n\n"
        f"{draft.body_markdown}\n\n"
        f"## References\n{draft.references_markdown}\n"
    )


def write_outputs(state: Any, output_path: Path) -> None:
    markdown = render_markdown(state)
    output_path.write_text(markdown, encoding="utf-8")

    state_path = output_path.with_suffix(".state.json")
    state_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

    print(f"Paper draft written to: {output_path}")
    print(f"Workflow state written to: {state_path}")
