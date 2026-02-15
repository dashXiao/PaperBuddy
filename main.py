from __future__ import annotations

import argparse
import os
from pathlib import Path

from langchain_openai import ChatOpenAI

from paper_agent import PaperSupervisor


def build_llm() -> ChatOpenAI:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    return ChatOpenAI(model=model, temperature=temperature)


def render_markdown(state) -> str:
    draft = state.draft
    assert draft is not None
    return (
        f"# {draft.title}\n\n"
        f"## Abstract\n{draft.abstract}\n\n"
        f"{draft.body_markdown}\n\n"
        f"## References\n{draft.references_markdown}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MVP paper-writing agent with LangChain subagents.")
    parser.add_argument("topic", help="宽泛论文题目，例如：AI 对教育公平的影响")
    parser.add_argument(
        "--output",
        default="output/paper.md",
        help="论文草稿输出路径",
    )
    parser.add_argument(
        "--max-retries-per-stage",
        type=int,
        default=1,
        help="每个阶段允许重试次数",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY.")

    llm = build_llm()
    supervisor = PaperSupervisor(llm=llm, max_retries_per_stage=args.max_retries_per_stage)
    state = supervisor.generate(topic=args.topic)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(state), encoding="utf-8")

    state_path = output_path.with_suffix(".state.json")
    state_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

    print(f"Paper draft written to: {output_path}")
    print(f"Workflow state written to: {state_path}")


if __name__ == "__main__":
    main()
