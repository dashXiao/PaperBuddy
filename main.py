from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from paper_agent import PaperSupervisor


load_dotenv()


def build_llm() -> ChatGoogleGenerativeAI:
    model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    temperature = float(os.getenv("GOOGLE_TEMPERATURE", "0.2"))
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


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
    parser.add_argument("topic", help="宽泛论文题目")
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
    parser.add_argument(
        "--research-top-k",
        type=int,
        default=8,
        help="ResearchAgent 最多保留的外部检索条数",
    )
    args = parser.parse_args()

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Missing GOOGLE_API_KEY.")
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("Missing TAVILY_API_KEY.")

    llm = build_llm()
    supervisor = PaperSupervisor(
        llm=llm,
        max_retries_per_stage=args.max_retries_per_stage,
        research_top_k=args.research_top_k,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_root = output_path.parent / f"{output_path.stem}.sources"

    state = supervisor.generate(topic=args.topic, artifacts_root=artifacts_root)

    output_path.write_text(render_markdown(state), encoding="utf-8")

    state_path = output_path.with_suffix(".state.json")
    state_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

    print(f"Paper draft written to: {output_path}")
    print(f"Workflow state written to: {state_path}")


if __name__ == "__main__":
    main()
