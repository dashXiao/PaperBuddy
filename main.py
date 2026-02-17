from __future__ import annotations

import argparse

from paper_agent import PaperSupervisor
from script.output import prepare_output_paths, write_outputs
from script.runtime import build_llm, init_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="MVP paper-writing agent with LangChain subagents.")
    parser.add_argument("topic", help="宽泛论文题目")
    parser.add_argument("--output", default="output/paper.md", help="输出路径")
    parser.add_argument("--max-retries-per-stage", type=int, default=0, help="每个阶段允许重试次数")
    parser.add_argument("--research-top-k", type=int, default=5, help="SearchAgent 搜索条目数")
    args = parser.parse_args()

    output_path, artifacts_root = prepare_output_paths(args.output)

    init_runtime()

    llm = build_llm()
    supervisor = PaperSupervisor(
        llm=llm,
        max_retries_per_stage=args.max_retries_per_stage,
        research_top_k=args.research_top_k,
    )
    state = supervisor.generate(topic=args.topic, artifacts_root=artifacts_root)

    write_outputs(state, output_path)


if __name__ == "__main__":
    main()
