# LangChain Subagents Paper Agent (MVP)

这是一个最小可运行骨架，用于探索 LangChain 中 `subagents` 的协作模式。

核心流程：
1. `DirectionAgent`：收敛论文方向
2. `ResearchAgent`：先检索，再生成结构化资料卡（可追溯来源）
3. `OutlineAgent`：形成论文大纲
4. `WriterAgent`：生成整篇草稿
5. `ReviewerAgent`（由 `Supervisor` 调用）：逐阶段评审并触发重试

## 目录结构

```text
.
├── .env.example
├── main.py
├── paper_agent
│   ├── __init__.py
│   ├── models.py
│   ├── research_tools.py
│   ├── subagents.py
│   └── supervisor.py
└── requirements.txt
```

## 快速开始

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 设置环境变量：

```bash
cp .env.example .env
# 然后编辑 .env，至少填入 GOOGLE_API_KEY 和 TAVILY_API_KEY
```

示例：

```dotenv
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
GOOGLE_MODEL=gemini-2.5-flash
GOOGLE_TEMPERATURE=0.2
```

`main.py` 会在启动时自动加载 `.env`。

3. 运行：

```bash
python3 main.py "人工智能对高等教育评价体系的影响"
```

可选参数：

```bash
python3 main.py "题目" --research-top-k 8
python3 main.py "题目" --max-retries-per-stage 0
python3 main.py "题目" --output output/topic_a.md
```

默认输出：
- 论文草稿：`output/paper.md`
- 工作流状态：`output/paper.state.json`

参数说明（与当前代码一致）：
- `--output`：论文草稿输出路径（默认 `output/paper.md`）
- `--max-retries-per-stage`：每阶段最大重试次数（默认 `1`）
- `--research-top-k`：Research 阶段保留的外部检索条数（默认 `8`）

注意：
- 不传 `--output` 时，再次运行会覆盖 `output/paper.md` 与 `output/paper.state.json`。
- `GOOGLE_API_KEY` 或 `TAVILY_API_KEY` 缺失时程序会直接报错退出。
- 检索源固定为 Tavily，`ResearchAgent` 若未检索到任何外部内容会直接报错并终止程序。

## 设计说明（MVP）

- 简化优先：重点演示多 subagent 分工与 supervisor 监管，不追求最强写作质量。
- 结构化输出：每个 subagent 使用 `with_structured_output` 输出 Pydantic 模型，便于主流程校验。
- 可控回退：每阶段支持固定次数重试（`--max-retries-per-stage`）。
- 轻量检索：`ResearchAgent` 固定使用 Tavily，并将检索片段注入 prompt。
