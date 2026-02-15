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
# 然后编辑 .env，填入 GOOGLE_API_KEY
```

3. 运行：

```bash
python3 main.py "人工智能对高等教育评价体系的影响"
```

离线调试（无 API Key / 无网络时）：

```bash
python3 main.py "人工智能对高等教育评价体系的影响" --llm-provider offline --search-provider none
```

可选参数：

```bash
python3 main.py "题目" --search-provider duckduckgo --research-top-k 8
python3 main.py "题目" --search-provider none
python3 main.py "题目" --llm-provider offline
```

默认输出：
- 论文草稿：`output/paper.md`
- 工作流状态：`output/paper.state.json`

## 设计说明（MVP）

- 简化优先：重点演示多 subagent 分工与 supervisor 监管，不追求最强写作质量。
- 结构化输出：每个 subagent 使用 `with_structured_output` 输出 Pydantic 模型，便于主流程校验。
- 可控回退：每阶段支持固定次数重试（`--max-retries-per-stage`）。
- 轻量检索：`ResearchAgent` 当前支持 `duckduckgo/none`，并将检索片段注入 prompt。
