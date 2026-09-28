# PaperPilot-Lite

> 面向论文与技术文档的本地 RAG + MCP 知识检索系统。

PaperPilot-Lite 实现了一套完整的论文 RAG Pipeline，支持 **Dense Retrieval、BM25、Hybrid Retrieval、Cross-Encoder Reranking、LLM 问答与离线评测**。

在此基础上，项目进一步通过 **MCP（Model Context Protocol）** 将本地论文库暴露为 Agent Tool，并完成与 Open Deep Research 的端到端集成，使 Researcher 能够自主选择 **本地论文检索 + Web Search** 完成多源研究。

---

## ✨ 核心能力

- 文档加载、清洗、切分与本地索引
- Dense Retrieval + BM25
- Min-Max / Reciprocal Rank Fusion（RRF）
- Cross-Encoder 二阶段重排序
- Mock / OpenAI-Compatible LLM
- grounded / extractive / explainer Prompt
- Recall@K、MRR@K、Latency 离线评测
- CLI：`ingest` / `search` / `ask` / `eval`
- MCP Server：`search_papers` / `ask_papers`
- Open Deep Research 多源 Agent 集成
- 174 个自动化测试通过

---

## 🏗️ 系统架构

### RAG Pipeline

```text
Documents
   ↓
Load / Clean / Chunk
   ↓
Embedding + Index
   ↓
┌──────────────────────────┐
│ Dense Retrieval          │
│        +                 │
│ BM25 Retrieval           │
│        ↓                 │
│ Hybrid Fusion (RRF)      │
└────────────┬─────────────┘
             ↓
       Candidate Pool
             ↓
 Optional Cross-Encoder
         Reranker
             ↓
         Final Top-K
             ↓
      Prompt Builder
             ↓
            LLM
             ↓
      Answer + Sources
```

### Agent / MCP

```text
                 Deep Research Agent
                        │
             Autonomous Tool Routing
                 ┌──────┴──────┐
                 ▼             ▼
         PaperPilot MCP     Web Search
                 │
          search_papers
                 │
                 ▼
        PaperPilotRuntime
                 │
          Shared Retriever
                 │
                 ▼
          Local Paper Index
```

`PaperPilotRuntime` 在 MCP Server 生命周期中只初始化一次，复用 Retriever、Embedding Model 和 Index，避免每次 Tool Call 重复加载重量级资源。

---

## 📊 Benchmark

Benchmark 使用 **8 篇公开论文、1376 个 chunk、64 条 query**，采用严格 `expected_chunk_ids` 进行 chunk-level evidence evaluation。

| 方法 | Recall@1 | Recall@3 | Recall@5 | MRR@5 | Mean Latency |
|---|---:|---:|---:|---:|---:|
| Dense | 0.2188 | 0.4375 | 0.5625 | 0.3432 | 103.24 ms |
| BM25 | 0.2969 | 0.6094 | 0.7656 | 0.4638 | 4.39 ms |
| Hybrid Min-Max | 0.3438 | 0.6406 | 0.7656 | 0.4977 | 101.83 ms |
| Hybrid RRF | 0.3594 | 0.6562 | 0.7500 | 0.5052 | 101.87 ms |
| Hybrid RRF + Reranker (`k=10`) | **0.5000** | **0.6875** | **0.7812** | **0.6068** | 355.37 ms |

主要结论：

- BM25 在当前技术论文数据上是较强的 baseline；
- Hybrid 的主要收益体现在 Recall@1、Recall@3 和 MRR@5，即改善前部排序；
- Cross-Encoder 将 **Recall@1 从 0.3594 提升至 0.5000、MRR@5 从 0.5052 提升至 0.6068**；
- `candidate_k=5/10/20/40` 消融表明，`k=10` 在当前实验中提供更合理的质量—延迟折中：继续扩大到 `k=20` 仅提高 Recall@5，但平均延迟由 **356 ms 增至 650 ms**；`k=40` 延迟超过 **1.1 s** 且排序指标出现下降。

完整实验设置、参数消融与复现方法见 [`docs/BENCHMARK.md`](docs/BENCHMARK.md)。

---

## 🚀 快速开始

### 安装

```bash
git clone <your-repository-url>
cd paperpilot-lite

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

配置环境变量：

```bash
cp .env.example .env
```

默认使用 Mock LLM，不会自动产生 API 调用费用。

### 构建索引

```bash
python -m src.paperpilot.cli ingest data/raw
```

### 检索

```bash
python -m src.paperpilot.cli search "What is RAG?"
```

### RAG 问答

```bash
python -m src.paperpilot.cli ask \
  "What is retrieval-augmented generation?"
```

使用真实 OpenAI-Compatible LLM：

```bash
python -m src.paperpilot.cli ask \
  "What is retrieval-augmented generation?" \
  --llm openai-compatible
```

### 评测

```bash
python -m src.paperpilot.cli eval \
  data/eval/papers_qa_set.jsonl \
  --index-dir data/benchmark/index \
  --retrieval-mode hybrid \
  --fusion-method rrf \
  --ks 1 3 5 \
  --mrr-k 5
```

---

## 🔌 MCP Server

PaperPilot 提供两个 MCP Tool：

| Tool | 功能 |
|---|---|
| `search_papers` | 从本地论文库返回原始 Evidence |
| `ask_papers` | 执行完整 RAG 并返回 Answer + Sources |

对于外部 Research Agent，推荐使用 `search_papers`：

```text
Agent
  ↓
search_papers
  ↓
Raw Evidence
  ↓
Agent Reasoning
```

这样可以避免：

```text
Outer Agent LLM
      ↓
PaperPilot LLM
```

形成不必要的嵌套模型调用。

### 启动

默认使用 `stdio`：

```bash
python -m src.paperpilot.mcp_server
```

Streamable HTTP：

```env
PAPERPILOT_MCP_TRANSPORT=streamable-http
PAPERPILOT_MCP_HOST=127.0.0.1
PAPERPILOT_MCP_PORT=8001
```

Endpoint：

```text
http://127.0.0.1:8001/mcp
```

---

## 🤖 Deep Research 集成

PaperPilot 已完成与 Open Deep Research 的端到端集成。

Researcher 可以自主决定：

```text
原始论文 / 本地知识
        ↓
 search_papers

最新信息 / 外部资料
        ↓
  Web Search
```

完整链路已经验证：

```text
User Question
      ↓
Research Brief
      ↓
Supervisor
      ↓
Researcher
      ↓
PaperPilot MCP + Web Search
      ↓
Evidence Synthesis
      ↓
Final Report
```

---

## 📁 项目结构

```text
paperpilot-lite/
├── configs/
│   └── rag_config.yaml
├── data/
├── docs/
│   └── BENCHMARK.md
├── scripts/
├── src/paperpilot/
│   ├── retriever.py
│   ├── keyword_retriever.py
│   ├── reranker.py
│   ├── rag_pipeline.py
│   ├── llm_client.py
│   ├── evaluator.py
│   ├── mcp_runtime.py
│   ├── mcp_server.py
│   └── cli.py
├── tests/
├── requirements.txt
└── README.md
```

---

## 🧪 测试

运行全部测试：

```bash
pytest -q
```

当前结果：

```text
174 passed
```

其中 MCP Runtime / Server：

```text
27 passed
```

---

## ⚠️ 已知限制

- Benchmark 当前仅包含 8 篇论文和 64 条 query。
- Dense Retriever 使用轻量级 Embedding Model。
- BM25 为本地轻量实现，并非生产级倒排索引。
- CPU Cross-Encoder 会显著增加检索延迟。
- MCP Server 当前主要面向本地 / 单机 Agent 场景。

---



