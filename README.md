# PaperPilot-Lite

一个面向论文与技术文档的轻量级本地 RAG（Retrieval-Augmented Generation，检索增强生成）系统。

当前版本已经完成从文档解析、切分、向量检索、BM25、Hybrid Retrieval（混合检索）、Cross-Encoder Reranker（交叉编码重排序）到 LLM 生成与离线评测的完整链路。

## 核心能力

- 文档加载、清洗、切分与本地向量索引
- Dense Retrieval（稠密检索）
- BM25 Keyword Retrieval（关键词检索）
- Hybrid Retrieval：Min-Max / Weighted RRF 两种融合方式
- Cross-Encoder Reranker：对第一阶段候选进行二阶段重排序
- Mock / OpenAI-Compatible LLM 接口
- grounded / extractive / explainer 多种 Prompt 模板
- CLI 支持 `ingest`、`search`、`ask`、`eval`
- 严格 chunk-level 检索评测
- Recall@K、MRR@K、Mean / P50 / P95 Latency
- YAML / `.env` / CLI 多层配置
- 148 个自动化测试通过

## 系统架构

```text
Raw Documents
     ↓
Loader / Cleaner / Chunker
     ↓
Embedding + Vector Index
     ↓
┌───────────────────────────┐
│      First-stage Retrieval │
│                           │
│  Dense Retrieval          │
│        +                  │
│  BM25 Retrieval           │
│        ↓                  │
│  Hybrid Fusion            │
│  ├── Min-Max              │
│  └── Weighted RRF         │
└─────────────┬─────────────┘
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
            Answer
```

默认检索配置：

```yaml
retrieval:
  mode: "hybrid"
  hybrid_alpha: 0.5
  fusion_method: "rrf"
  rrf_k: 60
  hybrid_candidate_k: 20

  reranker:
    enabled: false
    model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidate_k: 10
    device: "cpu"
    batch_size: 16
```

Reranker 默认关闭；需要更强前部排序质量时可显式开启。

## Benchmark

正式 benchmark 使用：

- 8 篇公开论文
- 1376 个 chunk
- 64 条 query
- 严格 `expected_chunk_ids` 作为 gold evidence

主要结果：

| 方法 | Recall@1 | Recall@3 | Recall@5 | MRR@5 | Mean Latency |
|---|---:|---:|---:|---:|---:|
| Dense | 0.2188 | 0.4375 | 0.5625 | 0.3432 | 105.33 ms |
| BM25 | 0.2969 | 0.6094 | 0.7656 | 0.4638 | 4.25 ms |
| Hybrid Min-Max | 0.3438 | 0.6406 | 0.7656 | 0.4977 | 104.03 ms |
| Hybrid RRF | 0.3594 | 0.6562 | 0.7500 | 0.5052 | 107.80 ms |
| Hybrid + Reranker (`k=10`) | **0.5000** | **0.6875** | 0.7812 | **0.6068** | 353.91 ms |
| Hybrid + Reranker (`k=20`) | 0.4844 | **0.6875** | **0.8125** | 0.6060 | 636.31 ms |

主要结论：

- 当前技术论文 benchmark 中，BM25 是很强的 baseline。
- Hybrid RRF 的主要收益体现在 Recall@1、Recall@3 与 MRR@5，即提升前部排序质量。
- Cross-Encoder Reranker 可进一步提升 Recall@1 与 MRR@5，但会增加 CPU 推理延迟。
- `candidate_k=10` 在当前实验中提供了较好的质量—延迟折中。
- 更完整的实验设置、消融实验与复现命令见 [`docs/BENCHMARK.md`](docs/BENCHMARK.md)。

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 配置环境变量

复制：

```bash
cp .env.example .env
```

如需调用真实 LLM，在 `.env` 中配置对应 API Key、模型名和 Base URL。

默认使用 Mock LLM，不会自动发起付费 API 请求。

## CLI 使用

### 构建索引

```bash
python -m src.paperpilot.cli ingest data/raw
```

### 检索

```bash
python -m src.paperpilot.cli search "What is RAG?"
```

使用 Hybrid + Reranker：

```bash
python -m src.paperpilot.cli search   "What search procedure finds RAG's top-K latent documents?"   --index-dir data/benchmark/index   --retrieval-mode hybrid   --fusion-method rrf   --reranker
```

### RAG 问答

Mock LLM：

```bash
python -m src.paperpilot.cli ask "What is retrieval-augmented generation?"
```

OpenAI-Compatible LLM：

```bash
python -m src.paperpilot.cli ask   "What is retrieval-augmented generation?"   --llm openai-compatible
```

### 检索评测

```bash
python -m src.paperpilot.cli eval   data/eval/papers_qa_set.jsonl   --index-dir data/benchmark/index   --retrieval-mode hybrid   --fusion-method rrf   --ks 1 3 5   --mrr-k 5
```

开启 Reranker：

```bash
python -m src.paperpilot.cli eval   data/eval/papers_qa_set.jsonl   --index-dir data/benchmark/index   --retrieval-mode hybrid   --fusion-method rrf   --reranker   --reranker-candidate-k 10   --ks 1 3 5   --mrr-k 5
```

## 项目结构

```text
paperpilot-lite/
├── configs/
│   └── rag_config.yaml
├── data/
│   ├── raw/
│   ├── eval/
│   └── benchmark/
│       └── papers.json
├── docs/
│   └── BENCHMARK.md
├── scripts/
│   ├── fetch_benchmark_papers.py
│   └── validate_benchmark.py
├── src/paperpilot/
│   ├── document_loader.py
│   ├── cleaner.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── keyword_retriever.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── prompt_builder.py
│   ├── llm_client.py
│   ├── rag_pipeline.py
│   ├── evaluator.py
│   ├── config.py
│   ├── logger.py
│   └── cli.py
└── tests/
```

## 设计要点

### Hybrid Retrieval

Dense Retrieval 提供 semantic signal（语义信号），BM25 提供 lexical signal（词法信号）。

项目支持：

```text
Dense + BM25
     ↓
Min-Max Fusion
或
Weighted RRF
```

当前默认使用 Weighted RRF。

### 两阶段检索

Reranker 不直接搜索整个语料库，而是只处理第一阶段 Retriever 返回的小规模 Candidate Pool：

```text
Hybrid Top-K Candidates
        ↓
Cross-Encoder
        ↓
Final Top-K
```

这种设计在检索质量与计算成本之间提供了更清晰的工程权衡。

### 严格评测

Benchmark 优先使用：

```text
expected_chunk_ids
```

只有检索到实际 gold evidence chunk 才算命中。

因此“找到正确论文但找到错误 chunk”不会被记为成功。

## 测试

运行全部测试：

```bash
pytest -q
```

当前：

```text
148 passed
```

## 已知限制

- Benchmark 仅包含 8 篇论文和 64 条 query，适合项目级比较，但不是大规模 IR benchmark。
- 当前 Dense Retriever 使用轻量 embedding model，不能代表最强稠密检索能力。
- 自实现 BM25 主要面向教学与中小规模 corpus，尚未采用生产级 inverted index（倒排索引）。
- CPU Cross-Encoder Reranker 会显著增加检索延迟。
- Benchmark label 与当前 chunking 规则绑定；修改解析、清洗或切分策略后需要重新校验 gold chunk。
- 当前 latency 主要用于比较数量级，尚未进行严格多轮 warm-up 与统计显著性测试。

## Roadmap

Goal 1 已完成：

```text
RAG Baseline
→ Real LLM
→ BM25
→ Hybrid Retrieval
→ Benchmark
→ Cross-Encoder Reranker
```

下一阶段 Goal 2：

```text
PaperPilot-Lite
      ↓
MCP Server
      ↓
search_papers / ask_papers
      ↓
open_deep_research
      ↓
End-to-End Research Agent
```
