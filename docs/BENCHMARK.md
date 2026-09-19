# PaperPilot-Lite 检索基准测试报告

本文档记录 PaperPilot-Lite 在 R4 阶段使用的检索基准测试、评估指标、对比实验、消融实验以及最终检索配置选择。

本次基准测试的主要目标包括：

- 在严格证据块级标签下评估检索质量；
- 对比 Dense、BM25 与 Hybrid 三类检索方法；
- 分析不同融合方法的效果差异；
- 分析 Hybrid 检索中关键超参数的敏感性；
- 为后续 R5 的 Reranker（重排序器）实验建立稳定、可复现的检索基线。

---

## 1. 基准测试设置

### 1.1 语料库

本次基准测试语料库包含 8 篇与检索增强生成、稠密检索以及 Agent 系统相关的公开论文：

- `rag_2005.11401v4.pdf`
- `dpr_2004.04906v3.pdf`
- `colbert_2004.12832v2.pdf`
- `fid_2007.01282v2.pdf`
- `hyde_2212.10496v1.pdf`
- `react_2210.03629v3.pdf`
- `self_rag_2310.11511v1.pdf`
- `toolformer_2302.04761v1.pdf`

经过预处理与文本切分后，正式 benchmark index（基准索引）包含：

- 8 篇论文；
- 1376 个 chunk（文本块）；
- 64 条评测 query（查询）。

基准语料目录：

```text
data/benchmark/raw
```

基准索引目录：

```text
data/benchmark/index
```

评测集：

```text
data/eval/papers_qa_set.jsonl
```

### 1.2 评测样本格式

`papers_qa_set.jsonl` 中每条评测数据包含如下字段：

```json
{
  "id": "rag-02",
  "question": "What search procedure finds RAG's top-K latent documents?",
  "answer": "Maximum Inner Product Search (MIPS).",
  "query_type": "abbreviation",
  "expected_source_file": "rag_2005.11401v4.pdf",
  "expected_chunk_ids": [
    "rag_2005.11401v4.pdf:2:2"
  ]
}
```

其中，检索评测最关键的字段是：

```text
expected_chunk_ids
```

该字段表示这条 query 对应的 gold evidence（标准证据块）。

### 1.3 严格的证据块级评测

项目早期的评测方式较宽松，只要检索结果来自正确论文，就可以视为命中。

当前 benchmark 使用更严格的 chunk-level evaluation（文本块级评测）。如果一条评测样本包含 `expected_chunk_ids`，那么只有当检索结果中的 `chunk_id` 与其中任意一个 gold chunk 完全匹配时，才算检索成功。

因此：

```text
正确论文 + 错误 chunk
```

不会被视为命中。

这使得当前 benchmark 真正评估的是：检索器是否找到了支持答案的实际证据块，而不仅仅是找到了正确论文。

### 1.4 Chunk ID 格式

当前 benchmark 的 chunk ID 采用：

```text
文件名:页码:chunk编号
```

例如：

```text
rag_2005.11401v4.pdf:2:2
```

因此 benchmark label（基准标签）与 benchmark index 必须来自同一套原始语料、PDF 解析逻辑、清洗逻辑、chunk size、chunk overlap 和 chunk 编号规则。

如果上述任何一个环节发生变化，都可能导致 chunk ID 改变，从而使旧的 `expected_chunk_ids` 失效。

---

## 2. 检索方法

### 2.1 Dense Retrieval（稠密检索）

Dense Retrieval 会将 query 与 chunk 编码成 embedding（向量表示），再根据向量相似度完成检索。

当前 embedding model（嵌入模型）为：

```text
sentence-transformers/all-MiniLM-L6-v2
```

Dense Retrieval 的优势是可以进行 semantic matching（语义匹配），即使 query 与证据文本没有完全相同的关键词，也可能通过语义相似度找到相关结果。

### 2.2 BM25 Keyword Retrieval（关键词检索）

BM25 根据 token overlap（词项重合）、term frequency（词频）、inverse document frequency（逆文档频率）以及文档长度归一化计算 query 与 chunk 的相关性。

当前 PaperPilot-Lite 使用自实现的轻量级 BM25，每个 chunk 都被当作一个独立的 BM25 document（文档）。

BM25 特别适合专有名词、模型名称、缩写、精确术语以及论文中的关键关键词。

### 2.3 Hybrid Min-Max Fusion（混合检索 + Min-Max 融合）

Hybrid Min-Max 同时使用 Dense 和 BM25。两路检索分数先分别进行 Min-Max normalization（最小-最大归一化），然后加权融合。

融合公式近似为：

```text
hybrid_score
    = alpha * dense_normalized_score
    + (1 - alpha) * bm25_normalized_score
```

基线实验中使用：

```text
alpha = 0.5
```

即 Dense 与 BM25 权重相同。

### 2.4 Hybrid RRF（混合检索 + 倒数排名融合）

RRF，全称 Reciprocal Rank Fusion（倒数排名融合），不直接融合原始 score，而是根据每一路检索中的 ranking position（排名位置）进行融合。

当前加权 RRF 近似为：

```text
score
    = alpha / (rrf_k + dense_rank)
    + (1 - alpha) / (rrf_k + bm25_rank)
```

其中 `alpha` 表示 Dense 权重，而 `1 - alpha` 表示 BM25 权重。

RRF 基线配置为：

```text
alpha = 0.5
rrf_k = 60
```

---

## 3. 评测指标

### 3.1 Recall@K

Recall@K 用于判断：在前 K 个检索结果中，是否至少出现一个 gold evidence chunk。

对于单条 query：

```text
如果 Top-K 中至少出现一个 gold chunk，则 Recall@K = 1
否则 Recall@K = 0
```

最终 Recall@K 是所有 query 的平均值。

本次 benchmark 统计：

```text
Recall@1
Recall@3
Recall@5
```

其中，Recall@1 更关注第一名是否正确，Recall@3 关注前三名覆盖能力，Recall@5 关注前五名中是否至少能找到一条正确证据。

### 3.2 MRR@5

MRR 全称 Mean Reciprocal Rank（平均倒数排名）。它关注第一条正确证据出现得有多靠前。

对于单条 query：

```text
RR = 1 / 第一条相关结果的排名
```

例如：

```text
正确结果位于第 1 名 -> RR = 1.0
正确结果位于第 2 名 -> RR = 0.5
正确结果位于第 4 名 -> RR = 0.25
```

如果 Top-5 中没有任何正确证据：

```text
RR = 0
```

最终 `MRR@5` 就是 64 条 query 的 RR 平均值。

MRR 可以补充 Recall 的不足。例如两种方法 Recall@5 相同，但一种方法总能把正确 chunk 排在第 1 名，另一种方法经常排在第 4、5 名，那么两者的 MRR 会明显不同。

### 3.3 Retrieval Latency（检索延迟）

当前 latency 只统计：

```python
retriever.retrieve(...)
```

本身的耗时。

计时不包含 embedding model 初始化、index 加载、benchmark 文件加载、指标计算和结果打印。

当前记录三个 latency 指标：

```text
Mean
P50
P95
```

其中：

- Mean：平均检索耗时；
- P50：中位数附近的典型耗时；
- P95：尾部延迟，即约 95% 的 query 会在该时间内完成。

当前 latency 主要用于比较不同方法的耗时量级，单次运行中几毫秒级的细小差异不应被过度解读。

---

## 4. 基线检索方法对比

所有方法均使用相同的 64 条 query、1376 个 chunk、8 篇论文以及严格的 `expected_chunk_ids`。

Hybrid Min-Max 与 Hybrid RRF 基线配置为：

```text
hybrid_alpha = 0.5
rrf_k = 60
```

### 4.1 检索质量结果

| 方法 | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|
| Dense | 0.2188 | 0.4375 | 0.5625 | 0.3432 |
| BM25 | 0.2969 | 0.6094 | **0.7656** | 0.4638 |
| Hybrid Min-Max | 0.3438 | 0.6406 | **0.7656** | 0.4977 |
| Hybrid RRF | **0.3594** | **0.6562** | 0.7500 | **0.5052** |

对应的命中 query 数量为：

| 方法 | Recall@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|
| Dense | 14 / 64 | 28 / 64 | 36 / 64 |
| BM25 | 19 / 64 | 39 / 64 | 49 / 64 |
| Hybrid Min-Max | 22 / 64 | 41 / 64 | 49 / 64 |
| Hybrid RRF | 23 / 64 | 42 / 64 | 48 / 64 |

### 4.2 检索延迟结果

| 方法 | Mean | P50 | P95 |
|---|---:|---:|---:|
| Dense | 105.33 ms | 103.79 ms | 113.56 ms |
| BM25 | 4.25 ms | 4.18 ms | 6.20 ms |
| Hybrid Min-Max | 104.03 ms | 103.45 ms | 117.38 ms |
| Hybrid RRF | 101.53 ms | 103.02 ms | 116.75 ms |

Dense 和 Hybrid 之间几毫秒的差异不应解释为 Hybrid 比 Dense 更快，因为 Hybrid 实际上同时运行了 Dense 与 BM25，再进行融合。

这些小幅差异更可能来自 CPU 调度、PyTorch inference 波动、cache（缓存）、后台系统负载以及首次调用与后续调用的差异。

当前真正有意义的 latency 结论是：BM25 大约为几毫秒，而 Dense / Hybrid 大约为 100 毫秒。

---

## 5. 基线实验分析

### 5.1 BM25 是一个很强的 Baseline

当前 benchmark 中，BM25 明显优于 Dense-only。

Recall@5 从：

```text
Dense = 0.5625
```

提升到：

```text
BM25 = 0.7656
```

绝对提升约 20.31 个百分点。

MRR@5 也从 0.3432 提升到 0.4638。

当前 benchmark 中包含大量技术术语、模型名称、缩写、精确关键词以及论文专有概念。这些 query 具有较强的 lexical signal（词法信号），因此 BM25 表现较强是合理的。

但这个结果不能被泛化成“BM25 永远优于 Dense Retrieval”。更准确的结论是：在当前语料、query 分布、embedding model 与 benchmark 条件下，BM25 比当前 Dense baseline 更强。

### 5.2 Hybrid 的主要价值是提升前部排序

Hybrid RRF 的 Recall@5 为 0.7500，实际上略低于 BM25 的 0.7656。

但 Hybrid RRF 获得了更高的：

```text
Recall@1 = 0.3594
Recall@3 = 0.6562
MRR@5 = 0.5052
```

这说明 Hybrid RRF 当前的主要优势并不是让更多 query 在 Top-5 中找到 gold evidence，而是将已经能找到的正确证据向更靠前的位置移动。

这对于后续 RAG Pipeline 很重要，因为排名更靠前的 chunk 更容易进入最终 prompt，并被 LLM 使用。

### 5.3 Min-Max 与 RRF 存在不同权衡

Hybrid Min-Max 的 Recall@5 为 0.7656，而 Hybrid RRF 为 0.7500。

但 RRF 的前部排序指标略高：

```text
RRF Recall@1 = 0.3594
Min-Max       = 0.3438

RRF Recall@3 = 0.6562
Min-Max       = 0.6406

RRF MRR@5    = 0.5052
Min-Max       = 0.4977
```

因此当前实验不能支持“RRF 在所有指标上全面优于 Min-Max”。更准确的结论是：Min-Max 在 Recall@5 上略强，而 RRF 在 Recall@1、Recall@3 与 MRR@5 上略强。

---

## 6. 消融实验说明

Ablation Study（消融实验）的目标是：在其他条件保持不变的情况下，只改变一个组件或参数，从而判断该组件或参数对系统性能的实际影响。

当前 R4.4 主要针对 Hybrid RRF 的两个关键超参数：

```text
hybrid_alpha
rrf_k
```

分别进行消融。

实验遵循控制变量原则：每轮实验只改变一个参数，其余 benchmark 条件保持完全一致。

---

## 7. Alpha 消融实验

### 7.1 实验设置

固定：

```text
retrieval_mode = hybrid
fusion_method = rrf
rrf_k = 60
```

只改变：

```text
hybrid_alpha
```

测试：

```text
0.3
0.5
0.7
```

对应含义：

```text
alpha = 0.3
Dense 权重 = 30%
BM25 权重  = 70%

alpha = 0.5
Dense 权重 = 50%
BM25 权重  = 50%

alpha = 0.7
Dense 权重 = 70%
BM25 权重  = 30%
```

### 7.2 实验结果

| Alpha | Dense 权重 | BM25 权重 | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.3 | 0.7 | 0.3281 | **0.6719** | 0.7344 | 0.5018 |
| 0.5 | 0.5 | 0.5 | **0.3594** | 0.6562 | **0.7500** | **0.5052** |
| 0.7 | 0.7 | 0.3 | 0.3438 | 0.5312 | **0.7500** | 0.4745 |

对应的 Recall 命中数量：

| Alpha | Recall@1 | Recall@3 | Recall@5 |
|---:|---:|---:|---:|
| 0.3 | 21 / 64 | 43 / 64 | 47 / 64 |
| 0.5 | 23 / 64 | 42 / 64 | 48 / 64 |
| 0.7 | 22 / 64 | 34 / 64 | 48 / 64 |

### 7.3 Alpha 消融分析

`alpha=0.3` 与 `alpha=0.5` 整体表现比较接近。

`alpha=0.3` 获得最高 Recall@3，为 0.6719，而 `alpha=0.5` 为 0.6562，两者仅相差 1 条 query，即 43/64 与 42/64。

但 `alpha=0.5` 同时获得最高 Recall@1、并列最高 Recall@5 以及最高 MRR@5。

当 Dense 权重提高到 `alpha=0.7` 时，Recall@3 与 MRR@5 明显下降。Recall@3 从 0.6562 下降到 0.5312，MRR@5 从 0.5052 下降到 0.4745。

这说明当前 benchmark 仍然需要保留较强的 BM25 信号。Dense Retrieval 可以提供互补信息，但 Dense 权重明显占主导时，会降低当前任务中的前部排序质量。

因此当前保留：

```text
hybrid_alpha = 0.5
```

作为默认值。原因不是 `0.5` 在每个指标上都绝对第一，而是它在 Recall@1、Recall@5 与 MRR@5 上表现最均衡。

---

## 8. RRF k 消融实验

### 8.1 实验设置

固定：

```text
retrieval_mode = hybrid
fusion_method = rrf
hybrid_alpha = 0.5
```

只改变：

```text
rrf_k
```

测试：

```text
20
40
60
```

`rrf_k` 控制排名位置对最终 RRF score 的影响程度。较小的 `rrf_k` 会放大高排名与低排名之间的差异，较大的 `rrf_k` 会使排名差异更加平滑。

### 8.2 实验结果

| RRF k | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---:|---:|---:|---:|---:|
| 20 | 0.3594 | **0.6562** | 0.7500 | **0.5052** |
| 40 | 0.3594 | 0.6406 | 0.7500 | 0.5013 |
| 60 | 0.3594 | **0.6562** | 0.7500 | **0.5052** |

对应的 Recall 命中数量：

| RRF k | Recall@1 | Recall@3 | Recall@5 |
|---:|---:|---:|---:|
| 20 | 23 / 64 | 42 / 64 | 48 / 64 |
| 40 | 23 / 64 | 41 / 64 | 48 / 64 |
| 60 | 23 / 64 | 42 / 64 | 48 / 64 |

### 8.3 RRF k 消融分析

实验结果表明，当前 benchmark 对 `rrf_k` 并不敏感。

`rrf_k=20` 和 `rrf_k=60` 在 Recall@1、Recall@3、Recall@5 和 MRR@5 四个指标上完全一致。

`rrf_k=40` 仅在 Recall@3 上少命中 1 条 query，即从 42/64 变为 41/64，MRR@5 也仅从 0.5052 变为 0.5013，差异很小。

因此当前实验并不能证明 `rrf_k=60` 是最优值。更准确的结论是：在当前 benchmark 中，RRF 在 `rrf_k=20~60` 范围内表现相对稳定，对该参数不敏感。

由于 `rrf_k=60` 原本就是项目默认值，且 `rrf_k=20` 与 `rrf_k=60` 指标一致，没有证据表明修改默认值能带来稳定收益，因此继续保留：

```text
rrf_k = 60
```

作为默认值。

---

## 9. 最终检索配置

综合基线实验与消融实验，当前项目默认 retrieval configuration（检索配置）为：

```yaml
retrieval:
  mode: hybrid
  fusion_method: rrf
  hybrid_alpha: 0.5
  rrf_k: 60
```

选择该配置的主要原因是 Hybrid RRF 在当前 benchmark 上提供了最强的前部排序质量。

在四种 baseline 中，它取得了最高的 Recall@1、Recall@3 和 MRR@5，但并没有获得最高 Recall@5。

因此这一配置本质上是一个 trade-off（权衡）：更强调将正确证据排到更靠前的位置，而不是单纯追求最大 Top-5 覆盖率。

---

## 10. 主要实验结论

### 10.1 BM25 在当前论文检索任务中表现很强

BM25 明显优于当前 Dense-only baseline。

当前 query 中包含大量技术术语与缩写，例如：

```text
RAG
MIPS
DPR
BART
ColBERT
FiD
ReAct
Toolformer
```

因此 lexical matching（词法匹配）具有明显优势。

### 10.2 Dense Retrieval 仍然具有互补价值

虽然 Dense-only 整体弱于 BM25，但加入 Dense 后，Recall@1、Recall@3 和 MRR@5 进一步提高。

说明 Dense Retrieval 仍然为部分 query 提供了 BM25 无法完全覆盖的 semantic signal（语义信号）。

### 10.3 Hybrid 的主要优势是排序质量

Hybrid RRF 并没有提高 Recall@5，它的主要优势体现在 Recall@1、Recall@3 与 MRR@5。

因此 Hybrid 当前最明显的作用是将正确证据向排名前部移动。

### 10.4 Dense 权重不应过高

Alpha 消融显示 `alpha=0.7` 时 Recall@3 与 MRR@5 明显下降，说明当前 benchmark 不适合明显偏向 Dense Retrieval。

### 10.5 RRF 对 rrf_k 不敏感

`rrf_k` 在 20、40、60 之间变化时，整体指标差异很小，说明当前 Hybrid RRF 对这一参数相对 robust（稳健）。

### 10.6 BM25 延迟显著低于 Dense / Hybrid

当前 1376 chunk benchmark 上：

```text
BM25：约几毫秒
Dense / Hybrid：约 100 ms
```

Dense 与 Hybrid 的主要耗时来自 query embedding inference（查询向量推理）。

但当前语料规模仍然较小，因此 BM25 的几毫秒结果不能直接外推到大规模生产级检索系统。

---

## 11. 当前 Benchmark 的限制

### 11.1 Benchmark 规模较小

当前只有 64 条 query 与 8 篇论文。

它足以支持项目级工程比较，但不能视为大规模 IR benchmark（信息检索基准）。

由于：

```text
1 / 64 = 1.5625%
```

因此仅相差 1~2 条 query 的结果不应该被过度解读。

### 11.2 Query 分布可能天然偏向 BM25

当前 benchmark 包含较多精确模型名、缩写、技术术语以及论文专有词汇，这会天然增强 BM25 的优势。

未来如果需要更全面测试 Dense Retrieval，可以增加更多 paraphrase query（改写问题）、semantic query（语义问题）以及不直接包含论文原词的问题。

### 11.3 Dense 使用的是轻量 Embedding Model

当前 Dense Retriever 使用：

```text
sentence-transformers/all-MiniLM-L6-v2
```

它适合作为轻量本地 baseline，但并不是当前最强的 retrieval-specific embedding model。

因此 `BM25 > Dense` 只代表当前项目配置下的实验结果，不能泛化成 Sparse Retrieval 永远优于 Dense Retrieval。

### 11.4 当前 BM25 不是生产级实现

当前 BM25 主要用于教学、项目实验与中小规模检索。

当前约 1376 chunks 的规模下，直接扫描仍然很快。但更大规模 corpus 通常需要 inverted index（倒排索引）或专业搜索引擎，而不是直接遍历所有 chunk。

### 11.5 Latency 仍然是单轮本地实验

当前 latency 受到 CPU scheduling、PyTorch inference 波动、cache、后台系统负载以及 first-query warm-up 等因素影响。

因此当前 latency 更适合比较数量级，而不是比较 `101 ms vs 104 ms` 这种细微差异。

未来如果需要更严格的性能测试，可以加入 warm-up queries、多轮重复实验、平均值和标准差。

### 11.6 Benchmark Overfitting

当前同一套 64-query benchmark 既用于评测，也用于分析参数。

如果不断进行非常细粒度的参数搜索，就可能对 benchmark 本身过拟合。

因此当前只使用较粗粒度的：

```text
alpha = 0.3 / 0.5 / 0.7
rrf_k = 20 / 40 / 60
```

目的不是榨取最后一点分数，而是理解系统行为。

### 11.7 Gold Label 依赖当前 Chunking 配置

当前严格评测依赖 `expected_chunk_ids`。

因此如果修改 loader、cleaner、chunk size、chunk overlap、page extraction 或 chunk 编号逻辑，都必须重新检查 benchmark label 是否仍然有效。

---

## 12. 实验复现命令

正式 benchmark index：

```text
data/benchmark/index
```

正式 evaluation set：

```text
data/eval/papers_qa_set.jsonl
```

### 12.1 Dense

```bash
python -m src.paperpilot.cli eval \
  data/eval/papers_qa_set.jsonl \
  --index-dir data/benchmark/index \
  --retrieval-mode dense
```

### 12.2 BM25

```bash
python -m src.paperpilot.cli eval \
  data/eval/papers_qa_set.jsonl \
  --index-dir data/benchmark/index \
  --retrieval-mode keyword
```

### 12.3 Hybrid Min-Max

```bash
python -m src.paperpilot.cli eval \
  data/eval/papers_qa_set.jsonl \
  --index-dir data/benchmark/index \
  --retrieval-mode hybrid \
  --fusion-method minmax \
  --hybrid-alpha 0.5
```

### 12.4 Hybrid RRF

```bash
python -m src.paperpilot.cli eval \
  data/eval/papers_qa_set.jsonl \
  --index-dir data/benchmark/index \
  --retrieval-mode hybrid \
  --fusion-method rrf \
  --hybrid-alpha 0.5 \
  --rrf-k 60
```

---

## 13. Alpha 消融复现

### Alpha = 0.3

```bash
python -m src.paperpilot.cli eval \
  data/eval/papers_qa_set.jsonl \
  --index-dir data/benchmark/index \
  --retrieval-mode hybrid \
  --fusion-method rrf \
  --hybrid-alpha 0.3 \
  --rrf-k 60
```

### Alpha = 0.5

```bash
python -m src.paperpilot.cli eval \
  data/eval/papers_qa_set.jsonl \
  --index-dir data/benchmark/index \
  --retrieval-mode hybrid \
  --fusion-method rrf \
  --hybrid-alpha 0.5 \
  --rrf-k 60
```

### Alpha = 0.7

```bash
python -m src.paperpilot.cli eval \
  data/eval/papers_qa_set.jsonl \
  --index-dir data/benchmark/index \
  --retrieval-mode hybrid \
  --fusion-method rrf \
  --hybrid-alpha 0.7 \
  --rrf-k 60
```

---

## 14. RRF k 消融复现

### RRF k = 20

```bash
python -m src.paperpilot.cli eval \
  data/eval/papers_qa_set.jsonl \
  --index-dir data/benchmark/index \
  --retrieval-mode hybrid \
  --fusion-method rrf \
  --hybrid-alpha 0.5 \
  --rrf-k 20
```

### RRF k = 40

```bash
python -m src.paperpilot.cli eval \
  data/eval/papers_qa_set.jsonl \
  --index-dir data/benchmark/index \
  --retrieval-mode hybrid \
  --fusion-method rrf \
  --hybrid-alpha 0.5 \
  --rrf-k 40
```

### RRF k = 60

```bash
python -m src.paperpilot.cli eval \
  data/eval/papers_qa_set.jsonl \
  --index-dir data/benchmark/index \
  --retrieval-mode hybrid \
  --fusion-method rrf \
  --hybrid-alpha 0.5 \
  --rrf-k 60
```

---

## 15. 下一阶段

当前 R4 已经建立了稳定的 retrieval baseline（检索基线）。

下一阶段 R5 将加入：

```text
Hybrid Retrieval
        ↓
Candidate Pool（候选集）
        ↓
Cross-Encoder Reranker（交叉编码重排序器）
        ↓
Final Top-K Evidence
```

R5 将继续使用同一套 benchmark，对比：

```text
Hybrid
vs
Hybrid + Reranker
```

评估指标保持：

```text
Recall@1
Recall@3
Recall@5
MRR@5
Mean Latency
P50 Latency
P95 Latency
```

最终判断：Reranker 带来的排序收益，是否值得额外的计算成本。
