# PaperPilot-Lite Retrieval Benchmark

本文档记录 PaperPilot-Lite 的检索评测、组件对比、关键参数消融及最终配置选择。

## 1. Benchmark 设置

正式评测集包含：

- 8 篇公开论文；
- 1376 个 chunk；
- 64 条 query；
- 使用 `expected_chunk_ids` 作为 gold evidence。

路径：

```text
Corpus: data/benchmark/raw
Index:  data/benchmark/index
QA:     data/eval/papers_qa_set.jsonl
```

采用严格 chunk-level evaluation：

```text
正确论文 + 错误 chunk ≠ 命中
```

仅当检索结果中的 `chunk_id` 与 `expected_chunk_ids` 匹配时计为成功。

主要指标：

- `Recall@1 / Recall@3 / Recall@5`
- `MRR@5`
- `Mean / P50 / P95 Latency`

Latency 仅统计 `retriever.retrieve()`，不包含模型初始化、索引加载和评测代码开销。

---

## 2. 检索架构

当前 Retrieval Pipeline：

```text
                  ┌── Dense Retrieval
Query ────────────┤
                  └── BM25 Retrieval
                           ↓
                    Hybrid Fusion
                  Min-Max / Weighted RRF
                           ↓
                     Candidate Pool
                           ↓
               Optional Cross-Encoder
                           ↓
                       Top-K
```

默认模型：

```text
Embedding:
sentence-transformers/all-MiniLM-L6-v2

Reranker:
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Hybrid 默认配置：

```text
hybrid_alpha = 0.5
fusion_method = rrf
rrf_k = 60
hybrid_candidate_k = 20
```

---

## 3. 组件对比实验

所有方法使用相同语料、Query、Embedding Model 和评测标签。

| Method | Recall@1 | Recall@3 | Recall@5 | MRR@5 | Mean Latency |
|---|---:|---:|---:|---:|---:|
| Dense | 0.2188 | 0.4375 | 0.5625 | 0.3432 | 103.24 ms |
| BM25 | 0.2969 | 0.6094 | 0.7656 | 0.4638 | 4.39 ms |
| Hybrid Min-Max | 0.3438 | 0.6406 | **0.7656** | 0.4977 | 101.83 ms |
| Hybrid RRF | 0.3594 | 0.6562 | 0.7500 | 0.5052 | 101.87 ms |
| Hybrid RRF + Cross-Encoder (`k=10`) | **0.5000** | **0.6875** | **0.7812** | **0.6068** | 355.37 ms |

对应 Recall 命中数：

| Method | R@1 | R@3 | R@5 |
|---|---:|---:|---:|
| Dense | 14/64 | 28/64 | 36/64 |
| BM25 | 19/64 | 39/64 | 49/64 |
| Hybrid Min-Max | 22/64 | 41/64 | 49/64 |
| Hybrid RRF | 23/64 | 42/64 | 48/64 |
| Hybrid RRF + Cross-Encoder | **32/64** | **44/64** | **50/64** |

### 结论

- **BM25 是当前数据上的强 baseline。** 当前 Query 含较多模型名、缩写和技术术语，lexical matching 较重要。
- **Dense 与 BM25 具有互补性。** Hybrid 没有明显提高 Recall@5，但改善了 Recall@1、Recall@3 和 MRR@5。
- **RRF 与 Min-Max 差异较小。** RRF 的前部排序指标略高，因此作为默认 Fusion。
- **Cross-Encoder 提升最明显。** Recall@1 从 `0.3594` 提升到 `0.5000`，MRR@5 从 `0.5052` 提升到 `0.6068`，代价是更高的 CPU 延迟。

由于 Benchmark 仅有 64 条 Query，1 条 Query 对应约 1.56 个百分点，因此不应过度解读 1～2 条 Query 的差异。

---

## 4. Hybrid 参数消融

### 4.1 Hybrid Alpha

固定 RRF，其余配置不变：

| Alpha | Dense Weight | BM25 Weight | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.3 | 0.7 | 0.3281 | **0.6719** | 0.7344 | 0.5018 |
| 0.5 | 0.5 | 0.5 | **0.3594** | 0.6562 | **0.7500** | **0.5052** |
| 0.7 | 0.7 | 0.3 | 0.3438 | 0.5312 | **0.7500** | 0.4745 |

`alpha=0.5` 在 Recall@1、Recall@5 和 MRR@5 上整体最均衡，因此保持为默认值。

Dense 权重增加到 `0.7` 后 Recall@3 和 MRR@5 明显下降，说明当前 Benchmark 仍依赖较强的 BM25 signal。

### 4.2 RRF k

固定 `alpha=0.5`：

| RRF k | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---:|---:|---:|---:|---:|
| 20 | 0.3594 | **0.6562** | 0.7500 | **0.5052** |
| 40 | 0.3594 | 0.6406 | 0.7500 | 0.5013 |
| 60 | 0.3594 | **0.6562** | 0.7500 | **0.5052** |

当前 Benchmark 对 `rrf_k=20~60` 不敏感，因此保留默认值：

```text
rrf_k = 60
```

---

## 5. Reranker Candidate-K 消融

固定：

```text
retrieval = Hybrid RRF
hybrid_alpha = 0.5
rrf_k = 60
Cross-Encoder = ms-marco-MiniLM-L-6-v2
device = CPU
```

仅改变 `reranker_candidate_k`。

每个配置独立运行 3 次；质量指标三次完全一致，Latency 取三次平均。

| candidate_k | Recall@1 | Recall@3 | Recall@5 | MRR@5 | Mean | P50 | P95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.4531 | **0.7031** | 0.7500 | 0.5734 | **225.03 ms** | 216.51 ms | 289.56 ms |
| 10 | **0.5000** | 0.6875 | 0.7812 | **0.6068** | 356.10 ms | 338.67 ms | 451.20 ms |
| 20 | 0.4844 | 0.6875 | **0.8125** | 0.6060 | 649.88 ms | 618.40 ms | 858.98 ms |
| 40 | 0.4688 | 0.6719 | 0.7812 | 0.5883 | 1178.59 ms | 1176.72 ms | 1456.46 ms |

### 结论

`candidate_k=5` 延迟最低，但 Recall@1 和 MRR@5 较低。

`candidate_k=10` 获得最高 Recall@1 和 MRR@5，是当前最均衡的质量—延迟配置。

`candidate_k=20` 将 Recall@5 从 `0.7812` 提升到 `0.8125`，但平均延迟从约 `356 ms` 增加到 `650 ms`，前部排序没有进一步改善。

`candidate_k=40` 平均延迟超过 `1.1 s`，且各项排序指标未继续提升，表现出明显的 diminishing returns。

因此默认选择：

```text
reranker_candidate_k = 10
```

---

## 6. 最终配置

```yaml
retrieval:
  mode: hybrid
  hybrid_alpha: 0.5
  fusion_method: rrf
  rrf_k: 60
  hybrid_candidate_k: 20

  reranker:
    enabled: false
    model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
    candidate_k: 10
    device: cpu
    batch_size: 16
```

Reranker 默认关闭，避免普通检索请求承担额外 Cross-Encoder 推理成本。

需要更强排序质量时显式开启：

```text
Hybrid RRF
    ↓
Top-10 Candidates
    ↓
Cross-Encoder
    ↓
Final Top-K
```

---

## 7. 最终结论

本轮实验得到以下工程结论：

1. BM25 在当前技术论文数据上明显强于 Dense-only；
2. Dense 与 BM25 具有互补性，Hybrid 主要改善前部排序；
3. Weighted RRF 与 Min-Max 表现接近，RRF 前部指标略优；
4. Cross-Encoder 是当前最有效的排序增强模块；
5. `candidate_k` 存在明显质量—延迟 trade-off；
6. `candidate_k=10` 是当前 Benchmark 下更合理的默认配置；
7. 更大的 Candidate Pool 不保证更好的最终排序。

---

## 8. Benchmark 限制

当前结果仅适用于本项目实验条件：

- 仅 8 篇论文、64 条 Query；
- Query 中包含较多技术术语和缩写，可能偏向 BM25；
- Dense 使用轻量 `all-MiniLM-L6-v2`，不代表更强 Embedding Model 的效果；
- BM25 为轻量自实现版本，不是生产级倒排索引；
- Latency 为本地 CPU 实验，主要用于比较数量级；
- `expected_chunk_ids` 与当前 Chunking 配置绑定。

因此实验目标是比较当前系统组件和工程 trade-off，而不是给出通用 IR 结论。

---

## 9. 复现

Hybrid RRF：

```bash
python -m src.paperpilot.cli eval \
  data/eval/papers_qa_set.jsonl \
  --index-dir data/benchmark/index \
  --retrieval-mode hybrid \
  --hybrid-alpha 0.5 \
  --fusion-method rrf \
  --rrf-k 60 \
  --hybrid-candidate-k 20 \
  --no-reranker \
  --ks 1 3 5 \
  --mrr-k 5
```

Hybrid RRF + Cross-Encoder：

```bash
python -m src.paperpilot.cli eval \
  data/eval/papers_qa_set.jsonl \
  --index-dir data/benchmark/index \
  --retrieval-mode hybrid \
  --hybrid-alpha 0.5 \
  --fusion-method rrf \
  --rrf-k 60 \
  --hybrid-candidate-k 20 \
  --reranker \
  --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --reranker-candidate-k 10 \
  --reranker-device cpu \
  --reranker-batch-size 16 \
  --ks 1 3 5 \
  --mrr-k 5
```