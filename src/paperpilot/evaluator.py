from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

from .retriever import Retriever


QAItem = dict[str, Any]
SearchResult = dict[str, Any]
EvaluationReport = dict[str, Any]


def load_qa_set(
    qa_path: str | Path,
) -> list[QAItem]:
    """从 JSONL 文件加载 QA evaluation set。"""

    path = Path(qa_path)

    if not path.exists():
        raise FileNotFoundError(
            f"QA file not found: {path}"
        )

    qa_items: list[QAItem] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_no, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_no}: {line}"
                ) from exc

            _validate_qa_item(
                item=item,
                line_no=line_no,
            )

            qa_items.append(item)

    if not qa_items:
        raise ValueError(
            f"QA file is empty: {path}"
        )

    return qa_items


def evaluate_retrieval(
    retriever: Retriever,
    qa_items: list[QAItem],
    ks: list[int] | tuple[int, ...] = (
        1,
        3,
        5,
    ),
) -> EvaluationReport:
    """评估 retrieval 的 Recall@K、MRR@K 和 retrieval latency。"""

    if retriever is None:
        raise ValueError(
            "retriever must not be None"
        )

    if not isinstance(
        qa_items,
        list,
    ):
        raise TypeError(
            "qa_items must be a list"
        )

    if not qa_items:
        raise ValueError(
            "qa_items must not be empty"
        )

    ks = _validate_ks(ks)
    max_k = max(ks)

    # Recall@K 的命中次数。
    hit_counts = {
        k: 0
        for k in ks
    }

    # 用于最后计算 MRR@max_k。
    reciprocal_rank_sum = 0.0

    # 保存每个 query 的 retrieval latency。
    latencies_ms: list[float] = []

    cases: list[dict[str, Any]] = []
    failed_cases: list[dict[str, Any]] = []

    for item_id, item in enumerate(
        qa_items,
        start=1,
    ):
        question = item["question"]

        # -------------------------------------------------
        # Retrieval latency
        #
        # 这里只测 retriever.retrieve()：
        # 不包括模型初始化、index 加载、QA 加载和报告打印。
        # -------------------------------------------------
        start_time = time.perf_counter()

        results = retriever.retrieve(
            query=question,
            top_k=max_k,
        )

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        latencies_ms.append(
            latency_ms
        )

        # -------------------------------------------------
        # MRR@K
        # -------------------------------------------------
        first_relevant_rank = (
            find_first_relevant_rank(
                results=results,
                item=item,
            )
        )

        reciprocal_rank = (
            1.0 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        )

        reciprocal_rank_sum += (
            reciprocal_rank
        )

        # -------------------------------------------------
        # Recall@K
        # -------------------------------------------------
        case_hits: dict[str, bool] = {}

        for k in ks:
            top_results = results[:k]

            hit_info = check_hit(
                top_results,
                item,
            )

            hit = hit_info["hit"]

            case_hits[
                f"Recall@{k}"
            ] = hit

            if hit:
                hit_counts[k] += 1

        # -------------------------------------------------
        # 保存单条 query 的详细 evaluation 信息
        # -------------------------------------------------
        case = {
            "item_id": item.get(
                "id",
                item_id,
            ),
            "question": question,
            "expected_source_file": item.get(
                "expected_source_file"
            ),
            "expected_keywords": item.get(
                "expected_keywords",
                [],
            ),
            "expected_chunk_ids": item.get(
                "expected_chunk_ids",
                [],
            ),
            "hits": case_hits,

            # Ranking quality
            "first_relevant_rank": (
                first_relevant_rank
            ),
            "reciprocal_rank": (
                reciprocal_rank
            ),

            # Retrieval efficiency
            "latency_ms": latency_ms,

            "top_results": summarize_results(
                results
            ),
        }

        cases.append(case)

        # max_k 内仍然没有命中 gold evidence。
        if not case_hits[
            f"Recall@{max_k}"
        ]:
            failed_cases.append(case)

    total = len(qa_items)

    # -------------------------------------------------
    # Recall@K
    # -------------------------------------------------
    recall = {
        k: hit_counts[k] / total
        for k in ks
    }

    # -------------------------------------------------
    # MRR@max_k
    # -------------------------------------------------
    mrr = (
        reciprocal_rank_sum
        / total
    )

    # -------------------------------------------------
    # Retrieval latency statistics
    # -------------------------------------------------
    latency_mean_ms = mean(
        latencies_ms
    )

    latency_p50_ms = _percentile(
        latencies_ms,
        0.50,
    )

    latency_p95_ms = _percentile(
        latencies_ms,
        0.95,
    )

    return {
        "total": total,
        "ks": list(ks),

        "hit_counts": hit_counts,
        "recall": recall,

        # 如果 ks = [1, 3, 5]，
        # 当前指标就是 MRR@5。
        "mrr_k": max_k,
        "mrr": mrr,

        "latency_ms": {
            "mean": latency_mean_ms,
            "p50": latency_p50_ms,
            "p95": latency_p95_ms,
        },

        "cases": cases,
        "failed_cases": failed_cases,
    }


def check_hit(
    results: list[SearchResult],
    item: QAItem,
) -> dict[str, Any]:
    """判断 retrieval results 是否命中 QA 样本的 gold evidence。"""

    if not isinstance(
        results,
        list,
    ):
        raise TypeError(
            "results must be a list"
        )

    expected_source_files = (
        _get_expected_source_files(item)
    )

    expected_keywords = item.get(
        "expected_keywords",
        [],
    )

    if expected_keywords is None:
        expected_keywords = []

    if not isinstance(
        expected_keywords,
        list,
    ):
        raise TypeError(
            "expected_keywords must be a list"
        )

    source_hit = False
    keyword_hit = False

    if expected_source_files:
        source_hit = any(
            _get_result_file_name(result)
            in expected_source_files
            for result in results
        )

    if expected_keywords:
        keyword_hit = (
            _contains_any_keyword(
                results,
                expected_keywords,
            )
        )

    chunk_hit = False

    # 如果存在 expected_chunk_ids，
    # 严格使用 gold chunk 作为最终判定标准。
    if "expected_chunk_ids" in item:
        expected_chunk_ids = set(
            item["expected_chunk_ids"]
        )

        for result in results:
            metadata = result.get(
                "metadata"
            )

            if (
                isinstance(metadata, dict)
                and metadata.get("chunk_id")
                in expected_chunk_ids
            ):
                chunk_hit = True
                break

        hit = chunk_hit

    else:
        # 兼容旧 benchmark。
        hit = (
            source_hit
            or keyword_hit
        )

    return {
        "hit": hit,
        "chunk_hit": chunk_hit,
        "source_hit": source_hit,
        "keyword_hit": keyword_hit,
    }


def find_first_relevant_rank(
    results: list[SearchResult],
    item: QAItem,
) -> int | None:
    """返回第一个 relevant result 的 1-based rank。

    如果当前 results 中没有命中 gold evidence，
    则返回 None。
    """

    if not isinstance(
        results,
        list,
    ):
        raise TypeError(
            "results must be a list"
        )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        # 复用 check_hit()，
        # 保证 Recall 和 MRR 使用完全相同的
        # relevance definition。
        hit_info = check_hit(
            results=[result],
            item=item,
        )

        if hit_info["hit"]:
            return rank

    return None


def summarize_results(
    results: list[SearchResult],
    max_text_chars: int = 200,
) -> list[dict[str, Any]]:
    """生成适合 evaluation report 展示的检索结果摘要。"""

    summaries: list[
        dict[str, Any]
    ] = []

    for rank, result in enumerate(
        results,
        start=1,
    ):
        metadata = result.get(
            "metadata",
            {},
        )

        if metadata is None:
            metadata = {}

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {}

        text = result.get(
            "text",
            "",
        )

        if not isinstance(
            text,
            str,
        ):
            text = str(text)

        score = result.get(
            "score"
        )

        summaries.append(
            {
                "rank": rank,
                "file_name": (
                    metadata.get(
                        "file_name"
                    )
                    or metadata.get(
                        "source"
                    )
                ),
                "page": metadata.get(
                    "page"
                ),
                "chunk_id": metadata.get(
                    "chunk_id"
                ),
                "score": (
                    float(score)
                    if isinstance(
                        score,
                        (int, float),
                    )
                    else score
                ),
                "text_preview": (
                    text[:max_text_chars]
                ),
            }
        )

    return summaries


def print_evaluation_report(
    report: EvaluationReport,
    show_failed_cases: bool = True,
    max_failed_cases: int = 10,
) -> None:
    """打印 Retrieval Evaluation 报告。"""

    print("=" * 80)
    print(
        "Retrieval Evaluation Report"
    )
    print("=" * 80)

    print(
        f"Total questions: "
        f"{report['total']}"
    )

    print()

    # -------------------------------------------------
    # Recall@K
    # -------------------------------------------------
    for k in report["ks"]:
        hit_count = (
            report["hit_counts"][k]
        )

        recall = (
            report["recall"][k]
        )

        print(
            f"Recall@{k}: "
            f"{recall:.4f} "
            f"({hit_count}/{report['total']})"
        )

    # -------------------------------------------------
    # MRR@K
    # -------------------------------------------------
    if (
        "mrr" in report
        and "mrr_k" in report
    ):
        print()

        print(
            f"MRR@{report['mrr_k']}: "
            f"{report['mrr']:.4f}"
        )

    # -------------------------------------------------
    # Retrieval latency
    # -------------------------------------------------
    latency = report.get(
        "latency_ms"
    )

    if isinstance(
        latency,
        dict,
    ):
        print()

        print(
            "Retrieval latency:"
        )

        print(
            f"  Mean: "
            f"{latency['mean']:.2f} ms"
        )

        print(
            f"  P50:  "
            f"{latency['p50']:.2f} ms"
        )

        print(
            f"  P95:  "
            f"{latency['p95']:.2f} ms"
        )

    failed_cases = report[
        "failed_cases"
    ]

    print()

    print(
        f"Failed cases at "
        f"Recall@{max(report['ks'])}: "
        f"{len(failed_cases)}"
    )

    if (
        show_failed_cases
        and failed_cases
    ):
        print()
        print("=" * 80)
        print(
            "Failed Case Details"
        )
        print("=" * 80)

        for case in failed_cases[
            :max_failed_cases
        ]:
            print()
            print("-" * 80)

            print(
                f"Item ID: "
                f"{case['item_id']}"
            )

            print(
                f"Question: "
                f"{case['question']}"
            )

            print(
                f"Expected source: "
                f"{case['expected_source_file']}"
            )

            print(
                f"Expected keywords: "
                f"{case['expected_keywords']}"
            )

            if case[
                "expected_chunk_ids"
            ]:
                print(
                    "Expected chunk IDs: "
                    f"{case['expected_chunk_ids']}"
                )

            print(
                "Top results:"
            )

            for result in case[
                "top_results"
            ]:
                print(
                    f"  Rank "
                    f"{result['rank']} | "
                    f"file="
                    f"{result['file_name']} | "
                    f"page="
                    f"{result['page']} | "
                    f"score="
                    f"{result['score']}"
                )

                print(
                    f"  Text: "
                    f"{result['text_preview']}"
                )

                print()


def evaluate_retrieval_from_index(
    index_dir: str | Path,
    qa_path: str | Path,
    model_name: str = (
        "sentence-transformers/"
        "all-MiniLM-L6-v2"
    ),
    device: str | None = "cpu",
    normalize_embeddings: bool = True,
    batch_size: int = 32,
    ks: list[int] | tuple[int, ...] = (
        1,
        3,
        5,
    ),
) -> EvaluationReport:
    """从保存的 index 直接执行 retrieval evaluation。"""

    ks = _validate_ks(ks)

    retriever = Retriever.from_index(
        index_dir=index_dir,
        model_name=model_name,
        device=device,
        normalize_embeddings=(
            normalize_embeddings
        ),
        batch_size=batch_size,
        default_top_k=max(ks),
    )

    qa_items = load_qa_set(
        qa_path
    )

    return evaluate_retrieval(
        retriever=retriever,
        qa_items=qa_items,
        ks=ks,
    )


def _percentile(
    values: list[float],
    percentile: float,
) -> float:
    """使用线性插值计算 percentile。

    percentile 使用 0~1 范围：
    0.50 -> P50
    0.95 -> P95
    """

    if not values:
        raise ValueError(
            "values must not be empty"
        )

    if not (
        0.0
        <= percentile
        <= 1.0
    ):
        raise ValueError(
            "percentile must be between 0 and 1"
        )

    sorted_values = sorted(
        values
    )

    if len(sorted_values) == 1:
        return sorted_values[0]

    # 例如 N=4、P50：
    # position = (4 - 1) * 0.5 = 1.5
    position = (
        len(sorted_values) - 1
    ) * percentile

    lower_index = int(
        position
    )

    upper_index = min(
        lower_index + 1,
        len(sorted_values) - 1,
    )

    fraction = (
        position
        - lower_index
    )

    lower_value = (
        sorted_values[
            lower_index
        ]
    )

    upper_value = (
        sorted_values[
            upper_index
        ]
    )

    return (
        lower_value
        + (
            upper_value
            - lower_value
        )
        * fraction
    )


def _validate_qa_item(
    item: Any,
    line_no: int,
) -> None:
    """检查单条 QA 样本是否合法。"""

    if not isinstance(
        item,
        dict,
    ):
        raise TypeError(
            f"QA item at line "
            f"{line_no} must be a dictionary"
        )

    if "question" not in item:
        raise KeyError(
            f"QA item at line "
            f"{line_no} missing question"
        )

    if not isinstance(
        item["question"],
        str,
    ):
        raise TypeError(
            f"question at line "
            f"{line_no} must be a string"
        )

    if not item[
        "question"
    ].strip():
        raise ValueError(
            f"question at line "
            f"{line_no} must not be empty"
        )

    if not any(
        key in item
        for key in (
            "expected_chunk_ids",
            "expected_source_file",
            "expected_keywords",
        )
    ):
        raise KeyError(
            f"QA item at line "
            f"{line_no} must contain "
            "expected_chunk_ids, "
            "expected_source_file, "
            "or expected_keywords"
        )

    if "expected_chunk_ids" in item:
        expected_chunk_ids = (
            item[
                "expected_chunk_ids"
            ]
        )

        if (
            not isinstance(
                expected_chunk_ids,
                list,
            )
            or not expected_chunk_ids
            or any(
                (
                    not isinstance(
                        chunk_id,
                        str,
                    )
                    or not chunk_id.strip()
                )
                for chunk_id
                in expected_chunk_ids
            )
        ):
            raise ValueError(
                f"expected_chunk_ids at line "
                f"{line_no} must be "
                "a non-empty list of "
                "non-empty strings"
            )

    if "expected_keywords" in item:
        expected_keywords = item[
            "expected_keywords"
        ]

        if (
            expected_keywords is not None
            and not isinstance(
                expected_keywords,
                list,
            )
        ):
            raise TypeError(
                f"expected_keywords at line "
                f"{line_no} must be a list"
            )

        if isinstance(
            expected_keywords,
            list,
        ):
            for keyword in (
                expected_keywords
            ):
                if not isinstance(
                    keyword,
                    str,
                ):
                    raise TypeError(
                        "each expected keyword "
                        f"at line {line_no} "
                        "must be a string"
                    )


def _validate_ks(
    ks: list[int] | tuple[int, ...],
) -> tuple[int, ...]:
    """检查 ks 是否合法，并返回排序后的 tuple。"""

    if not isinstance(
        ks,
        (list, tuple),
    ):
        raise TypeError(
            "ks must be a list or "
            "tuple of integers"
        )

    if not ks:
        raise ValueError(
            "ks must not be empty"
        )

    cleaned_ks: list[int] = []

    for k in ks:
        if not isinstance(
            k,
            int,
        ):
            raise TypeError(
                "each k must be an integer"
            )

        if k <= 0:
            raise ValueError(
                "each k must be positive"
            )

        cleaned_ks.append(k)

    return tuple(
        sorted(
            set(cleaned_ks)
        )
    )


def _get_expected_source_files(
    item: QAItem,
) -> set[str]:
    """从 QA 样本中取出期望来源文件名。"""

    source_files: set[str] = set()

    expected_source_file = item.get(
        "expected_source_file"
    )

    expected_source_files = item.get(
        "expected_source_files"
    )

    if (
        isinstance(
            expected_source_file,
            str,
        )
        and expected_source_file.strip()
    ):
        source_files.add(
            Path(
                expected_source_file
            ).name.lower()
        )

    if isinstance(
        expected_source_files,
        list,
    ):
        for file_name in (
            expected_source_files
        ):
            if (
                isinstance(
                    file_name,
                    str,
                )
                and file_name.strip()
            ):
                source_files.add(
                    Path(
                        file_name
                    ).name.lower()
                )

    return source_files


def _get_result_file_name(
    result: SearchResult,
) -> str:
    """从检索结果中取出文件名。"""

    if not isinstance(
        result,
        dict,
    ):
        return ""

    metadata = result.get(
        "metadata",
        {},
    )

    if metadata is None:
        metadata = {}

    if not isinstance(
        metadata,
        dict,
    ):
        return ""

    file_name = (
        metadata.get("file_name")
        or metadata.get("source")
        or ""
    )

    return Path(
        str(file_name)
    ).name.lower()


def _contains_any_keyword(
    results: list[SearchResult],
    expected_keywords: list[str],
) -> bool:
    """检查 results 文本中是否包含任一 expected keyword。"""

    combined_text_parts: list[str] = []

    for result in results:
        text = result.get(
            "text",
            "",
        )

        if not isinstance(
            text,
            str,
        ):
            text = str(text)

        combined_text_parts.append(
            text.lower()
        )

    combined_text = "\n".join(
        combined_text_parts
    )

    for keyword in expected_keywords:
        if (
            keyword.lower()
            in combined_text
        ):
            return True

    return False