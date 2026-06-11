from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .retriever import Retriever


QAItem = dict[str, Any]
SearchResult = dict[str, Any]
EvaluationReport = dict[str, Any]


def load_qa_set(qa_path: str | Path) -> list[QAItem]:

    path = Path(qa_path)

    if not path.exists():
        raise FileNotFoundError(f"QA file not found: {path}")
    
    qa_items: list[QAItem] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {line}") from exc
            
            _validate_qa_item(item=item, line_no=line_no)
            qa_items.append(item)

    if not qa_items:
        raise ValueError(f"QA file is empty: {path}")
    
    return qa_items


def evaluate_retrieval(
    retriever: Retriever,
    qa_items: list[QAItem],
    ks: list[int] | tuple[int, ...] = (1, 3, 5),
) -> EvaluationReport:
    
    if retriever is None:
        raise ValueError("retriever must not be None")

    if not isinstance(qa_items, list):
        raise TypeError("qa_items must be a list")

    if not qa_items:
        raise ValueError("qa_items must not be empty")
    
    ks = _validate_ks(ks)
    max_k = max(ks)

    hit_counts = {k: 0 for k in ks}
    cases: list[dict[str, Any]] = []
    failed_cases: list[dict[str, Any]] = []

    for item_id, item in enumerate(qa_items, start=1):
        question = item["question"]

        results = retriever.retrieve(
            query=question,
            top_k=max_k,
        )

        case_hits: dict[str, bool] = {}

        for k in ks:
            top_results = results[:k]
            hit_info = check_hit(top_results, item)
            hit = hit_info["hit"]
            case_hits[f"Recall@{k}"] = hit

            if hit:
                hit_counts[k] += 1
        
        case = {
            "item_id": item.get("id", item_id),
            "question": question,
            "expected_source_file": item.get("expected_source_file"),
            "expected_keywords": item.get("expected_keywords", []),
            "hits": case_hits,
            "top_results": summarize_results(results),
        }

        cases.append(case)

        if not case_hits[f"Recall@{max_k}"]:
            failed_cases.append(case)

    total = len(qa_items)

    recall = {
        k: hit_counts[k] / total
        for k in ks
    }

    return {
        "total": total,
        "ks": list(ks),
        "hit_counts": hit_counts,
        "recall": recall,
        "cases": cases,
        "failed_cases": failed_cases,
    }


def check_hit(results: list[SearchResult], item: QAItem) -> dict[str, Any]:

    if not isinstance(results, list):
        raise TypeError("results must be a list")

    expected_source_files = _get_expected_source_files(item)
    expected_keywords = item.get("expected_keywords", [])

    if expected_keywords is None:
        expected_keywords = []

    if not isinstance(expected_keywords, list):
        raise TypeError("expected_keywords must be a list")
    
    source_hit = False
    keyword_hit = False

    if expected_source_files:
        source_hit = any(
            _get_result_file_name(result) in expected_source_files
            for result in results
        )
    
    if expected_keywords:
        keyword_hit = _contains_any_keyword(results, expected_keywords)

    hit = source_hit or keyword_hit

    return {
        "hit": hit,
        "source_hit": source_hit,
        "keyword_hit": keyword_hit,
    }


def summarize_results(results: list[SearchResult], max_text_chars: int = 200) -> list[dict[str, Any]]:

    summaries: list[dict[str, Any]] = []

    for rank, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})

        if metadata is None:
            metadata = {}
        
        if not isinstance(metadata, dict):
            metadata = {}

        text = result.get("text", "")

        if not isinstance(text, str):
            text = str(text)
        
        score = result.get("score")

        summaries.append(
            {
                "rank": rank,
                "file_name": metadata.get("file_name") or metadata.get("source"),
                "page": metadata.get("page"),
                "chunk_id": metadata.get("chunk_id"),
                "score": float(score) if isinstance(score, (int, float)) else score,
                "text_preview": text[:max_text_chars],
            }
        )

    return summaries


def print_evaluation_report(
    report: EvaluationReport,
    show_failed_cases: bool = True,
    max_failed_cases: int = 10,
) -> None:
    
    print("=" * 80)
    print("Retrieval Evaluation Report")
    print("=" * 80)
    print(f"Total questions: {report['total']}")
    print()

    for k in report["ks"]:
        hit_count = report["hit_counts"][k]
        recall = report["recall"][k]
        print(f"Recall@{k}: {recall:.4f} ({hit_count}/{report['total']})")
    
    failed_cases = report["failed_cases"]
    print()
    print(f"Failed cases at Recall@{max(report['ks'])}: {len(failed_cases)}")

    if show_failed_cases and failed_cases:
        print()
        print("=" * 80)
        print("Failed Case Details")
        print("=" * 80)

        for case in failed_cases[:max_failed_cases]:
            print()
            print("-" * 80)
            print(f"Item ID: {case['item_id']}")
            print(f"Question: {case['question']}")
            print(f"Expected source: {case['expected_source_file']}")
            print(f"Expected keywords: {case['expected_keywords']}")
            print("Top results:")

            for result in case["top_results"]:
                print(
                    f"  Rank {result['rank']} | "
                    f"file={result['file_name']} | "
                    f"page={result['page']} | "
                    f"score={result['score']}"
                )
                print(f"  Text: {result['text_preview']}")
                print()
            

def evaluate_retrieval_from_index(
    index_dir: str | Path,
    qa_path: str | Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str | None = "cpu",
    normalize_embeddings: bool = True,
    batch_size: int = 32,
    ks: list[int] | tuple[int, ...] = (1, 3, 5),
) -> EvaluationReport:
    
    ks = _validate_ks(ks)

    retriever = Retriever.from_index(
        index_dir=index_dir,
        model_name=model_name,
        device=device,
        normalize_embeddings=normalize_embeddings,
        batch_size=batch_size,
        default_top_k=max(ks),
    )

    qa_items = load_qa_set(qa_path)

    return evaluate_retrieval(
        retriever=retriever,
        qa_items=qa_items,
        ks=ks,
    )


def _validate_qa_item(item: Any, line_no: int) -> None:
    """检查单条 QA 样本是否合法。"""

    if not isinstance(item, dict):
        raise TypeError(f"QA item at line {line_no} must be a dictionary")

    if "question" not in item:
        raise KeyError(f"QA item at line {line_no} missing question")

    if not isinstance(item["question"], str):
        raise TypeError(f"question at line {line_no} must be a string")

    if not item["question"].strip():
        raise ValueError(f"question at line {line_no} must not be empty")

    if "expected_source_file" not in item and "expected_keywords" not in item:
        raise KeyError(
            f"QA item at line {line_no} must contain expected_source_file or expected_keywords"
        )

    if "expected_keywords" in item:
        expected_keywords = item["expected_keywords"]

        if expected_keywords is not None and not isinstance(expected_keywords, list):
            raise TypeError(f"expected_keywords at line {line_no} must be a list")

        if isinstance(expected_keywords, list):
            for keyword in expected_keywords:
                if not isinstance(keyword, str):
                    raise TypeError(f"each expected keyword at line {line_no} must be a string")


def _validate_ks(ks: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """检查 ks 是否合法，并返回排序后的 tuple。"""

    if not isinstance(ks, (list, tuple)):
        raise TypeError("ks must be a list or tuple of integers")

    if not ks:
        raise ValueError("ks must not be empty")

    cleaned_ks: list[int] = []

    for k in ks:
        if not isinstance(k, int):
            raise TypeError("each k must be an integer")

        if k <= 0:
            raise ValueError("each k must be positive")

        cleaned_ks.append(k)

    return tuple(sorted(set(cleaned_ks)))


def _get_expected_source_files(item: QAItem) -> set[str]:
    """从 QA 样本中取出期望来源文件名。"""

    source_files: set[str] = set()

    expected_source_file = item.get("expected_source_file")
    expected_source_files = item.get("expected_source_files")

    if isinstance(expected_source_file, str) and expected_source_file.strip():
        source_files.add(Path(expected_source_file).name.lower())

    if isinstance(expected_source_files, list):
        for file_name in expected_source_files:
            if isinstance(file_name, str) and file_name.strip():
                source_files.add(Path(file_name).name.lower())

    return source_files


def _get_result_file_name(result: SearchResult) -> str:
    """从检索结果中取出文件名。"""

    if not isinstance(result, dict):
        return ""

    metadata = result.get("metadata", {})

    if metadata is None:
        metadata = {}

    if not isinstance(metadata, dict):
        return ""

    file_name = metadata.get("file_name") or metadata.get("source") or ""

    return Path(str(file_name)).name.lower()


def _contains_any_keyword(results: list[SearchResult], expected_keywords: list[str]) -> bool:

    combined_text_parts: list[str] = []

    for result in results:
        text = result.get("text", "")

        if not isinstance(text, str):
            text = str(text)

        combined_text_parts.append(text.lower())
    
    combined_text = "\n".join(combined_text_parts)

    for keyword in expected_keywords:
        if keyword.lower() in combined_text:
            return True
    
    return False