import json

import pytest

from src.paperpilot.evaluator import (
    _percentile,
    check_hit,
    evaluate_retrieval,
    find_first_relevant_rank,
    load_qa_set,
    summarize_results,
)


def test_check_hit_by_source_file():
    results = [
        {
            "text": "Some text about RAG.",
            "metadata": {
                "file_name": "rag_intro.txt",
            },
            "score": 0.9,
        }
    ]

    item = {
        "question": "What is RAG?",
        "expected_source_file": "rag_intro.txt",
        "expected_keywords": [
            "not existing keyword"
        ],
    }

    hit_info = check_hit(
        results,
        item,
    )

    assert hit_info["hit"] is True
    assert hit_info["source_hit"] is True


def test_check_hit_by_keyword():
    results = [
        {
            "text": (
                "Recall@K checks whether expected "
                "evidence appears in top K results."
            ),
            "metadata": {
                "file_name": "sample_rag.pdf",
            },
            "score": 0.8,
        }
    ]

    item = {
        "question": "What does Recall@K measure?",
        "expected_source_file": "wrong_file.txt",
        "expected_keywords": [
            "Recall@K"
        ],
    }

    hit_info = check_hit(
        results,
        item,
    )

    assert hit_info["hit"] is True
    assert hit_info["keyword_hit"] is True


def test_check_hit_returns_false_when_no_match():
    results = [
        {
            "text": "This text is about attention.",
            "metadata": {
                "file_name": "attention.md",
            },
            "score": 0.7,
        }
    ]

    item = {
        "question": "What is RAG?",
        "expected_source_file": "rag_intro.txt",
        "expected_keywords": [
            "Retrieval-Augmented Generation"
        ],
    }

    hit_info = check_hit(
        results,
        item,
    )

    assert hit_info["hit"] is False


def test_chunk_ids_take_priority_over_source_file():
    item = {
        "question": "What is ablation study?",
        "expected_source_file": "sample_rag.pdf",
        "expected_keywords": [
            "Retrieval evaluation"
        ],
        "expected_chunk_ids": [
            "sample_rag.pdf:2:1"
        ],
    }

    wrong_chunk = [
        {
            "text": (
                "Retrieval evaluation uses "
                "Recall@K."
            ),
            "metadata": {
                "file_name": "sample_rag.pdf",
                "chunk_id": (
                    "sample_rag.pdf:2:0"
                ),
            },
        }
    ]

    right_chunk = [
        {
            "text": (
                "Ablation study changes "
                "one variable at a time."
            ),
            "metadata": {
                "file_name": "sample_rag.pdf",
                "chunk_id": (
                    "sample_rag.pdf:2:1"
                ),
            },
        }
    ]

    wrong_hit = check_hit(
        wrong_chunk,
        item,
    )

    assert wrong_hit["source_hit"] is True
    assert wrong_hit["keyword_hit"] is True
    assert wrong_hit["chunk_hit"] is False
    assert wrong_hit["hit"] is False

    assert (
        check_hit(
            right_chunk,
            item,
        )["hit"]
        is True
    )


def test_load_qa_set_accepts_chunk_only_labels(
    tmp_path,
):
    qa_path = (
        tmp_path
        / "qa.jsonl"
    )

    item = {
        "question": "What is ablation study?",
        "expected_chunk_ids": [
            "sample_rag.pdf:2:1"
        ],
    }

    qa_path.write_text(
        json.dumps(item) + "\n",
        encoding="utf-8",
    )

    assert load_qa_set(
        qa_path
    ) == [item]


@pytest.mark.parametrize(
    "chunk_ids",
    [
        None,
        [],
        [""],
        ["valid", 1],
    ],
)
def test_load_qa_set_rejects_invalid_chunk_labels(
    tmp_path,
    chunk_ids,
):
    qa_path = (
        tmp_path
        / "qa.jsonl"
    )

    item = {
        "question": "What is ablation study?",
        "expected_chunk_ids": chunk_ids,
    }

    qa_path.write_text(
        json.dumps(item) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="expected_chunk_ids",
    ):
        load_qa_set(
            qa_path
        )


def test_summarize_results():
    results = [
        {
            "text": (
                "A long text about RAG."
            ),
            "metadata": {
                "file_name": "rag_intro.txt",
                "page": 1,
                "chunk_id": (
                    "rag_intro.txt:1:0"
                ),
            },
            "score": 0.9,
        }
    ]

    summaries = summarize_results(
        results
    )

    assert len(summaries) == 1
    assert summaries[0]["rank"] == 1

    assert (
        summaries[0]["file_name"]
        == "rag_intro.txt"
    )

    assert summaries[0]["page"] == 1

    assert (
        summaries[0]["chunk_id"]
        == "rag_intro.txt:1:0"
    )


def test_find_first_relevant_rank_returns_one_for_top_result():
    item = {
        "question": "What is RAG?",
        "expected_chunk_ids": [
            "rag_intro.txt:1:0",
        ],
    }

    results = [
        {
            "text": (
                "RAG combines retrieval "
                "and generation."
            ),
            "metadata": {
                "file_name": "rag_intro.txt",
                "chunk_id": (
                    "rag_intro.txt:1:0"
                ),
            },
            "score": 0.9,
        },
        {
            "text": "Some unrelated text.",
            "metadata": {
                "file_name": "other.txt",
                "chunk_id": (
                    "other.txt:1:0"
                ),
            },
            "score": 0.8,
        },
    ]

    rank = find_first_relevant_rank(
        results,
        item,
    )

    assert rank == 1


def test_find_first_relevant_rank_returns_correct_later_rank():
    item = {
        "question": "What is RAG?",
        "expected_chunk_ids": [
            "rag_intro.txt:1:2",
        ],
    }

    results = [
        {
            "text": "Wrong chunk one.",
            "metadata": {
                "file_name": "rag_intro.txt",
                "chunk_id": (
                    "rag_intro.txt:1:0"
                ),
            },
            "score": 0.9,
        },
        {
            "text": "Wrong chunk two.",
            "metadata": {
                "file_name": "rag_intro.txt",
                "chunk_id": (
                    "rag_intro.txt:1:1"
                ),
            },
            "score": 0.8,
        },
        {
            "text": "Correct evidence.",
            "metadata": {
                "file_name": "rag_intro.txt",
                "chunk_id": (
                    "rag_intro.txt:1:2"
                ),
            },
            "score": 0.7,
        },
    ]

    rank = find_first_relevant_rank(
        results,
        item,
    )

    assert rank == 3


def test_find_first_relevant_rank_uses_first_gold_chunk():
    item = {
        "question": (
            "What supports the answer?"
        ),
        "expected_chunk_ids": [
            "paper.pdf:1:2",
            "paper.pdf:1:4",
        ],
    }

    results = [
        {
            "text": "Wrong evidence.",
            "metadata": {
                "chunk_id": (
                    "paper.pdf:1:0"
                ),
            },
        },
        {
            "text": (
                "First valid evidence."
            ),
            "metadata": {
                "chunk_id": (
                    "paper.pdf:1:4"
                ),
            },
        },
        {
            "text": (
                "Second valid evidence."
            ),
            "metadata": {
                "chunk_id": (
                    "paper.pdf:1:2"
                ),
            },
        },
    ]

    rank = find_first_relevant_rank(
        results,
        item,
    )

    assert rank == 2


def test_find_first_relevant_rank_returns_none_when_missing():
    item = {
        "question": "What is RAG?",
        "expected_chunk_ids": [
            "rag_intro.txt:1:3",
        ],
    }

    results = [
        {
            "text": "Wrong chunk.",
            "metadata": {
                "chunk_id": (
                    "rag_intro.txt:1:0"
                ),
            },
        },
        {
            "text": (
                "Another wrong chunk."
            ),
            "metadata": {
                "chunk_id": (
                    "rag_intro.txt:1:1"
                ),
            },
        },
    ]

    rank = find_first_relevant_rank(
        results,
        item,
    )

    assert rank is None


class FakeMRRRetriever:
    """构造确定排名的 Retriever，用于验证 MRR 和 latency report。"""

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ):
        if query == "q1":
            # Gold rank = 1
            results = [
                {
                    "text": "gold",
                    "metadata": {
                        "chunk_id": "gold-1",
                    },
                },
                {
                    "text": "wrong",
                    "metadata": {
                        "chunk_id": "wrong-1",
                    },
                },
            ]

        elif query == "q2":
            # Gold rank = 2
            results = [
                {
                    "text": "wrong",
                    "metadata": {
                        "chunk_id": "wrong-2",
                    },
                },
                {
                    "text": "gold",
                    "metadata": {
                        "chunk_id": "gold-2",
                    },
                },
            ]

        else:
            # Gold completely missing
            results = [
                {
                    "text": "wrong",
                    "metadata": {
                        "chunk_id": "wrong-3",
                    },
                },
                {
                    "text": "still wrong",
                    "metadata": {
                        "chunk_id": "wrong-4",
                    },
                },
            ]

        return results[:top_k]


def test_evaluate_retrieval_calculates_mrr_and_latency():
    retriever = FakeMRRRetriever()

    qa_items = [
        {
            "question": "q1",
            "expected_chunk_ids": [
                "gold-1",
            ],
        },
        {
            "question": "q2",
            "expected_chunk_ids": [
                "gold-2",
            ],
        },
        {
            "question": "q3",
            "expected_chunk_ids": [
                "gold-3",
            ],
        },
    ]

    report = evaluate_retrieval(
        retriever=retriever,
        qa_items=qa_items,
        ks=[1, 2],
        mrr_k=2,
    )

    # -------------------------------------------------
    # MRR
    #
    # q1: rank 1 -> RR = 1
    # q2: rank 2 -> RR = 1/2
    # q3: missing -> RR = 0
    #
    # MRR@2 = (1 + 0.5 + 0) / 3 = 0.5
    # -------------------------------------------------
    assert report["mrr_k"] == 2

    assert report["mrr"] == pytest.approx(
        0.5
    )

    cases = report["cases"]

    assert (
        cases[0]["first_relevant_rank"]
        == 1
    )

    assert (
        cases[0]["reciprocal_rank"]
        == pytest.approx(1.0)
    )

    assert (
        cases[1]["first_relevant_rank"]
        == 2
    )

    assert (
        cases[1]["reciprocal_rank"]
        == pytest.approx(0.5)
    )

    assert (
        cases[2]["first_relevant_rank"]
        is None
    )

    assert (
        cases[2]["reciprocal_rank"]
        == pytest.approx(0.0)
    )

    # -------------------------------------------------
    # Latency
    #
    # 不能测试具体毫秒值，因为运行时间具有随机性。
    # 这里只验证统计结构和基本关系。
    # -------------------------------------------------
    latency = report[
        "latency_ms"
    ]

    assert (
        latency["mean"]
        >= 0.0
    )

    assert (
        latency["p50"]
        >= 0.0
    )

    assert (
        latency["p95"]
        >= 0.0
    )

    assert (
        latency["p50"]
        <= latency["p95"]
    )

    # 每一个 case 也应该保存自己的 latency。
    for case in cases:
        assert (
            case["latency_ms"]
            >= 0.0
        )


def test_percentile():
    values = [
        10.0,
        20.0,
        30.0,
        40.0,
    ]

    # 中位数：
    # 20 和 30 的中间值 = 25
    assert _percentile(
        values,
        0.50,
    ) == pytest.approx(
        25.0
    )

    # 最小值
    assert _percentile(
        values,
        0.0,
    ) == pytest.approx(
        10.0
    )

    # 最大值
    assert _percentile(
        values,
        1.0,
    ) == pytest.approx(
        40.0
    )


def test_percentile_single_value():
    assert _percentile(
        [42.0],
        0.95,
    ) == pytest.approx(
        42.0
    )


def test_percentile_rejects_empty_values():
    with pytest.raises(
        ValueError,
        match="values must not be empty",
    ):
        _percentile(
            [],
            0.50,
        )


@pytest.mark.parametrize(
    "percentile",
    [
        -0.1,
        1.1,
    ],
)
def test_percentile_rejects_invalid_percentile(
    percentile,
):
    with pytest.raises(
        ValueError,
        match=(
            "percentile must be "
            "between 0 and 1"
        ),
    ):
        _percentile(
            [
                10.0,
                20.0,
            ],
            percentile,
        )


class FakeDeepRecallRetriever:
    """用于测试 Recall@20 与 MRR@5 分离。"""

    def retrieve(
        self,
        query,
        top_k,
    ):
        results = []

        for rank in range(1, top_k + 1):
            chunk_id = (
                "paper.pdf:1:gold"
                if rank == 8
                else f"paper.pdf:1:wrong-{rank}"
            )

            results.append(
                {
                    "text": f"chunk at rank {rank}",
                    "metadata": {
                        "file_name": "paper.pdf",
                        "chunk_id": chunk_id,
                    },
                    "index": rank - 1,
                    "score": float(
                        top_k - rank
                    ),
                }
            )

        return results


def test_recall_depth_is_independent_from_mrr_depth():
    retriever = FakeDeepRecallRetriever()

    qa_items = [
        {
            "id": "q1",
            "question": "test question",
            "expected_source_file": "paper.pdf",
            "expected_chunk_ids": [
                "paper.pdf:1:gold",
            ],
        }
    ]

    report = evaluate_retrieval(
        retriever=retriever,
        qa_items=qa_items,
        ks=(1, 3, 5, 10, 20),
        mrr_k=5,
    )

    assert report["recall"][1] == pytest.approx(0.0)
    assert report["recall"][3] == pytest.approx(0.0)
    assert report["recall"][5] == pytest.approx(0.0)

    assert report["recall"][10] == pytest.approx(1.0)
    assert report["recall"][20] == pytest.approx(1.0)

    assert report["mrr_k"] == 5
    assert report["mrr"] == pytest.approx(0.0)


class FakeDeepRecallRetriever:
    """用于测试 Recall depth 与 MRR depth 相互独立。"""

    def retrieve(
        self,
        query,
        top_k,
    ):
        results = []

        for rank in range(
            1,
            top_k + 1,
        ):
            chunk_id = (
                "paper.pdf:1:gold"
                if rank == 8
                else (
                    f"paper.pdf:1:"
                    f"wrong-{rank}"
                )
            )

            results.append(
                {
                    "text": (
                        f"chunk at rank "
                        f"{rank}"
                    ),
                    "metadata": {
                        "file_name": (
                            "paper.pdf"
                        ),
                        "chunk_id": (
                            chunk_id
                        ),
                    },
                    "index": rank - 1,
                    "score": float(
                        top_k - rank
                    ),
                }
            )

        return results


def test_recall_depth_is_independent_from_mrr_depth():
    """
    gold 位于 rank 8。

    因此：
    Recall@5  = 0
    Recall@10 = 1
    Recall@20 = 1

    但 MRR@5 仍然必须为 0。
    """

    retriever = (
        FakeDeepRecallRetriever()
    )

    qa_items = [
        {
            "id": "q1",
            "question": (
                "test question"
            ),
            "expected_source_file": (
                "paper.pdf"
            ),
            "expected_chunk_ids": [
                "paper.pdf:1:gold",
            ],
        }
    ]

    report = evaluate_retrieval(
        retriever=retriever,
        qa_items=qa_items,
        ks=(
            1,
            3,
            5,
            10,
            20,
        ),
        mrr_k=5,
    )

    assert report[
        "recall"
    ][1] == pytest.approx(0.0)

    assert report[
        "recall"
    ][3] == pytest.approx(0.0)

    assert report[
        "recall"
    ][5] == pytest.approx(0.0)

    assert report[
        "recall"
    ][10] == pytest.approx(1.0)

    assert report[
        "recall"
    ][20] == pytest.approx(1.0)

    assert report[
        "mrr_k"
    ] == 5

    assert report[
        "mrr"
    ] == pytest.approx(0.0)