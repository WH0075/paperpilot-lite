import pytest

from src.paperpilot.reranker import CrossEncoderReranker
from src.paperpilot.retriever import Retriever

class FakeCrossEncoder:
    """用于单元测试的 fake Cross-Encoder。"""

    def __init__(self) -> None:
        self.last_pairs = None
        self.last_batch_size = None
        self.last_show_progress_bar = None

    def predict(
        self,
        pairs,
        batch_size,
        show_progress_bar,
    ):
        self.last_pairs = pairs
        self.last_batch_size = batch_size
        self.last_show_progress_bar = show_progress_bar

        return [
            0.2,
            0.9,
            -0.1,
        ]


def test_reranker_orders_results_by_cross_encoder_score():
    model = FakeCrossEncoder()

    reranker = CrossEncoderReranker(
        model=model,
        batch_size=8,
    )

    results = [
        {
            "text": "chunk A",
            "index": 0,
            "score": 0.95,
        },
        {
            "text": "chunk B",
            "index": 1,
            "score": 0.60,
        },
        {
            "text": "chunk C",
            "index": 2,
            "score": 0.30,
        },
    ]

    reranked = reranker.rerank(
        query="test query",
        results=results,
    )

    assert [
        result["index"]
        for result in reranked
    ] == [
        1,
        0,
        2,
    ]

    assert reranked[0]["rerank_score"] == pytest.approx(
        0.9
    )

    assert reranked[0]["score"] == pytest.approx(
        0.9
    )

    assert reranked[0]["retrieval_score"] == pytest.approx(
        0.60
    )


def test_reranker_passes_query_chunk_pairs_to_model():
    model = FakeCrossEncoder()

    reranker = CrossEncoderReranker(
        model=model,
        batch_size=8,
    )

    results = [
        {
            "text": "chunk A",
            "index": 0,
            "score": 1.0,
        },
        {
            "text": "chunk B",
            "index": 1,
            "score": 0.5,
        },
        {
            "text": "chunk C",
            "index": 2,
            "score": 0.1,
        },
    ]

    reranker.rerank(
        query="What is RAG?",
        results=results,
    )

    assert model.last_pairs == [
        ("What is RAG?", "chunk A"),
        ("What is RAG?", "chunk B"),
        ("What is RAG?", "chunk C"),
    ]

    assert model.last_batch_size == 8
    assert model.last_show_progress_bar is False


def test_reranker_respects_top_k():
    reranker = CrossEncoderReranker(
        model=FakeCrossEncoder(),
    )

    results = [
        {
            "text": "chunk A",
            "index": 0,
            "score": 0.9,
        },
        {
            "text": "chunk B",
            "index": 1,
            "score": 0.8,
        },
        {
            "text": "chunk C",
            "index": 2,
            "score": 0.7,
        },
    ]

    reranked = reranker.rerank(
        query="query",
        results=results,
        top_k=2,
    )

    assert len(reranked) == 2

    assert [
        result["index"]
        for result in reranked
    ] == [
        1,
        0,
    ]


def test_reranker_does_not_mutate_original_results():
    reranker = CrossEncoderReranker(
        model=FakeCrossEncoder(),
    )

    results = [
        {
            "text": "chunk A",
            "index": 0,
            "score": 0.9,
        },
        {
            "text": "chunk B",
            "index": 1,
            "score": 0.8,
        },
        {
            "text": "chunk C",
            "index": 2,
            "score": 0.7,
        },
    ]

    reranker.rerank(
        query="query",
        results=results,
    )

    assert "rerank_score" not in results[0]
    assert "retrieval_score" not in results[0]
    assert results[0]["score"] == pytest.approx(0.9)


def test_reranker_returns_empty_list_for_empty_results():
    reranker = CrossEncoderReranker(
        model=FakeCrossEncoder(),
    )

    assert reranker.rerank(
        query="query",
        results=[],
    ) == []


def test_reranker_rejects_empty_query():
    reranker = CrossEncoderReranker(
        model=FakeCrossEncoder(),
    )

    with pytest.raises(
        ValueError,
        match="query must not be empty",
    ):
        reranker.rerank(
            query="   ",
            results=[],
        )


def test_reranker_rejects_invalid_top_k():
    reranker = CrossEncoderReranker(
        model=FakeCrossEncoder(),
    )

    with pytest.raises(
        ValueError,
        match="top_k must be positive",
    ):
        reranker.rerank(
            query="query",
            results=[],
            top_k=0,
        )


def test_reranker_rejects_result_without_text():
    reranker = CrossEncoderReranker(
        model=FakeCrossEncoder(),
    )

    with pytest.raises(
        ValueError,
        match="must contain non-empty text",
    ):
        reranker.rerank(
            query="query",
            results=[
                {
                    "index": 0,
                    "score": 1.0,
                }
            ],
        )


class RecordingReranker:
    """记录 Retriever 传入参数的 fake reranker。"""

    def __init__(self) -> None:
        self.received_query = None
        self.received_results = None
        self.received_top_k = None

    def rerank(
        self,
        query,
        results,
        top_k=None,
    ):
        self.received_query = query
        self.received_results = list(results)
        self.received_top_k = top_k

        if top_k is None:
            return list(results)

        return list(results)[:top_k]


class MinimalVectorStore:
    """仅用于 Retriever integration test。"""

    def __init__(self, chunk_count: int = 100) -> None:
        self.chunks = [
            {
                "text": f"chunk {i}",
                "metadata": {
                    "chunk_id": f"test.pdf:1:{i}",
                },
            }
            for i in range(chunk_count)
        ]


class MinimalEmbedder:
    """测试中不会真正执行 embedding。"""

    pass


def _make_fake_results(count: int):
    return [
        {
            "text": f"chunk {i}",
            "metadata": {
                "chunk_id": f"test.pdf:1:{i}",
            },
            "index": i,
            "score": float(count - i),
        }
        for i in range(count)
    ]


def test_retriever_uses_candidate_pool_before_reranking(
    monkeypatch,
):
    """
    验证：

    final top_k = 5
    reranker candidate_k = 20

    第一阶段应检索 20 个候选，
    Reranker 收到 20 个，
    最终只返回 5 个。
    """

    reranker = RecordingReranker()

    retriever = Retriever(
        embedder=MinimalEmbedder(),
        vector_store=MinimalVectorStore(),
        default_mode="hybrid",
        reranker=reranker,
        reranker_candidate_k=20,
    )

    first_stage_call = {}

    def fake_retrieve_hybrid(
        *,
        query,
        top_k,
        alpha,
        fusion_method,
        rrf_k,
    ):
        first_stage_call["query"] = query
        first_stage_call["top_k"] = top_k
        first_stage_call["alpha"] = alpha
        first_stage_call["fusion_method"] = fusion_method
        first_stage_call["rrf_k"] = rrf_k

        return _make_fake_results(top_k)

    monkeypatch.setattr(
        retriever,
        "_retrieve_hybrid",
        fake_retrieve_hybrid,
    )

    results = retriever.retrieve(
        query="What is RAG?",
        top_k=5,
    )

    # 第一阶段不是只取最终 Top-5，
    # 而是先取 candidate_k=20。
    assert first_stage_call["top_k"] == 20

    # Reranker 确实收到完整的 20 个候选。
    assert len(reranker.received_results) == 20

    # Reranker 最终被要求返回 Top-5。
    assert reranker.received_top_k == 5

    # Retriever 最终也只返回 5 个结果。
    assert len(results) == 5

    assert reranker.received_query == "What is RAG?"


def test_runtime_candidate_k_overrides_default(
    monkeypatch,
):
    """运行时 candidate_k 应覆盖构造函数中的默认值。"""

    reranker = RecordingReranker()

    retriever = Retriever(
        embedder=MinimalEmbedder(),
        vector_store=MinimalVectorStore(),
        default_mode="hybrid",
        reranker=reranker,
        reranker_candidate_k=20,
    )

    first_stage_call = {}

    def fake_retrieve_hybrid(
        *,
        query,
        top_k,
        alpha,
        fusion_method,
        rrf_k,
    ):
        first_stage_call["top_k"] = top_k

        return _make_fake_results(top_k)

    monkeypatch.setattr(
        retriever,
        "_retrieve_hybrid",
        fake_retrieve_hybrid,
    )

    results = retriever.retrieve(
        query="query",
        top_k=5,
        candidate_k=12,
    )

    assert first_stage_call["top_k"] == 12
    assert len(reranker.received_results) == 12
    assert reranker.received_top_k == 5
    assert len(results) == 5


def test_candidate_k_must_not_be_smaller_than_top_k():
    """候选集不能比最终结果集还小。"""

    reranker = RecordingReranker()

    retriever = Retriever(
        embedder=MinimalEmbedder(),
        vector_store=MinimalVectorStore(),
        default_mode="hybrid",
        reranker=reranker,
        reranker_candidate_k=20,
    )

    with pytest.raises(
        ValueError,
        match=(
            "candidate_k must be greater than "
            "or equal to top_k"
        ),
    ):
        retriever.retrieve(
            query="query",
            top_k=5,
            candidate_k=3,
        )


def test_retriever_without_reranker_keeps_original_top_k(
    monkeypatch,
):
    """
    没有 Reranker 时保持 R4 行为：

    top_k=5
    -> 第一阶段直接取 5
    -> 返回 5
    """

    retriever = Retriever(
        embedder=MinimalEmbedder(),
        vector_store=MinimalVectorStore(),
        default_mode="hybrid",
        reranker=None,
        reranker_candidate_k=20,
    )

    first_stage_call = {}

    def fake_retrieve_hybrid(
        *,
        query,
        top_k,
        alpha,
        fusion_method,
        rrf_k,
    ):
        first_stage_call["top_k"] = top_k

        return _make_fake_results(top_k)

    monkeypatch.setattr(
        retriever,
        "_retrieve_hybrid",
        fake_retrieve_hybrid,
    )

    results = retriever.retrieve(
        query="query",
        top_k=5,
    )

    assert first_stage_call["top_k"] == 5
    assert len(results) == 5