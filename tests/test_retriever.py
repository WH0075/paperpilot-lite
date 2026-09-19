import numpy as np
import pytest

from src.paperpilot.retriever import Retriever


class FakeEmbedder:
    """测试用 Embedder，不真正加载 embedding 模型。"""

    def __init__(self):
        self.last_query = None

    def embed_text(self, text: str) -> np.ndarray:
        self.last_query = text

        return np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        )


class FakeVectorStore:
    """测试用 VectorStore。"""

    def __init__(self):
        self.last_query_embedding = None
        self.last_top_k = None

        # KeywordRetriever 初始化时需要读取 vector_store.chunks。
        self.chunks = [
            {
                "text": "RAG combines retrieval and generation.",
                "metadata": {
                    "chunk_id": "rag_intro.txt:1:0",
                    "file_name": "rag_intro.txt",
                    "page": 1,
                },
            },
            {
                "text": "A retriever finds relevant chunks.",
                "metadata": {
                    "chunk_id": "retriever.md:1:0",
                    "file_name": "retriever.md",
                    "page": 1,
                },
            },
            {
                "text": (
                    "Keyword retrieval matches exact terms "
                    "such as Catoni estimator."
                ),
                "metadata": {
                    "chunk_id": "keyword.md:1:0",
                    "file_name": "keyword.md",
                    "page": 1,
                },
            },
        ]

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ):
        self.last_query_embedding = query_embedding
        self.last_top_k = top_k

        results = [
            {
                "text": "RAG combines retrieval and generation.",
                "metadata": {
                    "chunk_id": "rag_intro.txt:1:0",
                    "file_name": "rag_intro.txt",
                    "page": 1,
                },
                "score": 0.95,
                "index": 0,
            },
            {
                "text": "A retriever finds relevant chunks.",
                "metadata": {
                    "chunk_id": "retriever.md:1:0",
                    "file_name": "retriever.md",
                    "page": 1,
                },
                "score": 0.82,
                "index": 1,
            },
        ]

        return results[:top_k]


def test_retrieve_returns_results():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        "What is RAG?",
        top_k=2,
    )

    assert len(results) == 2

    assert (
        results[0]["text"]
        == "RAG combines retrieval and generation."
    )

    assert (
        results[0]["metadata"]["chunk_id"]
        == "rag_intro.txt:1:0"
    )

    assert results[0]["score"] == 0.95
    assert results[0]["index"] == 0


def test_retrieve_uses_default_top_k():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        default_top_k=1,
    )

    results = retriever.retrieve(
        "What is RAG?"
    )

    assert len(results) == 1
    assert vector_store.last_top_k == 1


def test_retrieve_calls_embedder_with_query():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    retriever.retrieve(
        "What is RAG?",
        top_k=1,
    )

    assert embedder.last_query == "What is RAG?"


def test_retrieve_calls_vector_store_with_query_embedding():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    retriever.retrieve(
        "What is RAG?",
        top_k=1,
    )

    assert isinstance(
        vector_store.last_query_embedding,
        np.ndarray,
    )

    assert (
        vector_store.last_query_embedding.shape
        == (3,)
    )

    assert vector_store.last_top_k == 1


def test_dense_mode_uses_embedder_and_vector_store():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        "What is RAG?",
        top_k=1,
        mode="dense",
    )

    assert len(results) == 1

    assert embedder.last_query == "What is RAG?"

    assert (
        results[0]["metadata"]["chunk_id"]
        == "rag_intro.txt:1:0"
    )


def test_keyword_mode_uses_bm25_without_embedding():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        "Catoni estimator",
        top_k=1,
        mode="keyword",
    )

    assert len(results) == 1

    assert (
        results[0]["metadata"]["chunk_id"]
        == "keyword.md:1:0"
    )

    assert results[0]["score"] > 0

    # Keyword retrieval 不需要计算 query embedding。
    assert embedder.last_query is None

    # VectorStore.search() 也不应该被调用。
    assert vector_store.last_query_embedding is None


def test_default_keyword_mode_is_used():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        default_mode="keyword",
    )

    results = retriever.retrieve(
        "Catoni estimator",
        top_k=1,
    )

    assert (
        results[0]["metadata"]["chunk_id"]
        == "keyword.md:1:0"
    )

    assert embedder.last_query is None


def test_retrieve_texts_returns_only_texts():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    texts = retriever.retrieve_texts(
        "What is RAG?",
        top_k=2,
    )

    assert texts == [
        "RAG combines retrieval and generation.",
        "A retriever finds relevant chunks.",
    ]


def test_retrieve_texts_supports_keyword_mode():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    texts = retriever.retrieve_texts(
        "Catoni estimator",
        top_k=1,
        mode="keyword",
    )

    assert texts == [
        (
            "Keyword retrieval matches exact terms "
            "such as Catoni estimator."
        )
    ]


def test_retrieve_with_sources_returns_full_results():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    results = retriever.retrieve_with_sources(
        "What is RAG?",
        top_k=1,
    )

    assert len(results) == 1
    assert "text" in results[0]
    assert "metadata" in results[0]
    assert "score" in results[0]
    assert "index" in results[0]


def test_retrieve_rejects_invalid_mode():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    with pytest.raises(
        ValueError,
        match="mode must be one of: dense, keyword",
    ):
        retriever.retrieve(
            "What is RAG?",
            mode="invalid",
        )


def test_retrieve_rejects_empty_query():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_retrieve_rejects_non_string_query():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    with pytest.raises(TypeError):
        retriever.retrieve(
            123  # type: ignore[arg-type]
        )


def test_retrieve_rejects_invalid_top_k():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    with pytest.raises(ValueError):
        retriever.retrieve(
            "What is RAG?",
            top_k=0,
        )


def test_init_rejects_invalid_default_mode():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    with pytest.raises(
        ValueError,
        match="default_mode must be one of: dense, keyword",
    ):
        Retriever(
            embedder=embedder,
            vector_store=vector_store,
            default_mode="invalid",
        )


def test_init_rejects_none_embedder():
    vector_store = FakeVectorStore()

    with pytest.raises(ValueError):
        Retriever(
            embedder=None,  # type: ignore[arg-type]
            vector_store=vector_store,
        )


def test_init_rejects_none_vector_store():
    embedder = FakeEmbedder()

    with pytest.raises(ValueError):
        Retriever(
            embedder=embedder,
            vector_store=None,  # type: ignore[arg-type]
        )


def test_hybrid_rrf_combines_dense_and_keyword_results():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        "Catoni estimator",
        top_k=3,
        mode="hybrid",
        alpha=0.5,
        fusion_method="rrf",
        rrf_k=60,
    )

    assert len(results) > 0

    assert "dense_rank" in results[0]
    assert "keyword_rank" in results[0]

    assert "dense_rrf_score" in results[0]
    assert "keyword_rrf_score" in results[0]

    assert results[0]["score"] >= 0


def test_hybrid_minmax_contains_normalized_scores():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        "Catoni estimator",
        top_k=3,
        mode="hybrid",
        alpha=0.5,
        fusion_method="minmax",
    )

    assert len(results) > 0

    assert "dense_score_normalized" in results[0]
    assert "keyword_score_normalized" in results[0]

    assert 0.0 <= results[0]["score"] <= 1.0

def test_hybrid_rejects_invalid_fusion_method():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    with pytest.raises(
        ValueError,
        match="fusion_method must be one of: minmax, rrf",
    ):
        retriever.retrieve(
            "What is RAG?",
            mode="hybrid",
            fusion_method="invalid",
        )

def test_hybrid_rejects_invalid_alpha():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    with pytest.raises(
        ValueError,
        match="alpha must be between 0 and 1",
    ):
        retriever.retrieve(
            "What is RAG?",
            mode="hybrid",
            alpha=1.5,
        )

def test_hybrid_rejects_invalid_rrf_k():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    with pytest.raises(
        ValueError,
        match="rrf_k must be positive",
    ):
        retriever.retrieve(
            "What is RAG?",
            mode="hybrid",
            fusion_method="rrf",
            rrf_k=0,
        )