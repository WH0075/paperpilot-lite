import numpy as np
import pytest

from src.paperpilot.retriever import Retriever


class FakeEmbedder:
    def __init__(self):
        self.last_query = None

    def embed_text(self, text: str) -> np.ndarray:
        self.last_query = text
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class FakeVectorStore:
    def __init__(self):
        self.last_query_embedding = None
        self.last_top_k = None

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        self.last_query_embedding = query_embedding
        self.last_top_k = top_k

        return [
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
        ][:top_k]


def test_retrieve_returns_results():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    results = retriever.retrieve("What is RAG?", top_k=2)

    assert len(results) == 2
    assert results[0]["text"] == "RAG combines retrieval and generation."
    assert results[0]["metadata"]["chunk_id"] == "rag_intro.txt:1:0"
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

    results = retriever.retrieve("What is RAG?")

    assert len(results) == 1
    assert vector_store.last_top_k == 1


def test_retrieve_calls_embedder_with_query():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    retriever.retrieve("What is RAG?", top_k=1)

    assert embedder.last_query == "What is RAG?"


def test_retrieve_calls_vector_store_with_query_embedding():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    retriever.retrieve("What is RAG?", top_k=1)

    assert isinstance(vector_store.last_query_embedding, np.ndarray)
    assert vector_store.last_query_embedding.shape == (3,)
    assert vector_store.last_top_k == 1


def test_retrieve_texts_returns_only_texts():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    texts = retriever.retrieve_texts("What is RAG?", top_k=2)

    assert texts == [
        "RAG combines retrieval and generation.",
        "A retriever finds relevant chunks.",
    ]


def test_retrieve_with_sources_returns_full_results():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    results = retriever.retrieve_with_sources("What is RAG?", top_k=1)

    assert len(results) == 1
    assert "text" in results[0]
    assert "metadata" in results[0]
    assert "score" in results[0]
    assert "index" in results[0]


def test_retrieve_rejects_empty_query():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_retrieve_rejects_non_string_query():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    with pytest.raises(TypeError):
        retriever.retrieve(123)  # type: ignore[arg-type]


def test_retrieve_rejects_invalid_top_k():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    with pytest.raises(ValueError):
        retriever.retrieve("What is RAG?", top_k=0)


def test_init_rejects_none_embedder():
    vector_store = FakeVectorStore()

    with pytest.raises(ValueError):
        Retriever(embedder=None, vector_store=vector_store)  # type: ignore[arg-type]


def test_init_rejects_none_vector_store():
    embedder = FakeEmbedder()

    with pytest.raises(ValueError):
        Retriever(embedder=embedder, vector_store=None)  # type: ignore[arg-type]