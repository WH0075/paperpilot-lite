from src.paperpilot.keyword_retriever import (
    KeywordRetriever,
    tokenize,
)


def test_tokenize_normalizes_case():
    tokens = tokenize("RAG Uses Vector Search.")

    assert tokens == [
        "rag",
        "uses",
        "vector",
        "search",
    ]

def test_keyword_retriever_ranks_exact_match_first():
    chunks = [
        {
            "text": "Retrieval augmented generation uses vector search.",
            "metadata": {"chunk_id": "c1"},
        },
        {
            "text": "Catoni estimator is designed for robust mean estimation.",
            "metadata": {"chunk_id": "c2"},
        },
        {
            "text": "Transformer attention uses query key and value vectors.",
            "metadata": {"chunk_id": "c3"},
        },
    ]

    retriever = KeywordRetriever(chunks)

    results = retriever.retrieve(
        "Catoni estimator",
        top_k=3,
    )

    assert results[0]["metadata"]["chunk_id"] == "c2"
    assert results[0]["score"] > 0

import pytest


def test_keyword_retriever_rejects_empty_query():
    chunks = [
        {
            "text": "RAG uses vector retrieval.",
            "metadata": {"chunk_id": "c1"},
        }
    ]

    retriever = KeywordRetriever(chunks)

    with pytest.raises(
        ValueError,
        match="query must not be empty",
    ):
        retriever.retrieve("   ")

def test_keyword_retriever_respects_top_k():
    chunks = [
        {
            "text": "RAG uses vector retrieval.",
            "metadata": {"chunk_id": "c1"},
        },
        {
            "text": "RAG can use keyword retrieval.",
            "metadata": {"chunk_id": "c2"},
        },
        {
            "text": "Transformers use attention.",
            "metadata": {"chunk_id": "c3"},
        },
    ]

    retriever = KeywordRetriever(chunks)

    results = retriever.retrieve(
        "RAG retrieval",
        top_k=2,
    )

    assert len(results) == 2