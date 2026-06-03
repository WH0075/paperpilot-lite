from pathlib import Path

import numpy as np
import pytest

from src.paperpilot.vector_store import VectorStore


def make_fake_chunks():
    return [
        {
            "text": "RAG combines retrieval and generation.",
            "metadata": {
                "chunk_id": "rag_intro.txt:1:0",
                "file_name": "rag_intro.txt",
                "page": 1,
            },
        },
        {
            "text": "Attention computes similarity between queries and keys.",
            "metadata": {
                "chunk_id": "attention.md:1:0",
                "file_name": "attention.md",
                "page": 1,
            },
        },
        {
            "text": "Agents can use tools to complete tasks.",
            "metadata": {
                "chunk_id": "agent.md:1:0",
                "file_name": "agent.md",
                "page": 1,
            },
        },
    ]


def make_fake_embeddings():
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.8, 0.2, 0.0],
        ],
        dtype=np.float32,
    )


def test_build_index():
    chunks = make_fake_chunks()
    embeddings = make_fake_embeddings()

    store = VectorStore(similarity="cosine")
    store.build_index(embeddings=embeddings, chunks=chunks)

    assert store.get_num_chunks() == 3
    assert store.get_embedding_dim() == 3


def test_search_returns_top_k_results():
    chunks = make_fake_chunks()
    embeddings = make_fake_embeddings()

    store = VectorStore(similarity="cosine")
    store.build_index(embeddings=embeddings, chunks=chunks)

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(query_embedding=query, top_k=2)

    assert len(results) == 2
    assert results[0]["index"] == 0
    assert "text" in results[0]
    assert "metadata" in results[0]
    assert "score" in results[0]


def test_search_top_k_larger_than_num_chunks():
    chunks = make_fake_chunks()
    embeddings = make_fake_embeddings()

    store = VectorStore(similarity="cosine")
    store.build_index(embeddings=embeddings, chunks=chunks)

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(query_embedding=query, top_k=10)

    assert len(results) == 3


def test_save_and_load_index(tmp_path: Path):
    chunks = make_fake_chunks()
    embeddings = make_fake_embeddings()

    store = VectorStore(similarity="cosine")
    store.build_index(embeddings=embeddings, chunks=chunks)

    index_dir = tmp_path / "index"
    store.save_index(index_dir)

    assert (index_dir / "embeddings.npy").exists()
    assert (index_dir / "chunks.json").exists()
    assert (index_dir / "index_meta.json").exists()

    loaded_store = VectorStore.load_index(index_dir)

    assert loaded_store.get_num_chunks() == 3
    assert loaded_store.get_embedding_dim() == 3

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = loaded_store.search(query_embedding=query, top_k=1)

    assert len(results) == 1
    assert results[0]["index"] == 0


def test_build_index_rejects_mismatched_lengths():
    chunks = make_fake_chunks()
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    store = VectorStore()

    with pytest.raises(ValueError):
        store.build_index(embeddings=embeddings, chunks=chunks)


def test_search_rejects_wrong_query_dimension():
    chunks = make_fake_chunks()
    embeddings = make_fake_embeddings()

    store = VectorStore()
    store.build_index(embeddings=embeddings, chunks=chunks)

    wrong_query = np.array([1.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError):
        store.search(query_embedding=wrong_query, top_k=1)


def test_search_before_build_raises_error():
    store = VectorStore()
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError):
        store.search(query_embedding=query, top_k=1)


def test_invalid_similarity_raises_error():
    with pytest.raises(ValueError):
        VectorStore(similarity="invalid")


def test_l2_search():
    chunks = make_fake_chunks()
    embeddings = make_fake_embeddings()

    store = VectorStore(similarity="l2")
    store.build_index(embeddings=embeddings, chunks=chunks)

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(query_embedding=query, top_k=1)

    assert results[0]["index"] == 0