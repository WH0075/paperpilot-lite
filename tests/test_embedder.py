import numpy as np
import pytest

from src.paperpilot.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    return Embedder(device="cpu")


def test_embed_text_returns_numpy_array(embedder):
    embedding = embedder.embed_text("RAG combines retrieval and generation.")

    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert embedding.shape[0] == embedder.get_embedding_dim()
    assert embedding.dtype == np.float32


def test_embed_texts_returns_matrix(embedder):
    texts = [
        "RAG combines retrieval and generation.",
        "Attention computes similarity between queries and keys.",
        "Agents can use tools.",
    ]

    embeddings = embedder.embed_texts(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.ndim == 2
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] == embedder.get_embedding_dim()
    assert embeddings.dtype == np.float32


def test_embed_chunks_returns_matrix(embedder):
    chunks = [
        {
            "text": "RAG combines retrieval and generation.",
            "metadata": {"chunk_id": "a.txt:1:0"},
        },
        {
            "text": "Attention computes similarity between queries and keys.",
            "metadata": {"chunk_id": "b.txt:1:0"},
        },
    ]

    embeddings = embedder.embed_chunks(chunks)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.ndim == 2
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] == embedder.get_embedding_dim()


def test_embed_text_rejects_empty_text(embedder):
    with pytest.raises(ValueError):
        embedder.embed_text("   ")


def test_embed_texts_rejects_empty_list(embedder):
    with pytest.raises(ValueError):
        embedder.embed_texts([])


def test_embed_chunks_rejects_missing_text(embedder):
    chunks = [
        {
            "metadata": {"chunk_id": "bad_chunk"},
        }
    ]

    with pytest.raises(KeyError):
        embedder.embed_chunks(chunks)