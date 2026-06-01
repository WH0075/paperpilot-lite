import pytest

from src.paperpilot.chunker import (
    chunk_document,
    chunk_documents,
    chunk_text,
    validate_chunk_params,
)


def test_validate_chunk_params_accepts_valid_values():
    validate_chunk_params(chunk_size=100, overlap=20)


def test_validate_chunk_params_rejects_invalid_chunk_size():
    with pytest.raises(ValueError):
        validate_chunk_params(chunk_size=0, overlap=0)


def test_validate_chunk_params_rejects_negative_overlap():
    with pytest.raises(ValueError):
        validate_chunk_params(chunk_size=100, overlap=-1)


def test_validate_chunk_params_rejects_overlap_too_large():
    with pytest.raises(ValueError):
        validate_chunk_params(chunk_size=100, overlap=100)


def test_chunk_text_without_overlap():
    chunks = chunk_text("abcdefghij", chunk_size=5, overlap=0)

    assert chunks == [
        ("abcde", 0, 5),
        ("fghij", 5, 10),
    ]


def test_chunk_text_with_overlap():
    chunks = chunk_text("abcdefghij", chunk_size=5, overlap=2)

    assert chunks == [
        ("abcde", 0, 5),
        ("defgh", 3, 8),
        ("ghij", 6, 10),
    ]


def test_chunk_text_rejects_empty_text():
    with pytest.raises(ValueError):
        chunk_text("   ", chunk_size=5, overlap=0)


def test_chunk_document_preserves_metadata_and_adds_chunk_metadata():
    document = {
        "text": "abcdefghij",
        "metadata": {
            "doc_id": "sample.txt:1",
            "file_name": "sample.txt",
            "file_path": "data/raw/sample.txt",
            "page": 1,
            "file_type": ".txt",
        },
    }

    chunks = chunk_document(document, chunk_size=5, overlap=2)

    assert len(chunks) == 3
    assert chunks[0]["text"] == "abcde"
    assert chunks[0]["metadata"]["doc_id"] == "sample.txt:1"
    assert chunks[0]["metadata"]["file_name"] == "sample.txt"
    assert chunks[0]["metadata"]["source"] == "sample.txt"
    assert chunks[0]["metadata"]["chunk_id"] == "sample.txt:1:0"
    assert chunks[0]["metadata"]["start_char"] == 0
    assert chunks[0]["metadata"]["end_char"] == 5

    assert chunks[1]["text"] == "defgh"
    assert chunks[1]["metadata"]["chunk_id"] == "sample.txt:1:1"
    assert chunks[1]["metadata"]["start_char"] == 3
    assert chunks[1]["metadata"]["end_char"] == 8


def test_chunk_documents_batch():
    documents = [
        {
            "text": "abcdefghij",
            "metadata": {
                "doc_id": "a.txt:1",
                "file_name": "a.txt",
                "page": 1,
            },
        },
        {
            "text": "klmnopqrst",
            "metadata": {
                "doc_id": "b.txt:1",
                "file_name": "b.txt",
                "page": 1,
            },
        },
    ]

    chunks = chunk_documents(documents, chunk_size=5, overlap=0)

    assert len(chunks) == 4
    assert chunks[0]["metadata"]["chunk_id"] == "a.txt:1:0"
    assert chunks[1]["metadata"]["chunk_id"] == "a.txt:1:1"
    assert chunks[2]["metadata"]["chunk_id"] == "b.txt:1:0"
    assert chunks[3]["metadata"]["chunk_id"] == "b.txt:1:1"