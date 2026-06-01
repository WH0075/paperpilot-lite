import pytest

from src.paperpilot.cleaner import clean_document, clean_documents, clean_text


def test_clean_text_strips_outer_whitespace():
    text = "   hello RAG   "
    assert clean_text(text) == "hello RAG"


def test_clean_text_normalizes_windows_newlines():
    text = "hello\r\nRAG\r\nsystem"
    assert clean_text(text) == "hello\nRAG\nsystem"


def test_clean_text_compresses_spaces_and_tabs():
    text = "hello     RAG\t\t system"
    assert clean_text(text) == "hello RAG system"


def test_clean_text_compresses_blank_lines():
    text = "hello\n\n\n\nRAG"
    assert clean_text(text) == "hello\n\nRAG"


def test_clean_text_rejects_empty_text():
    with pytest.raises(ValueError):
        clean_text("   \n\n   ")


def test_clean_document_preserves_metadata():
    document = {
        "text": "   hello    RAG   ",
        "metadata": {
            "doc_id": "sample.txt:1",
            "file_name": "sample.txt",
            "page": 1,
        },
    }

    cleaned = clean_document(document)

    assert cleaned["text"] == "hello RAG"
    assert cleaned["metadata"]["doc_id"] == "sample.txt:1"
    assert cleaned["metadata"]["file_name"] == "sample.txt"
    assert cleaned["metadata"]["page"] == 1


def test_clean_documents_batch():
    documents = [
        {
            "text": " hello   world ",
            "metadata": {"doc_id": "a.txt:1"},
        },
        {
            "text": " RAG\n\n\nsystem ",
            "metadata": {"doc_id": "b.txt:1"},
        },
    ]

    cleaned = clean_documents(documents)

    assert len(cleaned) == 2
    assert cleaned[0]["text"] == "hello world"
    assert cleaned[1]["text"] == "RAG\n\nsystem"