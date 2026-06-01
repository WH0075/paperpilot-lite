import fitz
import pytest

from src.paperpilot.document_loader import (
    load_document,
    load_documents,
    load_pdf_document,
    load_text_document,
    read_text_file,
)


def create_sample_pdf(path, text: str = "This is a PDF document."):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_read_text_file(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello RAG", encoding="utf-8")

    text = read_text_file(file_path)

    assert text == "hello RAG"


def test_load_text_document_txt(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello txt", encoding="utf-8")

    documents = load_text_document(file_path)

    assert len(documents) == 1
    assert documents[0]["text"] == "hello txt"
    assert documents[0]["metadata"]["file_name"] == "sample.txt"
    assert documents[0]["metadata"]["page"] == 1
    assert documents[0]["metadata"]["file_type"] == ".txt"


def test_load_text_document_md(tmp_path):
    file_path = tmp_path / "sample.md"
    file_path.write_text("# Title\n\nhello markdown", encoding="utf-8")

    documents = load_text_document(file_path)

    assert len(documents) == 1
    assert "hello markdown" in documents[0]["text"]
    assert documents[0]["metadata"]["file_type"] == ".md"


def test_load_pdf_document(tmp_path):
    file_path = tmp_path / "sample.pdf"
    create_sample_pdf(file_path, "hello pdf")

    documents = load_pdf_document(file_path)

    assert len(documents) == 1
    assert "hello pdf" in documents[0]["text"]
    assert documents[0]["metadata"]["file_name"] == "sample.pdf"
    assert documents[0]["metadata"]["page"] == 1
    assert documents[0]["metadata"]["file_type"] == ".pdf"


def test_load_document_dispatch_txt(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("dispatch txt", encoding="utf-8")

    documents = load_document(file_path)

    assert len(documents) == 1
    assert "dispatch txt" in documents[0]["text"]


def test_load_documents_from_directory(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()

    txt_path = data_dir / "a.txt"
    md_path = data_dir / "b.md"
    pdf_path = data_dir / "c.pdf"
    ignored_path = data_dir / "ignored.csv"

    txt_path.write_text("txt content", encoding="utf-8")
    md_path.write_text("md content", encoding="utf-8")
    ignored_path.write_text("ignored", encoding="utf-8")
    create_sample_pdf(pdf_path, "pdf content")

    documents = load_documents(data_dir)

    file_names = {doc["metadata"]["file_name"] for doc in documents}

    assert len(documents) == 3
    assert "a.txt" in file_names
    assert "b.md" in file_names
    assert "c.pdf" in file_names
    assert "ignored.csv" not in file_names


def test_missing_file_raises_error(tmp_path):
    missing_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        read_text_file(missing_path)


def test_empty_text_file_raises_error(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        read_text_file(file_path)


def test_unsupported_file_type_raises_error(tmp_path):
    file_path = tmp_path / "sample.csv"
    file_path.write_text("a,b,c", encoding="utf-8")

    with pytest.raises(ValueError):
        load_document(file_path)