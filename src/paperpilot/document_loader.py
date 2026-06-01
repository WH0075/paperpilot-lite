from pathlib import Path
from typing import Any

import fitz


Document = dict[str, Any]


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def read_text_file(file_path: str | Path) -> str:
    """读取一个 txt 或 md 文件的原始文本。"""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    text = path.read_text(encoding="utf-8")
    
    if not text.strip():
        raise ValueError(f"File is empty: {path}")
    
    return text


def build_document(text: str, file_path: Path, page: int) -> Document:
    """把文本和来源信息包装成统一的 document 结构。"""
    return {
        "text": text,
        "metadata": {
            "doc_id": f"{file_path.name}:{page}",
            "file_name": file_path.name,
            "file_path": str(file_path),
            "page": page,
            "file_type": file_path.suffix.lower(),
        },
    }
    

def load_text_document(file_path: str | Path) -> list[Document]:
    """读取一个 txt 或 md 文件，并返回 document list。"""

    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix not in {".txt", ".md"}:
        raise ValueError(f"Unsupported text file type: {suffix}")

    text = read_text_file(path)

    return [build_document(text=text, file_path=path, page=1)]


def load_pdf_document(file_path: str | Path) -> list[Document]:
    """读取一个 PDF 文件，每一页生成一个 document。"""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Unsupported PDF file type: {path.suffix}")
    
    documents: list[Document] = []

    try:
        pdf = fitz.open(str(path))
    except Exception as exc:
        raise ValueError(f"Failed to open PDF file: {path}") from exc
    
    try:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            text = page.get_text("text")

            if text.strip():
                documents.append(
                    build_document(
                        text=text, 
                        file_path=path, 
                        page=page_index + 1,
                    )
                )
    finally:
        pdf.close()
    
    if not documents:
        raise ValueError(f"No extractable text found in PDF: {path}")

    return documents 
    


def load_document(file_path: str | Path) -> list[Document]:
    """读取单个文件，根据后缀自动选择 txt/md/pdf 加载函数。"""

    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return load_text_document(file_path=path)

    if suffix == ".pdf":
        return load_pdf_document(file_path=path)
    
    raise ValueError(f"Unsupported file type: {suffix}")



def load_documents(data_dir: str | Path) -> list[Document]:
    """读取一个目录下所有支持的文档。"""

    root = Path(data_dir)

    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")
    
    if not root.is_dir():
        raise ValueError(f"Path is not a directory: {root}")
    
    documents: list[Document] = []

    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        documents.extend(load_document(file_path=file_path))

    return documents
