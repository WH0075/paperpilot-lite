import re
from typing import Any


Document = dict[str, Any]


def clean_text(text: str) -> str:

    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if not text:
        raise ValueError("text is empty after cleaning")
    
    return text



def clean_document(document: Document) -> Document:

    if "text" not in document:
        raise KeyError("document must contain a text field")
    
    cleaned_text = clean_text(document["text"])
    metadata = dict(document.get("metadata", {}))

    return {
        "text": cleaned_text,
        "metadata": metadata,
    }
    


def clean_documents(documents: list[Document]) -> list[Document]:

    return [clean_document(document) for document in documents]
    