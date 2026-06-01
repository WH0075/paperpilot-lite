from typing import Any


Document = dict[str, Any]
Chunk = dict[str, Any]


def validate_chunk_params(chunk_size: int, overlap: int) -> None:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
   


def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> list[tuple[str, int, int]]:

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    
    if not text.strip():
        raise ValueError("text must not be empty")
    
    validate_chunk_params(chunk_size=chunk_size, overlap=overlap)

    chunks: list[tuple[str, int, int]] = []
    step = chunk_size - overlap
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append((chunk.strip(), start, end))
        
        if end == text_length:
            break

        start += step
    
    return chunks



def chunk_document(
    document: Document,
    chunk_size: int,
    overlap: int = 0,
) -> list[Chunk]:

    if "text" not in document:
        raise KeyError("document must contain a text field")
    
    text = document["text"]
    original_metadata = dict(document.get("metadata", {}))

    doc_id = original_metadata.get("doc_id", "unknown_doc")
    source = original_metadata.get("file_name", "unknown_source")

    text_chunks = chunk_text(
        text=text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    chunks: list[Chunk] = []

    for chunk_index, (chunk_content, start_char, end_char) in enumerate(text_chunks):
        chunk_metadata = {
            **original_metadata,
            "source": source,
            "chunk_id": f"{doc_id}:{chunk_index}",
            "start_char": start_char,
            "end_char": end_char,
        }

        chunks.append(
            {
                "text": chunk_content,
                "metadata": chunk_metadata,
            }
        )
    
    return chunks



def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    overlap: int = 0,
) -> list[Chunk]:

    all_chunks: list[Chunk] = []

    for document in documents:
        all_chunks.extend(
            chunk_document(
                document=document,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
    
    return all_chunks
