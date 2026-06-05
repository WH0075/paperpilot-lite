from __future__ import annotations

from typing import Any


SearchResult = dict[str, Any]


DEFAULT_SYSTEM_INSTRUCTION = """You are a helpful assistant for document question answering.

Answer the user's question using only the provided context.
If the context does not contain enough information, say that the context does not contain enough information.
Do not make up facts.
When possible, cite the context number such as [1], [2], or [3].
"""


def build_rag_prompt(
    query: str,
    search_results: list[SearchResult],
    max_context_chars: int = 4000,
    max_chunk_chars: int = 1200,
) -> str:
    """根据用户问题和检索结果构造 RAG prompt。"""

    _validate_query(query)
    _validate_search_results(search_results)
    _validate_positive_int(max_context_chars, "max_context_chars")
    _validate_positive_int(max_chunk_chars, "max_chunk_chars")

    if not search_results:
        return build_no_context_prompt(query=query)
    
    context = format_context(
        search_results=search_results,
        max_context_chars=max_context_chars,
        max_chunk_chars=max_chunk_chars,
    )

    prompt = f"""{DEFAULT_SYSTEM_INSTRUCTION}

Context:
{context}

Question:
{query.strip()}

Answer:
"""
    
    return prompt.strip()


def build_no_context_prompt(query: str) -> str:
    """当没有检索结果时，构造无上下文 prompt。"""

    _validate_query(query)

    prompt = f"""{DEFAULT_SYSTEM_INSTRUCTION}

Context:
No relevant context was retrieved.

Question:
{query.strip()}

Answer:
"""
    
    return prompt.strip()


def format_context(
    search_results: list[SearchResult],
    max_context_chars: int = 4000,
    max_chunk_chars: int = 1200,
) -> str:
    """把检索结果格式化成 prompt 中的 context 部分。"""

    _validate_search_results(search_results)
    _validate_positive_int(max_context_chars, "max_context_chars")
    _validate_positive_int(max_chunk_chars, "max_chunk_chars")
    
    context_blocks: list[str] = []
    current_length = 0

    for result_index, result in enumerate(search_results, start=1):
        text = result["text"].strip()
        metadata = result.get("metadata", {})
        score = result.get("score")

        truncated_text = truncate_text(text, max_chars=max_chunk_chars)
        source = format_source(metadata)

        block = format_context_block(
            block_id=result_index,
            text=truncated_text,
            source=source,
            score=score,
        )

        block_length = len(block)

        if current_length + block_length > max_context_chars:
            remaining_chars = max_context_chars - current_length

            if remaining_chars <= 0:
                break

            shortened_block = truncate_text(block, max_chars=remaining_chars)

            if shortened_block.strip():
                context_blocks.append(shortened_block)

            break

        context_blocks.append(block)
        current_length += block_length

    if not context_blocks:
        return "No relevant context was retrieved."
    
    return "\n\n".join(context_blocks)


def format_context_block(
    block_id: int,
    text: str,
    source: str,
    score: float | int | None = None,
) -> str:
    """格式化单个 context block。"""

    _validate_positive_int(block_id, "block_id")

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text.strip():
        raise ValueError("text must not be empty")

    if not isinstance(source, str):
        raise TypeError("source must be a string")

    source = source.strip() if source.strip() else "unknown source"

    score_line = ""

    if score is not None:
        score_line = f"\nScore: {float(score):.4f}"

    return f"""[{block_id}]
Source: {source}{score_line}
Content:
{text.strip()}"""


def format_source(metadata: dict[str, Any]) -> str:
    """把 metadata 格式化成来源字符串。"""

    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")
    
    file_name = metadata.get("file_name") or metadata.get("source") or "unknown file"
    page = metadata.get("page")
    chunk_id = metadata.get("chunk_id")

    parts: list[str] = [str(file_name)]

    if page is not None:
        parts.append(f"page {page}")

    if chunk_id is not None:
        parts.append(f"chunk_id {chunk_id}")

    return ", ".join(parts)

    
def truncate_text(text: str, max_chars: int) -> str:
    """把文本截断到指定字符数以内。"""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    
    _validate_positive_int(max_chars, "max_chars")

    text = text.strip()

    if len(text) <= max_chars:
        return text
    
    if max_chars <= 3:
        return text[:max_chars]
    
    return text[: max_chars - 3].rstrip() + "..."


def _validate_query(query: str) -> None:
    """检查 query 是否合法。"""

    if not isinstance(query, str):
        raise TypeError("query must be a string")

    if not query.strip():
        raise ValueError("query must not be empty")
    

def _validate_search_results(search_results: list[SearchResult]) -> None:
    """检查 search_results 是否合法。"""

    if not isinstance(search_results, list):
        raise TypeError("search_results must be a list")

    for result in search_results:
        if not isinstance(result, dict):
            raise TypeError("each search result must be a dictionary")

        if "text" not in result:
            raise KeyError("each search result must contain a text field")

        if not isinstance(result["text"], str):
            raise TypeError("search result text must be a string")

        if not result["text"].strip():
            raise ValueError("search result text must not be empty")

        if "metadata" in result and not isinstance(result["metadata"], dict):
            raise TypeError("search result metadata must be a dictionary")


def _validate_positive_int(value: int, name: str) -> None:
    """检查某个参数是否为正整数。"""

    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")

    if value <= 0:
        raise ValueError(f"{name} must be positive")