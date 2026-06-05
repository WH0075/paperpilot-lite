import pytest

from src.paperpilot.prompt_builder import (
    build_no_context_prompt,
    build_rag_prompt,
    format_context,
    format_context_block,
    format_source,
    truncate_text,
)


def make_search_results():
    return [
        {
            "text": "Retrieval-Augmented Generation combines retrieval and generation.",
            "metadata": {
                "file_name": "rag_intro.txt",
                "page": 1,
                "chunk_id": "rag_intro.txt:1:0",
            },
            "score": 0.8234,
            "index": 0,
        },
        {
            "text": "A retriever finds relevant chunks from a vector store.",
            "metadata": {
                "file_name": "retriever.md",
                "page": 2,
                "chunk_id": "retriever.md:2:0",
            },
            "score": 0.7567,
            "index": 1,
        },
    ]


def test_format_source_with_file_page_and_chunk_id():
    metadata = {
        "file_name": "rag_intro.txt",
        "page": 1,
        "chunk_id": "rag_intro.txt:1:0",
    }

    source = format_source(metadata)

    assert "rag_intro.txt" in source
    assert "page 1" in source
    assert "chunk_id rag_intro.txt:1:0" in source


def test_format_source_uses_unknown_file_when_missing():
    source = format_source({})

    assert source == "unknown file"


def test_format_context_block_contains_id_source_score_and_text():
    block = format_context_block(
        block_id=1,
        text="RAG combines retrieval and generation.",
        source="rag_intro.txt, page 1",
        score=0.823456,
    )

    assert "[1]" in block
    assert "Source: rag_intro.txt, page 1" in block
    assert "Score: 0.8235" in block
    assert "Content:" in block
    assert "RAG combines retrieval and generation." in block


def test_format_context_numbers_results():
    context = format_context(make_search_results())

    assert "[1]" in context
    assert "[2]" in context
    assert "rag_intro.txt" in context
    assert "retriever.md" in context


def test_build_rag_prompt_contains_query_and_context():
    prompt = build_rag_prompt(
        query="What is RAG?",
        search_results=make_search_results(),
    )

    assert "Context:" in prompt
    assert "Question:" in prompt
    assert "Answer:" in prompt
    assert "What is RAG?" in prompt
    assert "Retrieval-Augmented Generation" in prompt
    assert "rag_intro.txt" in prompt


def test_build_no_context_prompt():
    prompt = build_no_context_prompt("What is RAG?")

    assert "No relevant context was retrieved." in prompt
    assert "What is RAG?" in prompt
    assert "Answer:" in prompt


def test_build_rag_prompt_with_empty_results_uses_no_context_prompt():
    prompt = build_rag_prompt(
        query="What is RAG?",
        search_results=[],
    )

    assert "No relevant context was retrieved." in prompt


def test_truncate_text_short_text_unchanged():
    text = "short text"

    assert truncate_text(text, max_chars=20) == "short text"


def test_truncate_text_long_text_adds_ellipsis():
    text = "abcdefghij"

    assert truncate_text(text, max_chars=6) == "abc..."


def test_format_context_respects_max_chunk_chars():
    results = [
        {
            "text": "abcdefghij",
            "metadata": {"file_name": "a.txt"},
            "score": 0.9,
            "index": 0,
        }
    ]

    context = format_context(
        search_results=results,
        max_context_chars=1000,
        max_chunk_chars=6,
    )

    assert "abc..." in context


def test_format_context_respects_max_context_chars():
    results = make_search_results()

    context = format_context(
        search_results=results,
        max_context_chars=80,
        max_chunk_chars=100,
    )

    assert len(context) <= 80


def test_build_rag_prompt_rejects_empty_query():
    with pytest.raises(ValueError):
        build_rag_prompt(
            query="   ",
            search_results=make_search_results(),
        )


def test_build_rag_prompt_rejects_invalid_search_results():
    with pytest.raises(TypeError):
        build_rag_prompt(
            query="What is RAG?",
            search_results="bad input",  # type: ignore[arg-type]
        )


def test_format_source_rejects_non_dict_metadata():
    with pytest.raises(TypeError):
        format_source("bad metadata")  # type: ignore[arg-type]


def test_truncate_text_rejects_invalid_max_chars():
    with pytest.raises(ValueError):
        truncate_text("hello", max_chars=0)