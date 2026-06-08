import pytest

from src.paperpilot.llm_client import MockLLMClient
from src.paperpilot.rag_pipeline import RAGPipeline


class FakeRetriever:
    def __init__(self):
        self.last_query = None
        self.last_top_k = None

    def retrieve(self, query: str, top_k=None):
        self.last_query = query
        self.last_top_k = top_k

        return [
            {
                "text": "Retrieval-Augmented Generation combines retrieval and generation.",
                "metadata": {
                    "file_name": "rag_intro.txt",
                    "page": 1,
                    "chunk_id": "rag_intro.txt:1:0",
                },
                "score": 0.85,
                "index": 0,
            },
            {
                "text": "A retriever finds relevant chunks from a vector store.",
                "metadata": {
                    "file_name": "retriever.md",
                    "page": 2,
                    "chunk_id": "retriever.md:2:0",
                },
                "score": 0.75,
                "index": 1,
            },
        ]


def test_ask_returns_rag_response():
    retriever = FakeRetriever()
    llm_client = MockLLMClient(fixed_answer="Mock RAG answer.")

    pipeline = RAGPipeline(
        retriever=retriever,
        llm_client=llm_client,
    )

    response = pipeline.ask("What is RAG?", top_k=2)

    assert response["query"] == "What is RAG?"
    assert response["answer"] == "Mock RAG answer."
    assert "sources" in response
    assert "prompt" in response
    assert "search_results" in response
    assert len(response["sources"]) == 2
    assert len(response["search_results"]) == 2


def test_ask_calls_retriever_with_query_and_top_k():
    retriever = FakeRetriever()
    llm_client = MockLLMClient(fixed_answer="Mock answer.")

    pipeline = RAGPipeline(
        retriever=retriever,
        llm_client=llm_client,
    )

    pipeline.ask("What is RAG?", top_k=1)

    assert retriever.last_query == "What is RAG?"
    assert retriever.last_top_k == 1


def test_ask_sends_prompt_to_llm_client():
    retriever = FakeRetriever()
    llm_client = MockLLMClient(fixed_answer="Mock answer.")

    pipeline = RAGPipeline(
        retriever=retriever,
        llm_client=llm_client,
    )

    response = pipeline.ask("What is RAG?", top_k=1)

    assert llm_client.last_prompt is not None
    assert llm_client.last_prompt == response["prompt"]
    assert "Context:" in response["prompt"]
    assert "Question:" in response["prompt"]
    assert "Answer:" in response["prompt"]
    assert "What is RAG?" in response["prompt"]


def test_sources_are_built_from_search_results():
    retriever = FakeRetriever()
    llm_client = MockLLMClient(fixed_answer="Mock answer.")

    pipeline = RAGPipeline(
        retriever=retriever,
        llm_client=llm_client,
    )

    response = pipeline.ask("What is RAG?", top_k=2)

    sources = response["sources"]

    assert sources[0]["source_id"] == 1
    assert sources[0]["file_name"] == "rag_intro.txt"
    assert sources[0]["page"] == 1
    assert sources[0]["chunk_id"] == "rag_intro.txt:1:0"
    assert sources[0]["score"] == 0.85
    assert sources[0]["index"] == 0

    assert sources[1]["source_id"] == 2
    assert sources[1]["file_name"] == "retriever.md"


def test_ask_rejects_empty_query():
    retriever = FakeRetriever()
    llm_client = MockLLMClient()

    pipeline = RAGPipeline(
        retriever=retriever,
        llm_client=llm_client,
    )

    with pytest.raises(ValueError):
        pipeline.ask("   ")


def test_ask_rejects_non_string_query():
    retriever = FakeRetriever()
    llm_client = MockLLMClient()

    pipeline = RAGPipeline(
        retriever=retriever,
        llm_client=llm_client,
    )

    with pytest.raises(TypeError):
        pipeline.ask(123)  # type: ignore[arg-type]


def test_ask_rejects_invalid_top_k():
    retriever = FakeRetriever()
    llm_client = MockLLMClient()

    pipeline = RAGPipeline(
        retriever=retriever,
        llm_client=llm_client,
    )

    with pytest.raises(ValueError):
        pipeline.ask("What is RAG?", top_k=0)


def test_init_rejects_none_retriever():
    llm_client = MockLLMClient()

    with pytest.raises(ValueError):
        RAGPipeline(
            retriever=None,  # type: ignore[arg-type]
            llm_client=llm_client,
        )


def test_init_rejects_none_llm_client():
    retriever = FakeRetriever()

    with pytest.raises(ValueError):
        RAGPipeline(
            retriever=retriever,
            llm_client=None,  # type: ignore[arg-type]
        )


def test_init_rejects_invalid_max_context_chars():
    retriever = FakeRetriever()
    llm_client = MockLLMClient()

    with pytest.raises(ValueError):
        RAGPipeline(
            retriever=retriever,
            llm_client=llm_client,
            max_context_chars=0,
        )


def test_build_sources_handles_missing_metadata():
    search_results = [
        {
            "text": "Some text.",
            "score": 0.5,
            "index": 3,
        }
    ]

    sources = RAGPipeline._build_sources(search_results)

    assert sources[0]["source_id"] == 1
    assert sources[0]["file_name"] == "unknown file"
    assert sources[0]["page"] is None
    assert sources[0]["chunk_id"] is None
    assert sources[0]["score"] == 0.5
    assert sources[0]["index"] == 3