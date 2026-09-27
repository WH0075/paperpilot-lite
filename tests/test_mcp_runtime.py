from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.paperpilot import mcp_runtime
from src.paperpilot.mcp_runtime import (
    PaperPilotRuntime,
)


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
            }
        )

        return [
            {
                "text": "fake evidence",
                "score": 0.9,
                "metadata": {
                    "file_name": "fake.pdf",
                    "page": 1,
                    "chunk_id": "fake.pdf:1:1",
                },
            }
        ]


class FakePipeline:
    def __init__(
        self,
        retriever,
    ) -> None:
        self.retriever = retriever
        self.calls: list[dict] = []

    def ask(
        self,
        query: str,
        top_k: int | None = None,
        template_name: str = "grounded",
    ) -> dict:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "template_name": template_name,
            }
        )

        return {
            "query": query,
            "answer": "fake answer",
            "sources": [],
            "prompt": "fake prompt",
            "search_results": [],
            "template_name": template_name,
        }


def make_fake_config(
    *,
    reranker_enabled: bool = False,
):
    return SimpleNamespace(
        data=SimpleNamespace(
            index_dir="data/index",
        ),
        retrieval=SimpleNamespace(
            embedding_model=(
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            ),
            device="cpu",
            normalize_embeddings=True,
            batch_size=32,
            top_k=5,
            mode="hybrid",
            hybrid_alpha=0.5,
            fusion_method="rrf",
            rrf_k=60,
            hybrid_candidate_k=20,
            reranker=SimpleNamespace(
                enabled=reranker_enabled,
                model_name="fake-cross-encoder",
                candidate_k=10,
                device="cpu",
                batch_size=16,
            ),
        ),
        prompt=SimpleNamespace(
            template_name="grounded",
            max_context_chars=4000,
            max_chunk_chars=1200,
        ),
        llm=SimpleNamespace(
            provider="mock",
            model_name=None,
            base_url=None,
            temperature=0.2,
            max_tokens=512,
            thinking_enabled=False,
            timeout=60,
        ),
    )


def test_runtime_rejects_none_config():
    retriever = FakeRetriever()
    pipeline = FakePipeline(
        retriever
    )

    with pytest.raises(
        ValueError,
        match="config must not be None",
    ):
        PaperPilotRuntime(
            config=None,
            retriever=retriever,
            pipeline=pipeline,
        )


def test_runtime_rejects_none_retriever():
    config = make_fake_config()

    retriever = FakeRetriever()
    pipeline = FakePipeline(
        retriever
    )

    with pytest.raises(
        ValueError,
        match="retriever must not be None",
    ):
        PaperPilotRuntime(
            config=config,
            retriever=None,
            pipeline=pipeline,
        )


def test_runtime_rejects_none_pipeline():
    config = make_fake_config()
    retriever = FakeRetriever()

    with pytest.raises(
        ValueError,
        match="pipeline must not be None",
    ):
        PaperPilotRuntime(
            config=config,
            retriever=retriever,
            pipeline=None,
        )


def test_runtime_requires_shared_retriever():
    config = make_fake_config()

    runtime_retriever = FakeRetriever()
    pipeline_retriever = FakeRetriever()

    pipeline = FakePipeline(
        pipeline_retriever
    )

    with pytest.raises(
        ValueError,
        match=(
            "must share the same "
            "retriever instance"
        ),
    ):
        PaperPilotRuntime(
            config=config,
            retriever=runtime_retriever,
            pipeline=pipeline,
        )


def test_search_reuses_existing_retriever():
    config = make_fake_config()
    retriever = FakeRetriever()

    pipeline = FakePipeline(
        retriever
    )

    runtime = PaperPilotRuntime(
        config=config,
        retriever=retriever,
        pipeline=pipeline,
    )

    runtime.search(
        "first query",
        top_k=3,
    )

    runtime.search(
        "second query",
        top_k=5,
    )

    assert len(
        retriever.calls
    ) == 2

    assert (
        retriever.calls[0]
        == {
            "query": "first query",
            "top_k": 3,
        }
    )

    assert (
        retriever.calls[1]
        == {
            "query": "second query",
            "top_k": 5,
        }
    )


def test_ask_uses_shared_pipeline():
    config = make_fake_config()
    retriever = FakeRetriever()

    pipeline = FakePipeline(
        retriever
    )

    runtime = PaperPilotRuntime(
        config=config,
        retriever=retriever,
        pipeline=pipeline,
    )

    result = runtime.ask(
        "What is RAG?",
        top_k=4,
    )

    assert (
        result["answer"]
        == "fake answer"
    )

    assert (
        pipeline.calls[0]
        == {
            "query": "What is RAG?",
            "top_k": 4,
            "template_name": "grounded",
        }
    )


def test_reranker_disabled_returns_none():
    config = make_fake_config(
        reranker_enabled=False,
    )

    reranker = (
        PaperPilotRuntime
        ._create_reranker(
            config
        )
    )

    assert reranker is None


def test_reranker_enabled_builds_model(
    monkeypatch,
):
    config = make_fake_config(
        reranker_enabled=True,
    )

    captured = {}

    class FakeReranker:
        def __init__(
            self,
            *,
            model_name,
            device,
            batch_size,
        ):
            captured[
                "model_name"
            ] = model_name

            captured[
                "device"
            ] = device

            captured[
                "batch_size"
            ] = batch_size

    monkeypatch.setattr(
        mcp_runtime,
        "CrossEncoderReranker",
        FakeReranker,
    )

    reranker = (
        PaperPilotRuntime
        ._create_reranker(
            config
        )
    )

    assert isinstance(
        reranker,
        FakeReranker,
    )

    assert captured == {
        "model_name": "fake-cross-encoder",
        "device": "cpu",
        "batch_size": 16,
    }


def test_from_config_builds_one_shared_retriever(
    monkeypatch,
):
    config = make_fake_config()

    fake_retriever = FakeRetriever()

    captured = {
        "calls": 0,
    }

    def fake_from_index(
        cls,
        **kwargs,
    ):
        captured["calls"] += 1

        return fake_retriever

    monkeypatch.setattr(
        mcp_runtime.Retriever,
        "from_index",
        classmethod(
            fake_from_index
        ),
    )

    runtime = (
        PaperPilotRuntime.from_config(
            config=config,
        )
    )

    # Retriever.from_index() 只能初始化一次。
    assert captured["calls"] == 1

    # Runtime 持有的就是这一份 Retriever。
    assert (
        runtime.retriever
        is fake_retriever
    )

    # RAGPipeline 不能偷偷重新创建 Retriever。
    assert (
        runtime.pipeline.retriever
        is fake_retriever
    )

    # 直接保护 Goal 2 的核心架构不变量：
    # search() 与 ask() 共享同一个 Retriever。
    assert (
        runtime.pipeline.retriever
        is runtime.retriever
    )