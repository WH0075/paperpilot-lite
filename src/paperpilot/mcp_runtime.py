from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config import AppConfig, load_config
from .llm_client import (
    BaseLLMClient,
    MockLLMClient,
    OpenAICompatibleLLMClient,
)
from .rag_pipeline import RAGPipeline
from .reranker import CrossEncoderReranker
from .retriever import Retriever


SearchResult = dict[str, Any]
RAGResponse = dict[str, Any]


class PaperPilotRuntime:
    """
    PaperPilot-Lite 的长期运行时对象。

    负责在 MCP Server 生命周期内复用：
    - configuration（配置）
    - Retriever
    - embedding model（嵌入模型）
    - vector index（向量索引）
    - BM25 retriever
    - optional reranker（可选重排序器）
    - RAGPipeline
    - LLM client（大模型客户端）

    避免每次 MCP tool 调用都重新加载模型和索引。
    """

    def __init__(
        self,
        config: AppConfig,
        retriever: Retriever,
        pipeline: RAGPipeline,
    ) -> None:
        if config is None:
            raise ValueError(
                "config must not be None"
            )

        if retriever is None:
            raise ValueError(
                "retriever must not be None"
            )

        if pipeline is None:
            raise ValueError(
                "pipeline must not be None"
            )

        # Runtime 的关键设计要求：
        # search() 和 ask() 必须共享同一个 Retriever。
        if pipeline.retriever is not retriever:
            raise ValueError(
                "pipeline and runtime must share "
                "the same retriever instance"
            )

        self.config = config
        self.retriever = retriever
        self.pipeline = pipeline

    @classmethod
    def from_config(
        cls,
        config: AppConfig | None = None,
        index_dir: str | Path | None = None,
    ) -> "PaperPilotRuntime":
        """
        根据配置创建完整 Runtime。

        初始化顺序：

        config
          ↓
        optional reranker
          ↓
        Retriever.from_index()
          ↓
        LLM client
          ↓
        RAGPipeline(shared retriever)
          ↓
        PaperPilotRuntime
        """

        config = (
            config
            if config is not None
            else load_config()
        )

        selected_index_dir = (
            index_dir
            if index_dir is not None
            else config.data.index_dir
        )

        # --------------------------------------------------
        # Optional Cross-Encoder Reranker
        # --------------------------------------------------

        reranker = (
            cls._create_reranker(
                config
            )
        )

        # --------------------------------------------------
        # Retriever 只初始化一次
        # --------------------------------------------------

        retriever = Retriever.from_index(
            index_dir=selected_index_dir,
            model_name=(
                config.retrieval.embedding_model
            ),
            device=(
                config.retrieval.device
            ),
            normalize_embeddings=(
                config.retrieval.normalize_embeddings
            ),
            batch_size=(
                config.retrieval.batch_size
            ),
            default_top_k=(
                config.retrieval.top_k
            ),
            default_mode=(
                config.retrieval.mode
            ),
            hybrid_alpha=(
                config.retrieval.hybrid_alpha
            ),
            fusion_method=(
                config.retrieval.fusion_method
            ),
            rrf_k=(
                config.retrieval.rrf_k
            ),
            hybrid_candidate_k=(
                config.retrieval.hybrid_candidate_k
            ),
            reranker=reranker,
            reranker_candidate_k=(
                config.retrieval.reranker.candidate_k
            ),
        )

        # --------------------------------------------------
        # LLM Client
        # --------------------------------------------------

        llm_client = (
            cls._create_llm_client(
                config
            )
        )

        # --------------------------------------------------
        # Pipeline 直接注入上面的 shared Retriever
        #
        # 不使用 RAGPipeline.from_index()，
        # 否则会再次创建一套 Retriever。
        # --------------------------------------------------

        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=llm_client,
            max_context_chars=(
                config.prompt.max_context_chars
            ),
            max_chunk_chars=(
                config.prompt.max_chunk_chars
            ),
        )

        return cls(
            config=config,
            retriever=retriever,
            pipeline=pipeline,
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        使用已经初始化好的 Retriever 搜索论文证据。
        """

        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
        )

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        template_name: str | None = None,
    ) -> RAGResponse:
        """
        使用共享 Retriever 的 RAGPipeline 回答问题。
        """

        selected_template = (
            template_name
            if template_name is not None
            else self.config.prompt.template_name
        )

        return self.pipeline.ask(
            query=question,
            top_k=top_k,
            template_name=selected_template,
        )

    @staticmethod
    def _create_reranker(
        config: AppConfig,
    ) -> CrossEncoderReranker | None:
        """
        根据配置决定是否创建 Cross-Encoder Reranker。
        """

        reranker_config = (
            config.retrieval.reranker
        )

        if not reranker_config.enabled:
            return None

        return CrossEncoderReranker(
            model_name=(
                reranker_config.model_name
            ),
            device=(
                reranker_config.device
            ),
            batch_size=(
                reranker_config.batch_size
            ),
        )

    @staticmethod
    def _create_llm_client(
        config: AppConfig,
    ) -> BaseLLMClient:
        """
        根据 AppConfig 创建 LLM Client。

        MCP Runtime 不依赖 CLI，
        因此不能调用 cli.create_llm_client()。
        """

        llm_config = config.llm

        if llm_config.provider == "mock":
            return MockLLMClient()

        if (
            llm_config.provider
            == "openai-compatible"
        ):
            if not llm_config.model_name:
                raise ValueError(
                    "OpenAI-compatible LLM "
                    "requires model_name"
                )

            return (
                OpenAICompatibleLLMClient(
                    model_name=(
                        llm_config.model_name
                    ),
                    api_key=os.getenv(
                        "LLM_API_KEY"
                    ),
                    base_url=(
                        llm_config.base_url
                    ),
                    temperature=(
                        llm_config.temperature
                    ),
                    max_tokens=(
                        llm_config.max_tokens
                    ),
                    thinking_enabled=(
                        llm_config.thinking_enabled
                    ),
                    timeout=(
                        llm_config.timeout
                    ),
                )
            )

        raise ValueError(
            "Unsupported LLM provider: "
            f"{llm_config.provider}"
        )