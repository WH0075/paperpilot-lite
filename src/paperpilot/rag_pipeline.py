from __future__ import annotations

from pathlib import Path
from typing import Any

from .llm_client import BaseLLMClient, MockLLMClient
from .prompt_builder import build_rag_prompt
from .retriever import Retriever


SearchResult = dict[str, Any]
Source = dict[str, Any]
RAGResponse = dict[str, Any]


class RAGPipeline:
    """完整的 Retrieval-Augmented Generation pipeline。"""

    def __init__(
        self,
        retriever: Retriever,
        llm_client: BaseLLMClient,
        max_context_chars: int = 4000,
        max_chunk_chars: int = 1200,
    ) -> None:

        if retriever is None:
            raise ValueError(
                "retriever must not be None"
            )

        if llm_client is None:
            raise ValueError(
                "llm_client must not be None"
            )

        self._validate_positive_int(
            max_context_chars,
            "max_context_chars",
        )

        self._validate_positive_int(
            max_chunk_chars,
            "max_chunk_chars",
        )

        self.retriever = retriever
        self.llm_client = llm_client
        self.max_context_chars = max_context_chars
        self.max_chunk_chars = max_chunk_chars

    def ask(
        self,
        query: str,
        top_k: int | None = None,
        template_name: str = "grounded",
    ) -> RAGResponse:
        """执行完整 RAG 问答流程。"""

        self._validate_query(query)

        if top_k is not None:
            self._validate_positive_int(
                top_k,
                "top_k",
            )

        # 1. Retrieval
        search_results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        # 2. Prompt construction
        prompt = build_rag_prompt(
            query=query,
            search_results=search_results,
            max_context_chars=self.max_context_chars,
            max_chunk_chars=self.max_chunk_chars,
            template_name=template_name,
        )

        # 3. Generation
        answer = self.llm_client.generate(
            prompt
        )

        # 4. Source formatting
        sources = self._build_sources(
            search_results
        )

        return {
            "query": query.strip(),
            "answer": answer,
            "sources": sources,
            "prompt": prompt,
            "search_results": search_results,
            "template_name": template_name,
        }

    @classmethod
    def from_index(
        cls,
        index_dir: str | Path,
        llm_client: BaseLLMClient | None = None,
        model_name: str = (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        device: str | None = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        default_top_k: int = 3,
        default_mode: str = "dense",
        hybrid_alpha: float = 0.5,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        max_context_chars: int = 4000,
        max_chunk_chars: int = 1200,
    ) -> "RAGPipeline":
        """从已有索引构建完整 RAG pipeline。"""

        retriever = Retriever.from_index(
            index_dir=index_dir,
            model_name=model_name,
            device=device,
            normalize_embeddings=(
                normalize_embeddings
            ),
            batch_size=batch_size,
            default_top_k=default_top_k,
            default_mode=default_mode,
            hybrid_alpha=hybrid_alpha,
            fusion_method=fusion_method,
            rrf_k=rrf_k,
        )

        if llm_client is None:
            llm_client = MockLLMClient()

        return cls(
            retriever=retriever,
            llm_client=llm_client,
            max_context_chars=max_context_chars,
            max_chunk_chars=max_chunk_chars,
        )

    @staticmethod
    def _build_sources(
        search_results: list[SearchResult],
    ) -> list[Source]:
        """把 retrieval results 转换为用户可读的来源信息。"""

        if not isinstance(
            search_results,
            list,
        ):
            raise TypeError(
                "search_results must be a list"
            )

        sources: list[Source] = []

        for source_id, result in enumerate(
            search_results,
            start=1,
        ):
            if not isinstance(
                result,
                dict,
            ):
                raise TypeError(
                    "each search result "
                    "must be a dictionary"
                )

            metadata = result.get(
                "metadata",
                {},
            )

            if metadata is None:
                metadata = {}

            if not isinstance(
                metadata,
                dict,
            ):
                raise TypeError(
                    "metadata must be a dictionary"
                )

            file_name = (
                metadata.get("file_name")
                or metadata.get("source")
                or "unknown file"
            )

            page = metadata.get("page")
            chunk_id = metadata.get(
                "chunk_id"
            )

            score = result.get("score")
            index = result.get("index")

            source: Source = {
                "source_id": source_id,
                "file_name": file_name,
                "page": page,
                "chunk_id": chunk_id,
                "score": (
                    float(score)
                    if isinstance(
                        score,
                        (int, float),
                    )
                    else score
                ),
                "index": (
                    int(index)
                    if isinstance(index, int)
                    else index
                ),
            }

            sources.append(source)

        return sources

    @staticmethod
    def _validate_query(
        query: str,
    ) -> None:
        """检查 query 是否合法。"""

        if not isinstance(
            query,
            str,
        ):
            raise TypeError(
                "query must be a string"
            )

        if not query.strip():
            raise ValueError(
                "query must not be empty"
            )

    @staticmethod
    def _validate_positive_int(
        value: int,
        name: str,
    ) -> None:
        """检查某个参数是否为正整数。"""

        if not isinstance(
            value,
            int,
        ):
            raise TypeError(
                f"{name} must be an integer"
            )

        if value <= 0:
            raise ValueError(
                f"{name} must be positive"
            )