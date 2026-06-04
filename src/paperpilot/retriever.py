from __future__ import annotations

from pathlib import Path
from typing import Any

from .embedder import Embedder
from .vector_store import VectorStore


SearchResult = dict[str, Any]


class Retriever:
    """RAG 检索模块。"""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        default_top_k: int = 3,
    ) -> None:
        """初始化 Retriever。"""

        if embedder is None:
            raise ValueError("embedder must not be None")

        if vector_store is None:
            raise ValueError("vector_store must not be None")

        self._validate_top_k(default_top_k)

        self.embedder = embedder
        self.vector_store = vector_store
        self.default_top_k = default_top_k    


    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """根据用户 query 检索最相关的 chunks。"""

        self._validate_query(query)

        k = self.default_top_k if top_k is None else top_k
        self._validate_top_k(k)

        query_embedding = self.embedder.embed_text(query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=k,
        )

        return results


    def retrieve_texts(self, query: str, top_k: int | None = None) -> list[str]:
        """只返回检索结果中的文本内容。"""

        results = self.retrieve(query=query, top_k=top_k)

        return [result["text"] for result in results]


    def retrieve_with_sources(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """返回带来源信息的检索结果。"""

        return self.retrieve(query=query, top_k=top_k)
    

    @classmethod
    def from_index(
        cls,
        index_dir: str | Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        default_top_k: int = 3,
    ) -> "Retriever":
        """从已经保存的向量索引目录创建 Retriever。"""

        embedder = Embedder(
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
        )

        vector_store = VectorStore.load_index(index_dir)

        return cls(
            embedder=embedder,
            vector_store=vector_store,
            default_top_k=default_top_k,
        )

       
    @staticmethod
    def _validate_query(query: str) -> None:
        """检查 query 是否合法。"""

        if not isinstance(query,str):
            raise TypeError("query must be a string")
        
        if not query.strip():
            raise ValueError("query must not be empty")


    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        """检查 top_k 是否合法。"""

        if not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        