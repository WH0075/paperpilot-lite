from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


Chunk = dict[str, Any]
SearchResult = dict[str, Any]


class VectorStore:
    """RAG 向量索引模块。"""

    def __init__(self, similarity: str = "cosine") -> None:
        """初始化一个空的向量库。"""

        if similarity not in {"cosine", "inner_product", "l2"}:
            raise ValueError("similarity must be one of: 'cosine', 'inner_product', 'l2' ")
        
        self.similarity = similarity
        self.embeddings: np.ndarray | None = None
        self.chunks: list[Chunk] = []
        self.embedding_dim: int | None = None


    def build_index(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """根据向量矩阵和 chunk 列表建立内存索引。"""

        embeddings = self._validate_and_convert_embeddings(embeddings)
        self._validate_chunks(chunks)

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                "number of embeddings must match number of chunks: "
                f"{embeddings.shape[0]} != {len(chunks)}"
            )
        
        if self.similarity == "cosine":
            embeddings = self._normalize_matrix(embeddings)
        
        self.embeddings = embeddings
        self.chunks = list(chunks)
        self.embedding_dim = embeddings.shape[1]


    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        """根据 query 向量搜索最相似的 top-k chunk。"""

        if self.embeddings is None:
            raise ValueError("index has not been built or loaded")
        
        if not self.chunks:
            raise ValueError("chunks are empty")
        
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        query = self._validate_and_convert_query(query_embedding)

        if query.shape[0] != self.embedding_dim:
            raise ValueError(
                "query embedding dimension does not match index dimension: "
                f"{query.shape[0]} != {self.embedding_dim}"
            )
        
        if self.similarity == "cosine":
            query = self._normalize_vector(query)
            scores = self.embeddings @ query
            ranked_indices = np.argsort(scores)[::-1]
        
        elif self.similarity == "inner_product":
            scores = self.embeddings @ query
            ranked_indices = np.argsort(scores)[::-1]
        
        elif self.similarity == "l2":
            distances = np.sum((self.embeddings - query) ** 2, axis=1)
            scores = -distances
            ranked_indices = np.argsort(distances)
        
        k = min(top_k, len(self.chunks))
        selected_indices = ranked_indices[:k]

        results: list[SearchResult] = []

        for index in selected_indices:
            chunk = self.chunks[int(index)]

            results.append(
                {
                    "text": chunk["text"],
                    "metadata": dict(chunk.get("metadata", {})),
                    "score": float(scores[int(index)]),
                    "index": int(index),

                }
            )

        return results


    def save_index(self, index_dir: str | Path) -> None:
        """把当前向量库保存到磁盘。"""

        if self.embeddings is None:
            raise ValueError("index has not been built")
        
        if not self.chunks:
            raise ValueError("chunks are empty")
        
        root = Path(index_dir)
        root.mkdir(parents=True, exist_ok=True)

        embeddings_path = root / "embeddings.npy"
        chunks_path = root / "chunks.json"
        meta_path = root / "index_meta.json"

        np.save(embeddings_path, self.embeddings)

        with chunks_path.open("w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        
        meta = {
            "similarity": self.similarity,
            "embedding_dim": self.embedding_dim,
            "num_chunks": len(self.chunks),
            "embeddings_file": embeddings_path.name,
            "chunks_file": chunks_path.name,
        }

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        
    @classmethod
    def load_index(cls, index_dir: str | Path) -> "VectorStore":
        """从磁盘加载向量库。"""

        root = Path(index_dir)

        embeddings_path = root / "embeddings.npy"
        chunks_path = root / "chunks.json"
        meta_path = root / "index_meta.json"

        if not embeddings_path.exists():
            raise FileNotFoundError(f"embeddings file not found: {embeddings_path}")

        if not chunks_path.exists():
            raise FileNotFoundError(f"chunks file not found: {chunks_path}")

        if not meta_path.exists():
            raise FileNotFoundError(f"index metadata file not found: {meta_path}")

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        similarity = meta.get("similarity", "cosine")
        store = cls(similarity=similarity)

        embeddings = np.load(embeddings_path).astype(np.float32)

        with chunks_path.open("r", encoding="utf-8") as f:
            chunks = json.load(f)

        store.build_index(embeddings=embeddings, chunks=chunks)

        return store
   

    def get_num_chunks(self) -> int:
        """返回当前向量库里的 chunk 数量。"""

        return len(self.chunks)


    def get_embedding_dim(self) -> int:
        """返回当前向量库的向量维度。"""

        if self.embedding_dim is None:
            raise ValueError("embedding dimension is unknown")

        return int(self.embedding_dim)

    @staticmethod
    def _validate_and_convert_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """检查并转换向量矩阵。"""

        if not isinstance(embeddings, np.ndarray):
            raise TypeError("embeddings must be a NumPy ndarray")

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")

        if embeddings.shape[0] == 0:
            raise ValueError("embeddings must contain at least one vector")

        if embeddings.shape[1] == 0:
            raise ValueError("embedding dimension must be greater than zero")

        return np.asarray(embeddings, dtype=np.float32)

        
    @staticmethod
    def _validate_and_convert_query(query_embedding: np.ndarray) -> np.ndarray:
        """检查并转换 query 向量。"""

        if not isinstance(query_embedding, np.ndarray):
            raise TypeError("query_embedding must be a NumPy ndarray")

        query = np.asarray(query_embedding, dtype=np.float32)

        if query.ndim == 2:
            if query.shape[0] != 1:
                raise ValueError("2D query_embedding must have shape [1, embedding_dim]")
            query = query[0]

        if query.ndim != 1:
            raise ValueError("query_embedding must be a 1D array or shape [1, D]")

        if query.shape[0] == 0:
            raise ValueError("query_embedding must not be empty")
        
        return query
    

    @staticmethod
    def _validate_chunks(chunks: list[Chunk]) -> None:
        """检查 chunk 列表是否合法。"""

        if not isinstance(chunks, list):
            raise TypeError("chunks must be a list")

        if not chunks:
            raise ValueError("chunks must not be empty")

        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise TypeError("each chunk must be a dictionary")

            if "text" not in chunk:
                raise KeyError("each chunk must contain a text field")

            if not isinstance(chunk["text"], str):
                raise TypeError("chunk text must be a string")

            if not chunk["text"].strip():
                raise ValueError("chunk text must not be empty")

            if "metadata" in chunk and not isinstance(chunk["metadata"], dict):
                raise TypeError("chunk metadata must be a dictionary")

    @staticmethod
    def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        """对矩阵中的每一行向量做归一化。"""

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        """对单个向量做归一化。"""

        norm = np.linalg.norm(vector)

        if norm == 0:
            return vector
        
        return vector / norm
