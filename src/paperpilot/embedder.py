from __future__ import annotations

from typing import Any

import numpy as np


Chunk = dict[str, Any]


try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    SentenceTransformer = None
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = exc
else:
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = None


class Embedder:
    """RAG 文本向量化模块。"""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ) -> None:
        """初始化 embedding 模型。"""

        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed."
                "Please install it with: pip install sentence-transformers"
            ) from _SENTENCE_TRANSFORMERS_IMPORT_ERROR
        
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string")
        
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size

        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_embedding_dimension()
    

    def embed_text(self, text: str) -> np.ndarray:
        """把单条文本转成一个向量。"""

        self._validate_text(text)

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )

        return np.asarray(embedding, dtype=np.float32)
        

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """把多条文本批量转成向量矩阵。"""

        self._validate_texts(texts)

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )

        return np.asarray(embeddings, dtype=np.float32)


    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        """把 chunk 列表转成向量矩阵。"""

        if not isinstance(chunks, list):
            raise TypeError("chunks must be a list")
        
        if not chunks:
            raise ValueError("chunks must not be empty")
        
        texts: list[str] = []

        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise TypeError("each chunk must be a dictionary")
            
            if "text" not in chunk:
                raise KeyError("each chunk must contain a text field")
            
            text = chunk["text"]
            self._validate_text(text)
            texts.append(text)
        
        return self.embed_texts(texts)


    def get_embedding_dim(self) -> int:
        """返回 embedding 向量维度。"""

        if self.embedding_dim is None:
            raise ValueError("embedding dimension is unknown")
        
        return int(self.embedding_dim)
        

    @staticmethod
    def _validate_text(text: str) -> None:
        """检查单条文本是否合法。"""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        
        if not text.strip():
            raise ValueError("text must not be empty")


    @classmethod
    def _validate_texts(cls, texts: list[str]) -> None:
        """检查文本列表是否合法。"""

        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")
        
        if not texts:
            raise ValueError("texts must not be empty")
        
        for text in texts:
            cls._validate_text(text)

    