"""Cross-Encoder reranking for retrieved chunks."""

from typing import Any

from sentence_transformers import CrossEncoder


SearchResult = dict[str, Any]


class CrossEncoderReranker:
    """使用 Cross-Encoder 对第一阶段检索结果进行重排序。"""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        batch_size: int = 16,
        model: Any | None = None,
    ) -> None:
        """初始化 Cross-Encoder reranker。

        Args:
            model_name:
                Cross-Encoder 模型名称。
            device:
                推理设备，例如 "cpu"、"cuda"。
            batch_size:
                一次处理多少个 query-chunk pair。
            model:
                可选的外部模型对象，主要用于测试时注入 fake model，
                避免单元测试真实下载和加载 Hugging Face 模型。
        """

        if not model_name.strip():
            raise ValueError("model_name must not be empty")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        if model is not None:
            self.model = model
        else:
            self.model = CrossEncoder(
                model_name,
                device=device,
            )

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """根据 query 对候选检索结果重新排序。"""

        self._validate_query(query)

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")

        if not results:
            return []

        pairs: list[tuple[str, str]] = []

        for result in results:
            text = result.get("text")

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    "each search result must contain non-empty text"
                )

            pairs.append(
                (
                    query,
                    text,
                )
            )

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        reranked_results: list[SearchResult] = []

        for result, rerank_score in zip(
            results,
            scores,
            strict=True,
        ):
            reranked_result = dict(result)

            original_score = result.get(
                "retrieval_score",
                result.get("score"),
            )

            if original_score is not None:
                reranked_result["retrieval_score"] = float(
                    original_score
                )

            reranked_result["rerank_score"] = float(
                rerank_score
            )

            # score 始终表示当前最终排序依据。
            reranked_result["score"] = float(
                rerank_score
            )

            reranked_results.append(
                reranked_result
            )

        reranked_results.sort(
            key=lambda result: result["rerank_score"],
            reverse=True,
        )

        if top_k is None:
            return reranked_results

        return reranked_results[:top_k]

    @staticmethod
    def _validate_query(query: str) -> None:
        """检查 query 是否有效。"""

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be empty")