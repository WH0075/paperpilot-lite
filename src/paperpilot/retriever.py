from __future__ import annotations

from pathlib import Path
from typing import Any

from .embedder import Embedder
from .keyword_retriever import KeywordRetriever
from .reranker import CrossEncoderReranker
from .vector_store import VectorStore


SearchResult = dict[str, Any]


class Retriever:
    """RAG 检索模块。"""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        default_top_k: int = 3,
        default_mode: str = "dense",
        hybrid_alpha: float = 0.5,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        hybrid_candidate_k: int = 20,
        reranker: CrossEncoderReranker | None = None,
        reranker_candidate_k: int = 20,
    ) -> None:
        """初始化 Retriever。"""

        if embedder is None:
            raise ValueError(
                "embedder must not be None"
            )

        if vector_store is None:
            raise ValueError(
                "vector_store must not be None"
            )

        if default_mode not in {
            "dense",
            "keyword",
            "hybrid",
        }:
            raise ValueError(
                "default_mode must be one of: "
                "dense, keyword, hybrid"
            )

        if fusion_method not in {
            "minmax",
            "rrf",
        }:
            raise ValueError(
                "fusion_method must be one of: "
                "minmax, rrf"
            )

        self._validate_top_k(
            default_top_k
        )

        self._validate_top_k(
            hybrid_candidate_k
        )

        self._validate_top_k(
            reranker_candidate_k
        )

        self._validate_alpha(
            hybrid_alpha
        )

        self._validate_rrf_k(
            rrf_k
        )

        self.embedder = embedder
        self.vector_store = vector_store

        self.default_top_k = (
            default_top_k
        )

        self.default_mode = (
            default_mode
        )

        self.hybrid_alpha = (
            hybrid_alpha
        )

        self.fusion_method = (
            fusion_method
        )

        self.rrf_k = (
            rrf_k
        )

        # Hybrid 内部 Dense / BM25
        # 各自参与融合的候选深度。
        #
        # 它与最终 top_k 独立。
        self.hybrid_candidate_k = (
            hybrid_candidate_k
        )

        # 第二阶段 reranker。
        # None 表示关闭 reranking。
        self.reranker = (
            reranker
        )

        # 开启 reranker 时，
        # 第一阶段交给 reranker 的候选数量。
        self.reranker_candidate_k = (
            reranker_candidate_k
        )

        self.keyword_retriever = (
            KeywordRetriever(
                vector_store.chunks
            )
        )

    def _retrieve_dense(
        self,
        query: str,
        top_k: int,
    ) -> list[SearchResult]:
        """使用向量相似度执行 dense retrieval。"""

        query_embedding = (
            self.embedder.embed_text(
                query
            )
        )

        return self.vector_store.search(
            query_embedding=(
                query_embedding
            ),
            top_k=top_k,
        )

    @staticmethod
    def _normalize_scores(
        results: list[SearchResult],
    ) -> dict[int, float]:
        """Min-max normalize retrieval scores into [0, 1]."""

        if not results:
            return {}

        scores = [
            float(
                result["score"]
            )
            for result in results
        ]

        min_score = min(
            scores
        )

        max_score = max(
            scores
        )

        if max_score == min_score:
            normalized_value = (
                0.0
                if max_score == 0
                else 1.0
            )

            return {
                int(
                    result["index"]
                ): normalized_value
                for result in results
            }

        return {
            int(
                result["index"]
            ): (
                float(
                    result["score"]
                )
                - min_score
            )
            / (
                max_score
                - min_score
            )
            for result in results
        }

    def _retrieve_keyword(
        self,
        query: str,
        top_k: int,
    ) -> list[SearchResult]:
        """使用 BM25 执行 keyword retrieval。"""

        return (
            self.keyword_retriever.retrieve(
                query=query,
                top_k=top_k,
            )
        )

    def _get_hybrid_candidate_k(
        self,
        top_k: int,
    ) -> int:
        """计算 Hybrid 内部实际使用的候选深度。

        hybrid_candidate_k 与最终 top_k 相互独立。

        例如：

        hybrid_candidate_k = 20

        top_k = 5
        -> Dense / BM25 各取 Top-20
        -> Hybrid 返回 Top-5

        top_k = 20
        -> Dense / BM25 各取 Top-20
        -> Hybrid 返回 Top-20

        如果请求 top_k > hybrid_candidate_k，
        则候选深度至少扩展到 top_k，
        保证能够返回足够结果。
        """

        return min(
            max(
                self.hybrid_candidate_k,
                top_k,
            ),
            len(
                self.vector_store.chunks
            ),
        )

    def _retrieve_hybrid_minmax(
        self,
        query: str,
        top_k: int,
        alpha: float,
    ) -> list[SearchResult]:
        """使用 Min-Max score fusion 执行 hybrid retrieval。"""

        candidate_k = (
            self._get_hybrid_candidate_k(
                top_k
            )
        )

        dense_results = (
            self._retrieve_dense(
                query=query,
                top_k=candidate_k,
            )
        )

        keyword_results = (
            self._retrieve_keyword(
                query=query,
                top_k=candidate_k,
            )
        )

        keyword_results = [
            result
            for result in keyword_results
            if float(
                result["score"]
            ) > 0
        ]

        dense_normalized = (
            self._normalize_scores(
                dense_results
            )
        )

        keyword_normalized = (
            self._normalize_scores(
                keyword_results
            )
        )

        dense_by_index = {
            int(
                result["index"]
            ): result
            for result in dense_results
        }

        keyword_by_index = {
            int(
                result["index"]
            ): result
            for result in keyword_results
        }

        candidate_indices = (
            set(
                dense_by_index
            )
            | set(
                keyword_by_index
            )
        )

        hybrid_results: list[
            SearchResult
        ] = []

        for index in candidate_indices:
            dense_result = (
                dense_by_index.get(
                    index
                )
            )

            keyword_result = (
                keyword_by_index.get(
                    index
                )
            )

            base_result = (
                dense_result
                if dense_result
                is not None
                else keyword_result
            )

            if base_result is None:
                continue

            dense_score_normalized = (
                dense_normalized.get(
                    index,
                    0.0,
                )
            )

            keyword_score_normalized = (
                keyword_normalized.get(
                    index,
                    0.0,
                )
            )

            hybrid_score = (
                alpha
                * dense_score_normalized
                + (
                    1.0
                    - alpha
                )
                * keyword_score_normalized
            )

            hybrid_results.append(
                {
                    "text": (
                        base_result[
                            "text"
                        ]
                    ),

                    "metadata": dict(
                        base_result.get(
                            "metadata",
                            {},
                        )
                    ),

                    "index": (
                        index
                    ),

                    "score": float(
                        hybrid_score
                    ),

                    "dense_score": (
                        float(
                            dense_result[
                                "score"
                            ]
                        )
                        if dense_result
                        is not None
                        else None
                    ),

                    "keyword_score": (
                        float(
                            keyword_result[
                                "score"
                            ]
                        )
                        if keyword_result
                        is not None
                        else None
                    ),

                    "dense_score_normalized": (
                        float(
                            dense_score_normalized
                        )
                    ),

                    "keyword_score_normalized": (
                        float(
                            keyword_score_normalized
                        )
                    ),
                }
            )

        hybrid_results.sort(
            key=lambda result: (
                result["score"]
            ),
            reverse=True,
        )

        return (
            hybrid_results[
                :top_k
            ]
        )

    def _retrieve_hybrid_rrf(
        self,
        query: str,
        top_k: int,
        alpha: float,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """使用 weighted RRF 执行 hybrid retrieval。"""

        candidate_k = (
            self._get_hybrid_candidate_k(
                top_k
            )
        )

        dense_results = (
            self._retrieve_dense(
                query=query,
                top_k=candidate_k,
            )
        )

        keyword_results = (
            self._retrieve_keyword(
                query=query,
                top_k=candidate_k,
            )
        )

        keyword_results = [
            result
            for result in keyword_results
            if float(
                result["score"]
            ) > 0
        ]

        dense_by_index = {
            int(
                result["index"]
            ): result
            for result in dense_results
        }

        keyword_by_index = {
            int(
                result["index"]
            ): result
            for result in keyword_results
        }

        dense_ranks = {
            int(
                result["index"]
            ): rank
            for rank, result in enumerate(
                dense_results,
                start=1,
            )
        }

        keyword_ranks = {
            int(
                result["index"]
            ): rank
            for rank, result in enumerate(
                keyword_results,
                start=1,
            )
        }

        candidate_indices = (
            set(
                dense_by_index
            )
            | set(
                keyword_by_index
            )
        )

        hybrid_results: list[
            SearchResult
        ] = []

        for index in candidate_indices:
            dense_result = (
                dense_by_index.get(
                    index
                )
            )

            keyword_result = (
                keyword_by_index.get(
                    index
                )
            )

            base_result = (
                dense_result
                if dense_result
                is not None
                else keyword_result
            )

            if base_result is None:
                continue

            dense_rank = (
                dense_ranks.get(
                    index
                )
            )

            keyword_rank = (
                keyword_ranks.get(
                    index
                )
            )

            dense_rrf_score = (
                alpha
                / (
                    rrf_k
                    + dense_rank
                )
                if dense_rank
                is not None
                else 0.0
            )

            keyword_rrf_score = (
                (
                    1.0
                    - alpha
                )
                / (
                    rrf_k
                    + keyword_rank
                )
                if keyword_rank
                is not None
                else 0.0
            )

            hybrid_score = (
                dense_rrf_score
                + keyword_rrf_score
            )

            hybrid_results.append(
                {
                    "text": (
                        base_result[
                            "text"
                        ]
                    ),

                    "metadata": dict(
                        base_result.get(
                            "metadata",
                            {},
                        )
                    ),

                    "index": (
                        index
                    ),

                    "score": float(
                        hybrid_score
                    ),

                    "dense_score": (
                        float(
                            dense_result[
                                "score"
                            ]
                        )
                        if dense_result
                        is not None
                        else None
                    ),

                    "keyword_score": (
                        float(
                            keyword_result[
                                "score"
                            ]
                        )
                        if keyword_result
                        is not None
                        else None
                    ),

                    "dense_rank": (
                        dense_rank
                    ),

                    "keyword_rank": (
                        keyword_rank
                    ),

                    "dense_rrf_score": (
                        float(
                            dense_rrf_score
                        )
                    ),

                    "keyword_rrf_score": (
                        float(
                            keyword_rrf_score
                        )
                    ),
                }
            )

        hybrid_results.sort(
            key=lambda result: (
                result["score"]
            ),
            reverse=True,
        )

        return (
            hybrid_results[
                :top_k
            ]
        )

    def _retrieve_hybrid(
        self,
        query: str,
        top_k: int,
        alpha: float,
        fusion_method: str,
        rrf_k: int,
    ) -> list[SearchResult]:
        """调用指定的 Hybrid fusion strategy。"""

        if fusion_method == "minmax":
            return (
                self._retrieve_hybrid_minmax(
                    query=query,
                    top_k=top_k,
                    alpha=alpha,
                )
            )

        if fusion_method == "rrf":
            return (
                self._retrieve_hybrid_rrf(
                    query=query,
                    top_k=top_k,
                    alpha=alpha,
                    rrf_k=rrf_k,
                )
            )

        raise ValueError(
            "fusion_method must be one of: "
            "minmax, rrf"
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        mode: str | None = None,
        alpha: float | None = None,
        fusion_method: str | None = None,
        rrf_k: int | None = None,
        candidate_k: int | None = None,
    ) -> list[SearchResult]:
        """根据 query 和 retrieval mode 检索相关 chunks。

        未配置 reranker 时：

            Retrieval
            -> Final Top-K

        配置 reranker 时：

            Retrieval
            -> Candidate Top-K
            -> Reranker
            -> Final Top-K

        candidate_k 只表示交给 reranker 的候选数量。

        hybrid_candidate_k 则表示 Hybrid 内部
        Dense / BM25 各自参与融合的候选深度。
        """

        self._validate_query(
            query
        )

        k = (
            self.default_top_k
            if top_k is None
            else top_k
        )

        self._validate_top_k(
            k
        )

        # -------------------------------------------------
        # 第一阶段最终要返回多少候选。
        #
        # 没有 reranker：
        # first_stage_k = 最终 top_k
        #
        # 有 reranker：
        # first_stage_k = candidate_k
        # -------------------------------------------------
        first_stage_k = (
            k
        )

        if self.reranker is not None:
            selected_candidate_k = (
                self.reranker_candidate_k
                if candidate_k is None
                else candidate_k
            )

            self._validate_top_k(
                selected_candidate_k
            )

            if (
                selected_candidate_k
                < k
            ):
                raise ValueError(
                    "candidate_k must be "
                    "greater than or equal "
                    "to top_k"
                )

            first_stage_k = (
                selected_candidate_k
            )

        selected_mode = (
            self.default_mode
            if mode is None
            else mode
        )

        # -------------------------------------------------
        # Dense
        # -------------------------------------------------
        if selected_mode == "dense":
            results = (
                self._retrieve_dense(
                    query=query,
                    top_k=first_stage_k,
                )
            )

        # -------------------------------------------------
        # BM25
        # -------------------------------------------------
        elif selected_mode == "keyword":
            results = (
                self._retrieve_keyword(
                    query=query,
                    top_k=first_stage_k,
                )
            )

        # -------------------------------------------------
        # Hybrid
        # -------------------------------------------------
        elif selected_mode == "hybrid":
            selected_alpha = (
                self.hybrid_alpha
                if alpha is None
                else alpha
            )

            selected_fusion_method = (
                self.fusion_method
                if fusion_method
                is None
                else fusion_method
            )

            selected_rrf_k = (
                self.rrf_k
                if rrf_k is None
                else rrf_k
            )

            self._validate_alpha(
                selected_alpha
            )

            if (
                selected_fusion_method
                not in {
                    "minmax",
                    "rrf",
                }
            ):
                raise ValueError(
                    "fusion_method must be "
                    "one of: minmax, rrf"
                )

            self._validate_rrf_k(
                selected_rrf_k
            )

            results = (
                self._retrieve_hybrid(
                    query=query,
                    top_k=(
                        first_stage_k
                    ),
                    alpha=(
                        selected_alpha
                    ),
                    fusion_method=(
                        selected_fusion_method
                    ),
                    rrf_k=(
                        selected_rrf_k
                    ),
                )
            )

        else:
            raise ValueError(
                "mode must be one of: "
                "dense, keyword, hybrid"
            )

        # -------------------------------------------------
        # 第二阶段：Reranking
        # -------------------------------------------------
        if self.reranker is not None:
            return (
                self.reranker.rerank(
                    query=query,
                    results=results,
                    top_k=k,
                )
            )

        # -------------------------------------------------
        # 未开启 reranker 时，
        # 保持原始 retrieval 行为。
        # -------------------------------------------------
        return results[:k]

    def retrieve_texts(
        self,
        query: str,
        top_k: int | None = None,
        mode: str | None = None,
        alpha: float | None = None,
    ) -> list[str]:
        """只返回检索结果中的文本内容。"""

        results = self.retrieve(
            query=query,
            top_k=top_k,
            mode=mode,
            alpha=alpha,
        )

        return [
            result["text"]
            for result in results
        ]

    def retrieve_with_sources(
        self,
        query: str,
        top_k: int | None = None,
        mode: str | None = None,
        alpha: float | None = None,
    ) -> list[SearchResult]:
        """返回带来源信息的检索结果。"""

        return self.retrieve(
            query=query,
            top_k=top_k,
            mode=mode,
            alpha=alpha,
        )

    @classmethod
    def from_index(
        cls,
        index_dir: str | Path,
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
        hybrid_candidate_k: int = 20,
        reranker: CrossEncoderReranker | None = None,
        reranker_candidate_k: int = 20,
    ) -> "Retriever":
        """从已经保存的向量索引目录创建 Retriever。"""

        embedder = Embedder(
            model_name=model_name,
            device=device,
            normalize_embeddings=(
                normalize_embeddings
            ),
            batch_size=batch_size,
        )

        vector_store = (
            VectorStore.load_index(
                index_dir
            )
        )

        return cls(
            embedder=embedder,
            vector_store=vector_store,
            default_top_k=(
                default_top_k
            ),
            default_mode=(
                default_mode
            ),
            hybrid_alpha=(
                hybrid_alpha
            ),
            fusion_method=(
                fusion_method
            ),
            rrf_k=(
                rrf_k
            ),
            hybrid_candidate_k=(
                hybrid_candidate_k
            ),
            reranker=(
                reranker
            ),
            reranker_candidate_k=(
                reranker_candidate_k
            ),
        )

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
    def _validate_top_k(
        top_k: int,
    ) -> None:
        """检查 top_k 是否合法。"""

        if not isinstance(
            top_k,
            int,
        ):
            raise TypeError(
                "top_k must be an integer"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive"
            )

    @staticmethod
    def _validate_alpha(
        alpha: float,
    ) -> None:
        """检查 Hybrid fusion 权重是否合法。"""

        if not isinstance(
            alpha,
            (int, float),
        ):
            raise TypeError(
                "alpha must be a number"
            )

        if not (
            0.0
            <= float(alpha)
            <= 1.0
        ):
            raise ValueError(
                "alpha must be "
                "between 0 and 1"
            )

    @staticmethod
    def _validate_rrf_k(
        rrf_k: int,
    ) -> None:
        """检查 RRF rank constant 是否合法。"""

        if not isinstance(
            rrf_k,
            int,
        ):
            raise TypeError(
                "rrf_k must be an integer"
            )

        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be positive"
            )