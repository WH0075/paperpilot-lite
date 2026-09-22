from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .cleaner import clean_documents
from .chunker import chunk_documents
from .config import AppConfig, load_config
from .document_loader import load_documents
from .embedder import Embedder
from .evaluator import (
    evaluate_retrieval,
    load_qa_set,
    print_evaluation_report,
)
from .llm_client import (
    BaseLLMClient,
    MockLLMClient,
    OpenAICompatibleLLMClient,
)
from .logger import get_logger, setup_logging
from .rag_pipeline import RAGPipeline
from .reranker import CrossEncoderReranker
from .retriever import Retriever
from .vector_store import VectorStore


def handle_ingest(
    args: argparse.Namespace,
) -> None:
    """处理 ingest 命令：读取文档、清洗、切分、向量化、建索引并保存。"""

    data_dir = Path(
        args.data_dir
    )

    index_dir = Path(
        args.index_dir
    )

    print("=" * 80)
    print("PaperPilot-Lite Ingest")
    print("=" * 80)

    print(
        f"Data directory: "
        f"{data_dir}"
    )

    print(
        f"Index directory: "
        f"{index_dir}"
    )

    print(
        f"Chunk size: "
        f"{args.chunk_size}"
    )

    print(
        f"Overlap: "
        f"{args.overlap}"
    )

    print(
        f"Embedding model: "
        f"{args.model_name}"
    )

    print(
        f"Device: "
        f"{args.device}"
    )

    print(
        f"Similarity: "
        f"{args.similarity}"
    )

    print()

    documents = load_documents(
        data_dir
    )

    print(
        f"Loaded documents: "
        f"{len(documents)}"
    )

    cleaned_documents = (
        clean_documents(
            documents
        )
    )

    print(
        f"Cleaned documents: "
        f"{len(cleaned_documents)}"
    )

    chunks = chunk_documents(
        documents=cleaned_documents,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    print(
        f"Created chunks: "
        f"{len(chunks)}"
    )

    if not chunks:
        raise ValueError(
            "No chunks were created. "
            "Please check your input documents."
        )

    embedder = Embedder(
        model_name=args.model_name,
        device=args.device,
        normalize_embeddings=(
            args.normalize_embeddings
        ),
        batch_size=args.batch_size,
    )

    embeddings = (
        embedder.embed_chunks(
            chunks
        )
    )

    print(
        f"Embeddings shape: "
        f"{embeddings.shape}"
    )

    store = VectorStore(
        similarity=args.similarity,
    )

    store.build_index(
        embeddings=embeddings,
        chunks=chunks,
    )

    store.save_index(
        index_dir
    )

    print()
    print(
        "Ingest completed successfully."
    )

    print(
        f"Saved index to: "
        f"{index_dir}"
    )

    print(
        f"VectorStore chunks: "
        f"{store.get_num_chunks()}"
    )

    print(
        f"Embedding dim: "
        f"{store.get_embedding_dim()}"
    )


def add_retrieval_arguments(
    parser: argparse.ArgumentParser,
    config: AppConfig,
) -> None:
    """向 CLI 子命令添加通用 retrieval 和 reranker 参数。"""

    parser.add_argument(
        "--retrieval-mode",
        type=str,
        choices=[
            "dense",
            "keyword",
            "hybrid",
        ],
        default=(
            config.retrieval.mode
        ),
        help=(
            "Retrieval strategy: "
            "dense, keyword, or hybrid. "
            "Default comes from config."
        ),
    )

    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=(
            config.retrieval.hybrid_alpha
        ),
        help=(
            "Dense weight used by hybrid retrieval. "
            "Must be between 0 and 1."
        ),
    )

    parser.add_argument(
        "--fusion-method",
        type=str,
        choices=[
            "minmax",
            "rrf",
        ],
        default=(
            config.retrieval.fusion_method
        ),
        help=(
            "Hybrid fusion strategy: "
            "minmax or rrf."
        ),
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=(
            config.retrieval.rrf_k
        ),
        help=(
            "RRF smoothing constant. "
            "Default comes from config."
        ),
    )

    parser.add_argument(
        "--hybrid-candidate-k",
        type=int,
        default=(
            config.retrieval.hybrid_candidate_k
        ),
        help=(
            "Number of Dense/BM25 candidates "
            "used by Hybrid fusion. "
            "Default comes from config."
        ),
    )

    # -------------------------------------------------
    # Reranker enable / disable
    #
    # 使用 mutually exclusive group（互斥参数组），
    # 避免同时输入：
    #
    # --reranker --no-reranker
    # -------------------------------------------------
    reranker_group = (
        parser.add_mutually_exclusive_group()
    )

    reranker_group.add_argument(
        "--reranker",
        action="store_true",
        dest="reranker_enabled",
        help=(
            "Enable second-stage "
            "Cross-Encoder reranking."
        ),
    )

    reranker_group.add_argument(
        "--no-reranker",
        action="store_false",
        dest="reranker_enabled",
        help=(
            "Disable second-stage "
            "Cross-Encoder reranking."
        ),
    )

    parser.set_defaults(
        reranker_enabled=(
            config.retrieval.reranker.enabled
        ),
    )

    parser.add_argument(
        "--reranker-model",
        type=str,
        default=(
            config.retrieval.reranker.model_name
        ),
        help=(
            "Cross-Encoder reranker model name. "
            "Default comes from config."
        ),
    )

    parser.add_argument(
        "--reranker-candidate-k",
        type=int,
        default=(
            config.retrieval.reranker.candidate_k
        ),
        help=(
            "Number of first-stage candidates "
            "passed to the reranker. "
            "Default comes from config."
        ),
    )

    parser.add_argument(
        "--reranker-device",
        type=str,
        default=(
            config.retrieval.reranker.device
        ),
        help=(
            "Device used by reranker, "
            "such as cpu or cuda."
        ),
    )

    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=(
            config.retrieval.reranker.batch_size
        ),
        help=(
            "Cross-Encoder reranker batch size."
        ),
    )


def create_reranker(
    args: argparse.Namespace,
) -> CrossEncoderReranker | None:
    """根据 CLI/config 创建可选的 Cross-Encoder Reranker。

    reranker 未开启时返回 None，
    因此不会加载 Cross-Encoder 模型。
    """

    if not args.reranker_enabled:
        return None

    return CrossEncoderReranker(
        model_name=args.reranker_model,
        device=args.reranker_device,
        batch_size=(
            args.reranker_batch_size
        ),
    )


def create_retriever(
    args: argparse.Namespace,
    default_top_k: int | None = None,
) -> Retriever:
    """根据 CLI 参数构造统一 Retriever。

    search 和 eval 共用该函数，
    保证 Hybrid / Reranker 配置一致。
    """

    reranker = create_reranker(
        args
    )

    selected_top_k = (
        args.top_k
        if default_top_k is None
        else default_top_k
    )

    return Retriever.from_index(
        index_dir=args.index_dir,
        model_name=args.model_name,
        device=args.device,
        normalize_embeddings=(
            args.normalize_embeddings
        ),
        batch_size=args.batch_size,
        default_top_k=selected_top_k,
        default_mode=(
            args.retrieval_mode
        ),
        hybrid_alpha=(
            args.hybrid_alpha
        ),
        fusion_method=(
            args.fusion_method
        ),
        rrf_k=args.rrf_k,
        hybrid_candidate_k=(
            args.hybrid_candidate_k
        ),
        reranker=reranker,
        reranker_candidate_k=(
            args.reranker_candidate_k
        ),
    )


def print_retrieval_configuration(
    args: argparse.Namespace,
) -> None:
    """打印当前 Retrieval / Reranker 配置。"""

    print(
        f"Retrieval mode: "
        f"{args.retrieval_mode}"
    )

    if (
        args.retrieval_mode
        == "hybrid"
    ):
        print(
            f"Fusion method: "
            f"{args.fusion_method}"
        )

        print(
            f"Hybrid alpha: "
            f"{args.hybrid_alpha}"
        )

        print(
            f"Hybrid candidate k: "
            f"{args.hybrid_candidate_k}"
        )

        if (
            args.fusion_method
            == "rrf"
        ):
            print(
                f"RRF k: "
                f"{args.rrf_k}"
            )

    print(
        f"Reranker enabled: "
        f"{args.reranker_enabled}"
    )

    if args.reranker_enabled:
        print(
            f"Reranker model: "
            f"{args.reranker_model}"
        )

        print(
            f"Reranker candidate k: "
            f"{args.reranker_candidate_k}"
        )

        print(
            f"Reranker device: "
            f"{args.reranker_device}"
        )

        print(
            f"Reranker batch size: "
            f"{args.reranker_batch_size}"
        )


def handle_search(
    args: argparse.Namespace,
) -> None:
    """处理 search 命令：加载索引并返回 top-k 检索结果。"""

    retriever = create_retriever(
        args
    )

    results = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
    )

    print("=" * 80)
    print("PaperPilot-Lite Search")
    print("=" * 80)

    print(
        f"Query: "
        f"{args.query}"
    )

    print(
        f"Top-k: "
        f"{args.top_k}"
    )

    print_retrieval_configuration(
        args
    )

    print(
        f"Results: "
        f"{len(results)}"
    )

    print("=" * 80)

    print_search_results(
        results
    )


def create_llm_client(
    args: argparse.Namespace,
) -> BaseLLMClient:
    """Create the LLM client selected by CLI/configuration."""

    if args.llm == "mock":
        return MockLLMClient(
            fixed_answer=args.fixed_answer,
        )

    if (
        args.llm
        == "openai-compatible"
    ):
        if not args.llm_model:
            raise ValueError(
                "Real LLM mode requires a model name. "
                "Set LLM_MODEL_NAME in .env "
                "or pass --llm-model."
            )

        return (
            OpenAICompatibleLLMClient(
                model_name=(
                    args.llm_model
                ),
                api_key=os.getenv(
                    "LLM_API_KEY"
                ),
                base_url=(
                    args.llm_base_url
                ),
                temperature=(
                    args.llm_temperature
                ),
                max_tokens=(
                    args.llm_max_tokens
                ),
                thinking_enabled=(
                    args.llm_thinking
                ),
                timeout=(
                    args.llm_timeout
                ),
            )
        )

    raise ValueError(
        f"Unsupported LLM provider: "
        f"{args.llm}"
    )


def handle_ask(
    args: argparse.Namespace,
) -> None:
    """处理 ask 命令：执行完整 RAG 问答流程。"""

    llm_client = (
        create_llm_client(
            args
        )
    )

    # -------------------------------------------------
    # 先按照当前 RAGPipeline API 创建 Pipeline。
    #
    # RAGPipeline.from_index 已经负责构造：
    #
    # Embedder
    # VectorStore
    # Retriever
    # PromptBuilder
    #
    # 然后下面直接给其中的 Retriever
    # 注入 R5 新增的 candidate / reranker 配置。
    #
    # 这样当前阶段不需要同时修改 rag_pipeline.py。
    # -------------------------------------------------
    pipeline = RAGPipeline.from_index(
        index_dir=args.index_dir,
        llm_client=llm_client,
        model_name=args.model_name,
        device=args.device,
        normalize_embeddings=(
            args.normalize_embeddings
        ),
        batch_size=args.batch_size,
        default_top_k=args.top_k,

        # Retrieval configuration
        default_mode=(
            args.retrieval_mode
        ),
        hybrid_alpha=(
            args.hybrid_alpha
        ),
        fusion_method=(
            args.fusion_method
        ),
        rrf_k=args.rrf_k,

        # Prompt configuration
        max_context_chars=(
            args.max_context_chars
        ),
        max_chunk_chars=(
            args.max_chunk_chars
        ),
    )

    # -------------------------------------------------
    # R5 Retrieval / Reranker configuration
    # -------------------------------------------------
    pipeline.retriever.hybrid_candidate_k = (
        args.hybrid_candidate_k
    )

    pipeline.retriever.reranker_candidate_k = (
        args.reranker_candidate_k
    )

    pipeline.retriever.reranker = (
        create_reranker(
            args
        )
    )

    response = pipeline.ask(
        query=args.query,
        top_k=args.top_k,
        template_name=(
            args.template_name
        ),
    )

    print("=" * 80)
    print("PaperPilot-Lite Ask")
    print("=" * 80)

    print_retrieval_configuration(
        args
    )

    print(
        f"Template: "
        f"{response['template_name']}"
    )

    print()

    print("Question:")
    print(
        response["query"]
    )

    print()

    print("Answer:")
    print(
        response["answer"]
    )

    print()

    print("Sources:")

    print_sources(
        response["sources"]
    )

    if args.show_prompt:
        print()

        print("=" * 80)
        print("Prompt")
        print("=" * 80)

        print(
            response["prompt"]
        )

    if args.show_search_results:
        print()

        print("=" * 80)
        print("Raw Search Results")
        print("=" * 80)

        print_search_results(
            response[
                "search_results"
            ]
        )


def handle_eval(
    args: argparse.Namespace,
) -> None:
    """处理 eval 命令：评估 retrieval 的 Recall@K、MRR@K 和 latency。"""

    ks = args.ks

    retrieval_k = max(
        max(ks),
        args.mrr_k,
    )

    retriever = (
        create_retriever(
            args,
            default_top_k=retrieval_k,
        )
    )

    qa_items = load_qa_set(
        args.qa_path,
    )

    report = evaluate_retrieval(
        retriever=retriever,
        qa_items=qa_items,
        ks=ks,
        mrr_k=args.mrr_k,
    )

    print()
    print("=" * 80)
    print(
        "Retrieval Configuration"
    )
    print("=" * 80)

    print_retrieval_configuration(
        args
    )

    print("=" * 80)
    print()

    print_evaluation_report(
        report=report,
        show_failed_cases=(
            args.show_failed_cases
        ),
        max_failed_cases=(
            args.max_failed_cases
        ),
    )


def print_search_results(
    results: list[
        dict[str, Any]
    ],
) -> None:
    """格式化打印检索结果。"""

    if not results:
        print(
            "No results found."
        )
        return

    for i, result in enumerate(
        results,
        start=1,
    ):
        metadata = result.get(
            "metadata",
            {},
        )

        score = result.get(
            "score"
        )

        retrieval_score = (
            result.get(
                "retrieval_score"
            )
        )

        rerank_score = (
            result.get(
                "rerank_score"
            )
        )

        text = result.get(
            "text",
            "",
        )

        file_name = (
            metadata.get(
                "file_name"
            )
            or metadata.get(
                "source"
            )
            or "unknown file"
        )

        page = metadata.get(
            "page"
        )

        chunk_id = metadata.get(
            "chunk_id"
        )

        print()
        print("-" * 80)

        print(
            f"Result {i}"
        )

        # 当前最终排序使用的 score。
        if score is not None:
            try:
                print(
                    f"Score: "
                    f"{float(score):.4f}"
                )
            except (
                TypeError,
                ValueError,
            ):
                print(
                    f"Score: "
                    f"{score}"
                )

        # 开启 reranker 后保留第一阶段分数。
        if retrieval_score is not None:
            try:
                print(
                    f"Retrieval score: "
                    f"{float(retrieval_score):.6f}"
                )
            except (
                TypeError,
                ValueError,
            ):
                print(
                    f"Retrieval score: "
                    f"{retrieval_score}"
                )

        # Cross-Encoder 最终相关性分数。
        if rerank_score is not None:
            try:
                print(
                    f"Rerank score: "
                    f"{float(rerank_score):.4f}"
                )
            except (
                TypeError,
                ValueError,
            ):
                print(
                    f"Rerank score: "
                    f"{rerank_score}"
                )

        print(
            f"Source: "
            f"{file_name}"
        )

        if page is not None:
            print(
                f"Page: "
                f"{page}"
            )

        if chunk_id is not None:
            print(
                f"Chunk ID: "
                f"{chunk_id}"
            )

        print()
        print("Text:")

        print(
            text[:800]
        )

        if len(text) > 800:
            print("...")


def print_sources(
    sources: list[
        dict[str, Any]
    ],
) -> None:
    """格式化打印来源信息。"""

    if not sources:
        print(
            "No sources."
        )
        return

    for source in sources:
        source_id = source.get(
            "source_id"
        )

        file_name = (
            source.get(
                "file_name"
            )
            or "unknown file"
        )

        page = source.get(
            "page"
        )

        chunk_id = source.get(
            "chunk_id"
        )

        score = source.get(
            "score"
        )

        source_line = (
            f"[{source_id}] "
            f"{file_name}"
        )

        if page is not None:
            source_line += (
                f", page {page}"
            )

        if chunk_id is not None:
            source_line += (
                f", chunk_id "
                f"{chunk_id}"
            )

        if score is not None:
            try:
                source_line += (
                    f", score "
                    f"{float(score):.4f}"
                )
            except (
                TypeError,
                ValueError,
            ):
                source_line += (
                    f", score "
                    f"{score}"
                )

        print(
            source_line
        )


def build_parser(
    config: AppConfig | None = None,
) -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    config = (
        config
        or load_config()
    )

    parser = argparse.ArgumentParser(
        prog="paperpilot",
        description=(
            "PaperPilot-Lite: a local RAG "
            "document question-answering system."
        ),
    )

    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )

    # ============================================================
    # ingest
    # ============================================================

    ingest_parser = (
        subparsers.add_parser(
            "ingest",
            help=(
                "Load documents, create chunks, "
                "build embeddings, and save vector index."
            ),
        )
    )

    ingest_parser.add_argument(
        "data_dir",
        type=str,
        help=(
            "Directory containing raw documents, "
            "such as data/raw."
        ),
    )

    ingest_parser.add_argument(
        "--index-dir",
        type=str,
        default=(
            config.data.index_dir
        ),
        help=(
            "Directory to save vector index. "
            "Default: data/index."
        ),
    )

    ingest_parser.add_argument(
        "--chunk-size",
        type=int,
        default=(
            config.chunking.chunk_size
        ),
        help=(
            "Chunk size in characters. "
            "Default: 500."
        ),
    )

    ingest_parser.add_argument(
        "--overlap",
        type=int,
        default=(
            config.chunking.overlap
        ),
        help=(
            "Chunk overlap in characters. "
            "Default: 100."
        ),
    )

    ingest_parser.add_argument(
        "--model-name",
        type=str,
        default=(
            config.retrieval.embedding_model
        ),
        help=(
            "Sentence-transformers "
            "embedding model name."
        ),
    )

    ingest_parser.add_argument(
        "--device",
        type=str,
        default=(
            config.retrieval.device
        ),
        help=(
            "Device for embedding model, "
            "such as cpu or cuda."
        ),
    )

    ingest_parser.add_argument(
        "--batch-size",
        type=int,
        default=(
            config.retrieval.batch_size
        ),
        help=(
            "Embedding batch size."
        ),
    )

    ingest_parser.add_argument(
        "--similarity",
        type=str,
        default=(
            config.retrieval.similarity
        ),
        choices=[
            "cosine",
            "inner_product",
            "l2",
        ],
        help=(
            "Similarity metric for vector store."
        ),
    )

    ingest_parser.add_argument(
        "--no-normalize-embeddings",
        action="store_false",
        dest="normalize_embeddings",
        help=(
            "Disable embedding normalization."
        ),
    )

    ingest_parser.set_defaults(
        func=handle_ingest,
        normalize_embeddings=(
            config.retrieval.normalize_embeddings
        ),
    )

    # ============================================================
    # search
    # ============================================================

    search_parser = (
        subparsers.add_parser(
            "search",
            help=(
                "Search top-k relevant chunks "
                "for a query."
            ),
        )
    )

    search_parser.add_argument(
        "query",
        type=str,
        help="User query.",
    )

    search_parser.add_argument(
        "--index-dir",
        type=str,
        default=(
            config.data.index_dir
        ),
        help=(
            "Directory containing vector index."
        ),
    )

    search_parser.add_argument(
        "--top-k",
        type=int,
        default=(
            config.retrieval.top_k
        ),
        help=(
            "Number of final search results."
        ),
    )

    search_parser.add_argument(
        "--model-name",
        type=str,
        default=(
            config.retrieval.embedding_model
        ),
        help=(
            "Sentence-transformers "
            "embedding model name."
        ),
    )

    search_parser.add_argument(
        "--device",
        type=str,
        default=(
            config.retrieval.device
        ),
        help=(
            "Device for embedding model."
        ),
    )

    search_parser.add_argument(
        "--batch-size",
        type=int,
        default=(
            config.retrieval.batch_size
        ),
        help=(
            "Embedding batch size."
        ),
    )

    search_parser.add_argument(
        "--no-normalize-embeddings",
        action="store_false",
        dest="normalize_embeddings",
        help=(
            "Disable embedding normalization."
        ),
    )

    add_retrieval_arguments(
        search_parser,
        config,
    )

    search_parser.set_defaults(
        func=handle_search,
        normalize_embeddings=(
            config.retrieval.normalize_embeddings
        ),
    )

    # ============================================================
    # ask
    # ============================================================

    ask_parser = (
        subparsers.add_parser(
            "ask",
            help=(
                "Run full RAG question "
                "answering pipeline."
            ),
        )
    )

    ask_parser.add_argument(
        "query",
        type=str,
        help="User question.",
    )

    ask_parser.add_argument(
        "--index-dir",
        type=str,
        default=(
            config.data.index_dir
        ),
        help=(
            "Directory containing vector index."
        ),
    )

    ask_parser.add_argument(
        "--top-k",
        type=int,
        default=(
            config.retrieval.top_k
        ),
        help=(
            "Number of final retrieved chunks."
        ),
    )

    ask_parser.add_argument(
        "--template-name",
        type=str,
        default=(
            config.prompt.template_name
        ),
        choices=[
            "extractive",
            "grounded",
            "explainer",
        ],
        help=(
            "Prompt template name: "
            "extractive, grounded, or explainer."
        ),
    )

    ask_parser.add_argument(
        "--model-name",
        type=str,
        default=(
            config.retrieval.embedding_model
        ),
        help=(
            "Sentence-transformers "
            "embedding model name."
        ),
    )

    ask_parser.add_argument(
        "--device",
        type=str,
        default=(
            config.retrieval.device
        ),
        help=(
            "Device for embedding model."
        ),
    )

    ask_parser.add_argument(
        "--batch-size",
        type=int,
        default=(
            config.retrieval.batch_size
        ),
        help=(
            "Embedding batch size."
        ),
    )

    ask_parser.add_argument(
        "--max-context-chars",
        type=int,
        default=(
            config.prompt.max_context_chars
        ),
        help=(
            "Maximum context characters "
            "in prompt."
        ),
    )

    ask_parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=(
            config.prompt.max_chunk_chars
        ),
        help=(
            "Maximum characters per chunk "
            "in prompt."
        ),
    )

    ask_parser.add_argument(
        "--llm",
        type=str,
        default=(
            config.llm.provider
        ),
        choices=[
            "mock",
            "openai-compatible",
        ],
        help=(
            "LLM provider."
        ),
    )

    ask_parser.add_argument(
        "--llm-model",
        type=str,
        default=(
            config.llm.model_name
        ),
        help=(
            "Model name for a real "
            "OpenAI-compatible LLM."
        ),
    )

    ask_parser.add_argument(
        "--llm-base-url",
        type=str,
        default=(
            config.llm.base_url
        ),
        help=(
            "Base URL for an "
            "OpenAI-compatible API."
        ),
    )

    ask_parser.add_argument(
        "--llm-temperature",
        type=float,
        default=(
            config.llm.temperature
        ),
        help=(
            "Generation temperature."
        ),
    )

    ask_parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=(
            config.llm.max_tokens
        ),
        help=(
            "Maximum generated tokens."
        ),
    )

    ask_parser.add_argument(
        "--llm-thinking",
        action="store_true",
        default=(
            config.llm.thinking_enabled
        ),
        help=(
            "Enable LLM thinking/"
            "reasoning mode."
        ),
    )

    ask_parser.add_argument(
        "--llm-timeout",
        type=int,
        default=(
            config.llm.timeout
        ),
        help=(
            "LLM request timeout in seconds."
        ),
    )

    ask_parser.add_argument(
        "--fixed-answer",
        type=str,
        default=None,
        help=(
            "Fixed mock answer for testing."
        ),
    )

    ask_parser.add_argument(
        "--show-prompt",
        action="store_true",
        help=(
            "Print the full prompt "
            "sent to LLM client."
        ),
    )

    ask_parser.add_argument(
        "--show-search-results",
        action="store_true",
        help=(
            "Print raw search results."
        ),
    )

    ask_parser.add_argument(
        "--no-normalize-embeddings",
        action="store_false",
        dest="normalize_embeddings",
        help=(
            "Disable embedding normalization."
        ),
    )

    add_retrieval_arguments(
        ask_parser,
        config,
    )

    ask_parser.set_defaults(
        func=handle_ask,
        normalize_embeddings=(
            config.retrieval.normalize_embeddings
        ),
    )

    # ============================================================
    # eval
    # ============================================================

    eval_parser = (
        subparsers.add_parser(
            "eval",
            help=(
                "Evaluate retrieval quality "
                "with Recall@K and MRR@K."
            ),
        )
    )

    eval_parser.add_argument(
        "qa_path",
        type=str,
        help=(
            "Path to QA set jsonl file."
        ),
    )

    eval_parser.add_argument(
        "--index-dir",
        type=str,
        default=(
            config.data.index_dir
        ),
        help=(
            "Directory containing vector index."
        ),
    )

    eval_parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[
            1,
            3,
            5,
        ],
        help=(
            "K values for Recall@K. "
            "Default: 1 3 5."
        ),
    )

    eval_parser.add_argument(
        "--mrr-k",
        type=int,
        default=5,
        help=(
            "Evaluation depth for MRR@K. "
            "Default: 5."
        ),
    )

    eval_parser.add_argument(
        "--model-name",
        type=str,
        default=(
            config.retrieval.embedding_model
        ),
        help=(
            "Sentence-transformers "
            "embedding model name."
        ),
    )

    eval_parser.add_argument(
        "--device",
        type=str,
        default=(
            config.retrieval.device
        ),
        help=(
            "Device for embedding model."
        ),
    )

    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=(
            config.retrieval.batch_size
        ),
        help=(
            "Embedding batch size."
        ),
    )

    eval_parser.add_argument(
        "--show-failed-cases",
        action="store_true",
        help=(
            "Print failed retrieval cases."
        ),
    )

    eval_parser.add_argument(
        "--max-failed-cases",
        type=int,
        default=10,
        help=(
            "Maximum number of failed cases "
            "to print."
        ),
    )

    eval_parser.add_argument(
        "--no-normalize-embeddings",
        action="store_false",
        dest="normalize_embeddings",
        help=(
            "Disable embedding normalization."
        ),
    )

    add_retrieval_arguments(
        eval_parser,
        config,
    )

    eval_parser.set_defaults(
        func=handle_eval,
        normalize_embeddings=(
            config.retrieval.normalize_embeddings
        ),
    )

    return parser


def main() -> None:
    """CLI 主入口。"""

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    env_path = (
        project_root
        / ".env"
    )

    # Load local secrets/config overrides before
    # YAML + environment resolution.
    #
    # .env is ignored by git and should never
    # be committed.
    load_dotenv(
        dotenv_path=env_path,
    )

    config = load_config()

    setup_logging(
        level=config.logging.level,
        log_file=(
            config.logging.log_file
        ),
    )

    logger = get_logger(
        "cli"
    )

    parser = build_parser(
        config=config,
    )

    args = (
        parser.parse_args()
    )

    try:
        logger.info(
            "Running command: %s",
            args.command,
        )

        args.func(
            args
        )

    except Exception as exc:
        logger.exception(
            "Command failed: %s",
            args.command,
        )

        print()
        print("Error:")

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        raise SystemExit(
            1
        ) from exc


if __name__ == "__main__":
    main()