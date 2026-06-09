from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .cleaner import clean_documents
from .chunker import chunk_documents
from .document_loader import load_documents
from .embedder import Embedder
from .llm_client import MockLLMClient
from .rag_pipeline import RAGPipeline
from .retriever import Retriever
from .vector_store import VectorStore


def handle_ingest(args: argparse.Namespace) -> None:
    """处理 ingest 命令：读取文档、清洗、切分、向量化、建索引并保存。"""

    data_dir = Path(args.data_dir)
    index_dir = Path(args.index_dir)

    print("=" * 80)
    print("PaperPilot-Lite Ingest")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Index directory: {index_dir}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Overlap: {args.overlap}")
    print(f"Embedding model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Similarity: {args.similarity}")
    print()

    documents = load_documents(data_dir)
    print(f"Loaded documents: {len(documents)}")

    cleaned_documents = clean_documents(documents)
    print(f"Cleaned documents: {len(cleaned_documents)}")

    chunks = chunk_documents(
        documents=cleaned_documents,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(f"Created chunks: {len(chunks)}")

    if not chunks:
        raise ValueError("No chunks were created. Please check your input documents.")

    embedder = Embedder(
        model_name=args.model_name,
        device=args.device,
        normalize_embeddings=args.normalize_embeddings,
        batch_size=args.batch_size,
    )

    embeddings = embedder.embed_chunks(chunks)
    print(f"Embeddings shape: {embeddings.shape}")

    store = VectorStore(similarity=args.similarity)
    store.build_index(
        embeddings=embeddings,
        chunks=chunks,
    )

    store.save_index(index_dir)

    print()
    print("Ingest completed successfully.")
    print(f"Saved index to: {index_dir}")
    print(f"VectorStore chunks: {store.get_num_chunks()}")
    print(f"Embedding dim: {store.get_embedding_dim()}")


def handle_search(args: argparse.Namespace) -> None:
    """处理 search 命令：加载索引并返回 top-k 检索结果。"""

    retriever = Retriever.from_index(
        index_dir=args.index_dir,
        model_name=args.model_name,
        device=args.device,
        normalize_embeddings=args.normalize_embeddings,
        batch_size=args.batch_size,
        default_top_k=args.top_k,
    )

    results = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
    )

    print("=" * 80)
    print("PaperPilot-Lite Search")
    print("=" * 80)
    print(f"Query: {args.query}")
    print(f"Top-k: {args.top_k}")
    print(f"Results: {len(results)}")
    print("=" * 80)

    print_search_results(results)


def handle_ask(args: argparse.Namespace) -> None:
    """处理 ask 命令：执行完整 RAG 问答流程。"""

    llm_client = MockLLMClient(
        fixed_answer=args.fixed_answer,
    )

    pipeline = RAGPipeline.from_index(
        index_dir=args.index_dir,
        llm_client=llm_client,
        model_name=args.model_name,
        device=args.device,
        normalize_embeddings=args.normalize_embeddings,
        batch_size=args.batch_size,
        default_top_k=args.top_k,
        max_context_chars=args.max_context_chars,
        max_chunk_chars=args.max_chunk_chars,
    )

    response = pipeline.ask(
        query=args.query,
        top_k=args.top_k,
    )

    print("=" * 80)
    print("PaperPilot-Lite Ask")
    print("=" * 80)

    print("Question:")
    print(response["query"])
    print()

    print("Answer:")
    print(response["answer"])
    print()

    print("Sources:")
    print_sources(response["sources"])

    if args.show_prompt:
        print()
        print("=" * 80)
        print("Prompt")
        print("=" * 80)
        print(response["prompt"])

    if args.show_search_results:
        print()
        print("=" * 80)
        print("Raw Search Results")
        print("=" * 80)
        print_search_results(response["search_results"])


def print_search_results(results: list[dict[str, Any]]) -> None:
    """格式化打印检索结果。"""

    if not results:
        print("No results found.")
        return

    for i, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        score = result.get("score")
        text = result.get("text", "")

        file_name = metadata.get("file_name") or metadata.get("source") or "unknown file"
        page = metadata.get("page")
        chunk_id = metadata.get("chunk_id")

        print()
        print("-" * 80)
        print(f"Result {i}")

        if score is not None:
            try:
                print(f"Score: {float(score):.4f}")
            except (TypeError, ValueError):
                print(f"Score: {score}")

        print(f"Source: {file_name}")

        if page is not None:
            print(f"Page: {page}")

        if chunk_id is not None:
            print(f"Chunk ID: {chunk_id}")

        print()
        print("Text:")
        print(text[:800])

        if len(text) > 800:
            print("...")


def print_sources(sources: list[dict[str, Any]]) -> None:
    """格式化打印来源信息。"""

    if not sources:
        print("No sources.")
        return

    for source in sources:
        source_id = source.get("source_id")
        file_name = source.get("file_name") or "unknown file"
        page = source.get("page")
        chunk_id = source.get("chunk_id")
        score = source.get("score")

        source_line = f"[{source_id}] {file_name}"

        if page is not None:
            source_line += f", page {page}"

        if chunk_id is not None:
            source_line += f", chunk_id {chunk_id}"

        if score is not None:
            try:
                source_line += f", score {float(score):.4f}"
            except (TypeError, ValueError):
                source_line += f", score {score}"

        print(source_line)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        prog="paperpilot",
        description="PaperPilot-Lite: a local RAG document question-answering system.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Load documents, create chunks, build embeddings, and save vector index.",
    )
    ingest_parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing raw documents, such as data/raw.",
    )
    ingest_parser.add_argument(
        "--index-dir",
        type=str,
        default="data/index",
        help="Directory to save vector index. Default: data/index.",
    )
    ingest_parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size in characters. Default: 500.",
    )
    ingest_parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Chunk overlap in characters. Default: 100.",
    )
    ingest_parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers embedding model name.",
    )
    ingest_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for embedding model, such as cpu or cuda. Default: cpu.",
    )
    ingest_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size. Default: 32.",
    )
    ingest_parser.add_argument(
        "--similarity",
        type=str,
        default="cosine",
        choices=["cosine", "inner_product", "l2"],
        help="Similarity metric for vector store. Default: cosine.",
    )
    ingest_parser.add_argument(
        "--no-normalize-embeddings",
        action="store_false",
        dest="normalize_embeddings",
        help="Disable embedding normalization.",
    )
    ingest_parser.set_defaults(
        func=handle_ingest,
        normalize_embeddings=True,
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Search top-k relevant chunks for a query.",
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="User query.",
    )
    search_parser.add_argument(
        "--index-dir",
        type=str,
        default="data/index",
        help="Directory containing vector index. Default: data/index.",
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of search results. Default: 5.",
    )
    search_parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers embedding model name.",
    )
    search_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for embedding model, such as cpu or cuda. Default: cpu.",
    )
    search_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size. Default: 32.",
    )
    search_parser.add_argument(
        "--no-normalize-embeddings",
        action="store_false",
        dest="normalize_embeddings",
        help="Disable embedding normalization.",
    )
    search_parser.set_defaults(
        func=handle_search,
        normalize_embeddings=True,
    )

    ask_parser = subparsers.add_parser(
        "ask",
        help="Run full RAG question answering pipeline.",
    )
    ask_parser.add_argument(
        "query",
        type=str,
        help="User question.",
    )
    ask_parser.add_argument(
        "--index-dir",
        type=str,
        default="data/index",
        help="Directory containing vector index. Default: data/index.",
    )
    ask_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks. Default: 5.",
    )
    ask_parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers embedding model name.",
    )
    ask_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for embedding model, such as cpu or cuda. Default: cpu.",
    )
    ask_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size. Default: 32.",
    )
    ask_parser.add_argument(
        "--max-context-chars",
        type=int,
        default=4000,
        help="Maximum context characters in prompt. Default: 4000.",
    )
    ask_parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=1200,
        help="Maximum characters per chunk in prompt. Default: 1200.",
    )
    ask_parser.add_argument(
        "--fixed-answer",
        type=str,
        default=None,
        help="Fixed mock answer for testing.",
    )
    ask_parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the full prompt sent to LLM client.",
    )
    ask_parser.add_argument(
        "--show-search-results",
        action="store_true",
        help="Print raw search results.",
    )
    ask_parser.add_argument(
        "--no-normalize-embeddings",
        action="store_false",
        dest="normalize_embeddings",
        help="Disable embedding normalization.",
    )
    ask_parser.set_defaults(
        func=handle_ask,
        normalize_embeddings=True,
    )

    return parser


def main() -> None:
    """CLI 主入口。"""

    parser = build_parser()
    args = parser.parse_args()

    try:
        args.func(args)
    except Exception as exc:
        print()
        print("Error:")
        print(f"{type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()