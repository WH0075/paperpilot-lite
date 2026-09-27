from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from mcp.server.mcpserver import (
    Context,
    MCPServer,
)

from .mcp_runtime import PaperPilotRuntime


class SearchPaperItem(TypedDict):
    """单条论文检索结果。"""

    rank: int
    text: str
    file_name: str
    page: int | None
    chunk_id: str | None
    score: float | None


class SearchPapersResponse(TypedDict):
    """search_papers 的结构化返回结果。"""

    query: str
    results: list[SearchPaperItem]


class AskPaperSource(TypedDict):
    """ask_papers 返回的单条来源。"""

    source_id: int
    file_name: str
    page: int | None
    chunk_id: str | None
    score: float | None


class AskPapersResponse(TypedDict):
    """ask_papers 的结构化返回结果。"""

    question: str
    answer: str
    sources: list[AskPaperSource]


@dataclass
class AppContext:
    """
    MCP Server 生命周期内共享的应用状态。

    runtime 只初始化一次，
    后续所有 tool 调用都复用它。
    """

    runtime: PaperPilotRuntime


def load_project_env() -> None:
    """
    加载项目根目录下的 .env。

    这个函数既会被 Server 启动入口调用，
    也会被 lifespan 调用。

    这样 transport、host、port、
    index_dir 等配置都可以来自 .env。
    """

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    env_path = (
        project_root
        / ".env"
    )

    load_dotenv(
        dotenv_path=env_path,
    )


@asynccontextmanager
async def app_lifespan(
    server: MCPServer,
) -> AsyncGenerator[AppContext]:
    """
    MCP Server 生命周期。

    Server 启动：
        加载 .env
        创建 PaperPilotRuntime

    Server 运行：
        所有 tool 复用同一个 runtime

    Server 关闭：
        当前没有额外资源需要显式释放
    """

    load_project_env()

    index_dir = os.getenv(
        "PAPERPILOT_MCP_INDEX_DIR"
    )

    runtime = (
        PaperPilotRuntime.from_config(
            index_dir=index_dir,
        )
    )

    yield AppContext(
        runtime=runtime,
    )


mcp = MCPServer(
    name="PaperPilot-Lite",
    description=(
        "Paper and technical document "
        "retrieval server powered by "
        "PaperPilot-Lite."
    ),
    lifespan=app_lifespan,
)


@mcp.tool()
def search_papers(
    query: str,
    ctx: Context[AppContext],
    top_k: int = 5,
) -> SearchPapersResponse:
    """
    Search the local paper knowledge base for evidence
    relevant to a query.

    Use this tool when you need raw evidence chunks from
    papers rather than an LLM-generated answer.

    Args:
        query:
            Natural-language search query.

        top_k:
            Number of evidence chunks to return.

    Returns:
        Structured ranked evidence including source file,
        page, chunk ID, score, and text.
    """

    if not isinstance(
        query,
        str,
    ):
        raise TypeError(
            "query must be a string"
        )

    query = query.strip()

    if not query:
        raise ValueError(
            "query must not be empty"
        )

    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
    ):
        raise TypeError(
            "top_k must be an integer"
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be positive"
        )

    runtime = (
        ctx
        .request_context
        .lifespan_context
        .runtime
    )

    raw_results = (
        runtime.search(
            query=query,
            top_k=top_k,
        )
    )

    results: list[
        SearchPaperItem
    ] = []

    for rank, result in enumerate(
        raw_results,
        start=1,
    ):
        metadata = (
            result.get(
                "metadata",
                {},
            )
            or {}
        )

        score = result.get(
            "score"
        )

        item: SearchPaperItem = {
            "rank": rank,
            "text": result.get(
                "text",
                "",
            ),
            "file_name": (
                metadata.get(
                    "file_name"
                )
                or metadata.get(
                    "source"
                )
                or "unknown file"
            ),
            "page": metadata.get(
                "page"
            ),
            "chunk_id": (
                metadata.get(
                    "chunk_id"
                )
            ),
            "score": (
                float(score)
                if isinstance(
                    score,
                    (int, float),
                )
                else None
            ),
        }

        results.append(
            item
        )

    return {
        "query": query,
        "results": results,
    }


@mcp.tool()
def ask_papers(
    question: str,
    ctx: Context[AppContext],
    top_k: int = 5,
) -> AskPapersResponse:
    """
    Answer a question using the local paper knowledge base.

    This tool performs retrieval followed by RAG generation.
    Use search_papers instead when raw evidence chunks are preferred.

    Args:
        question:
            Natural-language question.

        top_k:
            Number of evidence chunks used for RAG.

    Returns:
        Generated answer together with supporting sources.
    """

    if not isinstance(
        question,
        str,
    ):
        raise TypeError(
            "question must be a string"
        )

    question = question.strip()

    if not question:
        raise ValueError(
            "question must not be empty"
        )

    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
    ):
        raise TypeError(
            "top_k must be an integer"
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be positive"
        )

    runtime = (
        ctx
        .request_context
        .lifespan_context
        .runtime
    )

    response = runtime.ask(
        question=question,
        top_k=top_k,
    )

    raw_sources = response.get(
        "sources",
        [],
    )

    sources: list[
        AskPaperSource
    ] = []

    for source in raw_sources:
        score = source.get(
            "score"
        )

        item: AskPaperSource = {
            "source_id": int(
                source.get(
                    "source_id",
                    len(sources) + 1,
                )
            ),
            "file_name": (
                source.get(
                    "file_name"
                )
                or "unknown file"
            ),
            "page": source.get(
                "page"
            ),
            "chunk_id": source.get(
                "chunk_id"
            ),
            "score": (
                float(score)
                if isinstance(
                    score,
                    (int, float),
                )
                else None
            ),
        }

        sources.append(
            item
        )

    return {
        "question": question,
        "answer": str(
            response.get(
                "answer",
                "",
            )
        ),
        "sources": sources,
    }


def run_server() -> None:
    """
    根据环境变量启动 MCP Server。

    支持：
        stdio
        streamable-http

    默认仍然使用 stdio，
    因此兼容之前的启动方式。
    """

    load_project_env()

    transport = os.getenv(
        "PAPERPILOT_MCP_TRANSPORT",
        "stdio",
    ).strip().lower()

    if transport == "stdio":
        mcp.run(
            transport="stdio",
        )
        return

    if transport == "streamable-http":
        host = os.getenv(
            "PAPERPILOT_MCP_HOST",
            "127.0.0.1",
        ).strip()

        port_text = os.getenv(
            "PAPERPILOT_MCP_PORT",
            "8001",
        ).strip()

        try:
            port = int(
                port_text
            )
        except ValueError as exc:
            raise ValueError(
                "PAPERPILOT_MCP_PORT "
                "must be an integer, "
                f"got {port_text!r}"
            ) from exc

        if not (
            1 <= port <= 65535
        ):
            raise ValueError(
                "PAPERPILOT_MCP_PORT "
                "must be between "
                "1 and 65535"
            )

        mcp.run(
            transport="streamable-http",
            host=host,
            port=port,
            streamable_http_path="/mcp",
        )

        return

    raise ValueError(
        "Unsupported "
        "PAPERPILOT_MCP_TRANSPORT: "
        f"{transport!r}. "
        "Expected 'stdio' or "
        "'streamable-http'."
    )


if __name__ == "__main__":
    run_server()