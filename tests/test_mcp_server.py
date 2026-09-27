from __future__ import annotations

import asyncio
from typing import Any

import pytest

from mcp import Client

from src.paperpilot import mcp_server


class FakeRuntime:
    """避免 MCP 测试真正加载 Embedding Model、Index 和 LLM。"""

    def __init__(self) -> None:
        self.search_calls: list[
            dict[str, Any]
        ] = []

        self.ask_calls: list[
            dict[str, Any]
        ] = []

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        self.search_calls.append(
            {
                "query": query,
                "top_k": top_k,
            }
        )

        return [
            {
                "text": (
                    "Maximum Inner Product Search "
                    "(MIPS) finds the top-K documents."
                ),
                "score": 0.9,
                "metadata": {
                    "file_name": "rag.pdf",
                    "page": 2,
                    "chunk_id": "rag.pdf:2:2",
                },
            },
            {
                "text": (
                    "The retriever returns "
                    "relevant latent documents."
                ),
                "score": 0.8,
                "metadata": {
                    "source": "rag.pdf",
                    "page": 3,
                    "chunk_id": "rag.pdf:3:1",
                },
            },
        ]

    def ask(
        self,
        question: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        self.ask_calls.append(
            {
                "question": question,
                "top_k": top_k,
            }
        )

        return {
            "query": question,
            "answer": (
                "RAG combines retrieval "
                "with generation."
            ),
            "sources": [
                {
                    "source_id": 1,
                    "file_name": "rag.pdf",
                    "page": 2,
                    "chunk_id": "rag.pdf:2:2",
                    "score": 0.9,
                }
            ],
            "prompt": "fake prompt",
            "search_results": [],
            "template_name": "grounded",
        }


def install_fake_runtime(
    monkeypatch,
) -> FakeRuntime:
    """
    替换 PaperPilotRuntime.from_config()。

    MCP lifespan 仍然真实运行，
    但不会加载真正的模型、索引和 LLM。
    """

    runtime = FakeRuntime()

    monkeypatch.setattr(
        mcp_server.PaperPilotRuntime,
        "from_config",
        classmethod(
            lambda cls, **kwargs: runtime
        ),
    )

    return runtime


def test_search_papers_is_registered(
    monkeypatch,
):
    """MCP tools/list 应能发现 search_papers。"""

    install_fake_runtime(
        monkeypatch
    )

    async def run():
        async with Client(
            mcp_server.mcp
        ) as client:
            tools_result = (
                await client.list_tools()
            )

            tool_names = {
                tool.name
                for tool in tools_result.tools
            }

            assert (
                "search_papers"
                in tool_names
            )

    asyncio.run(
        run()
    )


def test_search_papers_schema(
    monkeypatch,
):
    """
    验证 MCP SDK 根据 Python 类型注解
    正确生成 search_papers 输入/输出 Schema。
    """

    install_fake_runtime(
        monkeypatch
    )

    async def run():
        async with Client(
            mcp_server.mcp
        ) as client:
            tools_result = (
                await client.list_tools()
            )

            tool = next(
                tool
                for tool
                in tools_result.tools
                if (
                    tool.name
                    == "search_papers"
                )
            )

            input_schema = (
                tool.input_schema
            )

            properties = (
                input_schema[
                    "properties"
                ]
            )

            assert (
                properties["query"]["type"]
                == "string"
            )

            assert (
                properties["top_k"]["type"]
                == "integer"
            )

            assert (
                properties["top_k"][
                    "default"
                ]
                == 5
            )

            assert (
                "query"
                in input_schema["required"]
            )

            # ctx 由 MCP 自动注入，
            # 不应暴露给 Agent。
            assert (
                "ctx"
                not in properties
            )

            assert (
                tool.output_schema
                is not None
            )

    asyncio.run(
        run()
    )


def test_search_papers_returns_structured_content(
    monkeypatch,
):
    """
    tools/call 应调用 Runtime.search()，
    并返回稳定的结构化结果。
    """

    runtime = install_fake_runtime(
        monkeypatch
    )

    async def run():
        async with Client(
            mcp_server.mcp
        ) as client:
            result = (
                await client.call_tool(
                    "search_papers",
                    {
                        "query": (
                            "How does RAG "
                            "retrieve documents?"
                        ),
                        "top_k": 2,
                    },
                )
            )

            assert (
                result.is_error
                is False
            )

            content = (
                result.structured_content
            )

            assert content is not None

            assert (
                content["query"]
                == (
                    "How does RAG "
                    "retrieve documents?"
                )
            )

            results = (
                content["results"]
            )

            assert len(results) == 2

            first = results[0]

            assert (
                first["rank"] == 1
            )

            assert (
                first["file_name"]
                == "rag.pdf"
            )

            assert (
                first["page"] == 2
            )

            assert (
                first["chunk_id"]
                == "rag.pdf:2:2"
            )

            assert (
                first["score"] == 0.9
            )

            # 第二条只有 source，
            # 验证 file_name fallback。
            second = results[1]

            assert (
                second["file_name"]
                == "rag.pdf"
            )

    asyncio.run(
        run()
    )

    assert (
        runtime.search_calls
        == [
            {
                "query": (
                    "How does RAG "
                    "retrieve documents?"
                ),
                "top_k": 2,
            }
        ]
    )


def test_runtime_is_created_once_per_mcp_session(
    monkeypatch,
):
    """
    一个 MCP session 内多次 search_papers 调用
    应复用同一个 Runtime。
    """

    runtime = FakeRuntime()

    build_count = 0

    def fake_from_config(
        cls,
        **kwargs,
    ):
        nonlocal build_count

        build_count += 1

        return runtime

    monkeypatch.setattr(
        mcp_server.PaperPilotRuntime,
        "from_config",
        classmethod(
            fake_from_config
        ),
    )

    async def run():
        async with Client(
            mcp_server.mcp
        ) as client:
            await client.call_tool(
                "search_papers",
                {
                    "query": "query one",
                    "top_k": 1,
                },
            )

            await client.call_tool(
                "search_papers",
                {
                    "query": "query two",
                    "top_k": 2,
                },
            )

    asyncio.run(
        run()
    )

    assert build_count == 1

    assert (
        runtime.search_calls
        == [
            {
                "query": "query one",
                "top_k": 1,
            },
            {
                "query": "query two",
                "top_k": 2,
            },
        ]
    )


def test_search_papers_rejects_empty_query(
    monkeypatch,
):
    """空 query 应通过 MCP 返回 Tool Error。"""

    install_fake_runtime(
        monkeypatch
    )

    async def run():
        async with Client(
            mcp_server.mcp
        ) as client:
            result = (
                await client.call_tool(
                    "search_papers",
                    {
                        "query": "   ",
                        "top_k": 3,
                    },
                )
            )

            assert (
                result.is_error
                is True
            )

    asyncio.run(
        run()
    )


def test_search_papers_rejects_non_positive_top_k(
    monkeypatch,
):
    """top_k <= 0 应通过 MCP 返回 Tool Error。"""

    install_fake_runtime(
        monkeypatch
    )

    async def run():
        async with Client(
            mcp_server.mcp
        ) as client:
            result = (
                await client.call_tool(
                    "search_papers",
                    {
                        "query": "What is RAG?",
                        "top_k": 0,
                    },
                )
            )

            assert (
                result.is_error
                is True
            )

    asyncio.run(
        run()
    )


def test_ask_papers_is_registered(
    monkeypatch,
):
    """MCP tools/list 应能发现 ask_papers。"""

    install_fake_runtime(
        monkeypatch
    )

    async def run():
        async with Client(
            mcp_server.mcp
        ) as client:
            tools_result = (
                await client.list_tools()
            )

            tool_names = {
                tool.name
                for tool
                in tools_result.tools
            }

            assert (
                "ask_papers"
                in tool_names
            )

    asyncio.run(
        run()
    )


def test_ask_papers_schema(
    monkeypatch,
):
    """
    验证 MCP SDK 根据 Python 类型注解
    正确生成 ask_papers 输入/输出 Schema。
    """

    install_fake_runtime(
        monkeypatch
    )

    async def run():
        async with Client(
            mcp_server.mcp
        ) as client:
            tools_result = (
                await client.list_tools()
            )

            tool = next(
                tool
                for tool
                in tools_result.tools
                if (
                    tool.name
                    == "ask_papers"
                )
            )

            input_schema = (
                tool.input_schema
            )

            properties = (
                input_schema[
                    "properties"
                ]
            )

            assert (
                properties[
                    "question"
                ]["type"]
                == "string"
            )

            assert (
                properties[
                    "top_k"
                ]["type"]
                == "integer"
            )

            assert (
                properties[
                    "top_k"
                ]["default"]
                == 5
            )

            assert (
                "question"
                in input_schema["required"]
            )

            assert (
                "ctx"
                not in properties
            )

            assert (
                tool.output_schema
                is not None
            )

    asyncio.run(
        run()
    )


def test_ask_papers_returns_structured_content(
    monkeypatch,
):
    """
    tools/call 应调用 Runtime.ask()，
    并返回结构化 Answer + Sources。
    """

    runtime = install_fake_runtime(
        monkeypatch
    )

    async def run():
        async with Client(
            mcp_server.mcp
        ) as client:
            result = (
                await client.call_tool(
                    "ask_papers",
                    {
                        "question": (
                            "What is RAG?"
                        ),
                        "top_k": 3,
                    },
                )
            )

            assert (
                result.is_error
                is False
            )

            content = (
                result.structured_content
            )

            assert content is not None

            assert (
                content["question"]
                == "What is RAG?"
            )

            assert (
                content["answer"]
                == (
                    "RAG combines retrieval "
                    "with generation."
                )
            )

            sources = (
                content["sources"]
            )

            assert len(sources) == 1

            source = sources[0]

            assert (
                source["source_id"]
                == 1
            )

            assert (
                source["file_name"]
                == "rag.pdf"
            )

            assert (
                source["page"] == 2
            )

            assert (
                source["chunk_id"]
                == "rag.pdf:2:2"
            )

            assert (
                source["score"] == 0.9
            )

    asyncio.run(
        run()
    )

    assert (
        runtime.ask_calls
        == [
            {
                "question": (
                    "What is RAG?"
                ),
                "top_k": 3,
            }
        ]
    )


def test_ask_papers_rejects_empty_question(
    monkeypatch,
):
    """空 question 应通过 MCP 返回 Tool Error。"""

    install_fake_runtime(
        monkeypatch
    )

    async def run():
        async with Client(
            mcp_server.mcp
        ) as client:
            result = (
                await client.call_tool(
                    "ask_papers",
                    {
                        "question": "   ",
                        "top_k": 3,
                    },
                )
            )

            assert (
                result.is_error
                is True
            )

    asyncio.run(
        run()
    )


def test_ask_papers_rejects_non_positive_top_k(
    monkeypatch,
):
    """top_k <= 0 应通过 MCP 返回 Tool Error。"""

    install_fake_runtime(
        monkeypatch
    )

    async def run():
        async with Client(
            mcp_server.mcp
        ) as client:
            result = (
                await client.call_tool(
                    "ask_papers",
                    {
                        "question": (
                            "What is RAG?"
                        ),
                        "top_k": 0,
                    },
                )
            )

            assert (
                result.is_error
                is True
            )

    asyncio.run(
        run()
    )


def test_search_and_ask_share_one_runtime(
    monkeypatch,
):
    """
    search_papers 和 ask_papers
    应在同一个 MCP session 中共享 Runtime。
    """

    runtime = FakeRuntime()

    build_count = 0

    def fake_from_config(
        cls,
        **kwargs,
    ):
        nonlocal build_count

        build_count += 1

        return runtime

    monkeypatch.setattr(
        mcp_server.PaperPilotRuntime,
        "from_config",
        classmethod(
            fake_from_config
        ),
    )

    async def run():
        async with Client(
            mcp_server.mcp
        ) as client:
            await client.call_tool(
                "search_papers",
                {
                    "query": (
                        "What is RAG?"
                    ),
                    "top_k": 2,
                },
            )

            await client.call_tool(
                "ask_papers",
                {
                    "question": (
                        "What is RAG?"
                    ),
                    "top_k": 3,
                },
            )

    asyncio.run(
        run()
    )

    # 一个 MCP session 只构造一次 Runtime。
    assert build_count == 1

    assert (
        runtime.search_calls
        == [
            {
                "query": (
                    "What is RAG?"
                ),
                "top_k": 2,
            }
        ]
    )

    assert (
        runtime.ask_calls
        == [
            {
                "question": (
                    "What is RAG?"
                ),
                "top_k": 3,
            }
        ]
    )


def test_run_server_defaults_to_stdio(
    monkeypatch,
):
    """
    未配置 transport 时，
    MCP Server 应默认使用 stdio。
    """

    calls = []

    monkeypatch.setattr(
        mcp_server,
        "load_project_env",
        lambda: None,
    )

    monkeypatch.delenv(
        "PAPERPILOT_MCP_TRANSPORT",
        raising=False,
    )

    def fake_run(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        mcp_server.mcp,
        "run",
        fake_run,
    )

    mcp_server.run_server()

    assert calls == [
        {
            "transport": "stdio",
        }
    ]


def test_run_server_uses_streamable_http_config(
    monkeypatch,
):
    """
    streamable-http 应正确传递
    host、port 和 MCP path。
    """

    calls = []

    monkeypatch.setattr(
        mcp_server,
        "load_project_env",
        lambda: None,
    )

    monkeypatch.setenv(
        "PAPERPILOT_MCP_TRANSPORT",
        "streamable-http",
    )

    monkeypatch.setenv(
        "PAPERPILOT_MCP_HOST",
        "0.0.0.0",
    )

    monkeypatch.setenv(
        "PAPERPILOT_MCP_PORT",
        "9000",
    )

    def fake_run(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        mcp_server.mcp,
        "run",
        fake_run,
    )

    mcp_server.run_server()

    assert calls == [
        {
            "transport": "streamable-http",
            "host": "0.0.0.0",
            "port": 9000,
            "streamable_http_path": "/mcp",
        }
    ]


@pytest.mark.parametrize(
    "port_text",
    [
        "not-a-number",
        "0",
        "65536",
    ],
)
def test_run_server_rejects_invalid_port(
    monkeypatch,
    port_text,
):
    """
    HTTP port 必须是
    1 到 65535 之间的整数。
    """

    monkeypatch.setattr(
        mcp_server,
        "load_project_env",
        lambda: None,
    )

    monkeypatch.setenv(
        "PAPERPILOT_MCP_TRANSPORT",
        "streamable-http",
    )

    monkeypatch.setenv(
        "PAPERPILOT_MCP_PORT",
        port_text,
    )

    with pytest.raises(
        ValueError,
        match="PAPERPILOT_MCP_PORT",
    ):
        mcp_server.run_server()


def test_run_server_rejects_unsupported_transport(
    monkeypatch,
):
    """
    未支持的 transport
    应显式报错。
    """

    monkeypatch.setattr(
        mcp_server,
        "load_project_env",
        lambda: None,
    )

    monkeypatch.setenv(
        "PAPERPILOT_MCP_TRANSPORT",
        "websocket",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported "
            "PAPERPILOT_MCP_TRANSPORT"
        ),
    ):
        mcp_server.run_server()