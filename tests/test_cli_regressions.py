from argparse import Namespace

import pytest

from src.paperpilot import cli


class FakeRetriever:
    def __init__(self, *args, **kwargs):
        pass



def test_handle_eval_uses_args_ks(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        cli.Retriever,
        "from_index",
        classmethod(lambda cls, **kwargs: FakeRetriever()),
    )
    monkeypatch.setattr(cli, "load_qa_set", lambda path: [{"question": "q", "expected_keywords": ["x"]}])

    def fake_evaluate_retrieval(*, retriever, qa_items, ks):
        captured["ks"] = ks
        return {
            "total": 1,
            "ks": list(ks),
            "hit_counts": {k: 0 for k in ks},
            "recall": {k: 0.0 for k in ks},
            "cases": [],
            "failed_cases": [],
        }

    monkeypatch.setattr(cli, "evaluate_retrieval", fake_evaluate_retrieval)
    monkeypatch.setattr(cli, "print_evaluation_report", lambda **kwargs: None)

    args = Namespace(
        ks=[1, 3, 5],
        index_dir="data/index",
        model_name="fake-model",
        device="cpu",
        normalize_embeddings=True,
        batch_size=32,
        qa_path="data/eval/qa_set.jsonl",
        show_failed_cases=False,
        max_failed_cases=10,

        # R3 retrieval configuration
        retrieval_mode="hybrid",
        hybrid_alpha=0.5,
        fusion_method="rrf",
        rrf_k=60,
    )

    cli.handle_eval(args)

    assert captured["ks"] == [1, 3, 5]


def test_create_llm_client_mock():
    args = Namespace(
        llm="mock",
        fixed_answer="mocked",
        llm_model=None,
        llm_base_url=None,
        llm_temperature=0.2,
        llm_max_tokens=512,
        llm_timeout=60,
    )

    client = cli.create_llm_client(args)

    assert client.generate("prompt") == "mocked"


def test_create_llm_client_real_requires_model(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    args = Namespace(
        llm="openai-compatible",
        fixed_answer=None,
        llm_model=None,
        llm_base_url="https://example.com/v1",
        llm_temperature=0.2,
        llm_max_tokens=512,
        llm_thinking=False,
        llm_timeout=60,
    )

    with pytest.raises(ValueError, match="requires a model name"):
        cli.create_llm_client(args)


def test_create_llm_client_real(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    args = Namespace(
        llm="openai-compatible",
        fixed_answer=None,
        llm_model="test-model",
        llm_base_url="https://example.com/v1",
        llm_temperature=0.1,
        llm_max_tokens=256,
        llm_thinking=False,
        llm_timeout=30,
    )

    client = cli.create_llm_client(args)

    assert isinstance(client, cli.OpenAICompatibleLLMClient)
    assert client.model_name == "test-model"
    assert client.base_url == "https://example.com/v1"
    assert client.thinking_enabled is False
