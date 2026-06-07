import pytest

from src.paperpilot.llm_client import (
    BaseLLMClient,
    MockLLMClient,
    OpenAICompatibleLLMClient,
)


def test_mock_llm_client_returns_default_answer():
    client = MockLLMClient()

    answer = client.generate("This is a test prompt.")

    assert isinstance(answer, str)
    assert answer
    assert "mock answer" in answer.lower()
    assert client.last_prompt == "This is a test prompt."


def test_mock_llm_client_returns_fixed_answer():
    client = MockLLMClient(fixed_answer="Fixed answer.")

    answer = client.generate("Prompt.")

    assert answer == "Fixed answer."
    assert client.last_prompt == "Prompt."


def test_mock_llm_client_rejects_empty_prompt():
    client = MockLLMClient()

    with pytest.raises(ValueError):
        client.generate("   ")


def test_mock_llm_client_rejects_non_string_prompt():
    client = MockLLMClient()

    with pytest.raises(TypeError):
        client.generate(123)  # type: ignore[arg-type]


def test_base_llm_client_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseLLMClient()  # type: ignore[abstract]


def test_openai_compatible_client_rejects_empty_model_name():
    with pytest.raises(ValueError):
        OpenAICompatibleLLMClient(
            model_name="   ",
            api_key="fake-key",
        )


def test_openai_compatible_client_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(ValueError):
        OpenAICompatibleLLMClient(
            model_name="fake-model",
            api_key=None,
        )


def test_openai_compatible_client_rejects_invalid_temperature():
    with pytest.raises(ValueError):
        OpenAICompatibleLLMClient(
            model_name="fake-model",
            api_key="fake-key",
            temperature=-1,
        )


def test_openai_compatible_client_rejects_invalid_max_tokens():
    with pytest.raises(ValueError):
        OpenAICompatibleLLMClient(
            model_name="fake-model",
            api_key="fake-key",
            max_tokens=0,
        )


def test_openai_compatible_client_rejects_invalid_timeout():
    with pytest.raises(ValueError):
        OpenAICompatibleLLMClient(
            model_name="fake-model",
            api_key="fake-key",
            timeout=0,
        )