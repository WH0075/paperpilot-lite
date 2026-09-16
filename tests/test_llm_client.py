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

def test_openai_compatible_generate_builds_chat_completion_payload(monkeypatch):
    client = OpenAICompatibleLLMClient(
        model_name="test-model",
        api_key="fake-key",
        base_url="https://example.com/v1",
        temperature=0.3,
        thinking_enabled=False,
        max_tokens=123,
    )
    captured = {}

    def fake_post_json(*, url, payload):
        captured["url"] = url
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "Grounded answer [1]."}}]}

    monkeypatch.setattr(client, "_post_json", fake_post_json)

    answer = client.generate("Use only the context.")

    assert answer == "Grounded answer [1]."
    assert captured["url"] == "https://example.com/v1/chat/completions"
    assert captured["payload"]["model"] == "test-model"
    assert captured["payload"]["messages"][0]["content"] == "Use only the context."
    assert captured["payload"]["temperature"] == 0.3
    assert captured["payload"]["max_tokens"] == 123
    assert captured["payload"]["thinking"] == {
        "type": "disabled"
    }

def test_openai_compatible_generate_enables_thinking(
    monkeypatch,
):
    client = OpenAICompatibleLLMClient(
        model_name="test-model",
        api_key="fake-key",
        base_url="https://example.com/v1",
        thinking_enabled=True,
    )

    captured = {}

    def fake_post_json(*, url, payload):
        captured["url"] = url
        captured["payload"] = payload

        return {
            "choices": [
                {
                    "message": {
                        "content": "answer"
                    }
                }
            ]
        }

    monkeypatch.setattr(
        client,
        "_post_json",
        fake_post_json,
    )

    answer = client.generate("prompt")

    assert answer == "answer"
    assert captured["payload"]["thinking"] == {
        "type": "enabled"
    }
