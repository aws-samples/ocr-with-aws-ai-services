"""
Tests for the post-processing LLM output token limit

Bedrock's Converse API defaults maxTokens to 4096 when inferenceConfig is omitted
and truncates silently at the limit. The only symptom is a JSON parse error on an
unterminated string one step later, which reads as a model quality problem rather
than a configuration one.
"""

import json

import pytest

import shared.prompt_manager as prompt_manager


class FakeBedrockRuntime:
    """A stand-in Bedrock runtime client that records the converse call it received"""

    def __init__(self, response: dict) -> None:
        """
        Args:
            response: Converse response to return
        """
        self.response = response
        self.calls: list[dict] = []

    def converse(self, **kwargs) -> dict:
        """
        Record and answer a converse call

        Args:
            **kwargs: Converse parameters

        Returns:
            The canned response
        """
        self.calls.append(kwargs)
        return self.response


def converse_response(text: str, stop_reason: str = "end_turn") -> dict:
    """
    Build a minimal Converse response

    Args:
        text: Model output text
        stop_reason: Converse stopReason value

    Returns:
        A response dict shaped like the real API's
    """
    return {
        "output": {"message": {"content": [{"text": text}]}},
        "usage": {"inputTokens": 100, "outputTokens": 200, "totalTokens": 300},
        "stopReason": stop_reason,
    }


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch):
    """
    Install a fake Bedrock client into the prompt manager

    Args:
        monkeypatch: pytest monkeypatch fixture

    Returns:
        A factory taking a Converse response and returning the FakeBedrockRuntime
    """
    def install(response: dict) -> FakeBedrockRuntime:
        client = FakeBedrockRuntime(response)
        monkeypatch.setattr(prompt_manager, "get_aws_client", lambda *a, **k: client)
        return client

    return install


def test_max_tokens_is_sent_explicitly(fake_client) -> None:
    """
    The call sets maxTokens rather than accepting the 4096 default

    Regression test: omitting inferenceConfig capped structured output at 4096
    tokens, which a multi-section claim form exceeds.
    """
    client = fake_client(converse_response(json.dumps({"a": 1})))

    prompt_manager.process_text_with_llm("some text", '{"type":"object"}')

    inference_config = client.calls[0]["inferenceConfig"]
    assert inference_config["maxTokens"] == prompt_manager.LLM_MAX_OUTPUT_TOKENS
    assert inference_config["maxTokens"] > 4096


def test_truncated_response_raises_actionable_error(fake_client) -> None:
    """
    A max_tokens stop reason fails loudly instead of yielding invalid JSON

    Without this the caller sees only "Failed to parse JSON: Unterminated string",
    which points at the wrong cause.
    """
    fake_client(converse_response('{"a": "unterminat', stop_reason="max_tokens"))

    with pytest.raises(ValueError) as exc_info:
        prompt_manager.process_text_with_llm("some text", '{"type":"object"}')

    message = str(exc_info.value)
    assert "truncated" in message
    assert "OCR_LLM_MAX_OUTPUT_TOKENS" in message
    assert "OCR_LLM_STRUCTURING_CHAR_BUDGET" in message


def test_normal_stop_reason_returns_json(fake_client) -> None:
    """A complete response parses and returns normally"""
    fake_client(converse_response(json.dumps({"partA": {"name": "Smith"}})))

    result, token_usage = prompt_manager.process_text_with_llm(
        "some text", '{"type":"object"}'
    )

    assert result == {"partA": {"name": "Smith"}}
    assert token_usage["inputTokens"] == 100


def test_token_usage_uses_camel_case_keys(fake_client) -> None:
    """
    Token usage keys match what the cost calculator reads

    shared.cost_calculator reads inputTokens; the multi-page Textract path used to
    build and read snake_case keys against this camelCase source, so its token
    totals were always zero and post-processing cost was reported as $0.
    """
    fake_client(converse_response(json.dumps({"a": 1})))

    _, token_usage = prompt_manager.process_text_with_llm("text", '{"type":"object"}')

    assert set(token_usage) == {"inputTokens", "outputTokens", "totalTokens"}


def test_max_output_tokens_is_env_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    """OCR_LLM_MAX_OUTPUT_TOKENS controls the limit"""
    import importlib

    import shared.config

    monkeypatch.setenv("OCR_LLM_MAX_OUTPUT_TOKENS", "9999")
    try:
        reloaded = importlib.reload(shared.config)
        assert reloaded.LLM_MAX_OUTPUT_TOKENS == 9999
    finally:
        monkeypatch.delenv("OCR_LLM_MAX_OUTPUT_TOKENS", raising=False)
        importlib.reload(shared.config)
