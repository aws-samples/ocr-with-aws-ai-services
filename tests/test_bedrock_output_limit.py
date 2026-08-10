"""
Tests for the Bedrock engine's output token limit

The Bedrock engine does OCR and JSON structuring in a single model call, so its
response is the largest the app produces. Both of its file types used to accept a
truncating limit without noticing:

- the PDF path hardcoded max_tokens to 4000, below even the API default
- the image path sent no inferenceConfig, taking the 4096 default

Neither checked the stop reason, and the text-extraction loops concatenate a
half-finished response, so a truncated document was returned as a successful result
and scored against ground truth.

Both file types now go through Converse. The PDF path formerly used invoke_model with
a native Anthropic request body, which only the Claude models accept.
"""

import json

import pytest
from PIL import Image

from engines.bedrock_engine import BedrockEngine
from shared.config import LLM_MAX_OUTPUT_TOKENS

MODEL_ID = "us.anthropic.claude-sonnet-5"
HAIKU_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


class FakeBedrockRuntime:
    """A stand-in Bedrock runtime client recording the calls it received"""

    def __init__(self, *, converse_response: dict = None) -> None:
        """
        Args:
            converse_response: Response to return from converse
        """
        self.converse_response = converse_response
        self.converse_calls: list[dict] = []

    def converse(self, **kwargs) -> dict:
        """
        Record and answer a converse call

        Args:
            **kwargs: Converse parameters

        Returns:
            The canned response
        """
        self.converse_calls.append(kwargs)
        return self.converse_response


def converse_body(text: str, stop_reason: str = "end_turn") -> dict:
    """
    Build a Converse response

    Args:
        text: Model output text
        stop_reason: Converse stopReason value

    Returns:
        A response dict
    """
    return {
        "output": {"message": {"content": [{"text": text}]}},
        "usage": {"inputTokens": 100, "outputTokens": 200, "totalTokens": 300},
        "stopReason": stop_reason,
    }


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch):
    """
    Install a fake Bedrock client into the engine module

    Args:
        monkeypatch: pytest monkeypatch fixture

    Returns:
        A factory taking FakeBedrockRuntime kwargs and returning the client
    """
    import engines.bedrock_engine as bedrock_engine

    def install(**kwargs) -> FakeBedrockRuntime:
        client = FakeBedrockRuntime(**kwargs)
        monkeypatch.setattr(bedrock_engine, "get_aws_client", lambda *a, **k: client)
        return client

    return install


@pytest.fixture
def pdf_path(tmp_path) -> str:
    """
    Write a file with a .pdf name

    Nothing in the engine parses the PDF - it is base64-encoded and sent as-is, and
    the annotation is a synthetic placeholder - so the bytes need not be a real PDF.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Absolute path to the file as a string
    """
    path = tmp_path / "document.pdf"
    path.write_bytes(b"%PDF-1.4 not a real pdf")
    return str(path)


@pytest.fixture
def image() -> Image.Image:
    """
    Build a small in-memory image

    Returns:
        A 40x40 white RGB image
    """
    return Image.new("RGB", (40, 40), color="white")


# --- request shape ----------------------------------------------------------


def test_pdf_path_requests_the_configured_limit(fake_client, pdf_path) -> None:
    """
    The PDF call asks for the configured limit, not the old hardcoded 4000

    Regression test: 4000 was below even the API's own 4096 default, so the single
    call covering OCR and structuring for an entire multi-page PDF truncated.
    """
    client = fake_client(converse_response=converse_body(json.dumps({"a": 1})))

    BedrockEngine().process_image(pdf_path, {"model_id": MODEL_ID})

    inference_config = client.converse_calls[0]["inferenceConfig"]
    assert inference_config["maxTokens"] == LLM_MAX_OUTPUT_TOKENS
    assert inference_config["maxTokens"] > 4096


def test_pdf_path_sends_a_converse_document_block(fake_client, pdf_path) -> None:
    """
    A PDF is sent as a Converse document block with raw bytes

    Regression test: the old invoke_model body used Anthropic's own
    {"type": "document", "source": {"type": "base64", ...}} shape, which Nova rejects
    with "extraneous key [type] is not permitted". Every non-Claude model in
    BEDROCK_MODELS therefore failed on any PDF, and PDFs are the whole sample set.
    """
    client = fake_client(converse_response=converse_body(json.dumps({"a": 1})))

    BedrockEngine().process_image(pdf_path, {"model_id": MODEL_ID})

    content = client.converse_calls[0]["messages"][0]["content"]
    document_block = content[1]["document"]
    assert document_block["format"] == "pdf"
    # Raw bytes, not base64: Converse encodes the blob itself.
    assert document_block["source"]["bytes"] == b"%PDF-1.4 not a real pdf"
    # Bedrock rejects a document name outside its allowed character set, and a period
    # is not in that set - so the name must carry no ".pdf" extension.
    assert document_block["name"]
    assert "." not in document_block["name"]


def test_image_path_sends_an_inference_config(fake_client, image) -> None:
    """
    The image call sets maxTokens rather than accepting the 4096 default

    Regression test: converse_args carried no inferenceConfig at all.
    """
    client = fake_client(converse_response=converse_body(json.dumps({"a": 1})))

    BedrockEngine().process_image(image, {"model_id": MODEL_ID})

    inference_config = client.converse_calls[0]["inferenceConfig"]
    assert inference_config["maxTokens"] == LLM_MAX_OUTPUT_TOKENS
    assert inference_config["maxTokens"] > 4096


def test_haiku_uses_native_json_schema_output(fake_client, image) -> None:
    """Haiku 4.5 accepts the latest Converse structured-output field."""
    client = fake_client(converse_response=converse_body(json.dumps({"a": 1})))
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}

    BedrockEngine().process_image(
        image, {"model_id": HAIKU_MODEL_ID, "output_schema": json.dumps(schema)})

    json_schema = client.converse_calls[0]["outputConfig"]["textFormat"][
        "structure"]["jsonSchema"]
    assert json.loads(json_schema["schema"]) == schema
    assert json_schema["name"] == "ocr_result"


def test_sonnet_keeps_prompt_based_schema_fallback(fake_client, image) -> None:
    """Sonnet 5 currently rejects outputConfig, so it must not receive the field."""
    client = fake_client(converse_response=converse_body(json.dumps({"a": 1})))

    BedrockEngine().process_image(
        image, {"model_id": MODEL_ID, "output_schema": '{"type":"object"}'})

    assert "outputConfig" not in client.converse_calls[0]


# --- truncation is reported --------------------------------------------------


def test_truncated_pdf_response_is_an_error_not_a_result(fake_client, pdf_path) -> None:
    """
    A truncated PDF response is reported as an error

    Previously the half-finished text was returned as a successful result and scored
    against ground truth, so truncation looked like poor extraction accuracy.
    """
    fake_client(converse_response=converse_body('{"a": "unterminat', stop_reason="max_tokens"))

    result = BedrockEngine().process_image(pdf_path, {"model_id": MODEL_ID})

    assert result["operation_type"] == "error"
    assert "truncated" in result["text"]
    assert "OCR_LLM_MAX_OUTPUT_TOKENS" in result["text"]


def test_truncated_image_response_is_an_error_not_a_result(fake_client, image) -> None:
    """A truncated image response is reported as an error"""
    fake_client(converse_response=converse_body('{"a": "unterminat', stop_reason="max_tokens"))

    result = BedrockEngine().process_image(image, {"model_id": MODEL_ID})

    assert result["operation_type"] == "error"
    assert "truncated" in result["text"]


def test_complete_pdf_response_succeeds(fake_client, pdf_path) -> None:
    """A complete PDF response is returned normally, not flagged"""
    fake_client(converse_response=converse_body(json.dumps({"partA": {"name": "Smith"}})))

    result = BedrockEngine().process_image(pdf_path, {"model_id": MODEL_ID})

    assert result["operation_type"] == "bedrock"
    assert result["json"] == {"partA": {"name": "Smith"}}


def test_complete_image_response_succeeds(fake_client, image) -> None:
    """A complete image response is returned normally, not flagged"""
    fake_client(converse_response=converse_body(json.dumps({"partA": {"name": "Smith"}})))

    result = BedrockEngine().process_image(image, {"model_id": MODEL_ID})

    assert result["operation_type"] == "bedrock"
    assert result["json"] == {"partA": {"name": "Smith"}}


# --- the check itself --------------------------------------------------------


@pytest.mark.parametrize("stop_reason", [None, "end_turn", "stop_sequence", "tool_use"])
def test_non_truncating_stop_reasons_pass(stop_reason) -> None:
    """
    Only max_tokens is treated as truncation

    A missing stop reason must not be treated as failure: the check has to be
    tolerant of an API that does not report one.
    """
    BedrockEngine()._raise_if_truncated(stop_reason=stop_reason)


def test_max_tokens_stop_reason_raises() -> None:
    """The truncation signal raises, naming the limit and the override"""
    with pytest.raises(ValueError) as exc_info:
        BedrockEngine()._raise_if_truncated(stop_reason="max_tokens")

    message = str(exc_info.value)
    assert str(LLM_MAX_OUTPUT_TOKENS) in message
    assert "OCR_LLM_MAX_OUTPUT_TOKENS" in message
