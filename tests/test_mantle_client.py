"""
Tests for the bedrock-mantle transport and the routing that reaches it

The GPT-5.6 models cannot go through boto3: botocore ships no service model for the
bedrock-mantle endpoint, so `get_aws_client('bedrock-mantle')` fails outright, and
`bedrock_runtime.converse()` rejects their model IDs. They need a second transport
with a different request shape, a different response shape, and a different way of
reporting truncation.

Nothing here reaches the network. The request builder and the response readers are
pure functions over dicts, and the engine tests substitute the transport, which is
what lets them assert the thing that actually matters: that a Mantle model never
touches a boto3 client.
"""

import json

import pytest
from PIL import Image

from engines.bedrock_engine import BedrockEngine
from shared.config import LLM_MAX_OUTPUT_TOKENS
from shared.mantle_client import (
    build_responses_request,
    extract_output_text,
    extract_token_usage,
    get_mantle_endpoint_url,
    get_mantle_region,
)

MANTLE_MODEL_ID = "openai.gpt-5.6-luna"
RUNTIME_MODEL_ID = "us.anthropic.claude-sonnet-5"

IMAGE_BYTES = b"\xff\xd8\xff not really a jpeg"
PDF_BYTES = b"%PDF-1.4 not really a pdf"


def test_mantle_models_use_the_application_region_by_default(monkeypatch) -> None:
    """Models without a regional restriction follow the configured app region."""
    import shared.mantle_client as mantle_client

    class FakeSession:
        region_name = "us-west-2"

    monkeypatch.setattr(mantle_client, "get_aws_session", lambda *a, **k: FakeSession())

    assert get_mantle_region(model_id="openai.gpt-5.6-terra") == "us-west-2"
    assert "us-west-2" in get_mantle_endpoint_url(model_id="openai.gpt-5.6-terra")


def test_sol_is_always_routed_to_us_east_1(monkeypatch) -> None:
    """Sol is in-region us-east-1 and returns 404 from the us-west-2 endpoint."""
    import shared.mantle_client as mantle_client

    class FakeSession:
        region_name = "us-west-2"

    monkeypatch.setattr(mantle_client, "get_aws_session", lambda *a, **k: FakeSession())

    assert get_mantle_region(model_id="openai.gpt-5.6-sol") == "us-east-1"
    assert "us-east-1" in get_mantle_endpoint_url(model_id="openai.gpt-5.6-sol")


def mantle_response(
    text: str, *, status: str = "completed", incomplete_reason: str = None
) -> dict:
    """
    Build a Responses API result in the shape the real endpoint returns

    Includes a leading "reasoning" item, which these models always emit and which
    carries no output_text - the reason the text reader cannot simply flatten
    `output`.

    Args:
        text (str): The model's visible output text.
        status (str): Response status; "incomplete" when a budget ran out.
        incomplete_reason (str): Value for incomplete_details.reason, or None.

    Returns:
        dict: A decoded Responses API body.
    """
    body = {
        "status": status,
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [{"type": "output_text", "text": text}]},
        ],
        "usage": {
            "input_tokens": 3695,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 212,
            "output_tokens_details": {"reasoning_tokens": 64},
            "total_tokens": 3907,
        },
    }
    if incomplete_reason:
        body["incomplete_details"] = {"reason": incomplete_reason}
    return body


@pytest.fixture
def image() -> Image.Image:
    """
    Build a small in-memory image

    Returns:
        Image.Image: A 40x40 white RGB image.
    """
    return Image.new("RGB", (40, 40), color="white")


@pytest.fixture
def pdf_path(tmp_path) -> str:
    """
    Write a file with a .pdf name

    The engine base64-encodes the bytes and sends them as-is, so they need not be a
    real PDF.

    Args:
        tmp_path: pytest temporary directory fixture.

    Returns:
        str: Absolute path to the file.
    """
    path = tmp_path / "claim form.pdf"
    path.write_bytes(PDF_BYTES)
    return str(path)


@pytest.fixture
def fake_transport(monkeypatch: pytest.MonkeyPatch):
    """
    Replace the Mantle transport in the engine and record what it was called with

    Args:
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        A factory taking the transport's return value and yielding the call log.
    """
    import engines.bedrock_engine as bedrock_engine

    def install(result: dict) -> list:
        calls: list = []

        def transport(**kwargs) -> dict:
            calls.append(kwargs)
            return result

        monkeypatch.setattr(bedrock_engine, "invoke_mantle_responses", transport)
        return calls

    return install


@pytest.fixture
def forbid_boto3(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Make any attempt to build a boto3 client fail

    This is the assertion the routing exists to satisfy, expressed as a fixture so
    that a regression shows up as a failure in the test that cares rather than as a
    live API call.

    Args:
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        None
    """
    import engines.bedrock_engine as bedrock_engine

    def refuse(*args, **kwargs):
        raise AssertionError(
            f"A boto3 client was requested for a Mantle model: {args} {kwargs}"
        )

    monkeypatch.setattr(bedrock_engine, "get_aws_client", refuse)


# --- request shape -----------------------------------------------------------


def test_image_requests_carry_an_input_image_data_url() -> None:
    """An image is sent as an input_image content block holding a data URL"""
    request = build_responses_request(
        model_id=MANTLE_MODEL_ID,
        prompt="Transcribe this.",
        max_output_tokens=1024,
        image_bytes=IMAGE_BYTES,
    )

    content = request["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "Transcribe this."}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")


def test_pdf_requests_carry_an_input_file_data_url() -> None:
    """A PDF is sent as an input_file content block, not as an image"""
    request = build_responses_request(
        model_id=MANTLE_MODEL_ID,
        prompt="Transcribe this.",
        max_output_tokens=1024,
        pdf_bytes=PDF_BYTES,
        pdf_filename="claim-form.pdf",
    )

    document_block = request["input"][0]["content"][1]
    assert document_block["type"] == "input_file"
    assert document_block["filename"] == "claim-form.pdf"
    assert document_block["file_data"].startswith("data:application/pdf;base64,")


def test_the_system_prompt_is_sent_as_instructions() -> None:
    """
    The system prompt uses the API's own field rather than a system-role message

    The Responses API has no "system" role; a message with that role is rejected.
    """
    request = build_responses_request(
        model_id=MANTLE_MODEL_ID,
        prompt="Transcribe this.",
        max_output_tokens=1024,
        system_prompt="You are an OCR system.",
        image_bytes=IMAGE_BYTES,
    )

    assert request["instructions"] == "You are an OCR system."
    assert [message["role"] for message in request["input"]] == ["user"]


def test_no_instructions_field_when_there_is_no_system_prompt() -> None:
    """An absent system prompt is omitted rather than sent as an empty string"""
    request = build_responses_request(
        model_id=MANTLE_MODEL_ID,
        prompt="Transcribe this.",
        max_output_tokens=1024,
        image_bytes=IMAGE_BYTES,
    )

    assert "instructions" not in request


def test_the_output_limit_uses_the_snake_case_spelling() -> None:
    """
    The limit is sent as max_output_tokens

    Converse spells it maxTokens and invoke_model max_tokens; either spelling here is
    ignored, which would silently restore the API's own default.
    """
    request = build_responses_request(
        model_id=MANTLE_MODEL_ID,
        prompt="Transcribe this.",
        max_output_tokens=12_345,
        image_bytes=IMAGE_BYTES,
    )

    assert request["max_output_tokens"] == 12_345


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"image_bytes": IMAGE_BYTES, "pdf_bytes": PDF_BYTES},
    ],
    ids=["neither", "both"],
)
def test_exactly_one_document_is_required(kwargs: dict) -> None:
    """
    Neither or both documents is an error, not a silent preference

    Args:
        kwargs (dict): Document arguments to pass to the builder.
    """
    with pytest.raises(ValueError):
        build_responses_request(
            model_id=MANTLE_MODEL_ID,
            prompt="Transcribe this.",
            max_output_tokens=1024,
            **kwargs,
        )


# --- response reading --------------------------------------------------------


def test_output_text_skips_reasoning_items() -> None:
    """Only message items contribute text; reasoning items are internal"""
    assert extract_output_text(mantle_response('{"a": 1}')) == '{"a": 1}'


def test_output_text_joins_multiple_message_blocks() -> None:
    """Text split across blocks is concatenated in order, not just the first taken"""
    body = {
        "output": [
            {"type": "message", "content": [
                {"type": "output_text", "text": '{"a":'},
                {"type": "output_text", "text": " 1}"},
            ]}
        ]
    }

    assert extract_output_text(body) == '{"a": 1}'


def test_a_reasoning_only_response_yields_empty_text() -> None:
    """
    A response that produced no message item reads as empty rather than raising

    This is what a budget exhausted by reasoning alone looks like. The truncation
    check is what reports it; the reader must not mask that with an exception of its
    own.
    """
    body = {"output": [{"type": "reasoning", "summary": []}], "usage": {}}

    assert extract_output_text(body) == ""


def test_token_usage_is_translated_to_camel_case() -> None:
    """
    Usage is rewritten into the keys the cost calculator reads

    The API reports input_tokens; shared.cost_calculator reads inputTokens, and a
    missing key there is silently treated as zero tokens.
    """
    usage = extract_token_usage(mantle_response("{}"))

    assert usage == {"inputTokens": 3695, "outputTokens": 212, "totalTokens": 3907}


def test_token_usage_totals_are_derived_when_absent() -> None:
    """A response omitting total_tokens still reports a total"""
    body = {"usage": {"input_tokens": 10, "output_tokens": 4}}

    assert extract_token_usage(body)["totalTokens"] == 14


def test_missing_usage_reads_as_zero_rather_than_raising() -> None:
    """
    Absent usage yields zeros

    Reported cost is then $0.00, which is correct: there is no token count to charge
    for, and inventing one would be worse than reporting nothing.
    """
    assert extract_token_usage({}) == {
        "inputTokens": 0, "outputTokens": 0, "totalTokens": 0
    }


# --- engine routing ----------------------------------------------------------


def test_a_mantle_image_run_never_builds_a_boto3_client(
    fake_transport, forbid_boto3, image
) -> None:
    """
    An image sent to a Mantle model goes through the Mantle transport only

    Regression test for the whole point of the split: botocore has no service model
    for this endpoint, and converse() rejects these model IDs, so any boto3 call on
    this path fails at runtime.
    """
    calls = fake_transport({
        "text": json.dumps({"partA": {"name": "Smith"}}),
        "token_usage": {"inputTokens": 100, "outputTokens": 20, "totalTokens": 120},
        "stop_reason": None,
    })

    result = BedrockEngine().process_image(image, {"model_id": MANTLE_MODEL_ID})

    assert len(calls) == 1
    assert calls[0]["image_bytes"] is not None
    assert calls[0]["pdf_bytes"] is None
    assert result["operation_type"] == "bedrock"
    assert result["json"] == {"partA": {"name": "Smith"}}


def test_a_mantle_pdf_run_sends_the_pdf_not_an_image(
    fake_transport, forbid_boto3, pdf_path
) -> None:
    """A PDF sent to a Mantle model is passed as PDF bytes with a clean filename"""
    calls = fake_transport({
        "text": json.dumps({"partA": {"name": "Smith"}}),
        "token_usage": {"inputTokens": 100, "outputTokens": 20, "totalTokens": 120},
        "stop_reason": None,
    })

    result = BedrockEngine().process_image(pdf_path, {"model_id": MANTLE_MODEL_ID})

    assert calls[0]["pdf_bytes"] == PDF_BYTES
    assert calls[0]["image_bytes"] is None
    # Sanitised from "claim form.pdf" - the model is shown this name.
    assert calls[0]["pdf_filename"].endswith(".pdf")
    assert result["file_type"] == "pdf"


def test_the_configured_output_limit_is_passed_through(
    fake_transport, forbid_boto3, image
) -> None:
    """The Mantle path honours the same output limit as the bedrock-runtime paths"""
    calls = fake_transport({
        "text": "{}",
        "token_usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        "stop_reason": None,
    })

    BedrockEngine().process_image(image, {"model_id": MANTLE_MODEL_ID})

    assert calls[0]["max_output_tokens"] == LLM_MAX_OUTPUT_TOKENS


def test_a_fenced_json_response_still_parses(fake_transport, forbid_boto3, image) -> None:
    """
    A response wrapped in a markdown code fence is unwrapped before parsing

    These models fence their JSON far more readily than the Claude models do, and an
    unstripped fence leaves `json` as {"text": "```json ..."}, which scores 0% and
    reads as a failed extraction.
    """
    fake_transport({
        "text": '```json\n{"partA": {"name": "Smith"}}\n```',
        "token_usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        "stop_reason": None,
    })

    result = BedrockEngine().process_image(image, {"model_id": MANTLE_MODEL_ID})

    assert result["json"] == {"partA": {"name": "Smith"}}


def test_a_truncated_mantle_response_is_an_error_not_a_result(
    fake_transport, forbid_boto3, image
) -> None:
    """
    An exhausted output budget is reported as an error

    The Responses API spells this "max_output_tokens" where the other two APIs say
    "max_tokens", so the shared check has to accept both or a truncated response is
    scored as a poor extraction.
    """
    fake_transport({
        "text": '{"partA": {"name": "Smit',
        "token_usage": {"inputTokens": 100, "outputTokens": LLM_MAX_OUTPUT_TOKENS,
                        "totalTokens": 100 + LLM_MAX_OUTPUT_TOKENS},
        "stop_reason": "max_output_tokens",
    })

    result = BedrockEngine().process_image(image, {"model_id": MANTLE_MODEL_ID})

    assert result["operation_type"] == "error"
    assert "truncated" in result["text"]
    assert "OCR_LLM_MAX_OUTPUT_TOKENS" in result["text"]


def test_a_transport_failure_is_reported_as_an_engine_error(
    fake_transport, forbid_boto3, image, monkeypatch
) -> None:
    """
    A failed Mantle call surfaces as an error result naming the cause

    The endpoint reports an unrecognised model or a missing access grant in the HTTP
    body, which is the only place the actual reason appears.
    """
    import engines.bedrock_engine as bedrock_engine
    from shared.mantle_client import MantleApiError

    def failing_transport(**kwargs):
        raise MantleApiError("bedrock-mantle returned HTTP 403: access denied to model")

    monkeypatch.setattr(bedrock_engine, "invoke_mantle_responses", failing_transport)

    result = BedrockEngine().process_image(image, {"model_id": MANTLE_MODEL_ID})

    assert result["operation_type"] == "error"
    assert "403" in result["text"]
    assert "access denied to model" in result["text"]


def test_a_bedrock_runtime_model_does_not_use_the_mantle_transport(
    fake_transport, image, monkeypatch
) -> None:
    """
    Routing is by model ID, so the Claude models still go through boto3

    Without this the split could be satisfied by sending everything to Mantle, which
    would fail only against the live API.
    """
    calls = fake_transport({
        "text": "{}",
        "token_usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        "stop_reason": None,
    })

    import engines.bedrock_engine as bedrock_engine

    class FakeRuntime:
        """A stand-in bedrock-runtime client recording converse calls"""

        def __init__(self) -> None:
            self.converse_calls: list = []

        def converse(self, **kwargs) -> dict:
            """
            Record and answer a converse call

            Args:
                **kwargs: Converse parameters.

            Returns:
                dict: A minimal successful Converse response.
            """
            self.converse_calls.append(kwargs)
            return {
                "output": {"message": {"content": [{"text": "{}"}]}},
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                "stopReason": "end_turn",
            }

    runtime = FakeRuntime()
    monkeypatch.setattr(bedrock_engine, "get_aws_client", lambda *a, **k: runtime)

    BedrockEngine().process_image(image, {"model_id": RUNTIME_MODEL_ID})

    assert calls == []
    assert len(runtime.converse_calls) == 1
