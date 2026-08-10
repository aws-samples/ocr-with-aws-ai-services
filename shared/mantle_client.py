"""
Calling the models served by the bedrock-mantle endpoint

The GPT-5.6 models in `BEDROCK_MODELS` are not reachable through boto3 at all:
botocore ships no "bedrock-mantle" service model, so `get_aws_client()` cannot build
a client for them, and `bedrock_runtime.converse()` rejects their model IDs. They are
served by an OpenAI-compatible Responses API on a separate host, which takes a
different request shape from either Converse or InvokeModel.

This module is that second transport. It speaks plain HTTPS and signs each request
with SigV4 under the "bedrock" service name, using the same profile credentials as
every other AWS call in the app. AWS's own examples use a Bedrock API key with the
`openai` SDK instead; SigV4 is used here because it needs no second credential and
no extra dependency.

Response shape differences worth knowing, all handled below:

  - Text arrives under `output[].content[].text` for the items whose type is
    "message", not under a single top-level field. The list also contains
    "reasoning" items, which carry no user-visible text.
  - Usage keys are snake_case (`input_tokens`), where Converse uses camelCase
    (`inputTokens`). `shared.cost_calculator` reads camelCase, so usage is
    translated here rather than at each call site.
  - Truncation is reported as `status == "incomplete"` with
    `incomplete_details.reason`, not as a stop reason.
"""

import base64
import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from shared.aws_client import get_aws_session
from shared.config import (
    MANTLE_ENDPOINT_TEMPLATE,
    MANTLE_MODEL_REGION_OVERRIDES,
    logger,
)

# SigV4 service name to sign Mantle requests with.
#
# It is "bedrock", not "bedrock-mantle": the endpoint is a separate host but still
# part of the Bedrock service for signing and IAM purposes, so existing
# bedrock:InvokeModel permissions apply. Signing under "bedrock-mantle" is rejected
# with a signature mismatch.
MANTLE_SIGV4_SERVICE_NAME = "bedrock"

# Request timeout in seconds.
#
# A multi-page PDF sent to a reasoning model is a slow call - the models emit
# reasoning tokens before any output text - so this is deliberately generous. It
# exists to stop a hung connection from wedging the UI thread indefinitely, not to
# bound normal latency.
MANTLE_REQUEST_TIMEOUT_SECONDS = 600


class MantleApiError(RuntimeError):
    """Raised when the bedrock-mantle endpoint returns a non-2xx response"""


def get_mantle_region(*, model_id: str) -> str:
    """Resolve the endpoint region for one Mantle model."""
    overridden_region = MANTLE_MODEL_REGION_OVERRIDES.get(model_id)
    if overridden_region:
        return overridden_region

    region = get_aws_session().region_name
    if not region:
        raise RuntimeError("The AWS session did not resolve a region.")
    return region


def get_mantle_endpoint_url(*, model_id: str) -> str:
    """
    Build the Responses API URL for the model's resolved region

    Returns:
        str: Full URL of the /responses resource on the bedrock-mantle endpoint

    Raises:
        RuntimeError: If the AWS session unexpectedly has no region.
    """
    region = get_mantle_region(model_id=model_id)

    return f"{MANTLE_ENDPOINT_TEMPLATE.format(region=region)}/responses"


def build_responses_request(
    *,
    model_id: str,
    prompt: str,
    max_output_tokens: int,
    system_prompt: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    image_media_type: str = "image/jpeg",
    pdf_bytes: Optional[bytes] = None,
    pdf_filename: str = "document.pdf",
) -> Dict[str, Any]:
    """
    Assemble the Responses API request body for one document

    Kept separate from the call itself so the payload shape can be asserted in tests
    without reaching the network.

    Args:
        model_id (str): Mantle model ID, e.g. "openai.gpt-5.6-luna". Takes no "us."
            cross-region prefix.
        prompt (str): User instruction, including any JSON schema requirements.
        max_output_tokens (int): Ceiling on the response length. Note that reasoning
            tokens are charged against this same budget, so a value that would be
            ample for the answer alone can still be exhausted before any output text
            is produced.
        system_prompt (Optional[str]): System-level instruction, sent as the API's
            `instructions` field. Omitted when None.
        image_bytes (Optional[bytes]): Encoded image to transcribe.
        image_media_type (str): Media type of `image_bytes`. `convert_to_bytes()`
            always produces JPEG, hence the default.
        pdf_bytes (Optional[bytes]): PDF to transcribe.
        pdf_filename (str): Name reported for `pdf_bytes`. The API requires one; the
            model also sees it, so a descriptive name is preferable to a temp path.

    Returns:
        Dict[str, Any]: Request body ready to be JSON-encoded.

    Raises:
        ValueError: If neither or both of `image_bytes` and `pdf_bytes` are given.
            Exactly one document per call is the only case the caller needs, and
            silently preferring one over the other would send a document the caller
            did not intend.
    """
    if bool(image_bytes) == bool(pdf_bytes):
        raise ValueError(
            "Exactly one of image_bytes or pdf_bytes must be given, but "
            f"image_bytes was {'set' if image_bytes else 'empty'} and pdf_bytes was "
            f"{'set' if pdf_bytes else 'empty'}."
        )

    content: list[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]

    if image_bytes:
        # Binary is carried inline as a data URL rather than uploaded first, which
        # keeps this a single request with no file lifecycle to manage.
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        content.append({
            "type": "input_image",
            "image_url": f"data:{image_media_type};base64,{encoded}"
        })
    else:
        encoded = base64.b64encode(pdf_bytes).decode("utf-8")
        content.append({
            "type": "input_file",
            "filename": pdf_filename,
            "file_data": f"data:application/pdf;base64,{encoded}"
        })

    request_body: Dict[str, Any] = {
        "model": model_id,
        "max_output_tokens": max_output_tokens,
        "input": [{"role": "user", "content": content}]
    }

    if system_prompt:
        request_body["instructions"] = system_prompt

    return request_body


def extract_output_text(response_body: Dict[str, Any]) -> str:
    """
    Concatenate the visible text from a Responses API result

    Args:
        response_body (Dict[str, Any]): Decoded response from the Responses API.

    Returns:
        str: The model's text output. Reasoning items are skipped: they are internal
            and carry no `output_text` content, so including them is not possible
            anyway, but the filter documents why the loop is not simply flat.
    """
    texts: list[str] = []
    for item in response_body.get("output", []):
        if item.get("type") != "message":
            continue
        for block in item.get("content", []):
            if block.get("type") == "output_text":
                texts.append(block.get("text", ""))

    return "".join(texts)


def extract_token_usage(response_body: Dict[str, Any]) -> Dict[str, int]:
    """
    Translate Responses API usage into the camelCase keys the app reports costs from

    Args:
        response_body (Dict[str, Any]): Decoded response from the Responses API.

    Returns:
        Dict[str, int]: Keys 'inputTokens', 'outputTokens' and 'totalTokens', the
            spelling `shared.cost_calculator` and `engines.bedrock_engine.get_cost`
            read. `outputTokens` includes reasoning tokens, because they are billed
            as output tokens.
    """
    usage = response_body.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    return {
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        # Read rather than summed where present: the API reports it directly, and
        # the two need not agree if a future usage field is billed separately.
        "totalTokens": usage.get("total_tokens", input_tokens + output_tokens)
    }


def invoke_mantle_responses(
    *,
    model_id: str,
    prompt: str,
    max_output_tokens: int,
    system_prompt: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    image_media_type: str = "image/jpeg",
    pdf_bytes: Optional[bytes] = None,
    pdf_filename: str = "document.pdf",
) -> Dict[str, Any]:
    """
    Send one document to a bedrock-mantle model and return its text and usage

    Args:
        model_id (str): Mantle model ID, e.g. "openai.gpt-5.6-terra".
        prompt (str): User instruction, including any JSON schema requirements.
        max_output_tokens (int): Ceiling on the response length.
        system_prompt (Optional[str]): System-level instruction.
        image_bytes (Optional[bytes]): Encoded image to transcribe.
        image_media_type (str): Media type of `image_bytes`.
        pdf_bytes (Optional[bytes]): PDF to transcribe.
        pdf_filename (str): Name reported for `pdf_bytes`.

    Returns:
        Dict[str, Any]: Keys:
            'text' (str): The model's text output.
            'token_usage' (Dict[str, int]): camelCase usage counts.
            'stop_reason' (Optional[str]): Why the response ended early, or None if
                it completed. Named to match the field the Converse and InvokeModel
                paths report, so one truncation check covers all three.

    Raises:
        MantleApiError: If the endpoint returns a non-2xx status, or if credentials
            cannot be resolved. The HTTP body is included: it names the actual cause
            (unrecognised model, missing access grant, malformed content block), and
            urllib's own message is only the status line.
        RuntimeError: If no AWS region is configured.
    """
    region = get_mantle_region(model_id=model_id)
    url = get_mantle_endpoint_url(model_id=model_id)
    session = get_aws_session(region=region)

    credentials = session.get_credentials()
    if credentials is None:
        raise MantleApiError(
            "No AWS credentials could be resolved, so the bedrock-mantle request "
            "cannot be signed. Check OCR_AWS_PROFILE or AWS_PROFILE."
        )

    body = json.dumps(build_responses_request(
        model_id=model_id,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        system_prompt=system_prompt,
        image_bytes=image_bytes,
        image_media_type=image_media_type,
        pdf_bytes=pdf_bytes,
        pdf_filename=pdf_filename,
    ))

    # Signed with a throwaway AWSRequest purely to compute the Authorization header;
    # the request is then issued through urllib. get_frozen_credentials() is required
    # because a refreshable session credential could otherwise rotate between being
    # signed and being sent.
    signing_request = AWSRequest(
        method="POST",
        url=url,
        data=body,
        headers={"Content-Type": "application/json"}
    )
    SigV4Auth(
        credentials.get_frozen_credentials(), MANTLE_SIGV4_SERVICE_NAME, region
    ).add_auth(signing_request)

    logger.info(
        f"Calling bedrock-mantle model {model_id} at {url} "
        f"({len(body) / 1024:.2f}KB request body)"
    )

    http_request = urllib.request.Request(
        url,
        data=body.encode("utf-8"),
        method="POST",
        headers=dict(signing_request.headers)
    )

    try:
        with urllib.request.urlopen(
            http_request, timeout=MANTLE_REQUEST_TIMEOUT_SECONDS
        ) as http_response:
            response_body = json.loads(http_response.read())
    except urllib.error.HTTPError as http_error:
        # Read the body before the exception goes out of scope; it is the only place
        # the service says what was actually wrong.
        error_detail = http_error.read().decode("utf-8", errors="replace")
        raise MantleApiError(
            f"bedrock-mantle returned HTTP {http_error.code} for model {model_id}: "
            f"{error_detail}"
        ) from http_error
    except urllib.error.URLError as url_error:
        raise MantleApiError(
            f"Could not reach the bedrock-mantle endpoint at {url}: {url_error.reason}"
        ) from url_error

    token_usage = extract_token_usage(response_body)
    logger.info(
        f"Mantle token usage - Input: {token_usage['inputTokens']}, "
        f"Output: {token_usage['outputTokens']}, Total: {token_usage['totalTokens']}"
    )

    # A response that ran out of budget is reported as incomplete rather than as an
    # error, and its partial text parses as valid-looking prose, so the reason has to
    # be surfaced to the caller instead of being inferred from the text.
    stop_reason = None
    if response_body.get("status") == "incomplete":
        stop_reason = (response_body.get("incomplete_details") or {}).get("reason")

    return {
        "text": extract_output_text(response_body),
        "token_usage": token_usage,
        "stop_reason": stop_reason
    }
