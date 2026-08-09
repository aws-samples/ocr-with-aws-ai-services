# Init file for shared module
# Expose commonly used elements for convenience
from .config import logger, BEDROCK_MODELS, API_COSTS, STATUS_HTML, CUSTOM_THEME, POSTPROCESSING_MODEL, MAX_IMAGE_SIZE
from .aws_client import (
    get_aws_client,
    get_aws_session,
    get_account_id,
    get_current_region,
    AwsProfileError,
    describe_credential_error,
    log_aws_identity,
)
from .image_utils import convert_to_bytes, get_image_hash, get_image_object
from .prompt_manager import (
    get_prompt_for_document_type,
    process_text_with_llm,
    OCR_SYSTEM_PROMPT as SYSTEM_PROMPT,
    DOCUMENT_TYPE_PROMPT_MAP
)

# These names are re-exported deliberately, so `__all__` declares them as this
# package's public surface. Without it a linter reads every one as dead. Note
# `SYSTEM_PROMPT` is the alias this package exposes, not the underlying
# `OCR_SYSTEM_PROMPT`.
__all__ = [
    "logger",
    "BEDROCK_MODELS",
    "API_COSTS",
    "STATUS_HTML",
    "CUSTOM_THEME",
    "POSTPROCESSING_MODEL",
    "MAX_IMAGE_SIZE",
    "get_aws_client",
    "get_aws_session",
    "get_account_id",
    "get_current_region",
    "AwsProfileError",
    "describe_credential_error",
    "log_aws_identity",
    "convert_to_bytes",
    "get_image_hash",
    "get_image_object",
    "get_prompt_for_document_type",
    "process_text_with_llm",
    "SYSTEM_PROMPT",
    "DOCUMENT_TYPE_PROMPT_MAP",
]
