import logging
import os

# Re-exported so the many `from shared.config import CUSTOM_THEME` imports keep
# working; the theme itself is defined with the stylesheet it belongs to.
from shared.ui_theme import CUSTOM_THEME, banner  # noqa: F401

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Image size constants
MAX_IMAGE_SIZE = 5 * 1024 * 1024 - 100000  # 5MB minus buffer for Bedrock

# S3 buckets used to stage documents for processing.
#
# Textract's PDF path is asynchronous and can only read from S3, so a reachable
# bucket in the same account and region as the API call is mandatory - not a
# convenience. The previous defaults ('ocr-with-ai-services-demo-bucket' and
# 'my-bda-demo-bucket') are names in the global S3 namespace owned by other
# accounts: HeadBucket returns 403 Forbidden rather than 404, so every upload
# failed with AccessDenied no matter which credentials were used.
#
# Defined here once instead of being repeated across ui.py, processor.py,
# sample_handler.py and engines/textract_engine.py, and overridable per
# environment so the bucket does not have to be edited in code.
DEFAULT_S3_BUCKET = (
    os.environ.get("OCR_S3_BUCKET", "").strip() or "idp-testsetbucket-wxwjq6eoivpn"
)

# BDA writes its output alongside its input, but is otherwise no different, so it
# shares the main bucket unless given one of its own.
DEFAULT_BDA_S3_BUCKET = (
    os.environ.get("OCR_BDA_S3_BUCKET", "").strip() or DEFAULT_S3_BUCKET
)

# Available Bedrock models.
#
# These are served by two different endpoints, which need two different request
# formats, so the split matters more than it looks:
#
#   bedrock-runtime  the Converse / InvokeModel APIs reached through boto3. The
#                    "us." prefix marks a cross-region inference profile, so
#                    requests are served from whichever US region has capacity.
#                    Note that Sonnet 5 carries no date/version suffix.
#   bedrock-mantle   an OpenAI-compatible Responses API on a separate host. See
#                    MANTLE_MODEL_IDS below.
#
# Only vision-capable models belong here: this is an OCR benchmark, so every entry
# is sent an image or a PDF. That rules out the gpt-oss family, which is text-only.
BEDROCK_MODELS = {
    "Claude Sonnet 5": "us.anthropic.claude-sonnet-5",
    "Amazon Nova 2 Lite": "us.amazon.nova-2-lite-v1:0",
    "GPT-5.6 Terra": "openai.gpt-5.6-terra",
    "GPT-5.6 Luna": "openai.gpt-5.6-luna"
}

# Models reached through the bedrock-mantle endpoint rather than bedrock-runtime.
#
# These are NOT usable through boto3: botocore ships no "bedrock-mantle" service
# model, and they are absent from `aws bedrock list-foundation-models` (which only
# reports bedrock-runtime models — searching it for these returns nothing, which
# looks exactly like the models not existing). They are called over plain HTTPS
# against MANTLE_ENDPOINT_TEMPLATE, signed with SigV4 under the "bedrock" service
# name using ordinary AWS credentials — the published examples use a Bedrock API
# key, but SigV4 works and avoids needing a second credential.
#
# They also take no "us." cross-region prefix; neither geo nor global inference
# profiles exist for them, so the bare model ID is correct.
# https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-terra.html
MANTLE_MODEL_IDS = frozenset({
    "openai.gpt-5.6-terra",
    "openai.gpt-5.6-luna"
})

# The path is "/openai/v1" for the GPT-5.6 family specifically, not the bare "/v1"
# that other models on this endpoint use.
MANTLE_ENDPOINT_TEMPLATE = "https://bedrock-mantle.{region}.api.aws/openai/v1"

# Default model for the JSON-structuring step that runs after Textract and BDA.
# Kept in step with the default entry in BEDROCK_MODELS above so that
# post-processing quality is not a hidden variable between engines.
POSTPROCESSING_MODEL = "us.anthropic.claude-sonnet-5"

# Textract Analyze Document feature types.
# Any combination is valid; OCR (LINE/WORD blocks) is always returned regardless.
# Selection elements (checkboxes) are only returned when FORMS or TABLES is asked
# for, which is what most of the signal on scanned claim forms depends on.
# https://docs.aws.amazon.com/textract/latest/dg/API_AnalyzeDocument.html
TEXTRACT_FEATURE_TYPES = ["FORMS", "TABLES", "QUERIES", "SIGNATURES", "LAYOUT"]

# Polling settings for the asynchronous Textract APIs used for PDFs
# (StartDocumentTextDetection / StartDocumentAnalysis).
#
# Asynchronous jobs are queued service-side and the queue wait is not a function
# of page count or file size. The same 7-page scanned PDF measured 36 seconds on
# one attempt and over 300 seconds on another, so any fixed cap is a judgement
# call rather than a derived value - hence the environment overrides.
#
# 900 seconds is ~25x the measured job time: long enough that hitting it means
# something is genuinely wrong rather than merely busy.
TEXTRACT_ASYNC_TIMEOUT_SECONDS = int(
    os.environ.get("OCR_TEXTRACT_ASYNC_TIMEOUT_SECONDS", "").strip() or 900
)

# Poll with exponential backoff rather than at a fixed interval. Starting at 2s
# keeps short jobs responsive; growing to 15s keeps a 15-minute job to roughly 70
# GetDocument* calls instead of 450.
TEXTRACT_ASYNC_POLL_INITIAL_SECONDS = float(
    os.environ.get("OCR_TEXTRACT_ASYNC_POLL_INITIAL_SECONDS", "").strip() or 2.0
)
TEXTRACT_ASYNC_POLL_MAX_SECONDS = float(
    os.environ.get("OCR_TEXTRACT_ASYNC_POLL_MAX_SECONDS", "").strip() or 15.0
)
TEXTRACT_ASYNC_POLL_BACKOFF = float(
    os.environ.get("OCR_TEXTRACT_ASYNC_POLL_BACKOFF", "").strip() or 1.5
)

# Maximum characters of document text sent to the post-processing LLM in one call.
#
# Multi-page PDFs used to be structured one page at a time, which produced JSON
# keyed by page number instead of by schema field. Pages are now batched into as
# few calls as fit this budget, so a typical claim form goes in a single call and
# the model sees sections that span page boundaries whole.
#
# 120,000 characters is roughly 30,000 tokens, well inside the context of the
# models in BEDROCK_MODELS while leaving room for a 12-17 KB schema and the output.
LLM_STRUCTURING_CHAR_BUDGET = int(
    os.environ.get("OCR_LLM_STRUCTURING_CHAR_BUDGET", "").strip() or 120_000
)

# Maximum tokens any Bedrock model in this app may emit in one response.
#
# Bedrock's Converse API defaults maxTokens to 4096 when inferenceConfig is
# omitted, and truncates silently at the limit: the only visible symptom is a JSON
# parse error on an unterminated string. A 52-field claim form exceeds 4096.
# Claude Sonnet 5 allows far more than this.
#
# One knob covers both the post-processing structuring step and the Bedrock engine's
# own extraction, because there is no reason to want different ceilings: both emit
# the same structured JSON for the same document, and the Bedrock engine's call has
# to cover OCR and structuring together, so if anything it needs the larger budget.
LLM_MAX_OUTPUT_TOKENS = int(
    os.environ.get("OCR_LLM_MAX_OUTPUT_TOKENS", "").strip() or 16_384
)

# Per-page Textract AnalyzeDocument rates, US East / US West on-demand, for the
# first 1,000,000 pages per month. This app processes single documents
# interactively, so the first pricing tier is always the applicable one and the
# reduced rates above 1M pages/month are deliberately not modelled.
# https://aws.amazon.com/textract/pricing/
TEXTRACT_FEATURE_COSTS = {
    'FORMS': 0.05,        # pricing example 3
    'TABLES': 0.015,      # pricing example 3
    'QUERIES': 0.015,     # pricing example 5
    'SIGNATURES': 0.0035, # pricing example 8
    'LAYOUT': 0.004       # free when combined with TABLES - see pricing example 16
}

# AWS publishes discounted bundle rates for some feature combinations that are
# cheaper than the sum of their parts, so those are looked up rather than summed.
TEXTRACT_FEATURE_BUNDLE_COSTS = {
    frozenset({'TABLES', 'QUERIES'}): 0.020,           # pricing example 7
    frozenset({'FORMS', 'TABLES', 'QUERIES'}): 0.070   # pricing example 6
}

# API cost information - Only for APIs currently in use
API_COSTS = {
    # Currently used Textract APIs
    'textract_detect': 1.50 / 1000,  # DetectDocumentText API: $1.50 per 1,000 pages
    'textract_async': 1.50 / 1000,   # StartDocumentTextDetection API: $1.50 per 1,000 pages
    # AnalyzeDocument / StartDocumentAnalysis are priced per selected feature, so
    # their per-page rate is resolved at call time from TEXTRACT_FEATURE_COSTS by
    # shared.cost_calculator.calculate_textract_analyze_cost(). The zeros below
    # exist only so that operation-type lookups do not KeyError.
    'textract_analyze': 0.0,
    'textract_analyze_async': 0.0,

    # On-demand rates for us-east-1, from https://aws.amazon.com/bedrock/pricing/
    # as of 2026-08-07.
    #
    # UNITS: dollars per 1,000 tokens. Both consumers - calculate_bedrock_cost() and
    # BedrockEngine.get_cost() - compute (tokens / 1000) * rate, so a rate written as
    # "$3 per 1M" must appear here as 0.003 and NOT as 0.003 / 1000. Every entry used
    # to carry that extra division, which reported every Bedrock cost in this app at
    # 1/1000 of the real figure: a $0.0048 run showed as $0.000005. It read as
    # plausibly cheap rather than as wrong, which is why it survived.
    # tests/test_model_config.py pins the magnitude against a known example.
    #
    # Every model in BEDROCK_MODELS must have an entry, and so must
    # POSTPROCESSING_MODEL: calculate_bedrock_cost() reports $0.00 for a model it
    # cannot find rather than raising, so a missing entry here shows up as a free
    # engine rather than as an error. tests/test_model_config.py enforces that too.
    'bedrock': {
        'us.anthropic.claude-sonnet-5': {
            # Promotional launch pricing of $2/$10 per 1M runs through 2026-08-31,
            # after which the standard rate is $3/$15. Update this on 2026-09-01.
            'input': 0.002,     # $2 per 1M input tokens
            'output': 0.010     # $10 per 1M output tokens
        },
        'us.amazon.nova-2-lite-v1:0': {
            # Nova 2 Lite bills image and document-page input at a flat 230 tokens
            # per page regardless of resolution, so per-page cost here is stable.
            'input': 0.0003,    # $0.30 per 1M input tokens
            'output': 0.0025    # $2.50 per 1M output tokens
        },
        # The GPT-5.6 models are priced at two tiers by prompt size. These are the
        # short-context (<=272K tokens) rates, which is the tier every request in
        # this app falls into - a long document runs to tens of thousands of tokens,
        # not hundreds of thousands. The 1M-context tier is exactly double.
        'openai.gpt-5.6-terra': {
            'input': 0.0022,    # $2.20 per 1M input tokens
            'output': 0.0132    # $13.20 per 1M output tokens
        },
        'openai.gpt-5.6-luna': {
            'input': 0.00022,   # $0.22 per 1M input tokens
            'output': 0.00132   # $1.32 per 1M output tokens
        }
    },
    'bda': {
        'standard': {
            'document': 0.010,  # $0.010 per page
            'image': 0.003      # $0.003 per image
        },
        'custom': {
            'document': 0.040,  # $0.040 per page
            'image': 0.005,     # $0.005 per image
            'extra_field': 0.0005  # $0.0005 per additional field (beyond 30)
        }
    }
}


# Status banners.
#
# The six signatures are unchanged - callers across processor.py and sample_handler.py
# unpack them positionally - but the HTML now comes from shared.ui_theme.banner(), so
# no colour is chosen here. Each of these used to hardcode a saturated background with
# white text, which was readable but meant six more places asserting a palette.
STATUS_HTML = {
    "processing": lambda engine: banner(
        tone="info", text=f"Processing with <b>{engine}</b>…"),
    "completed": lambda engine, time, cost, token_info="": banner(
        tone="ok",
        text=f"<b>{engine}</b> completed in <code>{time:.3f}s</code> "
             f"· est. cost <code>${cost:.6f}</code>{token_info}"),
    "error": lambda engine, time, error: banner(
        tone="error",
        text=f"<b>{engine}</b> failed after <code>{time:.3f}s</code>: {error}"),
    "global_processing": lambda: banner(
        tone="info", text="Processing with the selected engines…"),
    # saved_note carries where the run's JSON record was written, or why it was
    # not. It is optional and last so the existing positional callers - the two
    # intermediate updates, which have nothing saved yet - are unaffected.
    "global_completed": lambda time, cost, saved_note="": banner(
        tone="ok",
        text=f"All engines completed in <code>{time:.3f}s</code> "
             f"· total est. cost <code>${cost:.6f}</code>{saved_note}"),
    "global_partial": lambda success, total, time, cost: banner(
        tone="warn",
        text=f"<b>{success}/{total}</b> engines completed in <code>{time:.3f}s</code> "
             f"· total est. cost <code>${cost:.6f}</code>")
}