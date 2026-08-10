"""
Tests for the default S3 bucket configuration.

The bug these guard against: 'ocr-with-ai-services-demo-bucket' and
'my-bda-demo-bucket' were hardcoded across four modules. Both are names in the
global S3 namespace owned by other accounts - HeadBucket returns 403 Forbidden, not
404 - so Textract's PDF path failed with AccessDenied regardless of credentials.

These tests assert the wiring, not the bucket's existence: reachability depends on
credentials and cannot be checked without them.
"""

import importlib
import tokenize

import pytest

# Buckets that are known to belong to other accounts. Re-introducing either of these
# anywhere would silently break PDF processing again.
FOREIGN_BUCKETS = ("ocr-with-ai-services-demo-bucket", "my-bda-demo-bucket")

# Every module that used to carry its own copy of the bucket name.
MODULES_WITH_BUCKET_DEFAULTS = (
    "shared.config",
    "ui",
    "processor",
    "sample_handler",
    "engines.textract_engine",
)


@pytest.fixture
def reloaded_config(monkeypatch):
    """
    Reload shared.config so module-level env lookups are re-evaluated

    DEFAULT_S3_BUCKET is resolved at import time, so monkeypatching the environment
    has no effect until the module is re-imported.

    Args:
        monkeypatch: pytest monkeypatch fixture

    Returns:
        Callable taking the desired environment and returning the reloaded module
    """
    import shared.config

    def _reload(**env):
        for name in ("OCR_S3_BUCKET", "OCR_BDA_S3_BUCKET"):
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return importlib.reload(shared.config)

    yield _reload
    # Restore the unpatched values for any later test in the session.
    monkeypatch.undo()
    importlib.reload(shared.config)


def test_default_bucket_is_not_a_foreign_bucket(reloaded_config):
    """The baked-in default must not be a bucket owned by another account"""
    config = reloaded_config()

    assert config.DEFAULT_S3_BUCKET not in FOREIGN_BUCKETS
    assert config.DEFAULT_BDA_S3_BUCKET not in FOREIGN_BUCKETS


def test_ocr_s3_bucket_overrides_the_default(reloaded_config):
    """OCR_S3_BUCKET must win, so the bucket need not be edited in code"""
    config = reloaded_config(OCR_S3_BUCKET="my-own-bucket")

    assert config.DEFAULT_S3_BUCKET == "my-own-bucket"


def test_bda_bucket_falls_back_to_the_main_bucket(reloaded_config):
    """Without its own setting, BDA shares the main bucket rather than a stale name"""
    config = reloaded_config(OCR_S3_BUCKET="my-own-bucket")

    assert config.DEFAULT_BDA_S3_BUCKET == "my-own-bucket"


def test_bda_bucket_can_be_set_independently(reloaded_config):
    """OCR_BDA_S3_BUCKET overrides the shared default"""
    config = reloaded_config(
        OCR_S3_BUCKET="my-own-bucket",
        OCR_BDA_S3_BUCKET="my-bda-bucket",
    )

    assert config.DEFAULT_S3_BUCKET == "my-own-bucket"
    assert config.DEFAULT_BDA_S3_BUCKET == "my-bda-bucket"


def test_blank_override_is_ignored(reloaded_config):
    """A blank override must not revive a repository-specific bucket."""
    config = reloaded_config(OCR_S3_BUCKET="   ")

    assert config.DEFAULT_S3_BUCKET == ""
    assert config.DEFAULT_S3_BUCKET not in FOREIGN_BUCKETS


def string_literals_in(file_path: str) -> list[str]:
    """
    Collect the value of every string literal in a Python source file

    Tokenizing rather than substring-matching the raw text is what lets a comment
    explaining why a bucket name is wrong coexist with a test forbidding that name
    as a value.

    Args:
        file_path: Path to the Python source file to scan

    Returns:
        List of string literal contents, with quotes and prefixes stripped
    """
    with open(file_path, "rb") as handle:
        tokens = list(tokenize.tokenize(handle.readline))

    literals: list[str] = []
    for token in tokens:
        if token.type != tokenize.STRING:
            continue
        # ast.literal_eval would choke on f-strings, and the raw text is enough to
        # spot a bucket name.
        literals.append(token.string.strip("rbuf").strip("\"'"))

    return literals


@pytest.mark.parametrize("module_name", MODULES_WITH_BUCKET_DEFAULTS)
def test_no_module_hardcodes_a_foreign_bucket(module_name):
    """
    No module may carry a foreign bucket name as a value.

    Checks the module source rather than behaviour, because the original defect was
    four independent literals that all had to be found and changed together. Only
    string literals are inspected - comments naming the old buckets are the point.
    """
    module = importlib.import_module(module_name)
    literals = string_literals_in(module.__file__)

    for foreign_bucket in FOREIGN_BUCKETS:
        assert foreign_bucket not in literals, (
            f"{module_name} hardcodes {foreign_bucket} as a value"
        )


def test_engine_and_ui_share_the_config_default():
    """
    The Textract fallback and the UI field must agree.

    They disagreeing is what made the original bug confusing: the field showed one
    bucket while a direct engine call used another.
    """
    from shared.config import DEFAULT_S3_BUCKET
    from engines.textract_engine import DEFAULT_S3_BUCKET as engine_default

    assert engine_default == DEFAULT_S3_BUCKET
