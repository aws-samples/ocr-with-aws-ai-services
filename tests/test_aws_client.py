"""
Tests for explicit AWS profile selection.

The bug these guard against: every client was built with boto3.client(), which uses
the default credential chain. With no AWS_PROFILE exported the app authenticated as
whatever [default] resolved to - a different account than the one the user had just
authenticated - and every call failed with an error that never mentioned profiles.

None of these tests require real credentials or a particular ~/.aws/config. The
profile list and the STS call are both stubbed.
"""

import logging

import botocore.session
import pytest
from botocore.exceptions import ClientError

from shared.aws_client import (
    AwsProfileError,
    describe_credential_error,
    get_aws_client,
    get_aws_session,
    log_aws_identity,
    resolve_aws_profile,
    resolve_aws_region,
)
from shared.config import DEFAULT_AWS_REGION

PROFILE_VARS = ("OCR_AWS_PROFILE", "AWS_PROFILE")
REGION_VARS = ("OCR_AWS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION")


@pytest.fixture(autouse=True)
def clean_aws_env(monkeypatch):
    """
    Clear AWS profile/region variables and the client caches before each test

    The lru_cache on get_aws_session/get_aws_client would otherwise leak a session
    built under one test's environment into the next test.

    Args:
        monkeypatch: pytest monkeypatch fixture

    Returns:
        None
    """
    for env_var in PROFILE_VARS + REGION_VARS:
        monkeypatch.delenv(env_var, raising=False)

    get_aws_session.cache_clear()
    get_aws_client.cache_clear()
    yield
    get_aws_session.cache_clear()
    get_aws_client.cache_clear()


@pytest.fixture
def stub_profiles(monkeypatch):
    """
    Make a fixed set of profile names appear to exist in the AWS config

    Args:
        monkeypatch: pytest monkeypatch fixture

    Returns:
        List of profile names presented as available
    """
    available = ["default", "demo-profile", "other-profile"]
    monkeypatch.setattr(
        botocore.session.Session,
        "available_profiles",
        property(lambda self: available),
    )
    return available


def test_ocr_profile_takes_precedence_over_aws_profile(monkeypatch, stub_profiles):
    """OCR_AWS_PROFILE must win, so the app can be pinned without touching AWS_PROFILE"""
    monkeypatch.setenv("OCR_AWS_PROFILE", "demo-profile")
    monkeypatch.setenv("AWS_PROFILE", "other-profile")

    profile_name, source = resolve_aws_profile()

    assert profile_name == "demo-profile"
    assert "OCR_AWS_PROFILE" in source


def test_aws_profile_used_when_ocr_profile_absent(monkeypatch, stub_profiles):
    """The standard variable is still honoured for anyone who exports it"""
    monkeypatch.setenv("AWS_PROFILE", "other-profile")

    profile_name, source = resolve_aws_profile()

    assert profile_name == "other-profile"
    assert "AWS_PROFILE" in source


def test_no_profile_configured_returns_none():
    """With nothing set, boto3's own default chain applies - reported, not guessed"""
    profile_name, source = resolve_aws_profile()

    assert profile_name is None
    assert source == "no profile configured"


def test_blank_profile_is_ignored(monkeypatch, stub_profiles):
    """An exported-but-empty variable must not be treated as a profile named ''"""
    monkeypatch.setenv("OCR_AWS_PROFILE", "   ")
    monkeypatch.setenv("AWS_PROFILE", "demo-profile")

    profile_name, _ = resolve_aws_profile()

    assert profile_name == "demo-profile"


def test_unknown_profile_raises_naming_the_profile(monkeypatch, stub_profiles):
    """
    A typo'd profile must fail loudly, not silently fall back to the default chain -
    falling back is what authenticated the wrong account in the first place.
    """
    monkeypatch.setenv("OCR_AWS_PROFILE", "typo-profile")

    with pytest.raises(AwsProfileError) as raised:
        resolve_aws_profile()

    message = str(raised.value)
    assert "typo-profile" in message
    assert "OCR_AWS_PROFILE" in message
    # The available profiles are listed so the fix is obvious from the error alone.
    assert "demo-profile" in message


@pytest.mark.parametrize(
    "env_var",
    ["OCR_AWS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"],
)
def test_region_read_from_each_supported_variable(monkeypatch, env_var):
    """Each supported region variable is honoured on its own"""
    monkeypatch.setenv(env_var, "eu-west-1")

    assert resolve_aws_region() == "eu-west-1"


def test_region_precedence(monkeypatch):
    """OCR_AWS_REGION > AWS_REGION > AWS_DEFAULT_REGION"""
    monkeypatch.setenv("OCR_AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")

    assert resolve_aws_region() == "us-east-1"

    monkeypatch.delenv("OCR_AWS_REGION")
    assert resolve_aws_region() == "us-west-2"

    monkeypatch.delenv("AWS_REGION")
    assert resolve_aws_region() == "eu-central-1"


def test_no_region_configured_uses_application_default():
    """A clean launch is deterministic rather than inheriting a profile region"""
    assert resolve_aws_region() == DEFAULT_AWS_REGION == "us-east-1"


def test_session_is_built_for_the_named_profile(monkeypatch, stub_profiles):
    """The resolved profile must actually reach boto3.session.Session"""
    monkeypatch.setenv("OCR_AWS_PROFILE", "demo-profile")
    monkeypatch.setenv("OCR_AWS_REGION", "us-east-1")

    captured = {}

    class FakeSession:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("boto3.session.Session", FakeSession)

    get_aws_session()

    assert captured == {"profile_name": "demo-profile", "region_name": "us-east-1"}


def test_client_is_built_from_the_configured_session(monkeypatch, stub_profiles):
    """
    get_aws_client must go through the session. Calling boto3.client() directly is
    the original defect: it silently ignores the selected profile.
    """
    monkeypatch.setenv("OCR_AWS_PROFILE", "demo-profile")
    requested = {}

    class FakeSession:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def client(self, service_name, **client_kwargs):
            requested["service_name"] = service_name
            requested["profile_name"] = self.kwargs.get("profile_name")
            return object()

    monkeypatch.setattr("boto3.session.Session", FakeSession)

    get_aws_client("s3")

    assert requested == {"service_name": "s3", "profile_name": "demo-profile"}


def test_describe_credential_error_names_the_profile(monkeypatch, stub_profiles):
    """The reported message must identify which profile needs re-authentication"""
    monkeypatch.setenv("OCR_AWS_PROFILE", "demo-profile")

    # botocore.exceptions.LoginRefreshRequired is not present in every botocore
    # version, and describe_credential_error matches on the type name for exactly
    # that reason, so a stand-in with the same name is a faithful test double.
    class LoginRefreshRequired(Exception):
        pass

    message = describe_credential_error(
        LoginRefreshRequired("Your session has expired or credentials have changed.")
    )

    assert message is not None
    assert "demo-profile" in message
    # The original text is preserved so nothing is lost in translation.
    assert "session has expired" in message


def test_describe_credential_error_warns_when_no_profile_is_set():
    """With no profile set the message must say the account may not be the intended one"""
    class TokenRetrievalError(Exception):
        pass

    message = describe_credential_error(TokenRetrievalError("token expired"))

    assert message is not None
    assert "OCR_AWS_PROFILE" in message


def test_describe_credential_error_matches_expired_token_client_error():
    """ClientError carries the reason in the response body, not the type name"""
    error = ClientError(
        {"Error": {"Code": "ExpiredToken", "Message": "The security token expired"}},
        "PutObject",
    )

    assert describe_credential_error(error) is not None


def test_describe_credential_error_ignores_unrelated_errors():
    """A genuine OCR failure must not be relabelled as a credential problem"""
    error = ClientError(
        {"Error": {"Code": "UnsupportedDocumentException", "Message": "bad PDF"}},
        "AnalyzeDocument",
    )

    assert describe_credential_error(error) is None
    assert describe_credential_error(ValueError("unparseable schema")) is None


def test_log_aws_identity_reports_profile_and_account(monkeypatch, stub_profiles, caplog):
    """The startup line must name the profile and the account it resolved to"""
    monkeypatch.setenv("OCR_AWS_PROFILE", "demo-profile")

    class FakeSts:
        def get_caller_identity(self):
            return {
                "Account": "111122223333",
                "Arn": "arn:aws:sts::111122223333:assumed-role/Admin/session",
            }

    monkeypatch.setattr("shared.aws_client.get_aws_client", lambda *a, **kw: FakeSts())

    with caplog.at_level(logging.INFO):
        log_aws_identity()

    logged = caplog.text
    assert "demo-profile" in logged
    assert "111122223333" in logged


def test_log_aws_identity_warns_loudly_with_no_profile(monkeypatch, caplog):
    """
    This is the line that would have made the original bug self-diagnosing: it names
    the account the default chain picked, at WARNING.
    """
    class FakeSts:
        def get_caller_identity(self):
            return {
                "Account": "444455556666",
                "Arn": "arn:aws:iam::444455556666:user/non-root-user",
            }

    monkeypatch.setattr("shared.aws_client.get_aws_client", lambda *a, **kw: FakeSts())

    with caplog.at_level(logging.WARNING):
        log_aws_identity()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "444455556666" in warnings[0].message
    assert "OCR_AWS_PROFILE" in warnings[0].message


def test_log_aws_identity_does_not_raise_on_expired_credentials(monkeypatch, stub_profiles, caplog):
    """Startup must survive bad credentials - the UI still has to come up"""
    monkeypatch.setenv("OCR_AWS_PROFILE", "demo-profile")

    def failing_client(*args, **kwargs):
        raise ClientError(
            {"Error": {"Code": "ExpiredToken", "Message": "expired"}},
            "GetCallerIdentity",
        )

    monkeypatch.setattr("shared.aws_client.get_aws_client", failing_client)

    with caplog.at_level(logging.ERROR):
        log_aws_identity()

    assert "demo-profile" in caplog.text


def test_log_aws_identity_reports_a_bad_profile_name(monkeypatch, stub_profiles, caplog):
    """A misconfigured profile is logged at startup rather than deferred to a click"""
    monkeypatch.setenv("OCR_AWS_PROFILE", "typo-profile")

    with caplog.at_level(logging.ERROR):
        log_aws_identity()

    assert "typo-profile" in caplog.text
