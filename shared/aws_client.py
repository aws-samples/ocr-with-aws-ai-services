import logging
import os
from functools import lru_cache

import boto3
import botocore.session
from botocore.exceptions import BotoCoreError, ClientError

from shared.config import DEFAULT_AWS_REGION

# Configure logging
logger = logging.getLogger(__name__)

# Environment variables consulted to pick the AWS profile, in precedence order.
#
# OCR_AWS_PROFILE comes first so this app can be pinned to one account without
# exporting AWS_PROFILE process-wide, which would also redirect any other AWS
# tooling sharing the same shell.
PROFILE_ENV_VARS = ("OCR_AWS_PROFILE", "AWS_PROFILE")

# Same idea for the region. AWS_DEFAULT_REGION is the older spelling of AWS_REGION
# and is still what some tools set, so both are honoured.
REGION_ENV_VARS = ("OCR_AWS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION")


class AwsProfileError(RuntimeError):
    """Raised when the configured AWS profile does not exist in the AWS config files"""


def resolve_aws_profile() -> tuple[str | None, str]:
    """
    Determine which named AWS profile the app should authenticate with

    Reads PROFILE_ENV_VARS in order and returns the first non-empty value, together
    with the name of the variable it came from so that callers can report where the
    setting originated.

    Returns:
        Tuple of (profile name or None when no profile is configured, human-readable
        description of the source)

    Raises:
        AwsProfileError: If a profile is named but is not present in the AWS config.
                        Falling back to the default chain here would silently
                        authenticate against a different account, which is exactly
                        the failure this function exists to prevent.
    """
    for env_var in PROFILE_ENV_VARS:
        profile_name = os.environ.get(env_var, "").strip()
        if not profile_name:
            continue

        # available_profiles reads ~/.aws/config and ~/.aws/credentials without
        # attempting to resolve any credentials, so this check is cheap and cannot
        # trigger an interactive auth flow.
        available_profiles = botocore.session.Session().available_profiles
        if profile_name not in available_profiles:
            raise AwsProfileError(
                f"{env_var} is set to '{profile_name}', which is not a profile in your AWS "
                f"config. Available profiles: {', '.join(sorted(available_profiles))}"
            )

        return profile_name, f"from {env_var}"

    return None, "no profile configured"


def resolve_aws_region() -> str:
    """
    Determine which AWS region the app should target

    Returns:
        Region name from the first populated entry of REGION_ENV_VARS, otherwise the
        application default
    """
    for env_var in REGION_ENV_VARS:
        region = os.environ.get(env_var, "").strip()
        if region:
            return region

    return DEFAULT_AWS_REGION


@lru_cache(maxsize=8)
def get_aws_session(region: str | None = None) -> boto3.session.Session:
    """
    Get a cached boto3 session bound to the explicitly configured profile

    Every AWS client in the app is built from this session, so this is the single
    place where credential selection is decided.

    Args:
        region: Region override. When None the region is resolved from the
                environment, then from the application default.

    Returns:
        Boto3 session for the configured profile and region

    Raises:
        AwsProfileError: If a profile is named but does not exist
    """
    profile_name, profile_source = resolve_aws_profile()
    effective_region = region or resolve_aws_region()

    session_kwargs: dict[str, str] = {}
    if profile_name:
        session_kwargs["profile_name"] = profile_name
    session_kwargs["region_name"] = effective_region

    logger.debug(
        f"Creating boto3 session (profile: {profile_name or 'default chain'} "
        f"[{profile_source}], region: {effective_region})"
    )

    return boto3.session.Session(**session_kwargs)


@lru_cache(maxsize=16)
def get_aws_client(
    service_name: str,
    region: str | None = None,
    endpoint_url: str | None = None,
):
    """
    Get a cached boto3 client for the configured profile

    Args:
        service_name: AWS service name, e.g. 's3'
        region: Region override, or None to inherit the session's region
        endpoint_url: Optional custom endpoint URL

    Returns:
        Boto3 client for the requested service
    """
    logger.debug(f"Getting AWS client for {service_name} in region {region}")

    client_kwargs: dict[str, str] = {}
    if region:
        client_kwargs["region_name"] = region
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url

    # Build from the configured session rather than boto3.client(), which would use
    # the default credential chain and ignore the selected profile entirely.
    return get_aws_session(region=region).client(service_name, **client_kwargs)


def get_account_id() -> str:
    """
    Get the current AWS account ID

    Returns:
        AWS account ID
    """
    sts_client = get_aws_client("sts")
    return sts_client.get_caller_identity()["Account"]


def get_current_region() -> str | None:
    """
    Get the current AWS region

    Returns:
        AWS region name
    """
    return get_aws_session().region_name


def describe_credential_error(error: BaseException) -> str | None:
    """
    Rewrite an expired- or unavailable-credential error as an actionable message

    Botocore reports expired credentials through several unrelated exception types
    depending on which provider was in use, and none of them mention which profile
    is at fault. Naming the profile is the whole point: authenticating the wrong one
    looks identical to not authenticating at all.

    Args:
        error: Exception raised while calling an AWS API

    Returns:
        Replacement message, or None if this is not a credential error and the
        original message should be used unchanged
    """
    # Matched by name rather than by class so that botocore version differences and
    # the dynamically generated ClientError subclasses are both covered.
    error_type = type(error).__name__
    credential_error_types = {
        "LoginRefreshRequired",
        "UnauthorizedSSOTokenError",
        "TokenRetrievalError",
        "CredentialRetrievalError",
        "NoCredentialsError",
        "PartialCredentialsError",
        "ProfileNotFound",
    }

    is_credential_error = error_type in credential_error_types
    if not is_credential_error and isinstance(error, ClientError):
        aws_error_code = error.response.get("Error", {}).get("Code", "")
        is_credential_error = aws_error_code in {
            "ExpiredToken",
            "ExpiredTokenException",
            "InvalidClientTokenId",
            "UnrecognizedClientException",
        }

    if not is_credential_error and not isinstance(error, AwsProfileError):
        return None

    try:
        profile_name, profile_source = resolve_aws_profile()
    except AwsProfileError:
        # The profile name itself is the problem, so there is nothing to add.
        return str(error)

    if profile_name:
        return (
            f"AWS credentials for profile '{profile_name}' ({profile_source}) are expired "
            f"or unavailable. Re-authenticate that profile, then retry. "
            f"Original error: {error}"
        )

    return (
        f"AWS credentials are expired or unavailable. No "
        f"{' or '.join(PROFILE_ENV_VARS)} is set, so boto3's default credential chain is "
        f"being used - this may not be the account you intended. "
        f"Original error: {error}"
    )


def log_aws_identity() -> None:
    """
    Log the profile, region and account the app will actually authenticate as

    Called once at startup. Invisible credential selection is the failure this
    guards against: a caller who has just authenticated one profile has no way to
    tell that the app resolved a different one, and every AWS call then fails with
    an error that says nothing about profiles.

    Deliberately swallows its own errors - a credential problem must still leave a
    usable UI, and the per-engine error handlers report it in context.

    Returns:
        None
    """
    try:
        profile_name, profile_source = resolve_aws_profile()
    except AwsProfileError as profile_error:
        logger.error(f"AWS profile misconfigured: {profile_error}")
        return

    region = resolve_aws_region()

    try:
        identity = get_aws_client("sts").get_caller_identity()
        account_id = identity["Account"]
        caller_arn = identity["Arn"]
    except (BotoCoreError, ClientError, KeyError) as identity_error:
        described = describe_credential_error(identity_error) or str(identity_error)
        logger.error(
            f"Could not determine the AWS identity for "
            f"{profile_name or 'the default credential chain'}: {described}"
        )
        return

    if profile_name:
        logger.info(f"AWS profile: {profile_name} ({profile_source}) | region: {region}")
        logger.info(f"AWS identity: account {account_id}, {caller_arn}")
    else:
        # Loud on purpose. This is the line that makes "why is it not using my
        # profile?" answerable without reading any code.
        logger.warning(
            f"No {' or '.join(PROFILE_ENV_VARS)} set - using boto3's default credential "
            f"chain, which resolved to account {account_id} ({caller_arn}). Set "
            f"OCR_AWS_PROFILE to pin this app to a specific account."
        )
