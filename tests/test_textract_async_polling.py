"""
Tests for asynchronous Textract job polling

Covers the polling loop in TextractEngine._await_async_job: the timeout boundary,
terminal status handling (including PARTIAL_SUCCESS, which the original loop
rejected as an unrecognised status), backoff growth, and the diagnostic content of
the error messages.

Sleeps are stubbed out and the clock is driven by a fake monotonic source, so the
whole suite runs in well under a second despite exercising 15-minute timeouts.
"""

import re

import pytest

import engines.textract_engine as textract_engine
from engines.textract_engine import TextractEngine


class FakeClock:
    """A monotonic clock advanced only by the sleeps the code under test performs"""

    def __init__(self) -> None:
        """Start the clock at an arbitrary non-zero offset to catch zero assumptions"""
        self.now: float = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        """
        Read the current fake time

        Returns:
            Seconds since an arbitrary origin
        """
        return self.now

    def sleep(self, seconds: float) -> None:
        """
        Advance the clock instead of blocking

        Args:
            seconds: Requested sleep duration
        """
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """
    Replace time.sleep and time.monotonic inside the engine module with a fake clock

    Args:
        monkeypatch: pytest monkeypatch fixture

    Returns:
        The FakeClock driving the module's notion of time
    """
    fake = FakeClock()
    monkeypatch.setattr(textract_engine.time, "sleep", fake.sleep)
    monkeypatch.setattr(textract_engine.time, "monotonic", fake.monotonic)
    return fake


@pytest.fixture
def engine() -> TextractEngine:
    """
    Build a TextractEngine

    Returns:
        A TextractEngine instance; no AWS client is created by the constructor
    """
    return TextractEngine()


def make_getter(responses: list[dict]):
    """
    Build a fake GetDocument* callable that returns each response in turn

    The final response repeats indefinitely so a test can describe a job that
    stays IN_PROGRESS forever without listing hundreds of entries.

    Args:
        responses: Responses to return, in order

    Returns:
        A callable accepting JobId and MaxResults keywords, with a .calls list
    """
    calls: list[dict] = []

    def getter(**kwargs) -> dict:
        calls.append(kwargs)
        index = min(len(calls) - 1, len(responses) - 1)
        return responses[index]

    getter.calls = calls
    return getter


def test_succeeded_returns_status(engine: TextractEngine, clock: FakeClock) -> None:
    """A job that succeeds on the first poll returns SUCCEEDED"""
    getter = make_getter([{"JobStatus": "SUCCEEDED"}])

    status = engine._await_async_job(
        get_results=getter, job_id="job-1", api_name="get-document-text-detection"
    )

    assert status == "SUCCEEDED"
    assert len(getter.calls) == 1


def test_polls_until_success(engine: TextractEngine, clock: FakeClock) -> None:
    """An IN_PROGRESS job is polled repeatedly until it succeeds"""
    getter = make_getter(
        [
            {"JobStatus": "IN_PROGRESS"},
            {"JobStatus": "IN_PROGRESS"},
            {"JobStatus": "SUCCEEDED"},
        ]
    )

    status = engine._await_async_job(
        get_results=getter, job_id="job-1", api_name="get-document-text-detection"
    )

    assert status == "SUCCEEDED"
    assert len(getter.calls) == 3


def test_poll_uses_max_results_one(engine: TextractEngine, clock: FakeClock) -> None:
    """
    Poll calls request a single block

    Without MaxResults the poll that observes SUCCEEDED downloads a full page of
    blocks that the caller immediately discards and re-fetches.
    """
    getter = make_getter([{"JobStatus": "SUCCEEDED"}])

    engine._await_async_job(
        get_results=getter, job_id="job-1", api_name="get-document-text-detection"
    )

    assert getter.calls[0] == {"JobId": "job-1", "MaxResults": 1}


def test_partial_success_returns_results(engine: TextractEngine, clock: FakeClock) -> None:
    """
    PARTIAL_SUCCESS is a terminal success, not an error

    The original loop sent PARTIAL_SUCCESS to its "unexpected status" branch and
    raised, discarding the pages Textract had processed successfully.
    """
    getter = make_getter(
        [
            {
                "JobStatus": "PARTIAL_SUCCESS",
                "StatusMessage": "1 page could not be processed",
                "Warnings": [{"ErrorCode": "PAGE_CORRUPT", "Pages": [4]}],
            }
        ]
    )

    status = engine._await_async_job(
        get_results=getter, job_id="job-1", api_name="get-document-text-detection"
    )

    assert status == "PARTIAL_SUCCESS"


def test_partial_success_warns_with_detail(
    engine: TextractEngine, clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    """PARTIAL_SUCCESS logs the service's warnings so the missing pages are visible"""
    getter = make_getter(
        [
            {
                "JobStatus": "PARTIAL_SUCCESS",
                "StatusMessage": "1 page could not be processed",
                "Warnings": [{"ErrorCode": "PAGE_CORRUPT", "Pages": [4]}],
            }
        ]
    )

    with caplog.at_level("WARNING"):
        engine._await_async_job(
            get_results=getter, job_id="job-1", api_name="get-document-text-detection"
        )

    logged = caplog.text
    assert "partially succeeded" in logged
    assert "PAGE_CORRUPT" in logged
    assert "1 page could not be processed" in logged


def test_failed_raises_with_status_message(engine: TextractEngine, clock: FakeClock) -> None:
    """A FAILED job surfaces Textract's own explanation rather than a bare message"""
    getter = make_getter(
        [
            {
                "JobStatus": "FAILED",
                "StatusMessage": "Document is password protected",
            }
        ]
    )

    with pytest.raises(Exception, match="Document is password protected"):
        engine._await_async_job(
            get_results=getter, job_id="job-1", api_name="get-document-text-detection"
        )


def test_failed_includes_warnings(engine: TextractEngine, clock: FakeClock) -> None:
    """A FAILED job includes Warnings alongside StatusMessage"""
    getter = make_getter(
        [
            {
                "JobStatus": "FAILED",
                "StatusMessage": "Unable to process",
                "Warnings": [{"ErrorCode": "INTERNAL_ERROR", "Pages": [1, 2]}],
            }
        ]
    )

    with pytest.raises(Exception) as exc_info:
        engine._await_async_job(
            get_results=getter, job_id="job-1", api_name="get-document-text-detection"
        )

    assert "INTERNAL_ERROR" in str(exc_info.value)


def test_failed_without_status_message_says_so(
    engine: TextractEngine, clock: FakeClock
) -> None:
    """
    A FAILED job with no StatusMessage says the service returned none

    Silently substituting an empty string would read as though the failure had no
    cause rather than no explanation.
    """
    getter = make_getter([{"JobStatus": "FAILED"}])

    with pytest.raises(Exception, match="no StatusMessage returned by Textract"):
        engine._await_async_job(
            get_results=getter, job_id="job-1", api_name="get-document-text-detection"
        )


def test_unrecognised_status_raises(engine: TextractEngine, clock: FakeClock) -> None:
    """An unknown JobStatus is refused rather than guessed at"""
    getter = make_getter([{"JobStatus": "SOMETHING_NEW"}])

    with pytest.raises(Exception, match="unrecognised status 'SOMETHING_NEW'"):
        engine._await_async_job(
            get_results=getter, job_id="job-1", api_name="get-document-text-detection"
        )


def test_timeout_raises_after_configured_limit(
    engine: TextractEngine, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job that never finishes raises once the configured timeout elapses"""
    monkeypatch.setattr(textract_engine, "TEXTRACT_ASYNC_TIMEOUT_SECONDS", 60)
    getter = make_getter([{"JobStatus": "IN_PROGRESS"}])

    with pytest.raises(Exception, match="did not finish within 60s"):
        engine._await_async_job(
            get_results=getter, job_id="job-1", api_name="get-document-text-detection"
        )

    assert clock.now - 1000.0 >= 60


def test_timeout_never_sleeps_past_the_deadline(
    engine: TextractEngine, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The final sleep is trimmed so the deadline is honoured within one request

    With a 15s maximum backoff, sleeping a full interval near the deadline would
    overshoot the configured timeout by up to 15 seconds.
    """
    monkeypatch.setattr(textract_engine, "TEXTRACT_ASYNC_TIMEOUT_SECONDS", 10)
    monkeypatch.setattr(textract_engine, "TEXTRACT_ASYNC_POLL_INITIAL_SECONDS", 4.0)
    monkeypatch.setattr(textract_engine, "TEXTRACT_ASYNC_POLL_MAX_SECONDS", 4.0)
    getter = make_getter([{"JobStatus": "IN_PROGRESS"}])

    with pytest.raises(Exception, match="did not finish within 10s"):
        engine._await_async_job(
            get_results=getter, job_id="job-1", api_name="get-document-text-detection"
        )

    # 4 + 4 + 2, not 4 + 4 + 4
    assert clock.sleeps == [4.0, 4.0, 2.0]
    assert clock.now - 1000.0 == pytest.approx(10.0)


def test_success_at_the_timeout_boundary_is_not_a_timeout(
    engine: TextractEngine, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A job that succeeds exactly at the deadline returns its results

    Regression test for the original off-by-one: the loop counter reached exactly
    max_wait_time on the final iteration, so a SUCCEEDED job broke out of the loop
    and was then reported as a timeout by a guard that tested the counter instead
    of the outcome.
    """
    monkeypatch.setattr(textract_engine, "TEXTRACT_ASYNC_TIMEOUT_SECONDS", 10)
    monkeypatch.setattr(textract_engine, "TEXTRACT_ASYNC_POLL_INITIAL_SECONDS", 10.0)
    monkeypatch.setattr(textract_engine, "TEXTRACT_ASYNC_POLL_MAX_SECONDS", 10.0)
    getter = make_getter([{"JobStatus": "SUCCEEDED"}])

    status = engine._await_async_job(
        get_results=getter, job_id="job-1", api_name="get-document-text-detection"
    )

    # The single sleep consumes the entire budget, so the status arrives at
    # exactly elapsed == timeout.
    assert clock.now - 1000.0 == pytest.approx(10.0)
    assert status == "SUCCEEDED"


def test_timeout_message_names_job_and_recovery_command(
    engine: TextractEngine, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The timeout message carries everything needed to recover the job

    Textract keeps results for 7 days, so the job id is the difference between
    recoverable and wasted work.
    """
    monkeypatch.setattr(textract_engine, "TEXTRACT_ASYNC_TIMEOUT_SECONDS", 30)
    getter = make_getter([{"JobStatus": "IN_PROGRESS"}])

    with pytest.raises(Exception) as exc_info:
        engine._await_async_job(
            get_results=getter, job_id="abc123", api_name="get-document-analysis"
        )

    message = str(exc_info.value)
    assert "abc123" in message
    assert "aws textract get-document-analysis --job-id abc123" in message
    assert "OCR_TEXTRACT_ASYNC_TIMEOUT_SECONDS" in message
    # The distinction matters: a queued job is not a broken one.
    assert "not" in message and "failed" in message


def test_timeout_message_reports_real_elapsed_time(
    engine: TextractEngine, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Elapsed time comes from the clock, not from summed sleep intervals

    Each poll is a network round trip. The original loop accumulated only its
    sleep interval, so its reported wait was always an understatement.
    """
    monkeypatch.setattr(textract_engine, "TEXTRACT_ASYNC_TIMEOUT_SECONDS", 20)
    monkeypatch.setattr(textract_engine, "TEXTRACT_ASYNC_POLL_INITIAL_SECONDS", 5.0)
    monkeypatch.setattr(textract_engine, "TEXTRACT_ASYNC_POLL_MAX_SECONDS", 5.0)

    calls: list[dict] = []

    def slow_getter(**kwargs) -> dict:
        # Each request itself takes 5s of wall clock, which summed sleeps ignore.
        calls.append(kwargs)
        clock.now += 5.0
        return {"JobStatus": "IN_PROGRESS"}

    with pytest.raises(Exception) as exc_info:
        engine._await_async_job(
            get_results=slow_getter, job_id="job-1", api_name="get-document-text-detection"
        )

    # Two polls of 5s sleep + 5s request = 20s, so the loop stops after 2 polls
    # rather than the 4 it would take if only sleeps were counted.
    assert len(calls) == 2
    waited = int(re.search(r"waited (\d+)s", str(exc_info.value)).group(1))
    assert waited == 20


def test_poll_interval_backs_off(
    engine: TextractEngine, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The interval grows geometrically rather than staying fixed"""
    monkeypatch.setattr(textract_engine, "TEXTRACT_ASYNC_TIMEOUT_SECONDS", 10_000)
    monkeypatch.setattr(textract_engine, "TEXTRACT_ASYNC_POLL_INITIAL_SECONDS", 2.0)
    monkeypatch.setattr(textract_engine, "TEXTRACT_ASYNC_POLL_MAX_SECONDS", 15.0)
    monkeypatch.setattr(textract_engine, "TEXTRACT_ASYNC_POLL_BACKOFF", 2.0)
    getter = make_getter(
        [{"JobStatus": "IN_PROGRESS"}] * 4 + [{"JobStatus": "SUCCEEDED"}]
    )

    engine._await_async_job(
        get_results=getter, job_id="job-1", api_name="get-document-text-detection"
    )

    assert clock.sleeps == [2.0, 4.0, 8.0, 15.0, 15.0]


def test_poll_interval_is_capped(
    engine: TextractEngine, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backoff stops growing at the configured maximum"""
    monkeypatch.setattr(textract_engine, "TEXTRACT_ASYNC_TIMEOUT_SECONDS", 10_000)
    monkeypatch.setattr(textract_engine, "TEXTRACT_ASYNC_POLL_INITIAL_SECONDS", 2.0)
    monkeypatch.setattr(textract_engine, "TEXTRACT_ASYNC_POLL_MAX_SECONDS", 6.0)
    monkeypatch.setattr(textract_engine, "TEXTRACT_ASYNC_POLL_BACKOFF", 10.0)
    getter = make_getter(
        [{"JobStatus": "IN_PROGRESS"}] * 3 + [{"JobStatus": "SUCCEEDED"}]
    )

    engine._await_async_job(
        get_results=getter, job_id="job-1", api_name="get-document-text-detection"
    )

    assert max(clock.sleeps) == 6.0


def test_backoff_bounds_poll_count_for_a_long_job(
    engine: TextractEngine, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A full 15-minute wait costs far fewer API calls than fixed 2s polling would

    Fixed 2s polling for 900s is 450 GetDocument* calls per document; the batch
    path multiplies that by the sample count.
    """
    getter = make_getter([{"JobStatus": "IN_PROGRESS"}])

    with pytest.raises(Exception, match="did not finish within 900s"):
        engine._await_async_job(
            get_results=getter, job_id="job-1", api_name="get-document-text-detection"
        )

    assert len(getter.calls) < 100
    assert clock.now - 1000.0 == pytest.approx(900.0)


def test_default_timeout_is_longer_than_the_old_five_minutes() -> None:
    """
    The shipped default exceeds the 300s cap that produced the reported failure

    The reported job succeeded service-side; only the client gave up.
    """
    assert textract_engine.TEXTRACT_ASYNC_TIMEOUT_SECONDS > 300


@pytest.mark.parametrize(
    "env_var,attribute,value,expected",
    [
        ("OCR_TEXTRACT_ASYNC_TIMEOUT_SECONDS", "TEXTRACT_ASYNC_TIMEOUT_SECONDS", "120", 120),
        ("OCR_TEXTRACT_ASYNC_POLL_INITIAL_SECONDS", "TEXTRACT_ASYNC_POLL_INITIAL_SECONDS", "0.5", 0.5),
        ("OCR_TEXTRACT_ASYNC_POLL_MAX_SECONDS", "TEXTRACT_ASYNC_POLL_MAX_SECONDS", "30", 30.0),
        ("OCR_TEXTRACT_ASYNC_POLL_BACKOFF", "TEXTRACT_ASYNC_POLL_BACKOFF", "3", 3.0),
    ],
)
def test_polling_settings_are_env_overridable(
    monkeypatch: pytest.MonkeyPatch,
    env_var: str,
    attribute: str,
    value: str,
    expected: float,
) -> None:
    """
    Each polling constant can be overridden from the environment

    Args:
        monkeypatch: pytest monkeypatch fixture
        env_var: Environment variable to set
        attribute: shared.config attribute it controls
        value: Value to set in the environment
        expected: Value the reloaded module should expose
    """
    import importlib

    import shared.config

    monkeypatch.setenv(env_var, value)
    try:
        reloaded = importlib.reload(shared.config)
        assert getattr(reloaded, attribute) == expected
    finally:
        # Restore the module for every other test in the session, since reload
        # rebinds the constants other modules already imported by value.
        monkeypatch.delenv(env_var, raising=False)
        importlib.reload(shared.config)
