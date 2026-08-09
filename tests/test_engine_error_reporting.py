"""A failed engine run must report itself as a failure, not as a completed one.

Every engine signals failure by *returning* a dict with operation_type "error"
rather than by raising, so `process_engine_result` is the only place that can
notice. Before this was checked, a failed run produced a green "completed" banner
with a cost attached and silently no accuracy figure — which is how a BDA failure
came to look like "BDA doesn't show accuracy".
"""

from unittest.mock import MagicMock, patch

import pytest

from engines.bda_engine import BDAEngine
from processor import _error_message, process_engine_result

ENGINES = ["Textract", "Bedrock", "BDA"]

# The exact `text` each engine puts on its error result. Kept verbatim from the
# engines so a change to their wording breaks these tests rather than the UI.
ERROR_TEXTS = {
    "Textract": "Textract Error: An error occurred (InvalidS3ObjectException)",
    "Bedrock": "Amazon Bedrock Error: An error occurred (AccessDeniedException)",
    "BDA": "BDA Error: The provided S3 bucket does not exist",
}

TRUTH = {"sectionA": {"name": "Jane Roe", "amount": "100.00"}}


def error_result(engine, *, process_time=1.25):
    """Build the result dict an engine returns when it fails.

    Args:
        engine (str): Engine name, used to pick that engine's error wording.
        process_time (float): Seconds the failed attempt took.

    Returns:
        dict: The engine's error result, shaped as the engines actually return it.
    """
    return {
        "text": ERROR_TEXTS[engine],
        "json": None,
        "image": None,
        "process_time": process_time,
        "operation_type": "error",
        "pages": 0,
    }


class TestErrorMessage:
    """The engine-name prefix is stripped, because the banner already names it."""

    @pytest.mark.parametrize("engine", ENGINES)
    def test_strips_engine_prefix(self, engine):
        message = _error_message(text=ERROR_TEXTS[engine])
        assert "Error:" not in message
        assert not message.startswith(engine)
        # The actual cause survives.
        assert message.endswith(")") or "S3 bucket" in message

    def test_keeps_text_with_no_prefix(self):
        assert _error_message(text="something broke") == "something broke"

    def test_splits_on_first_marker_only(self):
        # A message that itself mentions "Error:" must not be truncated further.
        message = _error_message(text="BDA Error: upstream said Error: nope")
        assert message == "upstream said Error: nope"

    @pytest.mark.parametrize("empty", ["", None])
    def test_reports_absence_of_a_message(self, empty):
        assert "no message" in _error_message(text=empty)


class TestFailedRunIsReportedAsFailed:
    @pytest.mark.parametrize("engine", ENGINES)
    def test_status_is_an_error_not_a_completion(self, engine):
        processed = process_engine_result(engine, error_result(engine), TRUTH, True)
        status = processed["status_html"]
        # The error tone class, not the bare word "error": the word also occurs inside
        # the AWS message some engines return, so substring-matching it passed for two
        # of the three engines regardless of how the banner was actually marked up.
        assert "ocr-banner--error" in status
        assert "completed" not in status, (
            f"{engine} failed but reported completion")

    @pytest.mark.parametrize("engine", ENGINES)
    def test_status_names_the_real_cause(self, engine):
        processed = process_engine_result(engine, error_result(engine), TRUTH, True)
        # The specific cause, not just a generic failure notice.
        tail = ERROR_TEXTS[engine].split("Error:", 1)[1].strip()
        assert tail in processed["status_html"]

    @pytest.mark.parametrize("engine", ENGINES)
    def test_no_cost_is_reported_for_work_that_produced_nothing(self, engine):
        processed = process_engine_result(engine, error_result(engine), TRUTH, True)
        assert processed["cost"] == 0.0
        # BDA's per-page rate is $0.040; it must not appear on a failed run.
        assert "0.040" not in processed["status_html"]

    @pytest.mark.parametrize("engine", ENGINES)
    def test_accuracy_and_json_are_empty(self, engine):
        processed = process_engine_result(engine, error_result(engine), TRUTH, True)
        assert processed["accuracy"] == 0.0
        assert processed["json"] is None
        assert "Accuracy:" not in processed["status_html"]

    def test_process_time_is_preserved(self):
        processed = process_engine_result(
            "BDA", error_result("BDA", process_time=7.5), TRUTH, True)
        assert processed["time"] == 7.5
        assert "7.500" in processed["status_html"]


class TestBdaBucketFailureIsShapedAsAnError:
    """A bucket that cannot be reached is a failure, not a zero-page success.

    This path used to return early with only `text` and `process_time` - no
    `operation_type` and no `pages`. `process_engine_result` therefore read it as a
    successful run, and the first thing to complain was the per-page cost
    calculation, which reported "BDA completed but reported 0 pages" and pointed
    nowhere near the misconfigured bucket.
    """

    def run_against_an_unreachable_bucket(self, *, tmp_path):
        """Run BDA with a bucket whose HeadBucket call fails.

        Args:
            tmp_path: pytest temporary directory, used for a stand-in PDF.

        Returns:
            dict: The engine's result.
        """
        document = tmp_path / "claim.pdf"
        # The bucket check happens before the bytes are used, so the contents of
        # this file never matter.
        document.write_bytes(b"%PDF-1.4 not a real document")

        s3_client = MagicMock()
        s3_client.head_bucket.side_effect = Exception(
            'Invalid bucket name "": Bucket name must match the regex')

        with patch("engines.bda_engine.get_aws_client", return_value=s3_client), \
                patch("engines.bda_engine.get_account_id", return_value="111122223333"), \
                patch("engines.bda_engine.get_current_region", return_value="us-east-1"):
            return BDAEngine()._process_with_bda(
                str(document), s3_bucket="", document_type="generic",
                output_schema=None, use_blueprint=False, is_pdf=True)

    def test_the_result_is_marked_as_an_error(self, tmp_path):
        result = self.run_against_an_unreachable_bucket(tmp_path=tmp_path)
        assert result["operation_type"] == "error"

    def test_the_result_carries_a_page_count_so_costing_does_not_raise(self, tmp_path):
        result = self.run_against_an_unreachable_bucket(tmp_path=tmp_path)
        assert result["pages"] == 0

    def test_the_message_names_the_bucket_as_the_cause(self, tmp_path):
        """The user has to be told which setting to fix."""
        result = self.run_against_an_unreachable_bucket(tmp_path=tmp_path)
        assert "S3 bucket" in result["text"]

    def test_the_failure_is_reported_as_a_failure_to_the_user(self, tmp_path):
        """End to end: the banner this result produces must not say "completed"."""
        result = self.run_against_an_unreachable_bucket(tmp_path=tmp_path)
        processed = process_engine_result("BDA", result, TRUTH, True)

        assert "ocr-banner--error" in processed["status_html"]
        assert "completed" not in processed["status_html"]
        assert processed["cost"] == 0.0


class TestSuccessfulRunStillScores:
    """The error check must not intercept anything that actually worked."""

    def test_bda_success_reports_completion_and_accuracy(self):
        result = {
            "text": "Jane Roe 100.00",
            # Schema-shaped, matching truth exactly, so accuracy is unambiguous.
            "json": TRUTH,
            "image": None,
            "process_time": 2.0,
            "token_usage": None,
            "field_count": 2,
            "use_blueprint": False,
            "pages": 1,
            "operation_type": "bda",
        }
        processed = process_engine_result("BDA", result, TRUTH, True)
        assert "completed" in processed["status_html"]
        assert processed["accuracy"] == 100.0
        assert "Accuracy: 100.0%" in processed["status_html"]

    def test_success_without_truth_reports_completion_and_no_accuracy(self):
        result = {
            "text": "some text",
            "json": {"sectionA": {}},
            "image": None,
            "process_time": 1.0,
            "field_count": 0,
            "use_blueprint": False,
            # BDA is billed per page, so a successful result always names its page
            # count. Omitting it now raises rather than billing a single page.
            "pages": 1,
            "operation_type": "bda",
        }
        processed = process_engine_result("BDA", result, None, False)
        assert "completed" in processed["status_html"]
        assert "Accuracy:" not in processed["status_html"]
