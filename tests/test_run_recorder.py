"""Tests for persisting a run, so repeated runs can be compared afterwards.

A run used to exist only on screen: the comparison table was rebuilt on every
click and the previous run's figures were gone. These tests pin the record that
replaces that - that it names the document and the timestamp, that it carries the
cost arithmetic and the configuration that produced the figures, that the history
file accumulates rather than being rewritten, and that a run which cannot be saved
says so instead of vanishing.

`recorded_at` is injected throughout rather than read from the clock, which is why
an exact filename can be asserted here.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pytest
from PIL import Image

import sample_handler
from shared.cost_calculator import CostComponent
from shared.results_table import RunRow
from shared.run_recorder import HISTORY_FILENAME, record_run, save_run_record

RECORDED_AT = datetime(2026, 8, 7, 14, 22, 3)
LATER = datetime(2026, 8, 7, 15, 4, 19)

DOCUMENT = "claims/PFL/form - 4410772"

TEXTRACT_ROW = RunRow(
    engine="Textract",
    documents=1,
    pages=7,
    total_time_s=50.533,
    total_cost_usd=0.074868,
    accuracy_pct=67.31,
    cost_breakdown=[
        CostComponent(
            label="Textract StartDocumentTextDetection",
            formula="7 pages x $0.001500 per page (text detection only)",
            amount=0.0105,
            source="https://aws.amazon.com/textract/pricing/"),
        CostComponent(
            label="JSON structuring (us.anthropic.claude-sonnet-5)",
            formula="21,432 input tokens x $0.003000/1K",
            amount=0.064368,
            source="https://aws.amazon.com/bedrock/pricing/"),
    ])

BDA_ROW = RunRow(
    engine="BDA",
    documents=1,
    pages=7,
    total_time_s=52.838,
    total_cost_usd=0.13262,
    accuracy_pct=88.46,
    cost_breakdown=[
        CostComponent(
            label="BDA standard output",
            formula="7 pages x $0.010000 per document",
            amount=0.07,
            source="https://aws.amazon.com/bedrock/pricing/"),
    ])

# A run where the page count could not be established - Bedrock on an unreadable
# page tree. Its per-page figures must be absent, not zero.
UNKNOWN_PAGES_ROW = RunRow(
    engine="Bedrock",
    documents=1,
    pages=0,
    total_time_s=52.911,
    total_cost_usd=0.064988,
    accuracy_pct=92.31,
    cost_breakdown=[])

CONFIGURATION: Dict[str, Any] = {
    "engines": ["BDA", "Textract"],
    "bedrock_model": "Claude Sonnet 5",
    "document_type": "generic",
    "structured_output": True,
    "bda_blueprint": False,
}


def write_run(*, results_dir: Path, **overrides: Any):
    """Record a two-engine run into the given directory.

    Args:
        results_dir: Directory to write into.
        **overrides: Arguments to replace, e.g. rows or recorded_at.

    Returns:
        RunRecord: The paths written.
    """
    arguments: Dict[str, Any] = {
        "document_name": DOCUMENT,
        "rows": [TEXTRACT_ROW, BDA_ROW],
        "recorded_at": RECORDED_AT,
        "total_time_s": 53.9,
        "configuration": CONFIGURATION,
        "ground_truth_available": True,
        "results_dir": results_dir,
    }
    arguments.update(overrides)
    return record_run(**arguments)


def read_record(*, path: Path) -> Dict[str, Any]:
    """Parse a written record.

    Args:
        path: Path to the record JSON.

    Returns:
        Dict[str, Any]: The parsed record.
    """
    return json.loads(path.read_text())


def read_history(*, results_dir: Path) -> List[Dict[str, Any]]:
    """Parse every line of the history file.

    Args:
        results_dir: Directory the history was written into.

    Returns:
        List[Dict[str, Any]]: One dict per line, in file order.
    """
    lines = (results_dir / HISTORY_FILENAME).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


class TestWhereTheRecordGoes:
    """The filename has to identify the run without opening it."""

    def test_the_filename_carries_the_timestamp_and_the_document(self, tmp_path) -> None:
        """`ls results/` is the index, so the name must say which run this was."""
        run_record = write_run(results_dir=tmp_path)
        assert run_record.record_path.name == "20260807-142203-form---4410772.json"

    def test_the_timestamp_leads_so_runs_sort_chronologically(self, tmp_path) -> None:
        """Comparing runs starts with putting them in order."""
        first = write_run(results_dir=tmp_path)
        second = write_run(results_dir=tmp_path, recorded_at=LATER)
        assert sorted([second.record_path.name, first.record_path.name]) == [
            first.record_path.name, second.record_path.name]

    def test_a_sub_foldered_document_stays_inside_the_results_directory(
        self, tmp_path
    ) -> None:
        """A grouped sample label carries a slash, which must not escape the dir."""
        run_record = write_run(results_dir=tmp_path)
        assert run_record.record_path.parent == tmp_path
        assert not (tmp_path / "PFL").exists()

    def test_the_results_directory_is_created(self, tmp_path) -> None:
        """The first run of a fresh clone must not fail for want of a directory."""
        results_dir = tmp_path / "results"
        run_record = write_run(results_dir=results_dir)
        assert run_record.record_path.exists()

    def test_a_document_name_with_nothing_usable_raises(self, tmp_path) -> None:
        """A record named after nothing cannot be matched to a document later."""
        with pytest.raises(ValueError, match="record filename"):
            write_run(results_dir=tmp_path, document_name="///")


class TestWhatTheRecordHolds:
    """Enough to answer "why did that cost that much?" without the app running."""

    def test_the_document_and_the_time_are_recorded(self, tmp_path) -> None:
        """A record of figures with no document name compares nothing."""
        record = read_record(path=write_run(results_dir=tmp_path).record_path)
        assert record["document"] == DOCUMENT
        assert record["recorded_at"] == "2026-08-07T14:22:03"
        assert record["total_time_s"] == 53.9

    def test_every_engine_that_ran_is_recorded(self, tmp_path) -> None:
        """The side-by-side comparison is the point of the run."""
        record = read_record(path=write_run(results_dir=tmp_path).record_path)
        assert [entry["engine"] for entry in record["engines"]] == ["Textract", "BDA"]

    def test_the_total_is_the_sum_of_the_engines(self, tmp_path) -> None:
        """The banner reports this figure, so the record must agree with it."""
        record = read_record(path=write_run(results_dir=tmp_path).record_path)
        assert record["total_cost_usd"] == pytest.approx(0.074868 + 0.13262)

    def test_pages_and_per_page_figures_are_recorded(self, tmp_path) -> None:
        """The reported gap: nothing persisted said how many pages a figure covered."""
        record = read_record(path=write_run(results_dir=tmp_path).record_path)
        textract = record["engines"][0]

        assert textract["pages"] == 7
        assert textract["time_per_page_s"] == pytest.approx(50.533 / 7)
        assert textract["cost_per_page_usd"] == pytest.approx(0.074868 / 7)

    def test_an_unknown_page_count_records_null_not_zero(self, tmp_path) -> None:
        """Zero would read as free. Null reads as "we could not tell"."""
        run_record = write_run(results_dir=tmp_path, rows=[UNKNOWN_PAGES_ROW])
        engine = read_record(path=run_record.record_path)["engines"][0]

        assert engine["pages"] == 0
        assert engine["time_per_page_s"] is None
        assert engine["cost_per_page_usd"] is None

    def test_the_cost_breakdown_is_recorded_with_its_formulas(self, tmp_path) -> None:
        """The tooltip is on screen only; the record is what survives the run."""
        record = read_record(path=write_run(results_dir=tmp_path).record_path)
        breakdown = record["engines"][0]["cost_breakdown"]

        assert len(breakdown) == 2
        assert breakdown[0]["label"] == "Textract StartDocumentTextDetection"
        assert "7 pages x $0.001500 per page" in breakdown[0]["formula"]
        assert breakdown[0]["amount_usd"] == pytest.approx(0.0105)
        assert breakdown[0]["source"].startswith("https://")

    def test_the_breakdown_sums_to_the_engine_total(self, tmp_path) -> None:
        """A breakdown that does not add up to the total explains the wrong number."""
        record = read_record(path=write_run(results_dir=tmp_path).record_path)
        bda = record["engines"][1]

        assert sum(charge["amount_usd"] for charge in bda["cost_breakdown"]) == (
            pytest.approx(0.07))

    def test_the_configuration_is_recorded(self, tmp_path) -> None:
        """Two runs of one document are only comparable if the settings are known."""
        record = read_record(path=write_run(results_dir=tmp_path).record_path)
        assert record["configuration"] == CONFIGURATION

    def test_whether_accuracy_could_be_scored_is_recorded(self, tmp_path) -> None:
        """Without this an accuracy of 0.0 could mean "no ground truth"."""
        with_truth = read_record(path=write_run(results_dir=tmp_path).record_path)
        assert with_truth["ground_truth_available"] is True

        without = read_record(path=write_run(
            results_dir=tmp_path, recorded_at=LATER,
            ground_truth_available=False).record_path)
        assert without["ground_truth_available"] is False

    def test_a_run_with_no_engines_raises_rather_than_writing_an_empty_record(
        self, tmp_path
    ) -> None:
        """An empty record looks like a run that cost nothing."""
        with pytest.raises(ValueError, match="no engine results"):
            write_run(results_dir=tmp_path, rows=[])

        assert list(tmp_path.iterdir()) == []


class TestTheHistoryFile:
    """One flat line per observation, appended, so jq can compare runs."""

    def test_one_line_per_engine(self, tmp_path) -> None:
        """A line per engine is what makes "how has BDA moved?" a filter."""
        write_run(results_dir=tmp_path)
        assert [entry["engine"] for entry in read_history(results_dir=tmp_path)] == [
            "Textract", "BDA"]

    def test_a_second_run_appends_rather_than_replacing(self, tmp_path) -> None:
        """The whole purpose is running repeatedly and comparing."""
        write_run(results_dir=tmp_path)
        write_run(results_dir=tmp_path, recorded_at=LATER)

        history = read_history(results_dir=tmp_path)
        assert len(history) == 4
        assert {entry["recorded_at"] for entry in history} == {
            "2026-08-07T14:22:03", "2026-08-07T15:04:19"}

    def test_each_line_carries_the_run_and_the_document(self, tmp_path) -> None:
        """A line has to stand alone to be usable without the nested record."""
        write_run(results_dir=tmp_path)
        entry = read_history(results_dir=tmp_path)[0]

        assert entry["recorded_at"] == "2026-08-07T14:22:03"
        assert entry["document"] == DOCUMENT

    def test_each_line_carries_the_comparable_figures(self, tmp_path) -> None:
        """These are the columns a comparison across runs is drawn from."""
        write_run(results_dir=tmp_path)
        entry = read_history(results_dir=tmp_path)[0]

        assert entry["pages"] == 7
        assert entry["total_cost_usd"] == pytest.approx(0.074868)
        assert entry["cost_per_page_usd"] == pytest.approx(0.074868 / 7)
        assert entry["accuracy_pct"] == pytest.approx(67.31)

    def test_a_line_points_at_its_full_record(self, tmp_path) -> None:
        """The line is a summary; the formulas live in the record it names."""
        run_record = write_run(results_dir=tmp_path)
        entry = read_history(results_dir=tmp_path)[0]

        assert entry["record"] == run_record.record_path.name
        assert (tmp_path / entry["record"]).exists()

    def test_the_breakdown_is_left_out_of_the_line(self, tmp_path) -> None:
        """A nested list in a JSONL line defeats the point of the flat format."""
        write_run(results_dir=tmp_path)
        assert "cost_breakdown" not in read_history(results_dir=tmp_path)[0]

    def test_every_line_is_valid_json_on_its_own(self, tmp_path) -> None:
        """A missing newline between runs would merge two observations into one."""
        write_run(results_dir=tmp_path)
        write_run(results_dir=tmp_path, recorded_at=LATER)

        for line in (tmp_path / HISTORY_FILENAME).read_text().splitlines():
            assert json.loads(line)


class TestABatchRunIsRecordedToo:
    """"Process All Samples" is the button that does the most work.

    It writes its own `results/run_<timestamp>/` tree, which is per-sample and
    per-engine. That is not the same thing as an entry in history.jsonl: comparing
    the batch against a later batch meant walking two directories. These tests pin
    that a batch also records itself through the shared path, so both kinds of run
    land in one comparable history.

    The engines are replaced with a stand-in, so this makes no AWS calls.
    """

    def run_a_batch(self, *, tmp_path, monkeypatch) -> str:
        """Run a one-sample batch with a stand-in Textract engine.

        Args:
            tmp_path: pytest temporary directory, used as the working directory.
            monkeypatch: pytest monkeypatch fixture.

        Returns:
            str: The final status banner.
        """
        # One sample bundle: a directory holding exactly one document.
        bundle_dir = tmp_path / "sample" / "receipt"
        bundle_dir.mkdir(parents=True)
        Image.new("RGB", (8, 8), "white").save(bundle_dir / "receipt.png")

        # results/ and sample/ are both resolved against the working directory.
        monkeypatch.chdir(tmp_path)

        class StandInTextract:
            """A Textract engine that returns a fixed three-page result."""

            def process_image(self, image, options=None) -> Dict[str, Any]:
                """Return a result shaped as TextractEngine's success path returns.

                Args:
                    image: Ignored.
                    options: Ignored.

                Returns:
                    Dict[str, Any]: A three-page text-detection result.
                """
                return {
                    "text": "TOTAL 100.00",
                    "json": {"total": "100.00"},
                    "image": None,
                    "process_time": 4.5,
                    "token_usage": None,
                    "pages": 3,
                    "operation_type": "textract_detect",
                    "feature_types": None,
                }

        monkeypatch.setattr(sample_handler, "TextractEngine", StandInTextract)

        payloads = list(sample_handler.process_all_samples(
            True, False, False, "Claude Sonnet 5"))
        return payloads[-1][0]

    def test_the_batch_appears_in_the_history(self, tmp_path, monkeypatch) -> None:
        """A batch and a single run have to be comparable in one file."""
        self.run_a_batch(tmp_path=tmp_path, monkeypatch=monkeypatch)
        history = read_history(results_dir=tmp_path / "results")

        assert [entry["engine"] for entry in history] == ["Textract"]
        assert history[0]["pages"] == 3

    def test_the_record_marks_itself_as_a_batch_and_counts_the_samples(
        self, tmp_path, monkeypatch
    ) -> None:
        """Otherwise a one-sample batch is indistinguishable from a single run."""
        self.run_a_batch(tmp_path=tmp_path, monkeypatch=monkeypatch)
        record_path = next((tmp_path / "results").glob("*-all-samples-1.json"))
        record = read_record(path=record_path)

        assert record["configuration"]["batch"] is True
        assert record["configuration"]["samples"] == 1
        assert record["configuration"]["engines"] == ["Textract"]

    def test_the_batch_record_names_its_own_run_directory(
        self, tmp_path, monkeypatch
    ) -> None:
        """The per-sample outputs are the other half of the batch's results."""
        self.run_a_batch(tmp_path=tmp_path, monkeypatch=monkeypatch)
        record_path = next((tmp_path / "results").glob("*-all-samples-1.json"))
        run_directory = read_record(path=record_path)["configuration"]["run_directory"]

        assert (tmp_path / run_directory / "summary.json").exists()

    def test_the_banner_names_the_saved_record(self, tmp_path, monkeypatch) -> None:
        """A file written and never mentioned is a file nobody reads."""
        banner = self.run_a_batch(tmp_path=tmp_path, monkeypatch=monkeypatch)

        assert "saved to" in banner
        assert "all-samples-1.json" in banner

    def test_a_failed_batch_attempt_is_not_recorded_as_a_run(
        self, tmp_path, monkeypatch
    ) -> None:
        """The batch summary records the failure, but history gets no fake run."""
        bundle = tmp_path / "sample" / "receipt"
        bundle.mkdir(parents=True)
        Image.new("RGB", (8, 8), "white").save(bundle / "receipt.png")
        monkeypatch.chdir(tmp_path)

        class FailingTextract:
            def process_image(self, image, options=None) -> Dict[str, Any]:
                return {
                    "text": "Textract Error: access denied",
                    "json": None,
                    "image": None,
                    "process_time": 1.5,
                    "operation_type": "error",
                    "pages": 0,
                }

        monkeypatch.setattr(sample_handler, "TextractEngine", FailingTextract)

        payloads = list(sample_handler.process_all_samples(
            True, False, False, "Claude Sonnet 5"))
        final_status = payloads[-1][0]
        run_dir = next((tmp_path / "results").glob("run_*"))
        summary = json.loads((run_dir / "summary.json").read_text())

        assert "ocr-banner--error" in final_status
        assert summary["results"]["Textract"]["documents_processed"] == 0
        assert summary["results"]["Textract"]["documents_failed"] == 1
        assert not (tmp_path / "results" / HISTORY_FILENAME).exists()
        assert not list((tmp_path / "results").glob("*-all-samples-*.json"))


class TestReportingTheSaveToTheUser:
    """A run that was not saved must say so; the figures are gone otherwise."""

    def test_a_saved_run_names_the_file_in_the_banner(self, tmp_path, monkeypatch) -> None:
        """The path is the only way to find the record afterwards."""
        monkeypatch.chdir(tmp_path)

        note = save_run_record(
            document_name=DOCUMENT, rows=[TEXTRACT_ROW], total_time_s=53.9,
            configuration=CONFIGURATION, ground_truth_available=True,
            recorded_at=RECORDED_AT)

        assert "saved to" in note
        assert "20260807-142203-form---4410772.json" in note
        assert (tmp_path / "results"
                / "20260807-142203-form---4410772.json").exists()

    def test_a_failed_save_reports_it_instead_of_raising(self, tmp_path, monkeypatch) -> None:
        """A paid extraction must not be discarded because a write failed."""
        monkeypatch.chdir(tmp_path)

        note = save_run_record(
            document_name=DOCUMENT, rows=[], total_time_s=53.9,
            configuration=CONFIGURATION, ground_truth_available=True,
            recorded_at=RECORDED_AT)

        assert "not saved" in note

    def test_a_failed_save_is_logged_as_an_error(self, tmp_path, monkeypatch, caplog) -> None:
        """The banner is transient; the log is what a later diagnosis reads."""
        monkeypatch.chdir(tmp_path)

        with caplog.at_level("ERROR"):
            save_run_record(
                document_name=DOCUMENT, rows=[], total_time_s=53.9,
                configuration=CONFIGURATION, ground_truth_available=True,
                recorded_at=RECORDED_AT)

        assert "Could not save this run's results" in caplog.text

    def test_an_unwritable_results_directory_is_reported_not_raised(
        self, tmp_path, monkeypatch
    ) -> None:
        """A read-only working directory is a configuration problem, not a crash."""
        monkeypatch.chdir(tmp_path)
        # A plain file where the results directory should be: mkdir then fails.
        (tmp_path / "results").write_text("not a directory")

        note = save_run_record(
            document_name=DOCUMENT, rows=[TEXTRACT_ROW], total_time_s=53.9,
            configuration=CONFIGURATION, ground_truth_available=True,
            recorded_at=RECORDED_AT)

        assert "not saved" in note
