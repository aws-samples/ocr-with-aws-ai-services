"""Tests for the Comparison Results table: columns, per-page maths and tooltips.

The table used to report "Samples Processed: 1" for a seven-page PDF and a single
dollar figure with no provenance. These tests pin the replacement: that Pages is
distinct from Documents, that per-page figures are derived rather than invented,
that an engine with no page count gets blank per-page cells instead of a division
by zero, and that hovering a cost shows the arithmetic behind that exact number.
"""

from typing import Any, Dict, List

import pytest

from sample_handler import build_batch_rows
from shared.cost_calculator import CostComponent, merge_cost_components
from shared.results_table import (
    COLUMNS,
    COST_COLUMNS,
    UNKNOWN,
    RunRow,
    build_run_rows,
    rows_to_dataframe,
    rows_to_html,
)

TEXTRACT_COMPONENTS: List[CostComponent] = [
    CostComponent(
        label="Textract StartDocumentTextDetection",
        formula="7 pages x $0.001500 per page (text detection only)",
        amount=0.0105,
        source="https://aws.amazon.com/textract/pricing/"),
    CostComponent(
        label="JSON structuring (us.anthropic.claude-sonnet-5)",
        formula="21,432 input tokens x $0.003000/1K = $0.064296",
        amount=0.064368,
        source="https://aws.amazon.com/bedrock/pricing/"),
]


def seven_page_result(**overrides: Any) -> Dict[str, Any]:
    """Build a processed result for a successful seven-page run.

    Args:
        **overrides: Fields to set or replace.

    Returns:
        Dict[str, Any]: A result dict shaped like process_engine_result's return.
    """
    result: Dict[str, Any] = {
        "time": 50.533,
        "cost": 0.074868,
        "accuracy": 67.31,
        "pages": 7,
        "cost_breakdown": TEXTRACT_COMPONENTS,
    }
    result.update(overrides)
    return result


class TestColumns:
    """The header set is the contract, for the screen and the saved record alike."""

    def test_pages_is_a_column_of_its_own(self) -> None:
        """The reported defect: nothing on screen said how many pages were read."""
        assert "Pages" in COLUMNS

    def test_documents_replaces_samples_processed(self) -> None:
        """"Samples" next to no page count is what read as "one page"."""
        assert "Documents" in COLUMNS
        assert "Samples Processed" not in COLUMNS

    def test_totals_and_per_page_figures_are_both_shown(self) -> None:
        """The total is what the run cost; the per-page figure predicts the next."""
        for column in (
            "Total Time (s)", "Time / Page (s)",
            "Total Cost ($)", "Cost / Page ($)",
        ):
            assert column in COLUMNS

    def test_accuracy_has_no_per_page_column(self) -> None:
        """Accuracy is already a proportion; dividing it by pages means nothing."""
        assert "Accuracy / Page (%)" not in COLUMNS
        assert "Accuracy (%)" in COLUMNS

    def test_the_dataframe_uses_the_same_columns_in_the_same_order(self) -> None:
        """The persisted record and the rendered table must not diverge."""
        frame = rows_to_dataframe(
            rows=build_run_rows(engine_results={"Textract": seven_page_result()}))
        assert list(frame.columns) == COLUMNS

    def test_an_empty_run_still_has_every_column(self) -> None:
        """Callers rely on the shape before any engine has finished."""
        assert list(rows_to_dataframe(rows=[]).columns) == COLUMNS


class TestPerPageArithmetic:
    """Per-page figures are derived from the totals, not reported separately."""

    def test_time_per_page_divides_the_total_by_the_pages(self) -> None:
        """50.533s over 7 pages is 7.219s a page."""
        row = build_run_rows(engine_results={"Textract": seven_page_result()})[0]
        assert row.time_per_page_s == pytest.approx(50.533 / 7)

    def test_cost_per_page_divides_the_total_by_the_pages(self) -> None:
        """The per-page cost is what makes two engines' rates comparable."""
        row = build_run_rows(engine_results={"Textract": seven_page_result()})[0]
        assert row.cost_per_page_usd == pytest.approx(0.074868 / 7)

    def test_the_rendered_figures_match_the_derived_ones(self) -> None:
        """Formatting must not round a figure into disagreeing with its own total."""
        frame = rows_to_dataframe(
            rows=build_run_rows(engine_results={"Textract": seven_page_result()}))
        record = frame.iloc[0]

        assert record["Pages"] == "7"
        assert record["Documents"] == "1"
        assert record["Total Time (s)"] == "50.533"
        assert record["Time / Page (s)"] == f"{50.533 / 7:.3f}"
        assert record["Total Cost ($)"] == "0.074868"
        assert record["Cost / Page ($)"] == f"{0.074868 / 7:.6f}"
        assert record["Accuracy (%)"] == "67.31"

    def test_a_single_page_run_has_equal_totals_and_per_page_figures(self) -> None:
        """The case that made the old table look right while being ambiguous."""
        row = build_run_rows(
            engine_results={"BDA": seven_page_result(pages=1)})[0]
        assert row.time_per_page_s == pytest.approx(row.total_time_s)
        assert row.cost_per_page_usd == pytest.approx(row.total_cost_usd)

    def test_the_document_count_is_not_the_page_count(self) -> None:
        """One seven-page PDF is one document. Conflating them caused the report."""
        row = build_run_rows(
            engine_results={"Textract": seven_page_result()}, document_count=1)[0]
        assert row.documents == 1
        assert row.pages == 7


class TestUnknownPageCount:
    """An engine that could not report pages gets blanks, not zeros or crashes."""

    def test_no_pages_means_no_per_page_figures(self) -> None:
        """0 pages means unknown. Dividing by it would raise."""
        row = build_run_rows(
            engine_results={"Bedrock": seven_page_result(pages=0)})[0]
        assert row.time_per_page_s is None
        assert row.cost_per_page_usd is None

    def test_the_unknown_cells_render_as_a_dash(self) -> None:
        """A blank cell reads as a bug; a dash reads as "not available"."""
        frame = rows_to_dataframe(
            rows=build_run_rows(engine_results={"Bedrock": seven_page_result(pages=0)}))
        record = frame.iloc[0]

        assert record["Pages"] == UNKNOWN
        assert record["Time / Page (s)"] == UNKNOWN
        assert record["Cost / Page ($)"] == UNKNOWN
        # The whole-document figures are still known and still shown.
        assert record["Total Time (s)"] == "50.533"

    def test_a_failed_engine_does_not_break_the_table(self) -> None:
        """A failed engine has no time, cost, pages or components at all."""
        html = rows_to_html(rows=build_run_rows(engine_results={
            "BDA": {"time": 1.2, "cost": 0.0, "accuracy": 0.0,
                    "pages": 0, "cost_breakdown": []}}))
        assert "BDA" in html
        assert UNKNOWN in html


class TestCostTooltips:
    """Hovering a cost shows how that exact number was reached."""

    def _tooltip(self) -> str:
        """Build the tooltip for the seven-page Textract row.

        Returns:
            str: The tooltip text.
        """
        return build_run_rows(
            engine_results={"Textract": seven_page_result()})[0].cost_tooltip()

    def test_every_component_is_named_with_its_own_formula_and_amount(self) -> None:
        """Two services billed this run, and the tooltip itemises both."""
        tooltip = self._tooltip()
        for component in TEXTRACT_COMPONENTS:
            assert component.label in tooltip
            assert component.formula in tooltip
            assert f"${component.amount:.6f}" in tooltip

    def test_the_total_is_stated(self) -> None:
        """The tooltip explains the figure in the cell it is attached to."""
        assert "Total: $0.074868" in self._tooltip()

    def test_the_per_page_division_is_shown(self) -> None:
        """The Cost / Page cell carries this tooltip too, so it explains itself."""
        assert f"$0.074868 / 7 pages = ${0.074868 / 7:.6f}" in self._tooltip()

    def test_the_published_rates_are_cited(self) -> None:
        """A figure that cannot be checked against a published price is not useful."""
        tooltip = self._tooltip()
        assert "https://aws.amazon.com/textract/pricing/" in tooltip
        assert "https://aws.amazon.com/bedrock/pricing/" in tooltip

    def test_a_repeated_source_is_cited_once(self) -> None:
        """BDA and its structuring step share a pricing page; listing it twice is noise."""
        row = RunRow(
            engine="BDA", documents=1, pages=7, total_time_s=52.8,
            total_cost_usd=0.12, accuracy_pct=88.46,
            cost_breakdown=[
                CostComponent(label="BDA standard output", formula="7 x $0.01",
                              amount=0.07, source="https://aws.amazon.com/bedrock/pricing/"),
                CostComponent(label="JSON structuring (m)", formula="tokens",
                              amount=0.05, source="https://aws.amazon.com/bedrock/pricing/"),
            ])
        assert row.cost_tooltip().count("https://aws.amazon.com/bedrock/pricing/") == 1

    def test_an_unbilled_run_says_so_rather_than_showing_an_empty_tooltip(self) -> None:
        """A failed engine is not billed an estimate, and the tooltip explains why."""
        row = RunRow(
            engine="BDA", documents=1, pages=0, total_time_s=1.2,
            total_cost_usd=0.0, accuracy_pct=0.0, cost_breakdown=[])
        assert "No charge was recorded" in row.cost_tooltip()

    def test_an_unknown_page_count_omits_the_per_page_line(self) -> None:
        """Without a page count there is no per-page figure to explain."""
        row = RunRow(
            engine="Bedrock", documents=1, pages=0, total_time_s=52.9,
            total_cost_usd=0.065, accuracy_pct=92.31,
            cost_breakdown=[TEXTRACT_COMPONENTS[1]])
        assert "Per page:" not in row.cost_tooltip()


class TestHtmlRendering:
    """The markup carries the tooltips and does not trust its inputs."""

    def test_both_cost_cells_carry_the_formula(self) -> None:
        """The total and the per-page figure are explained by the same arithmetic."""
        html = rows_to_html(
            rows=build_run_rows(engine_results={"Textract": seven_page_result()}))
        # One `has-formula` cell per cost column.
        assert html.count('has-formula') == len(COST_COLUMNS)

    def test_every_header_carries_its_column_tooltip(self) -> None:
        """Including the one saying Documents is not Pages."""
        html = rows_to_html(
            rows=build_run_rows(engine_results={"Textract": seven_page_result()}))
        assert "Documents processed, not pages" in html
        assert "as reported by the service itself" in html

    def test_the_accuracy_header_explains_why_it_is_not_per_page(self) -> None:
        """The one column deliberately left as a whole-document figure."""
        html = rows_to_html(
            rows=build_run_rows(engine_results={"Textract": seven_page_result()}))
        assert "a fraction of a fraction" in html

    def test_the_costs_are_labelled_as_estimates(self) -> None:
        """These are computed from published rates, not read off a bill."""
        html = rows_to_html(
            rows=build_run_rows(engine_results={"Textract": seven_page_result()}))
        assert "not billed amounts" in html

    def test_all_three_engines_render_one_row_each(self) -> None:
        """The table's whole purpose is the side-by-side comparison."""
        html = rows_to_html(rows=build_run_rows(engine_results={
            "Textract": seven_page_result(),
            "Bedrock": seven_page_result(cost=0.064988, accuracy=92.31),
            "BDA": seven_page_result(cost=0.12262, accuracy=88.46),
        }))
        assert html.count("<tr>") == 4  # one header row plus three engines

    def test_markup_in_a_value_is_escaped(self) -> None:
        """Engine names and formulas end up in attributes and in cell text."""
        html = rows_to_html(rows=[RunRow(
            engine='<script>alert("x")</script>',
            documents=1, pages=1, total_time_s=1.0, total_cost_usd=0.01,
            accuracy_pct=0.0,
            cost_breakdown=[CostComponent(
                label='" onmouseover="alert(1)', formula="f", amount=0.01,
                source="s")])]
        )
        assert "<script>" not in html
        assert 'onmouseover="alert(1)"' not in html

    def test_an_empty_run_invites_a_run_rather_than_showing_a_bare_header(self) -> None:
        """The panel is visible before the first run, so it says what to do."""
        html = rows_to_html(rows=[])
        assert "Run a document" in html
        assert "<table" not in html


def batch_totals(**overrides: Any) -> Dict[str, Dict[str, Any]]:
    """Build the batch accumulator for three documents of seven pages each.

    Args:
        **overrides: Per-engine entries to set or replace.

    Returns:
        Dict[str, Dict[str, Any]]: The structure process_all_samples accumulates.
    """
    totals: Dict[str, Dict[str, Any]] = {
        "Textract": {
            "count": 3,
            "total_pages": 21,
            "total_time": 150.0,
            "total_cost": 0.225,
            "accuracy_values": [60.0, 70.0, 80.0],
            "cost_components": TEXTRACT_COMPONENTS * 3,
        },
        "Bedrock": {
            "count": 0, "total_pages": 0, "total_time": 0, "total_cost": 0,
            "accuracy_values": [], "cost_components": [],
        },
        "BDA": {
            "count": 0, "total_pages": 0, "total_time": 0, "total_cost": 0,
            "accuracy_values": [], "cost_components": [],
        },
    }
    totals.update(overrides)
    return totals


class TestBatchRows:
    """Process All Samples feeds the same table, with the same column meanings."""

    def test_documents_and_pages_are_both_the_batch_totals(self) -> None:
        """Three seven-page PDFs are three documents and twenty-one pages."""
        row = build_batch_rows(results_by_engine=batch_totals())[0]
        assert row.documents == 3
        assert row.pages == 21

    def test_per_page_figures_divide_by_every_page_in_the_batch(self) -> None:
        """Averaging per document made a PDF batch incomparable with an image batch."""
        row = build_batch_rows(results_by_engine=batch_totals())[0]
        assert row.time_per_page_s == pytest.approx(150.0 / 21)
        assert row.cost_per_page_usd == pytest.approx(0.225 / 21)

    def test_accuracy_is_the_mean_over_the_documents(self) -> None:
        """Accuracy is a proportion per document, so it averages rather than sums."""
        row = build_batch_rows(results_by_engine=batch_totals())[0]
        assert row.accuracy_pct == pytest.approx(70.0)

    def test_an_engine_that_ran_nothing_is_left_out(self) -> None:
        """A row of zeros reads as "it ran and cost nothing", which is not the case."""
        rows = build_batch_rows(results_by_engine=batch_totals())
        assert [row.engine for row in rows] == ["Textract"]

    def test_no_engine_ran_yields_no_rows(self) -> None:
        """Before the first document completes there is nothing to compare."""
        empty = {name: {"count": 0, "total_pages": 0, "total_time": 0,
                        "total_cost": 0, "accuracy_values": [],
                        "cost_components": []}
                 for name in ("Textract", "Bedrock", "BDA")}
        assert build_batch_rows(results_by_engine=empty) == []

    def test_the_recurring_charges_are_merged_into_one_line_each(self) -> None:
        """Nine components over three documents are still two kinds of charge."""
        row = build_batch_rows(results_by_engine=batch_totals())[0]
        assert len(row.cost_breakdown) == len(TEXTRACT_COMPONENTS)

    def test_a_merged_charge_sums_every_document(self) -> None:
        """The tooltip's figures must add up to the total in the cell."""
        row = build_batch_rows(results_by_engine=batch_totals())[0]
        by_label = {component.label: component for component in row.cost_breakdown}
        assert by_label["Textract StartDocumentTextDetection"].amount == pytest.approx(
            0.0105 * 3)

    def test_a_merged_formula_says_it_was_summed_rather_than_quoting_one_document(
        self,
    ) -> None:
        """"7 pages x $0.0015" would understate a summed figure by a factor of three."""
        row = build_batch_rows(results_by_engine=batch_totals())[0]
        for component in row.cost_breakdown:
            assert "summed across 3 documents" in component.formula

    def test_merging_keeps_the_pricing_source(self) -> None:
        """The published rate is still what the merged figure should be checked against."""
        merged = merge_cost_components(
            components=TEXTRACT_COMPONENTS, document_count=1)
        assert [component.source for component in merged] == [
            component.source for component in TEXTRACT_COMPONENTS]

    def test_merging_nothing_yields_nothing(self) -> None:
        """An engine that failed on every document is not billed an estimate."""
        assert merge_cost_components(components=[], document_count=3) == []
