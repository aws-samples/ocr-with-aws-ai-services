"""
The Compare tab renders ground truth against every engine at once

The tab used to require picking one engine before it would show anything, which made
the one question the app exists to answer - which engine got this field right? - the
one question it could not display. These tests pin the side-by-side table: a column per
engine that ran, a per-cell verdict rather than a per-row one, and a filter that
narrows the table instead of gating it.
"""

import os
import re
import sys
from typing import Any, Dict, List

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.comparison_utils import (  # noqa: E402
    ALL_ENGINES_LABEL,
    ENGINE_FILTER_CHOICES,
    ENGINE_NAMES,
    create_comparison_view,
    create_multi_engine_diff_view,
)

TRUTH: Dict[str, Any] = {
    "claimant": {
        "name": "Ada Lovelace",
        "dateOfHire": "2019-03-04",
    },
    "policyNumber": "SYN-3319864",
}

# Textract gets everything; Bedrock omits the hire date; BDA reads the date but mangles
# the name and the policy number. No engine is right about all three and no field is
# agreed on by all three - so neither a row-level verdict nor a single column would do.
EXTRACTED: Dict[str, Dict[str, Any]] = {
    "Textract": {
        "claimant": {"name": "Ada Lovelace", "dateOfHire": "2019-03-04"},
        "policyNumber": "SYN-3319864",
    },
    "Bedrock": {
        "claimant": {"name": "Ada Lovelace"},
        "policyNumber": "SYN-3319864",
    },
    "BDA": {
        "claimant": {"name": "Ada Loveless", "dateOfHire": "2019-03-04"},
        "policyNumber": "5304179OO",
    },
}


def header_cells(*, html_output: str) -> List[str]:
    """
    Read the table's header labels in document order

    Args:
        html_output (str): Rendered comparison HTML.

    Returns:
        List[str]: The text of every <th>.
    """
    return re.findall(r"<th>(.*?)</th>", html_output)


def row_for(*, html_output: str, field_name: str) -> str:
    """
    Return the single <tr> whose field column holds the given name

    Args:
        html_output (str): Rendered comparison HTML.
        field_name (str): Leaf field name, as shown in the first column.

    Returns:
        str: The row's markup.

    Raises:
        AssertionError: If no row, or more than one row, names the field. Either would
            make any assertion about "the" row meaningless.
    """
    rows = [
        row for row in re.findall(r"<tr.*?</tr>", html_output, flags=re.DOTALL)
        if f"<td>{field_name}</td>" in row
    ]

    assert len(rows) == 1, (
        f"expected exactly one row for {field_name!r}, found {len(rows)}")

    return rows[0]


class TestColumns:
    """One column per engine that produced a result, in a stable order"""

    def test_every_engine_gets_a_column(self) -> None:
        """The default view is ground truth against all three engines"""
        html_output = create_comparison_view(
            truth_data=TRUTH, engine_json_by_name=EXTRACTED)

        assert header_cells(html_output=html_output) == [
            "Field", "Expected", "Textract", "Bedrock", "BDA"]

    def test_column_order_follows_engine_names_not_completion_order(self) -> None:
        """
        Columns do not move when a different engine finishes first

        The processing generator re-renders this table as each engine completes, so the
        dict it passes is in completion order. Columns that reordered mid-run would be
        unreadable.
        """
        reversed_results = {
            name: EXTRACTED[name] for name in reversed(ENGINE_NAMES)}

        html_output = create_comparison_view(
            truth_data=TRUTH, engine_json_by_name=reversed_results)

        assert header_cells(html_output=html_output)[2:] == list(ENGINE_NAMES)

    def test_an_engine_that_did_not_run_gets_no_column(self) -> None:
        """One engine's worth of results gives one engine column, not three"""
        html_output = create_comparison_view(
            truth_data=TRUTH,
            engine_json_by_name={
                "Textract": EXTRACTED["Textract"], "Bedrock": None, "BDA": {}})

        assert header_cells(html_output=html_output) == [
            "Field", "Expected", "Textract", "Match"]

    def test_one_engine_is_headed_with_its_own_name(self) -> None:
        """
        A single-engine table says whose values it is showing

        The column used to be headed "Extracted" whichever engine was selected, so a
        screenshot of the tab did not record which engine it was of.
        """
        html_output = create_comparison_view(
            truth_data=TRUTH, engine_json_by_name=EXTRACTED,
            engine_filter="BDA")

        assert header_cells(html_output=html_output) == [
            "Field", "Expected", "BDA", "Match"]


class TestPerCellVerdict:
    """The verdict is per engine cell, because engines disagree about the same field"""

    def test_a_field_two_engines_disagree_on_carries_both_verdicts(self) -> None:
        """
        One row holds a match and a mismatch at once

        This is the case a row-level tint cannot express, and the whole reason the
        multi-engine table tints cells instead of rows.
        """
        row = row_for(
            html_output=create_multi_engine_diff_view(
                truth_data=TRUTH, engine_json_by_name=EXTRACTED),
            field_name="name")

        assert row.count("class='cell-match'") == 2, row      # Textract, Bedrock
        assert row.count("class='cell-mismatch'") == 1, row   # BDA

    def test_a_value_one_engine_omitted_is_marked_missing_in_its_cell(self) -> None:
        """Only the engine that omitted the field is tinted as a miss"""
        row = row_for(
            html_output=create_multi_engine_diff_view(
                truth_data=TRUTH, engine_json_by_name=EXTRACTED),
            field_name="dateOfHire")

        assert row.count("class='cell-match'") == 2       # Textract, BDA
        assert row.count("class='cell-mismatch'") == 1    # Bedrock omitted it
        assert "MISSING" in row

    def test_the_verdict_is_also_a_glyph(self) -> None:
        """
        Colour is not the only signal

        Two 13%-alpha tints are not reliably distinguishable, so each cell states its
        verdict as a mark as well.
        """
        html_output = create_multi_engine_diff_view(
            truth_data=TRUTH, engine_json_by_name=EXTRACTED)

        assert "<span class='cell-mark'>✓</span>" in html_output
        assert "<span class='cell-mark'>✗</span>" in html_output

    def test_rows_carry_no_row_level_verdict_class(self) -> None:
        """
        The multi-engine table does not reuse the single-engine row classes

        A row tint here would have to pick one engine's verdict for the whole row.
        """
        html_output = create_multi_engine_diff_view(
            truth_data=TRUTH, engine_json_by_name=EXTRACTED)

        assert "<tr class='match'>" not in html_output
        assert "<tr class='mismatch'>" not in html_output

    def test_the_expected_column_is_not_tinted(self) -> None:
        """Ground truth is neither a match nor a miss, so it stays plain"""
        row = row_for(
            html_output=create_multi_engine_diff_view(
                truth_data=TRUTH, engine_json_by_name=EXTRACTED),
            field_name="policyNumber")

        # Field, Expected, then one cell per engine - so exactly three tinted cells.
        assert row.count("<td class='cell-") == len(ENGINE_NAMES)


class TestFieldCoverage:
    """Every field reachable from any engine's evaluation is on screen"""

    # A mixed list of a dict and a scalar. get_detailed_accuracy walks the truth, but
    # which branch it takes depends on the EXTRACTED value: an engine that returned a
    # list is scored by compare_lists, which treats the mixed list as scalars and
    # reports 'items[0]'; an engine that returned a string instead falls to
    # add_missing_fields, which recurses and reports 'items[0].code'. The two engines
    # therefore describe different field paths for the same document.
    MIXED_TRUTH: Dict[str, Any] = {"items": [{"code": "A1"}, "loose"]}
    RETURNED_A_LIST: Dict[str, Any] = {"items": [{"code": "A1"}, "loose"]}
    RETURNED_A_STRING: Dict[str, Any] = {"items": "A1, loose"}

    @pytest.mark.parametrize("first_engine_returned_a_list", [True, False])
    def test_paths_only_one_engine_reports_still_appear(
            self, first_engine_returned_a_list: bool) -> None:
        """
        The rows are the union of the engines' paths, not the first engine's

        Rendering only the first engine's paths would silently drop fields depending on
        which engine happened to be selected - and a dropped row reads as agreement.
        """
        left, right = (
            (self.RETURNED_A_LIST, self.RETURNED_A_STRING)
            if first_engine_returned_a_list
            else (self.RETURNED_A_STRING, self.RETURNED_A_LIST))

        html_output = create_multi_engine_diff_view(
            truth_data=self.MIXED_TRUTH,
            engine_json_by_name={"Textract": left, "BDA": right})

        assert "<td>items[0]</td>" in html_output   # reported by the list engine only
        assert "<td>code</td>" in html_output       # reported by the string engine only

    def test_an_engine_silent_about_a_path_says_so_rather_than_matching(self) -> None:
        """
        A path one engine never evaluated is called out, not left blank

        An empty cell would read as "no difference"; NOT REPORTED distinguishes it from
        MISSING, which means the engine was scored on the field and returned nothing.
        """
        html_output = create_multi_engine_diff_view(
            truth_data=self.MIXED_TRUTH,
            engine_json_by_name={
                "Textract": self.RETURNED_A_LIST, "BDA": self.RETURNED_A_STRING})

        assert "NOT REPORTED" in html_output

    def test_a_parent_path_header_spans_every_column(self) -> None:
        """
        The section header stretches across the whole table

        A colspan left at the single-engine table's 4 would stop short of the last
        engine's column and break the row.
        """
        html_output = create_multi_engine_diff_view(
            truth_data=TRUTH, engine_json_by_name=EXTRACTED)

        assert f"colspan='{2 + len(ENGINE_NAMES)}'" in html_output
        assert "<b>claimant</b>" in html_output


class TestFilter:
    """The dropdown narrows the table rather than gating it"""

    def test_all_engines_is_the_first_choice(self) -> None:
        """Nothing has to be selected for the tab to show something"""
        assert ENGINE_FILTER_CHOICES[0] == ALL_ENGINES_LABEL
        assert list(ENGINE_FILTER_CHOICES[1:]) == list(ENGINE_NAMES)

    def test_all_engines_is_the_default_argument(self) -> None:
        """A caller that does not pass a filter gets the side-by-side table"""
        assert create_comparison_view(
            truth_data=TRUTH, engine_json_by_name=EXTRACTED
        ) == create_comparison_view(
            truth_data=TRUTH, engine_json_by_name=EXTRACTED,
            engine_filter=ALL_ENGINES_LABEL)

    def test_filtering_to_an_engine_drops_the_other_columns(self) -> None:
        """Selecting Bedrock leaves Bedrock's values only"""
        headers = header_cells(
            html_output=create_comparison_view(
                truth_data=TRUTH, engine_json_by_name=EXTRACTED,
                engine_filter="Bedrock"))

        assert "Bedrock" in headers
        assert "Textract" not in headers
        assert "BDA" not in headers

    def test_filtering_to_an_engine_with_no_result_says_so(self) -> None:
        """
        An engine that was not run reports that, rather than rendering empty

        An empty table is indistinguishable from an engine that extracted nothing.
        """
        html_output = create_comparison_view(
            truth_data=TRUTH,
            engine_json_by_name={"Textract": EXTRACTED["Textract"]},
            engine_filter="BDA")

        assert "<table" not in html_output
        assert "BDA" in html_output

    def test_an_unknown_filter_raises(self) -> None:
        """
        A filter value that is not a choice fails loudly

        Falling through to "all engines" would misreport whose numbers are on screen.
        """
        with pytest.raises(ValueError, match="Unknown engine filter"):
            create_comparison_view(
                truth_data=TRUTH, engine_json_by_name=EXTRACTED,
                engine_filter="Nova")


class TestNothingToCompare:
    """The empty states explain themselves"""

    def test_no_ground_truth_is_explained(self) -> None:
        """Without truth there is no comparison to render"""
        html_output = create_comparison_view(
            truth_data=None, engine_json_by_name=EXTRACTED)

        assert "<table" not in html_output
        assert "ground truth" in html_output

    def test_no_engine_results_is_explained(self) -> None:
        """Truth without any extraction is also an empty state"""
        html_output = create_comparison_view(
            truth_data=TRUTH,
            engine_json_by_name={name: None for name in ENGINE_NAMES})

        assert "<table" not in html_output
        assert "comparison" in html_output


class TestMarkupUsesTheStylesheet:
    """The multi-engine table obeys the same rules as the single-engine one"""

    def test_no_inline_style_or_style_block(self) -> None:
        """
        Colour decisions live in APP_CSS and nowhere else

        Every readability defect this stylesheet replaced was a hardcoded colour in an
        inline style attribute that assumed the background it would land on.
        """
        html_output = create_multi_engine_diff_view(
            truth_data=TRUTH, engine_json_by_name=EXTRACTED)

        assert "<style>" not in html_output
        assert "style=" not in html_output

    def test_values_are_wrapped_so_the_height_cap_applies(self) -> None:
        """
        Every value sits in a .value-box

        max-height is ignored on a <td>, so a value rendered straight into the cell is
        uncapped - and claim-form values run to paragraphs.
        """
        html_output = create_multi_engine_diff_view(
            truth_data=TRUTH, engine_json_by_name=EXTRACTED)

        value_cells = re.findall(r"<td class='cell-\w+'>(.*?)</td>", html_output)

        assert value_cells
        for cell in value_cells:
            assert cell.startswith("<div class='value-box'>"), cell

    def test_the_summary_bar_names_every_engine(self) -> None:
        """
        The headline figures are per engine

        A single total would say nothing about which engine earned it.
        """
        html_output = create_multi_engine_diff_view(
            truth_data=TRUTH, engine_json_by_name=EXTRACTED)
        summary = html_output.split("<table")[0]

        for name in ENGINE_NAMES:
            assert name in summary
