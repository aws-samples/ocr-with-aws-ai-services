"""Tests for the feature-aware Amazon Textract cost model.

Every expected figure below is taken from a worked example on
https://aws.amazon.com/textract/pricing/ so that a future rate change shows up as
a failing test naming the example it came from, rather than as a quietly wrong
number in the benchmark comparison table.
"""

import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.cost_calculator import (  # noqa: E402  (path setup must precede import)
    calculate_textract_analyze_cost,
    calculate_textract_cost,
    resolve_textract_feature_rate,
)


@pytest.mark.parametrize(
    "feature_types, expected_rate, source",
    [
        (["TABLES"], 0.015, "pricing example 3"),
        (["FORMS"], 0.05, "pricing example 3"),
        (["QUERIES"], 0.015, "pricing example 5"),
        (["SIGNATURES"], 0.0035, "pricing example 8"),
        (["LAYOUT"], 0.004, "pricing table"),
    ],
)
def test_single_feature_rates(feature_types: List[str], expected_rate: float, source: str) -> None:
    """Assert the per-page rate of each individual Textract feature.

    Args:
        feature_types (List[str]): The single-element feature list under test.
        expected_rate (float): Published per-page rate in USD.
        source (str): Which pricing-page example the rate comes from.

    Returns:
        None
    """
    rate, _ = resolve_textract_feature_rate(feature_types)
    assert rate == pytest.approx(expected_rate), f"rate disagrees with {source}"


def test_forms_and_tables_are_additive() -> None:
    """Forms + Tables has no bundle discount: $0.05 + $0.015 (pricing example 3).

    Returns:
        None
    """
    rate, _ = resolve_textract_feature_rate(["FORMS", "TABLES"])
    assert rate == pytest.approx(0.065)


def test_tables_and_queries_use_the_bundle_rate() -> None:
    """Tables + Queries bundles to $0.020, below the additive $0.030 (example 7).

    Returns:
        None
    """
    rate, breakdown = resolve_textract_feature_rate(["TABLES", "QUERIES"])
    assert rate == pytest.approx(0.020)
    assert any("bundle" in line for line in breakdown)


def test_forms_tables_and_queries_use_the_bundle_rate() -> None:
    """Forms + Tables + Queries bundles to $0.070, below additive $0.080 (example 6).

    Returns:
        None
    """
    rate, _ = resolve_textract_feature_rate(["FORMS", "TABLES", "QUERIES"])
    assert rate == pytest.approx(0.070)


def test_layout_is_free_alongside_tables() -> None:
    """Layout costs nothing when Tables is also requested (pricing example 16).

    Returns:
        None
    """
    rate, breakdown = resolve_textract_feature_rate(["LAYOUT", "TABLES"])
    assert rate == pytest.approx(0.015)
    assert any("no charge" in line for line in breakdown)


def test_layout_is_charged_without_tables() -> None:
    """Layout is billable when requested on its own alongside Forms.

    Returns:
        None
    """
    rate, _ = resolve_textract_feature_rate(["LAYOUT", "FORMS"])
    assert rate == pytest.approx(0.054)


def test_feature_order_does_not_change_the_rate() -> None:
    """The rate depends on the set of features, not the order they arrive in.

    Returns:
        None
    """
    forward, _ = resolve_textract_feature_rate(["FORMS", "TABLES", "QUERIES"])
    reverse, _ = resolve_textract_feature_rate(["QUERIES", "TABLES", "FORMS"])
    assert forward == pytest.approx(reverse)


def test_empty_feature_list_raises() -> None:
    """An empty feature list is a programming error, not a $0 AnalyzeDocument call.

    Returns:
        None
    """
    with pytest.raises(ValueError, match="at least one feature type"):
        resolve_textract_feature_rate([])


def test_unknown_feature_raises() -> None:
    """An unrecognised feature name must fail loudly rather than cost nothing.

    Returns:
        None
    """
    with pytest.raises(ValueError, match="Unknown Textract feature"):
        resolve_textract_feature_rate(["FORMS", "HANDWRITING"])


def test_analyze_cost_scales_with_page_count() -> None:
    """Total AnalyzeDocument cost is the per-page rate times the page count.

    Returns:
        None
    """
    _, cost = calculate_textract_analyze_cost(["FORMS", "TABLES"], page_count=7)
    assert cost == pytest.approx(0.065 * 7)


def test_detect_operations_keep_the_flat_rate() -> None:
    """Text-only detection stays at $1.50 per 1,000 pages for sync and async.

    Returns:
        None
    """
    _, sync_cost = calculate_textract_cost("textract_detect", page_count=2)
    _, async_cost = calculate_textract_cost("textract_async", page_count=2)
    assert sync_cost == pytest.approx(0.003)
    assert async_cost == pytest.approx(0.003)


def test_analyze_operations_route_to_the_feature_model() -> None:
    """Both analyze operation types are priced from the feature list.

    Returns:
        None
    """
    _, sync_cost = calculate_textract_cost(
        "textract_analyze", page_count=1, feature_types=["FORMS"]
    )
    _, async_cost = calculate_textract_cost(
        "textract_analyze_async", page_count=1, feature_types=["FORMS"]
    )
    assert sync_cost == pytest.approx(0.05)
    assert async_cost == pytest.approx(0.05)


def test_analyze_without_features_raises() -> None:
    """Routing an analyze operation with no features selected must raise.

    Returns:
        None
    """
    with pytest.raises(ValueError):
        calculate_textract_cost("textract_analyze", page_count=1, feature_types=[])


def test_forms_on_a_multipage_form_costs_more_than_text_detection() -> None:
    """Sanity check the benchmark's headline trade-off on a 7-page PFL packet.

    Returns:
        None
    """
    _, text_only = calculate_textract_cost("textract_async", page_count=7)
    _, with_forms = calculate_textract_cost(
        "textract_analyze_async", page_count=7, feature_types=["FORMS", "TABLES"]
    )
    assert with_forms > text_only * 40
