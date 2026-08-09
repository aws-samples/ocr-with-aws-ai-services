"""
Tests for grouping page text into LLM calls and merging the structured results

The behaviour these protect is the reason multi-page PDFs scored 0%: structured
output has to end up keyed by schema field, not by page number, and a field split
across a page boundary has to survive being seen in two different calls.
"""

import json
import os

import pytest

from shared.json_merge import (
    group_pages_for_structuring,
    is_empty_value,
    merge_structured_results,
)

# Real, tracked ground truth from a multi-section claim form, so the merge is checked
# against a document shape that actually occurs rather than a hand-made stub. The
# synthetic PFL bundle is used because it ships in the repository, so this test is
# runnable on a fresh clone.
TRUTH_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sample", "pfl-synthetic", "truth.json",
)


# --- is_empty_value ---------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [None, "", "null", "None", {}, []],
)
def test_empty_values_are_empty(value) -> None:
    """Null-like scalars and empty containers count as nothing extracted"""
    assert is_empty_value(value) is True


@pytest.mark.parametrize(
    "value",
    ["Smith", 0, 0.0, False, True, {"a": 1}, [1], "0", "false"],
)
def test_real_values_are_not_empty(value) -> None:
    """
    Falsy-but-real values are not empty

    0 and False are legitimate extracted values - a zero dollar amount or an
    unchecked box - and must not be overwritten by a later chunk.
    """
    assert is_empty_value(value) is False


# --- group_pages_for_structuring -------------------------------------------


def test_pages_within_budget_become_one_chunk() -> None:
    """
    A document that fits the budget is sent in a single call

    This is the common case and the reason the merge is usually a no-op: the model
    sees every page at once, so sections spanning pages need no reconciling.
    """
    pages = {1: "a" * 100, 2: "b" * 100, 3: "c" * 100}

    chunks = group_pages_for_structuring(page_texts=pages, char_budget=1000)

    assert len(chunks) == 1
    assert chunks[0][0] == [1, 2, 3]


def test_pages_are_split_when_over_budget() -> None:
    """Pages are grouped into as few chunks as fit the budget"""
    # Two 40-character pages fit in 100; a third would not.
    pages = {1: "a" * 40, 2: "b" * 40, 3: "c" * 40, 4: "d" * 40}

    chunks = group_pages_for_structuring(page_texts=pages, char_budget=100)

    assert [pages_in_chunk for pages_in_chunk, _ in chunks] == [[1, 2], [3, 4]]


def test_each_page_stands_alone_when_two_cannot_fit() -> None:
    """When no two pages fit together, every page becomes its own chunk"""
    pages = {1: "a" * 60, 2: "b" * 60, 3: "c" * 60}

    chunks = group_pages_for_structuring(page_texts=pages, char_budget=100)

    assert [pages_in_chunk for pages_in_chunk, _ in chunks] == [[1], [2], [3]]


def test_chunks_stay_in_page_order() -> None:
    """Pages are ordered numerically regardless of dict insertion order"""
    pages = {3: "c" * 10, 1: "a" * 10, 2: "b" * 10}

    chunks = group_pages_for_structuring(page_texts=pages, char_budget=1000)

    assert chunks[0][0] == [1, 2, 3]
    text = chunks[0][1]
    assert text.index("Page 1") < text.index("Page 2") < text.index("Page 3")


def test_chunk_text_labels_each_page() -> None:
    """Chunk text keeps page markers so the model can attribute fields"""
    pages = {1: "first", 2: "second"}

    chunks = group_pages_for_structuring(page_texts=pages, char_budget=1000)

    assert "--- Page 1 ---" in chunks[0][1]
    assert "--- Page 2 ---" in chunks[0][1]


def test_blank_pages_are_skipped() -> None:
    """Pages with no text produce no chunk content and cost no LLM call"""
    pages = {1: "real text", 2: "   \n  ", 3: ""}

    chunks = group_pages_for_structuring(page_texts=pages, char_budget=1000)

    assert len(chunks) == 1
    assert chunks[0][0] == [1]


def test_no_text_produces_no_chunks() -> None:
    """A document with no extracted text produces no calls at all"""
    assert group_pages_for_structuring(page_texts={1: "", 2: "  "}, char_budget=100) == []
    assert group_pages_for_structuring(page_texts={}, char_budget=100) == []


def test_oversized_single_page_is_not_truncated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    A page bigger than the budget is sent whole with a warning

    Truncating would silently discard text Textract had already extracted and paid
    for, which would look like an extraction accuracy problem.
    """
    pages = {1: "x" * 500}

    with caplog.at_level("WARNING"):
        chunks = group_pages_for_structuring(page_texts=pages, char_budget=100)

    assert len(chunks) == 1
    assert chunks[0][1].count("x") == 500
    assert "over the 100-character chunk budget" in caplog.text


def test_oversized_page_does_not_absorb_neighbours() -> None:
    """An over-budget page stands alone rather than dragging other pages in"""
    pages = {1: "x" * 500, 2: "y" * 10}

    chunks = group_pages_for_structuring(page_texts=pages, char_budget=100)

    assert [pages_in_chunk for pages_in_chunk, _ in chunks] == [[1], [2]]


def test_non_positive_budget_raises() -> None:
    """A zero or negative budget is a configuration error, not a silent default"""
    with pytest.raises(ValueError, match="char_budget must be positive"):
        group_pages_for_structuring(page_texts={1: "a"}, char_budget=0)


# --- merge_structured_results ----------------------------------------------


def test_single_result_passes_through() -> None:
    """One chunk merges to an equal object, with no page wrapper"""
    result = {"partA": {"name": "Smith"}}

    merged = merge_structured_results(results=[result], labels=["pages 1-3"])

    assert merged == result
    assert "pages" not in merged


def test_disjoint_sections_are_united() -> None:
    """Fields found in different chunks all appear in the merged object"""
    merged = merge_structured_results(
        results=[{"partA": {"name": "Smith"}}, {"partB": {"employer": "Acme"}}],
        labels=["pages 1-2", "pages 3-4"],
    )

    assert merged == {"partA": {"name": "Smith"}, "partB": {"employer": "Acme"}}


def test_section_split_across_chunks_is_merged() -> None:
    """
    A section spanning a chunk boundary has its fields united

    This is why stripping the old "pages" wrapper would not have been enough on its
    own: both chunks return the same top-level section, each holding half the
    fields.
    """
    merged = merge_structured_results(
        results=[
            {"partA": {"name": "Smith", "dob": None}},
            {"partA": {"name": None, "dob": "1980-01-01"}},
        ],
        labels=["pages 1-2", "pages 3-4"],
    )

    assert merged == {"partA": {"name": "Smith", "dob": "1980-01-01"}}


def test_empty_values_do_not_overwrite_real_ones() -> None:
    """
    A later chunk's nulls never erase an earlier chunk's extracted value

    Each chunk is structured against the full schema, so most fields come back null
    in most chunks; letting those win would empty the result.
    """
    merged = merge_structured_results(
        results=[{"a": "value", "b": {"c": "nested"}}, {"a": "", "b": {"c": None}}],
        labels=["one", "two"],
    )

    assert merged == {"a": "value", "b": {"c": "nested"}}


def test_real_value_fills_an_earlier_empty() -> None:
    """An earlier null is replaced by a later chunk's real value"""
    merged = merge_structured_results(
        results=[{"a": None}, {"a": "found"}],
        labels=["one", "two"],
    )

    assert merged == {"a": "found"}


def test_false_is_not_overwritten() -> None:
    """
    A False value survives a later null

    An unchecked box is an extracted answer, not a missing one.
    """
    merged = merge_structured_results(
        results=[{"signed": False}, {"signed": None}],
        labels=["one", "two"],
    )

    assert merged["signed"] is False


def test_lists_accumulate_across_chunks() -> None:
    """Table rows continuing past a boundary are appended, not replaced"""
    merged = merge_structured_results(
        results=[{"rows": [{"n": 1}]}, {"rows": [{"n": 2}]}],
        labels=["one", "two"],
    )

    assert merged["rows"] == [{"n": 1}, {"n": 2}]


def test_duplicate_list_items_are_not_repeated() -> None:
    """
    Items restated in both chunks appear once

    A header row or a field the model repeats per page should not inflate the list.
    """
    merged = merge_structured_results(
        results=[{"rows": [{"n": 1}, {"n": 2}]}, {"rows": [{"n": 2}, {"n": 3}]}],
        labels=["one", "two"],
    )

    assert merged["rows"] == [{"n": 1}, {"n": 2}, {"n": 3}]


def test_list_dedup_ignores_key_order() -> None:
    """Dicts differing only in key order are the same item"""
    merged = merge_structured_results(
        results=[{"rows": [{"a": 1, "b": 2}]}, {"rows": [{"b": 2, "a": 1}]}],
        labels=["one", "two"],
    )

    assert merged["rows"] == [{"a": 1, "b": 2}]


def test_conflicting_scalars_keep_first_and_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    A disagreement between chunks is logged rather than silently resolved

    Keeping the first value is arbitrary, and on a benchmark the discarded one may
    be the correct answer, so the choice has to be visible.
    """
    with caplog.at_level("WARNING"):
        merged = merge_structured_results(
            results=[{"partA": {"name": "Smith"}}, {"partA": {"name": "Smyth"}}],
            labels=["pages 1-2", "pages 3-4"],
        )

    assert merged == {"partA": {"name": "Smith"}}
    assert "partA.name" in caplog.text
    assert "Smith" in caplog.text and "Smyth" in caplog.text
    assert "pages 3-4" in caplog.text


def test_identical_scalars_do_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    """Agreement between chunks is not a conflict"""
    with caplog.at_level("WARNING"):
        merge_structured_results(
            results=[{"a": "same"}, {"a": "same"}],
            labels=["one", "two"],
        )

    assert "Conflicting" not in caplog.text


def test_non_object_result_raises() -> None:
    """
    A chunk that structured to a list or scalar is refused

    Merging it into a schema-shaped object is undefined; guessing would corrupt the
    accuracy numbers rather than report a problem.
    """
    with pytest.raises(ValueError, match="is a list, not a JSON object"):
        merge_structured_results(results=[{"a": 1}, ["not", "an", "object"]], labels=["one", "two"])


def test_mismatched_labels_raise() -> None:
    """A caller that loses track of its labels gets an error, not a bad warning"""
    with pytest.raises(ValueError, match="same length"):
        merge_structured_results(results=[{"a": 1}, {"b": 2}], labels=["only one"])


def test_deeply_nested_sections_merge() -> None:
    """Merging recurses to arbitrary depth"""
    merged = merge_structured_results(
        results=[
            {"a": {"b": {"c": {"d": "deep"}}}},
            {"a": {"b": {"c": {"e": "also deep"}}}},
        ],
        labels=["one", "two"],
    )

    assert merged == {"a": {"b": {"c": {"d": "deep", "e": "also deep"}}}}


# --- the behaviour that actually motivated the change ----------------------


def test_merged_output_scores_against_real_ground_truth() -> None:
    """
    A perfect extraction split across chunks scores 100% against real ground truth

    The regression this guards: the old code returned
    {"pages": {"page_1": {...}, ...}}, whose top level is page numbers while the
    ground truth's top level is form sections, so the evaluator found every field
    missing and a perfect extraction scored 0%.
    """
    from shared.evaluator import get_detailed_accuracy

    with open(TRUTH_FILE, "r", encoding="utf-8") as handle:
        truth = json.load(handle)

    # Split the truth across two chunks by section, the way a real document splits
    # across pages: each chunk carries some sections and nulls for the rest.
    section_names = list(truth)
    first_half = {name: truth[name] for name in section_names[:2]}
    second_half = {name: truth[name] for name in section_names[2:]}
    chunk_one = {**first_half, **{name: None for name in second_half}}
    chunk_two = {**second_half, **{name: None for name in first_half}}

    merged = merge_structured_results(
        results=[chunk_one, chunk_two], labels=["pages 1-4", "pages 5-7"]
    )

    assert get_detailed_accuracy(merged, truth)["total_accuracy"] == 100.0


def test_old_page_wrapper_shape_would_score_zero() -> None:
    """
    Confirms the regression test above is not vacuous

    The same perfect extraction, wrapped the way the old code wrapped it, scores 0%
    - so the 100% above is really attributable to the shape change.
    """
    from shared.evaluator import get_detailed_accuracy

    with open(TRUTH_FILE, "r", encoding="utf-8") as handle:
        truth = json.load(handle)

    old_shape = {"pages": {"page_1": truth}}

    assert get_detailed_accuracy(old_shape, truth)["total_accuracy"] == 0
