"""
Tests for the generator contract of the two Gradio streaming handlers

process_image_with_engines() and process_all_samples() are generator functions, so
`return value` discards the payload - Gradio substitutes None for every output and the
UI sits on the spinner with no error shown. Every early exit must therefore yield.

These tests make no AWS calls: they only exercise the guard clauses that run before any
engine is constructed. The schema tests use a synthetic bundle tree rather than the real
sample/ corpus, so they do not depend on which samples happen to be preloaded.
"""
import json
import os

import pytest
from gradio.utils import is_prop_update

from event_handler import clear_sample_context_for_upload, handle_sample_selection
from processor import process_image_with_engines
from sample_handler import process_all_samples
from shared.comparison_utils import ENGINE_FILTER_CHOICES
from shared.sample_paths import SCHEMA_FILENAME

# The Process File click is wired to output_components + [results_table]
EXPECTED_OUTPUT_COUNT = 20

SCHEMA = {"type": "object", "properties": {"claim_number": {"type": "string"}}}


def drain(generator):
    """
    Collect everything a generator yields

    Args:
        generator: The generator to exhaust

    Returns:
        List of yielded payloads
    """
    return list(generator)


@pytest.fixture
def sample_tree(tmp_path, monkeypatch):
    """
    Build a synthetic sample tree with one schema-bearing and one schema-less PDF

    Args:
        tmp_path: pytest temporary directory
        monkeypatch: pytest monkeypatch fixture

    Returns:
        Path to the synthetic sample root
    """
    with_schema = tmp_path / "sample" / "PFL" / "with-schema"
    without_schema = tmp_path / "sample" / "PFL" / "without-schema"
    with_schema.mkdir(parents=True)
    without_schema.mkdir(parents=True)

    (with_schema / "with-schema.pdf").write_bytes(b"%PDF-1.4 fake")
    (with_schema / SCHEMA_FILENAME).write_text(json.dumps(SCHEMA))
    (without_schema / "without-schema.pdf").write_bytes(b"%PDF-1.4 fake")

    # SAMPLE_DIR is relative, so the working directory is what redirects discovery,
    # schema lookup and load_truth_data() at once.
    monkeypatch.chdir(tmp_path)

    return tmp_path


def test_no_document_yields_an_error_payload():
    """An empty file input must produce a visible error, not a silent hang"""
    payloads = drain(
        process_image_with_engines(None, True, False, False, "Claude Sonnet 5")
    )

    assert len(payloads) == 1, "a returned payload never reaches Gradio - it must yield"
    assert "No image uploaded" in payloads[0][0]


def test_no_engine_selected_yields_an_error_payload():
    """Clearing every engine checkbox must say so rather than hang"""
    payloads = drain(
        process_image_with_engines("any-document.pdf", False, False, False, "Claude Sonnet 5")
    )

    assert len(payloads) == 1
    assert "at least one OCR engine" in payloads[0][0]


@pytest.mark.parametrize(
    "image,use_textract",
    [
        (None, True),                 # no document
        ("any-document.pdf", False),  # no engine
    ],
)
def test_guard_payloads_match_the_wired_output_count(image, use_textract):
    """
    A short payload raises "Number of output components does not match"

    The guards are the only paths that build their payload by hand, so a component
    added to the results panel without updating them fails here rather than at runtime.
    """
    payloads = drain(
        process_image_with_engines(image, use_textract, False, False, "Claude Sonnet 5")
    )

    assert len(payloads[0]) == EXPECTED_OUTPUT_COUNT


@pytest.mark.parametrize(
    "image,use_textract",
    [
        (None, True),
        ("any-document.pdf", False),
    ],
)
def test_guards_do_not_write_html_into_the_comparison_dropdown(image, use_textract):
    """
    Output 17 is the diff_engine Dropdown, not an HTML block

    Its only valid values are ENGINE_FILTER_CHOICES, so anything else makes Gradio warn
    that the value is not in the list of choices and leaves a junk selection. The tuple
    is imported rather than spelled out here: the label of the default choice has
    changed once already, when the tab stopped requiring a single engine to be picked.
    """
    payloads = drain(
        process_image_with_engines(image, use_textract, False, False, "Claude Sonnet 5")
    )
    diff_engine_value = payloads[0][17]

    assert (is_prop_update(diff_engine_value)
            or diff_engine_value in ENGINE_FILTER_CHOICES)


def test_guard_payloads_end_with_table_markup():
    """
    The last output is the comparison table, which is now a gr.HTML

    It became HTML rather than a DataFrame because each cost cell carries a `title`
    with the formula behind the figure, which a DataFrame cell cannot hold. A guard
    that still yielded a DataFrame here would render as its repr.
    """
    payloads = drain(
        process_image_with_engines(None, True, False, False, "Claude Sonnet 5")
    )
    table = payloads[0][-1]

    assert isinstance(table, str)
    assert table.lstrip().startswith("<")


def test_no_samples_yields_a_message(monkeypatch, tmp_path):
    """Batch mode over an empty sample tree must report it"""
    # No sample/ directory here at all, which is the state a fresh clone is in once the
    # tracked samples are removed - discovery returns an empty list rather than raising.
    monkeypatch.chdir(tmp_path)

    payloads = drain(process_all_samples(True, False, False, "Claude Sonnet 5"))

    assert len(payloads) == 1
    status_html, table = payloads[0]
    assert "No sample documents found" in status_html
    # The batch path feeds the same gr.HTML component as the single-document path.
    assert isinstance(table, str)


def test_sample_without_schema_leaves_the_editor_alone(sample_tree):
    """
    gr.Code cannot take None, and an uploaded schema must survive the next selection

    Batch mode already falls back to the editor's schema when a sample has none, so
    the dropdown must not clear it.
    """
    # A bundle label is a path relative to sample/ and carries no file extension.
    _label, path, schema_update, _truth, _status = handle_sample_selection(
        os.path.join("PFL", "without-schema")
    )

    assert path.endswith("without-schema.pdf")
    assert is_prop_update(schema_update), "a schema-less sample must yield a no-op update"


def test_sample_with_schema_returns_its_schema_text(sample_tree):
    """A sample that has a schema still overrides the editor"""
    _label, _path, schema_update, _truth, _status = handle_sample_selection(
        os.path.join("PFL", "with-schema")
    )

    assert isinstance(schema_update, str)
    assert json.loads(schema_update) == SCHEMA


def test_empty_selection_returns_five_values(sample_tree):
    """The dropdown change handler is wired to five outputs"""
    result = handle_sample_selection("")

    assert len(result) == 5
    assert is_prop_update(result[2]), "an empty selection must not clear the schema either"


def test_manual_upload_clears_sample_truth_context():
    """An uploaded file must not inherit the previously selected sample's truth."""
    dropdown_update, sample_name, truth, truth_status = (
        clear_sample_context_for_upload()
    )

    assert is_prop_update(dropdown_update)
    assert dropdown_update["value"] is None
    assert sample_name == ""
    assert truth is None
    assert truth_status == "<div></div>"
