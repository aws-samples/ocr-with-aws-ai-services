"""Tests for Textract feature-block serialisation and option validation.

The block fixtures below reproduce the shapes documented for AnalyzeDocument:
KEY_VALUE_SET, CELL and QUERY_RESULT blocks carry no Text of their own, so their
content has to be resolved through CHILD relationships to WORD and
SELECTION_ELEMENT blocks. The engine previously read `item["Text"]` straight off a
KEY_VALUE_SET block, which never produces anything, so these tests pin the
relationship walk rather than trusting it.

No AWS call is made - only the response-parsing and option-validation code is
under test.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines.textract_engine import TextractEngine  # noqa: E402


@pytest.fixture
def engine() -> TextractEngine:
    """Provide a Textract engine instance.

    Returns:
        TextractEngine: A fresh engine; the constructor makes no AWS calls.
    """
    return TextractEngine()


def _word(block_id: str, text: str, page: int = 1) -> Dict[str, Any]:
    """Build a WORD block.

    Args:
        block_id (str): The block Id.
        text (str): The word's text.
        page (int): 1-indexed page number.

    Returns:
        Dict[str, Any]: A WORD block.
    """
    return {"Id": block_id, "BlockType": "WORD", "Text": text, "Page": page}


def _children(*ids: str) -> List[Dict[str, Any]]:
    """Build a CHILD relationship list.

    Args:
        *ids (str): Child block Ids.

    Returns:
        List[Dict[str, Any]]: A single CHILD relationship entry.
    """
    return [{"Type": "CHILD", "Ids": list(ids)}]


# --- form fields ------------------------------------------------------------


FORM_BLOCKS: List[Dict[str, Any]] = [
    {
        "Id": "key-1",
        "BlockType": "KEY_VALUE_SET",
        "EntityTypes": ["KEY"],
        "Page": 2,
        "Relationships": [
            {"Type": "CHILD", "Ids": ["w-1", "w-2"]},
            {"Type": "VALUE", "Ids": ["value-1"]},
        ],
    },
    {
        "Id": "value-1",
        "BlockType": "KEY_VALUE_SET",
        "EntityTypes": ["VALUE"],
        "Page": 2,
        "Relationships": _children("w-3"),
    },
    _word("w-1", "Nature", page=2),
    _word("w-2", "of disability", page=2),
    _word("w-3", "Total knee replacement", page=2),
    # A checkbox: the value half resolves to a SELECTION_ELEMENT, not a WORD.
    {
        "Id": "key-2",
        "BlockType": "KEY_VALUE_SET",
        "EntityTypes": ["KEY"],
        "Page": 2,
        "Relationships": [
            {"Type": "CHILD", "Ids": ["w-4"]},
            {"Type": "VALUE", "Ids": ["value-2"]},
        ],
    },
    {
        "Id": "value-2",
        "BlockType": "KEY_VALUE_SET",
        "EntityTypes": ["VALUE"],
        "Page": 2,
        "Relationships": _children("sel-1"),
    },
    _word("w-4", "Recovered?", page=2),
    {
        "Id": "sel-1",
        "BlockType": "SELECTION_ELEMENT",
        "SelectionStatus": "NOT_SELECTED",
        "Page": 2,
    },
]


def test_form_fields_resolve_key_and_value_text(engine: TextractEngine) -> None:
    """Key and value text is assembled from WORD children, not from Text.

    Args:
        engine (TextractEngine): The engine under test.

    Returns:
        None
    """
    block_map = engine._build_block_map(FORM_BLOCKS)
    by_page = engine._extract_form_fields(FORM_BLOCKS, block_map)

    assert list(by_page) == [2], "form fields must be grouped by their own page number"
    assert "Nature of disability: Total knee replacement" in by_page[2]


def test_form_checkbox_state_is_preserved(engine: TextractEngine) -> None:
    """An unchecked box serialises as NOT_SELECTED rather than as an empty value.

    Distinguishing "answered No" from "left blank" is a real quality signal on
    these claim forms, so the selection status must survive serialisation.

    Args:
        engine (TextractEngine): The engine under test.

    Returns:
        None
    """
    block_map = engine._build_block_map(FORM_BLOCKS)
    by_page = engine._extract_form_fields(FORM_BLOCKS, block_map)

    assert "Recovered?: NOT_SELECTED" in by_page[2]


# --- tables -----------------------------------------------------------------


TABLE_BLOCKS: List[Dict[str, Any]] = [
    {
        "Id": "table-1",
        "BlockType": "TABLE",
        "Page": 3,
        "Relationships": _children("cell-11", "cell-12", "cell-21", "cell-22"),
    },
    {
        "Id": "cell-11",
        "BlockType": "CELL",
        "RowIndex": 1,
        "ColumnIndex": 1,
        "Page": 3,
        "Relationships": _children("t-1"),
    },
    {
        "Id": "cell-12",
        "BlockType": "CELL",
        "RowIndex": 1,
        "ColumnIndex": 2,
        "Page": 3,
        "Relationships": _children("t-2"),
    },
    {
        "Id": "cell-21",
        "BlockType": "CELL",
        "RowIndex": 2,
        "ColumnIndex": 1,
        "Page": 3,
        "Relationships": _children("t-3"),
    },
    # Deliberately empty cell: no CHILD relationship at all.
    {"Id": "cell-22", "BlockType": "CELL", "RowIndex": 2, "ColumnIndex": 2, "Page": 3},
    _word("t-1", "Week", page=3),
    _word("t-2", "Gross", page=3),
    _word("t-3", "1", page=3),
]


def test_tables_serialise_as_ordered_pipe_separated_rows(engine: TextractEngine) -> None:
    """Cells are placed by RowIndex/ColumnIndex, with empty cells left blank.

    Args:
        engine (TextractEngine): The engine under test.

    Returns:
        None
    """
    block_map = engine._build_block_map(TABLE_BLOCKS)
    by_page = engine._extract_tables(TABLE_BLOCKS, block_map)

    assert by_page[3][0] == "[Table 1]"
    assert by_page[3][1] == "Week | Gross"
    assert by_page[3][2] == "1 | "


# --- queries ----------------------------------------------------------------


QUERY_BLOCKS: List[Dict[str, Any]] = [
    {
        "Id": "query-1",
        "BlockType": "QUERY",
        "Page": 1,
        "Query": {"Text": "What is the diagnosis code?", "Alias": "diagnosis_code"},
        "Relationships": [{"Type": "ANSWER", "Ids": ["answer-1"]}],
    },
    {"Id": "answer-1", "BlockType": "QUERY_RESULT", "Text": "M17.12", "Page": 1},
    {
        "Id": "query-2",
        "BlockType": "QUERY",
        "Page": 1,
        "Query": {"Text": "What is the policy number?", "Alias": "policy_number"},
    },
]


def test_queries_serialise_alias_and_answer(engine: TextractEngine) -> None:
    """An answered query serialises as "alias: answer".

    Args:
        engine (TextractEngine): The engine under test.

    Returns:
        None
    """
    block_map = engine._build_block_map(QUERY_BLOCKS)
    by_page = engine._extract_queries(QUERY_BLOCKS, block_map)

    assert "diagnosis_code: M17.12" in by_page[1]


def test_unanswered_query_is_recorded_not_dropped(engine: TextractEngine) -> None:
    """A query with no ANSWER relationship is reported as unanswered.

    A redacted or blank field producing no answer is a finding about the document,
    so dropping the line would hide it.

    Args:
        engine (TextractEngine): The engine under test.

    Returns:
        None
    """
    block_map = engine._build_block_map(QUERY_BLOCKS)
    by_page = engine._extract_queries(QUERY_BLOCKS, block_map)

    assert "policy_number: <no answer found>" in by_page[1]


# --- signatures -------------------------------------------------------------


def test_signatures_are_reported_per_page(engine: TextractEngine) -> None:
    """Each SIGNATURE block yields one line on its own page.

    Args:
        engine (TextractEngine): The engine under test.

    Returns:
        None
    """
    blocks = [
        {"Id": "sig-1", "BlockType": "SIGNATURE", "Page": 1, "Confidence": 99.4},
        {"Id": "sig-2", "BlockType": "SIGNATURE", "Page": 4, "Confidence": 87.0},
    ]
    by_page = engine._extract_signatures(blocks)

    assert by_page[1] == ["Signature detected (confidence 99.4%)"]
    assert by_page[4] == ["Signature detected (confidence 87.0%)"]


# --- section assembly -------------------------------------------------------


def test_no_features_produces_no_sections(engine: TextractEngine) -> None:
    """With no features requested the serialiser adds nothing.

    Args:
        engine (TextractEngine): The engine under test.

    Returns:
        None
    """
    assert engine._serialize_feature_blocks(blocks=FORM_BLOCKS, feature_types=[]) == {}


def test_unrequested_feature_blocks_are_ignored(engine: TextractEngine) -> None:
    """Blocks for a feature that was not requested are not serialised.

    Args:
        engine (TextractEngine): The engine under test.

    Returns:
        None
    """
    sections = engine._serialize_feature_blocks(
        blocks=FORM_BLOCKS + TABLE_BLOCKS, feature_types=["TABLES"]
    )
    flattened = "\n".join(line for lines in sections.values() for line in lines)

    assert "[Table 1]" in flattened
    assert "[Form fields]" not in flattened


def test_multiple_features_produce_separate_headed_sections(engine: TextractEngine) -> None:
    """Requesting several features yields one headed section per feature.

    Args:
        engine (TextractEngine): The engine under test.

    Returns:
        None
    """
    sections = engine._serialize_feature_blocks(
        blocks=FORM_BLOCKS + TABLE_BLOCKS + QUERY_BLOCKS,
        feature_types=["FORMS", "TABLES", "QUERIES"],
    )

    assert "[Form fields]" in sections[2]
    assert "[Table 1]" in sections[3]
    assert "[Queries]" in sections[1]


# --- option validation ------------------------------------------------------


def test_feature_types_are_normalised_to_canonical_order(engine: TextractEngine) -> None:
    """Case and ordering of the caller's feature list do not matter.

    Args:
        engine (TextractEngine): The engine under test.

    Returns:
        None
    """
    assert engine._validate_feature_types(["tables", "FORMS", "forms"]) == ["FORMS", "TABLES"]


def test_absent_feature_types_means_text_only(engine: TextractEngine) -> None:
    """None and empty both mean "text detection only", not an error.

    Args:
        engine (TextractEngine): The engine under test.

    Returns:
        None
    """
    assert engine._validate_feature_types(None) == []
    assert engine._validate_feature_types([]) == []


def test_unsupported_feature_type_raises(engine: TextractEngine) -> None:
    """An unsupported feature name fails before any AWS call is made.

    Args:
        engine (TextractEngine): The engine under test.

    Returns:
        None
    """
    with pytest.raises(ValueError, match="Unsupported Textract feature type"):
        engine._validate_feature_types(["FORMS", "EXPENSE"])


def test_queries_config_is_built_with_aliases(engine: TextractEngine) -> None:
    """Each query is sent with a readable alias derived from its text.

    Args:
        engine (TextractEngine): The engine under test.

    Returns:
        None
    """
    queries = engine._normalize_queries("What is the diagnosis code?\n\n  What is the ICD-10 code?  ")
    config = engine._build_queries_config(["QUERIES"], queries)

    assert config is not None
    assert [entry["Text"] for entry in config["Queries"]] == [
        "What is the diagnosis code?",
        "What is the ICD-10 code?",
    ]
    assert config["Queries"][0]["Alias"] == "what_is_the_diagnosis_code"
    assert all(
        char.isalnum() or char in "_-"
        for entry in config["Queries"]
        for char in entry["Alias"]
    )


def test_queries_feature_without_queries_raises(engine: TextractEngine) -> None:
    """QUERIES with no queries raises instead of silently downgrading to text-only.

    Textract rejects the request anyway, and a silent fallback would change what
    is being measured without saying so.

    Args:
        engine (TextractEngine): The engine under test.

    Returns:
        None
    """
    with pytest.raises(ValueError, match="requires at least one query"):
        engine._build_queries_config(["QUERIES"], [])


def test_overlong_query_raises(engine: TextractEngine) -> None:
    """A query beyond the 200-character API limit is rejected up front.

    Args:
        engine (TextractEngine): The engine under test.

    Returns:
        None
    """
    with pytest.raises(ValueError, match="200 characters"):
        engine._build_queries_config(["QUERIES"], ["x" * 201])


def test_queries_without_the_queries_feature_are_ignored(engine: TextractEngine) -> None:
    """Stale query text does not get sent when QUERIES is not selected.

    Args:
        engine (TextractEngine): The engine under test.

    Returns:
        None
    """
    assert engine._build_queries_config(["FORMS"], ["What is the diagnosis?"]) is None
