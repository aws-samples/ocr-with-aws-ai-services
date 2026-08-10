"""Tests for the nested-schema <-> BDA-blueprint translation.

BDA blueprints support one level of nesting, so a nested output schema has to be
translated on the way out and put back on the way in. When the put-back step was
missing, BDA returned flat keys (`requestPartA_gender`) that no nested truth path
(`requestPartA.gender`) could match, and every run scored exactly 0% however good
the extraction was. `TestRoundTripScoresFull` is the test that would have caught
that: it feeds the ground truth's own values back through the round trip and
asserts the evaluator sees 100%.

The blueprint syntax asserted here was verified against `CreateBlueprint` in
us-east-1 on 2026-08-07 and cross-checked against the public catalog blueprints
(`Invoice`, `Bank-Statement`, `W2-Form`), which use the same `$ref` form.
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

from shared.blueprint_schema import (
    HOIST_SEPARATOR,
    build_blueprint_schema,
    resolve_scalar_type,
    restore_nested_result,
)
from shared.evaluator import calculate_accuracy

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
# The synthetic PFL bundle is used because it ships in the repository and was built
# to reproduce every shape this translation has to survive: a group inside a group
# (requestPartB.insuranceCarrier), an array of objects inside a group
# (requestPartB.grossWages) and an optional numeric column (numberOfDaysWorked).
# Reading a tracked bundle is what keeps this test runnable on a fresh clone.
SAMPLE_NAME: str = "pfl-synthetic"
SCHEMA_PATH: Path = PROJECT_ROOT / "sample" / SAMPLE_NAME / "schema.json"
TRUTH_PATH: Path = PROJECT_ROOT / "sample" / SAMPLE_NAME / "truth.json"


def value_at(*, data: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    """Read the value a path points at, or None when the path is absent.

    Args:
        data: Nested mapping to read from.
        path: Path segments to follow.

    Returns:
        Any: The value at that path, or None if any segment is missing.
    """
    current: Any = data
    for segment in path:
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]
    return current


def simulate_bda_result(
    *, truth: Dict[str, Any], field_map: Dict[str, Tuple[str, ...]]
) -> Dict[str, Any]:
    """Build the flat `inference_result` a perfect BDA run would return.

    BDA returns one top-level key per blueprint field, already shaped for groups
    and tables. This reads each mapped path out of the ground truth to produce
    that shape, so a round trip can be scored without calling AWS.

    Args:
        truth: Ground-truth data, nested as the output schema declares it.
        field_map: Blueprint key -> original path, from `build_blueprint_schema`.

    Returns:
        Dict[str, Any]: A flat result keyed by blueprint field name.
    """
    return {
        key: value_at(data=truth, path=path) for key, path in field_map.items()
    }


@pytest.fixture(scope="module")
def sample_schema() -> Dict[str, Any]:
    """Load the real PFL output schema.

    Returns:
        Dict[str, Any]: The parsed schema.
    """
    return json.loads(SCHEMA_PATH.read_text())


@pytest.fixture(scope="module")
def sample_truth() -> Dict[str, Any]:
    """Load the real PFL ground truth.

    Returns:
        Dict[str, Any]: The parsed ground truth.
    """
    return json.loads(TRUTH_PATH.read_text())


class TestRoundTripScoresFull:
    """The round trip must preserve shape well enough to score 100%."""

    def test_perfect_extraction_scores_one_hundred(
        self, sample_schema: Dict[str, Any], sample_truth: Dict[str, Any]
    ) -> None:
        """Ground-truth values routed through the round trip score 100%.

        Before the fix this scored 0.0 with identical values, because the result
        stayed flat.
        """
        build = build_blueprint_schema(schema=sample_schema, document_type="form")
        assert build is not None

        returned = simulate_bda_result(truth=sample_truth, field_map=build.field_map)
        restored = restore_nested_result(
            inference_result=returned, field_map=build.field_map
        )

        assert calculate_accuracy(restored, sample_truth) == 100.0

    def test_flat_result_would_score_zero(
        self, sample_schema: Dict[str, Any], sample_truth: Dict[str, Any]
    ) -> None:
        """The unrestored flat result scores 0%, pinning the original defect.

        This guards the test above from passing for the wrong reason: if the
        evaluator were shape-insensitive, both would score the same.
        """
        build = build_blueprint_schema(schema=sample_schema, document_type="form")
        assert build is not None

        flat: Dict[str, Any] = {}
        for key, path in build.field_map.items():
            value = value_at(data=sample_truth, path=path)
            if isinstance(value, dict):
                # A group comes back as an object; flattening it the way the old
                # code named its fields is what produced the unmatched keys.
                for child_key, child_value in value.items():
                    flat[f"{key}_{child_key}"] = child_value
            else:
                flat[key] = value

        assert calculate_accuracy(flat, sample_truth) == 0.0


class TestGroups:
    """Nested objects become `$ref` groups rather than underscore-joined names."""

    def test_nested_object_becomes_a_ref(self) -> None:
        """A top-level object is emitted as a `$ref` into `definitions`."""
        build = build_blueprint_schema(
            schema={
                "properties": {
                    "partA": {
                        "type": "object",
                        "properties": {
                            "gender": {"type": "string", "description": "Q8 gender"}
                        },
                    }
                }
            }
        )
        assert build is not None

        field = build.schema["properties"]["partA"]
        assert "$ref" in field
        assert field["$ref"].startswith("#/definitions/")

        definition = build.schema["definitions"][field["$ref"].rsplit("/", 1)[1]]
        assert definition["properties"]["gender"] == {
            "type": "string",
            "inferenceType": "explicit",
            "instruction": "Q8 gender",
        }

    def test_no_underscore_joined_field_names(
        self, sample_schema: Dict[str, Any]
    ) -> None:
        """No blueprint field is named `parent_child` for the real schema.

        The old builder produced `requestPartA_gender` and friends. Hoisted tables
        use `HOIST_SEPARATOR` and are the only compound names allowed.
        """
        build = build_blueprint_schema(schema=sample_schema)
        assert build is not None

        for name in build.schema["properties"]:
            if HOIST_SEPARATOR in name:
                continue
            assert "_" not in name, f"{name} looks like a flattened field name"

    def test_group_children_are_not_top_level(
        self, sample_schema: Dict[str, Any]
    ) -> None:
        """Scalar children stay inside their group."""
        build = build_blueprint_schema(schema=sample_schema)
        assert build is not None

        assert "gender" not in build.schema["properties"]
        assert "requestPartA_gender" not in build.schema["properties"]
        assert set(build.schema["properties"]) >= {
            "requestPartA",
            "requestPartB",
            "bondingCertification",
            "paymentEnrollment",
        }


    def test_group_inside_a_group_is_hoisted_not_dropped(
        self, sample_schema: Dict[str, Any]
    ) -> None:
        """`requestPartB.insuranceCarrier` is a group inside a group.

        BDA allows one level of nesting, so it becomes its own top-level group.
        Dropping it instead would silently lose its five address fields and score
        them all as missing.
        """
        build = build_blueprint_schema(schema=sample_schema)
        assert build is not None

        hoisted_name = f"requestPartB{HOIST_SEPARATOR}insuranceCarrier"
        assert hoisted_name in build.schema["properties"]
        assert build.field_map[hoisted_name] == ("requestPartB", "insuranceCarrier")

        definition_name = build.schema["properties"][hoisted_name]["$ref"].rsplit(
            "/", 1
        )[1]
        columns = build.schema["definitions"][definition_name]["properties"]
        assert set(columns) == {
            "name",
            "streetAddress",
            "city",
            "state",
            "zipCode",
        }

    def test_deeply_hoisted_group_restores_to_its_path(
        self, sample_schema: Dict[str, Any]
    ) -> None:
        """A hoisted group lands back at `requestPartB.insuranceCarrier`."""
        build = build_blueprint_schema(schema=sample_schema)
        assert build is not None

        carrier = {"name": "Guardian", "city": "Bethlehem", "state": "PA"}
        restored = restore_nested_result(
            inference_result={
                f"requestPartB{HOIST_SEPARATOR}insuranceCarrier": carrier
            },
            field_map=build.field_map,
        )

        assert restored["requestPartB"]["insuranceCarrier"] == carrier


class TestTables:
    """Arrays of objects become tables, hoisted when BDA cannot nest them."""

    def test_array_of_objects_inside_a_group_is_hoisted(
        self, sample_schema: Dict[str, Any]
    ) -> None:
        """`requestPartB.grossWages` is hoisted, because BDA rejects it nested.

        Verified against `CreateBlueprint`: an array of objects inside a group
        returns `ValidationException: Request has invalid blueprint schema`.
        """
        build = build_blueprint_schema(schema=sample_schema)
        assert build is not None

        hoisted_name = f"requestPartB{HOIST_SEPARATOR}grossWages"
        assert hoisted_name in build.schema["properties"]
        assert build.field_map[hoisted_name] == ("requestPartB", "grossWages")
        assert ("requestPartB", "grossWages") in build.hoisted_paths

        table = build.schema["properties"][hoisted_name]
        assert table["type"] == "array"
        assert "$ref" in table["items"]

    def test_hoisted_table_restores_into_its_parent(
        self, sample_schema: Dict[str, Any]
    ) -> None:
        """A hoisted table lands back at `requestPartB.grossWages` as a list."""
        build = build_blueprint_schema(schema=sample_schema)
        assert build is not None

        rows = [
            {"weekNumber": 1, "weekEndingDate": "2025-04-20", "grossAmountPaid": 1738.64},
            {"weekNumber": 2, "weekEndingDate": "2025-04-06", "grossAmountPaid": 2249.52},
        ]
        restored = restore_nested_result(
            inference_result={f"requestPartB{HOIST_SEPARATOR}grossWages": rows},
            field_map=build.field_map,
        )

        assert restored["requestPartB"]["grossWages"] == rows

    @pytest.mark.parametrize("group_first", [True, False])
    def test_group_and_its_hoisted_table_merge(self, group_first: bool) -> None:
        """A group and a field hoisted out of it must not overwrite each other.

        Both write into the same parent, so the result depends on nothing but the
        merge being correct - which means it must hold in either arrival order.
        """
        field_map = {
            "partB": ("partB",),
            f"partB{HOIST_SEPARATOR}rows": ("partB", "rows"),
        }
        entries = [
            ("partB", {"title": "Manager"}),
            (f"partB{HOIST_SEPARATOR}rows", [{"amount": 10}]),
        ]
        if not group_first:
            entries.reverse()

        restored = restore_nested_result(
            inference_result=dict(entries), field_map=field_map
        )

        assert restored == {"partB": {"title": "Manager", "rows": [{"amount": 10}]}}

    def test_top_level_array_of_objects_stays_put(self) -> None:
        """A table at the top level needs no hoisting."""
        build = build_blueprint_schema(
            schema={
                "properties": {
                    "lineItems": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"amount": {"type": "number"}},
                        },
                    }
                }
            }
        )
        assert build is not None

        assert build.field_map["lineItems"] == ("lineItems",)
        assert build.hoisted_paths == []
        assert build.schema["properties"]["lineItems"]["type"] == "array"


class TestScalarArrays:
    """Arrays of scalars are supported inside a group and need no hoisting."""

    def test_scalar_array_stays_inside_its_group(self) -> None:
        """An array of strings in a group is emitted in place."""
        build = build_blueprint_schema(
            schema={
                "properties": {
                    "partA": {
                        "type": "object",
                        "properties": {
                            "race": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Q10 race categories",
                            }
                        },
                    }
                }
            }
        )
        assert build is not None

        assert build.hoisted_paths == []
        definition_name = build.schema["properties"]["partA"]["$ref"].rsplit("/", 1)[1]
        race = build.schema["definitions"][definition_name]["properties"]["race"]
        assert race["type"] == "array"
        assert race["items"] == {"type": "string"}


class TestTypeResolution:
    """Union types must resolve to something BDA accepts."""

    @pytest.mark.parametrize(
        "declared,expected",
        [
            (["number", "null"], "number"),
            (["null", "string"], "string"),
            ("integer", "integer"),
            ("boolean", "boolean"),
            ("date", "string"),
            (None, "string"),
        ],
    )
    def test_resolves_to_a_scalar(self, declared: Any, expected: str) -> None:
        """Each declaration reduces to one blueprint-acceptable scalar type."""
        assert resolve_scalar_type(declared_type=declared) == expected

    def test_all_null_union_raises(self) -> None:
        """A type naming only null is a schema error, not a string field."""
        with pytest.raises(ValueError, match="no extractable type"):
            resolve_scalar_type(declared_type=["null"])

    def test_optional_number_column_is_a_number(
        self, sample_schema: Dict[str, Any]
    ) -> None:
        """`numberOfDaysWorked` is `["number", "null"]` and must stay a number."""
        build = build_blueprint_schema(schema=sample_schema)
        assert build is not None

        table = build.schema["properties"][f"requestPartB{HOIST_SEPARATOR}grossWages"]
        row = build.schema["definitions"][table["items"]["$ref"].rsplit("/", 1)[1]]
        assert row["properties"]["numberOfDaysWorked"]["type"] == "number"


class TestFieldCount:
    """Field count drives cost, so it must follow the emitted blueprint."""

    def test_counts_leaves_and_table_columns(
        self, sample_schema: Dict[str, Any]
    ) -> None:
        """Every scalar leaf and every table column is counted once."""
        build = build_blueprint_schema(schema=sample_schema)
        assert build is not None

        def count(definition: Dict[str, Any]) -> int:
            """Count the blueprint fields a schema property contributes.

            Args:
                definition: A schema property object.

            Returns:
                int: Number of billable blueprint fields.
            """
            if definition.get("type") == "array" and isinstance(
                definition.get("items"), dict
            ):
                items = definition["items"]
                if items.get("type") == "object":
                    # A table bills one field per column.
                    return len(items["properties"])
                return 1
            if definition.get("type") == "object":
                return sum(
                    count(child) for child in definition["properties"].values()
                )
            return 1

        expected = sum(count(prop) for prop in sample_schema["properties"].values())
        assert build.field_count == expected

    def test_stays_within_the_blueprint_field_limit(
        self, sample_schema: Dict[str, Any]
    ) -> None:
        """BDA allows 100 fields per blueprint for InvokeDataAutomationAsync."""
        build = build_blueprint_schema(schema=sample_schema)
        assert build is not None
        assert build.field_count <= 100


class TestFailsLoudly:
    """Bad input raises or reports rather than producing a silent wrong answer."""

    def test_hoist_name_collision_raises(self) -> None:
        """A generated hoist name that is already taken is an error."""
        schema = {
            "properties": {
                "partB": {
                    "type": "object",
                    "properties": {
                        "rows": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"amount": {"type": "number"}},
                            },
                        }
                    },
                },
                f"partB{HOIST_SEPARATOR}rows": {"type": "string"},
            }
        }
        with pytest.raises(ValueError, match="already taken"):
            build_blueprint_schema(schema=schema)

    def test_unparseable_schema_returns_none(self) -> None:
        """A schema that is not JSON yields None rather than a partial build."""
        assert build_blueprint_schema(schema="{not json") is None

    def test_schema_without_properties_returns_none(self) -> None:
        """A schema declaring no properties yields None."""
        assert build_blueprint_schema(schema={"type": "object"}) is None

    def test_unmapped_key_is_kept_not_dropped(self) -> None:
        """A returned key absent from the map stays visible at the top level."""
        restored = restore_nested_result(
            inference_result={"surprise": "value"}, field_map={}
        )
        assert restored == {"surprise": "value"}
