"""Tests that every preloaded sample bundle is internally consistent.

These tests run over the real `sample/` tree rather than a fixture, because their
subject is the corpus itself: a bundle whose `truth.json` names fields its
`schema.json` does not declare cannot ever score 100%, and the symptom is a
plausible-looking benchmark number rather than an error. Asserting it here is what
turns that into a failing test.

Whatever the tree happens to hold is checked, so a fresh clone simply has fewer
bundles to walk - no particular bundle is required.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from shared.sample_paths import (
    DOCUMENT_EXTENSIONS,
    SCHEMA_FILENAME,
    TRUTH_FILENAME,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SAMPLE_ROOT: Path = PROJECT_ROOT / "sample"


def _bundles() -> List[Path]:
    """Collect every sample bundle directory under sample/.

    A bundle is any directory holding exactly one document, which is the same rule
    `shared.sample_paths.list_sample_bundles()` applies - restated here rather than
    imported so that a bug in discovery cannot hide a broken bundle from these tests.

    Returns:
        List[Path]: Sorted absolute paths of the bundle directories.
    """
    bundles: List[Path] = []

    for directory in SAMPLE_ROOT.rglob("*"):
        if not directory.is_dir():
            continue
        documents = [entry for entry in directory.iterdir()
                     if entry.is_file()
                     and entry.suffix.lower() in DOCUMENT_EXTENSIONS]
        if len(documents) == 1:
            bundles.append(directory)

    return sorted(bundles)


def _bundles_with_truth() -> List[Path]:
    """Collect the bundles that carry transcribed ground truth.

    Returns:
        List[Path]: Sorted absolute paths of the bundle directories that have a
            truth.json. A bundle without one is a valid demo sample; it just cannot
            be scored.
    """
    return [bundle for bundle in _bundles() if (bundle / TRUTH_FILENAME).exists()]


def _load_json(*, path: Path) -> Dict[str, Any]:
    """Read and parse a JSON file, failing the test if it is malformed.

    Args:
        path (Path): Absolute path of the JSON file to read.

    Returns:
        Dict[str, Any]: The parsed JSON object.
    """
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _label(bundle: Path) -> str:
    """Render a bundle's dropdown label for use as a test id.

    Args:
        bundle (Path): Absolute path of the bundle directory.

    Returns:
        str: The bundle's path relative to sample/.
    """
    return str(bundle.relative_to(SAMPLE_ROOT))


def test_the_sample_tree_holds_bundles() -> None:
    """Guard against the walk silently matching nothing.

    Every assertion below is parameterised over the walk, so an empty result would
    turn this whole file into a no-op that still reports as passing.

    Returns:
        None
    """
    assert _bundles(), f"No sample bundles found under {SAMPLE_ROOT}"


def test_ground_truth_is_tracked_for_at_least_one_bundle() -> None:
    """A fresh clone has to ship something scoreable.

    Accuracy is the point of the comparison, so the tracked corpus cannot consist
    entirely of demo images with no ground truth to score against.

    Returns:
        None
    """
    assert _bundles_with_truth(), (
        "No bundle under sample/ carries ground truth, so nothing in a fresh clone "
        "can be scored for accuracy")


@pytest.mark.parametrize("bundle", _bundles_with_truth(), ids=_label)
def test_a_bundle_with_truth_also_has_a_schema(bundle: Path) -> None:
    """Ground truth without a schema scores against the default one-field schema.

    Args:
        bundle (Path): Absolute path of the bundle directory under test.

    Returns:
        None
    """
    schema_path: Path = bundle / SCHEMA_FILENAME

    assert schema_path.exists(), (
        f"Bundle {_label(bundle)} has {TRUTH_FILENAME} but no {SCHEMA_FILENAME}. "
        f"Without one the engines are asked for the editor's default schema and the "
        f"run scores near zero against ground truth it was never asked to produce.")


@pytest.mark.parametrize("bundle", _bundles_with_truth(), ids=_label)
def test_the_schema_declares_every_field_the_truth_expects(bundle: Path) -> None:
    """A truth key the schema never asks for can only ever be scored as missing.

    Args:
        bundle (Path): Absolute path of the bundle directory under test.

    Returns:
        None
    """
    truth: Dict[str, Any] = _load_json(path=bundle / TRUTH_FILENAME)
    schema: Dict[str, Any] = _load_json(path=bundle / SCHEMA_FILENAME)

    # Only object-rooted schemas can be checked key-by-key; the samples all are.
    assert schema.get("type") == "object", (
        f"{_label(bundle)}/{SCHEMA_FILENAME} must have a top-level "
        f"\"type\": \"object\"")

    schema_properties: Dict[str, Any] = schema.get("properties", {})
    assert schema_properties, f"{_label(bundle)}/{SCHEMA_FILENAME} declares no properties"

    missing = sorted(set(truth.keys()) - set(schema_properties.keys()))
    assert not missing, (
        f"{_label(bundle)}/{SCHEMA_FILENAME} is missing top-level properties present "
        f"in the ground truth: {missing}")


@pytest.mark.parametrize("bundle", _bundles(), ids=_label)
def test_a_bundle_schema_is_valid_json(bundle: Path) -> None:
    """An unparseable schema is logged and dropped, so the run silently uses the default.

    Args:
        bundle (Path): Absolute path of the bundle directory under test.

    Returns:
        None
    """
    schema_path: Path = bundle / SCHEMA_FILENAME

    if not schema_path.exists():
        pytest.skip(f"{_label(bundle)} has no {SCHEMA_FILENAME}")

    schema = _load_json(path=schema_path)

    assert isinstance(schema, dict), f"{_label(bundle)}/{SCHEMA_FILENAME} is not an object"


@pytest.mark.parametrize("bundle", _bundles(), ids=_label)
def test_a_bundle_holds_no_stray_schema_or_truth_names(bundle: Path) -> None:
    """Schema and truth are found by fixed filename, so a near miss is invisible.

    A file called `schemas.json` or `groundtruth.json` is never read: the sample loads
    with no schema and scores against nothing, with only a log line to say so.

    Args:
        bundle (Path): Absolute path of the bundle directory under test.

    Returns:
        None
    """
    expected = {SCHEMA_FILENAME, TRUTH_FILENAME}
    near_misses = sorted(
        entry.name for entry in bundle.iterdir()
        if entry.is_file()
        and entry.suffix.lower() == ".json"
        and entry.name not in expected)

    assert not near_misses, (
        f"Bundle {_label(bundle)} holds JSON files that are neither {SCHEMA_FILENAME} "
        f"nor {TRUTH_FILENAME} and will never be read: {near_misses}")
