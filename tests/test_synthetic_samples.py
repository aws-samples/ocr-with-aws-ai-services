"""Tests for the tracked synthetic claim-form samples and the rules that keep
local-only bundles out of the repository.

The forms in `sample/*-synthetic/` are generated, not scanned, so a fresh clone can run
a realistic multi-page benchmark on documents the repository is allowed to publish, and
`tools/generate_synthetic_samples.py` is what produces them.

Two properties of those samples are load-bearing and neither is obvious from reading
the files, which is why they are asserted here:

- **No text layer.** The forms these stand in for are fax-grade scans, so the engines
  have to actually OCR them. A PDF with a text layer would let Bedrock and BDA read
  the values straight out of the file, and every accuracy and cost figure measured
  against it would describe a different task from the one being benchmarked.
- **Ground truth by construction.** Each `truth.json` is written from the same Python
  value table the pages are drawn from, so it cannot drift from the document. What has
  to be checked is that the tracked files still match what the generator produces.
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import fitz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.generate_synthetic_samples import (  # noqa: E402
    FORM_BUILDERS,
    build_pdf,
    check_sample,
    page_fingerprints,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SAMPLE_ROOT: Path = PROJECT_ROOT / "sample"
GITIGNORE: Path = PROJECT_ROOT / ".gitignore"

# Built once: rendering three multi-page forms at 150 DPI is the slow part of this file.
FORMS: List[Dict[str, Any]] = [builder() for builder in FORM_BUILDERS]


def form_id(form: Dict[str, Any]) -> str:
    """Render a form's bundle name for use as a test id.

    Args:
        form (Dict[str, Any]): A form specification from FORM_BUILDERS.

    Returns:
        str: The form's bundle name.
    """
    return form["id"]


@pytest.fixture(scope="module")
def rendered() -> Dict[str, bytes]:
    """Render every synthetic form once for the whole module.

    Returns:
        Dict[str, bytes]: Bundle name mapped to freshly generated PDF bytes.
    """
    return {form["id"]: build_pdf(form=form, seed=form["seed"]) for form in FORMS}


# --- the tracked bundles ----------------------------------------------------


def test_three_synthetic_bundles_are_tracked() -> None:
    """A fresh clone has to ship the multi-page benchmark, not just single images.

    Returns:
        None
    """
    tracked = sorted(path.name for path in SAMPLE_ROOT.glob("*-synthetic"))

    assert tracked == ["pfl-synthetic", "pml-synthetic", "std-synthetic"]


@pytest.mark.parametrize("form", FORMS, ids=form_id)
def test_the_bundle_holds_all_three_files(form: Dict[str, Any]) -> None:
    """Document, schema and ground truth together are what makes a sample scoreable.

    Args:
        form (Dict[str, Any]): The form specification under test.

    Returns:
        None
    """
    bundle = SAMPLE_ROOT / form["id"]

    assert (bundle / f"{form['id']}.pdf").exists()
    assert (bundle / "schema.json").exists()
    assert (bundle / "truth.json").exists()


@pytest.mark.parametrize("form", FORMS, ids=form_id)
def test_the_tracked_bundle_matches_the_generator(form: Dict[str, Any]) -> None:
    """The generator is the source of truth, so the tracked files must agree with it.

    This is `--check` run as a test: editing a value in the generator without
    regenerating leaves a document whose pages disagree with the ground truth
    transcribed from them, and the only symptom is a benchmark that scores below 100%
    for no visible reason.

    Args:
        form (Dict[str, Any]): The form specification under test.

    Returns:
        None
    """
    differences = check_sample(form=form, sample_root=str(SAMPLE_ROOT))

    assert differences == [], (
        "The tracked bundle no longer matches the generator. Regenerate with "
        "`python tools/generate_synthetic_samples.py`. Differences: "
        + "; ".join(differences))


@pytest.mark.parametrize("form", FORMS, ids=form_id)
def test_the_tracked_truth_is_the_generator_truth(form: Dict[str, Any]) -> None:
    """Ground truth is written from the values the pages are drawn from.

    Args:
        form (Dict[str, Any]): The form specification under test.

    Returns:
        None
    """
    tracked = json.loads((SAMPLE_ROOT / form["id"] / "truth.json").read_text())

    assert tracked == form["truth"]


@pytest.mark.parametrize("form", FORMS, ids=form_id)
def test_the_tracked_schema_is_the_generator_schema(form: Dict[str, Any]) -> None:
    """A schema that drifts from the truth cannot score 100% however good the OCR.

    Args:
        form (Dict[str, Any]): The form specification under test.

    Returns:
        None
    """
    tracked = json.loads((SAMPLE_ROOT / form["id"] / "schema.json").read_text())

    assert tracked == form["schema"]


@pytest.mark.parametrize("form", FORMS, ids=form_id)
def test_the_document_is_multi_page(form: Dict[str, Any]) -> None:
    """Single-page samples cannot exercise page chunking, merging or per-page stats.

    Args:
        form (Dict[str, Any]): The form specification under test.

    Returns:
        None
    """
    document = fitz.open(SAMPLE_ROOT / form["id"] / f"{form['id']}.pdf")
    try:
        page_count = document.page_count
    finally:
        document.close()

    assert page_count == len(form["pages"]), (
        f"{form['id']} was drawn with {len(form['pages'])} pages but the tracked PDF "
        f"has {page_count}")
    assert page_count > 1


# --- no text layer ----------------------------------------------------------


@pytest.mark.parametrize("form", FORMS, ids=form_id)
def test_the_tracked_document_has_no_text_layer(form: Dict[str, Any]) -> None:
    """Every page is a raster, so the engines have to OCR rather than read.

    The generator draws a crisp page, rasterises it and discards the original. If a
    text layer survived, Bedrock and BDA would extract from the text rather than from
    the image and the accuracy and cost figures would describe a different task.

    Args:
        form (Dict[str, Any]): The form specification under test.

    Returns:
        None
    """
    document = fitz.open(SAMPLE_ROOT / form["id"] / f"{form['id']}.pdf")
    try:
        with_text = [index for index, page in enumerate(document, start=1)
                     if page.get_text().strip()]
    finally:
        document.close()

    assert with_text == [], (
        f"{form['id']} carries an extractable text layer on page(s) {with_text}, so "
        f"the engines would read it instead of performing OCR")


@pytest.mark.parametrize("form", FORMS, ids=form_id)
def test_a_freshly_generated_document_has_no_text_layer_either(
    form: Dict[str, Any], rendered: Dict[str, bytes]
) -> None:
    """The no-text-layer property comes from the generator, not from a one-off edit.

    Args:
        form (Dict[str, Any]): The form specification under test.
        rendered (Dict[str, bytes]): Freshly generated PDF bytes per bundle.

    Returns:
        None
    """
    document = fitz.open(stream=rendered[form["id"]], filetype="pdf")
    try:
        with_text = [index for index, page in enumerate(document, start=1)
                     if page.get_text().strip()]
    finally:
        document.close()

    assert with_text == []


# --- determinism ------------------------------------------------------------


@pytest.mark.parametrize("form", FORMS, ids=form_id)
def test_generating_twice_renders_identically(form: Dict[str, Any]) -> None:
    """Regenerating must not churn the tracked PDFs.

    The pages carry seeded noise and skew, so an unseeded generator would produce a
    different multi-hundred-kilobyte binary on every run and every regeneration would
    show up as a diff nobody can review.

    Args:
        form (Dict[str, Any]): The form specification under test.

    Returns:
        None
    """
    first = page_fingerprints(pdf_bytes=build_pdf(form=form, seed=form["seed"]))
    second = page_fingerprints(pdf_bytes=build_pdf(form=form, seed=form["seed"]))

    assert first == second


@pytest.mark.parametrize("form", FORMS, ids=form_id)
def test_a_different_seed_renders_differently(form: Dict[str, Any]) -> None:
    """Confirms the determinism test above is not vacuous.

    If the seed were ignored, the fingerprints would match for any seed and the test
    above would pass without proving anything about seeding.

    Args:
        form (Dict[str, Any]): The form specification under test.

    Returns:
        None
    """
    seeded = page_fingerprints(pdf_bytes=build_pdf(form=form, seed=form["seed"]))
    other = page_fingerprints(pdf_bytes=build_pdf(form=form, seed=form["seed"] + 1))

    assert seeded != other


def test_the_check_flag_passes_against_the_tracked_corpus() -> None:
    """`--check` is the command CI and a reviewer run, so it has to work end to end.

    Returns:
        None
    """
    completed = subprocess.run(
        [sys.executable, "tools/generate_synthetic_samples.py", "--check"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)

    assert completed.returncode == 0, (
        f"--check reported differences:\n{completed.stdout}{completed.stderr}")


# --- the samples are synthetic ---------------------------------------------


@pytest.mark.parametrize("form", FORMS, ids=form_id)
def test_the_claimant_identifier_is_marked_synthetic(form: Dict[str, Any]) -> None:
    """A reader must not mistake a tracked sample for a real claim.

    The identifier a human would quote when looking a claim up carries a `SYN-` prefix,
    so a page, a run record or a support question naming it is unambiguously about a
    made-up document.

    Args:
        form (Dict[str, Any]): The form specification under test.

    Returns:
        None
    """
    identifiers = [(key, value) for key, value in _leaves(form["truth"])
                   if isinstance(value, str) and value.startswith("SYN-")]

    assert identifiers, (
        f"{form['id']} carries no SYN- prefixed identifier, so nothing in its ground "
        f"truth marks the sample as made up")


@pytest.mark.parametrize("form", FORMS, ids=form_id)
def test_no_value_looks_like_a_real_claim_number(form: Dict[str, Any]) -> None:
    """No tracked value may carry the shape of a bare nine-digit claim number.

    Nine digits with no prefix is the shape a carrier claim number takes, and a value
    of that shape reads as a real one however invented it is. Every identifier here is
    deliberately shorter and `SYN-` prefixed.

    Args:
        form (Dict[str, Any]): The form specification under test.

    Returns:
        None
    """
    serialised = json.dumps(form["truth"]) + json.dumps(form["schema"])
    matches = sorted(set(re.findall(r"\b\d{9}\b", serialised)))

    assert matches == [], (
        f"{form['id']} contains values shaped like a bare claim number: {matches}")


def _leaves(value: Any, key: str = ""):
    """Walk a nested structure yielding (key, scalar) for every leaf.

    Args:
        value (Any): The structure to walk.
        key (str): The key the current value was reached by.

    Yields:
        Tuple[str, Any]: Each leaf's key and value.
    """
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _leaves(child, child_key)
    elif isinstance(value, list):
        for child in value:
            yield from _leaves(child, key)
    else:
        yield key, value


# --- local-only bundles stay out -------------------------------------------


def test_the_local_only_directory_is_ignored() -> None:
    """One rule has to cover a local-only sample's document, schema and truth.

    Under the four-directory layout a document and the values transcribed out of it sat
    in different directories, so holding the document back said nothing about its
    ground truth. Bundling all three files in one directory is what makes a single
    ignore rule sufficient, and this asserts the rule is actually there.

    Returns:
        None
    """
    assert "sample/private/" in GITIGNORE.read_text().splitlines(), (
        f"{GITIGNORE.name} must ignore the local-only sample directory, or a bundle "
        f"placed there would be committed")


@pytest.mark.parametrize(
    "relative_path",
    [
        "sample/private/claims/STD/case-77315/case-77315.pdf",
        "sample/private/claims/STD/case-77315/schema.json",
        "sample/private/claims/STD/case-77315/truth.json",
    ],
    ids=["document", "schema", "truth"],
)
def test_git_ignores_every_file_in_a_local_only_bundle(relative_path: str) -> None:
    """Asks git directly, because reading the rule is not the same as it applying.

    `git check-ignore` is the only authority on whether a path would be committed; a
    later negation rule or a different pattern ordering could undo the line asserted
    above.

    Args:
        relative_path (str): Path inside a hypothetical local-only bundle.

    Returns:
        None
    """
    completed = subprocess.run(
        ["git", "check-ignore", "-q", relative_path],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)

    assert completed.returncode == 0, (
        f"git would track {relative_path}, so a local-only sample placed there could "
        f"be committed")


def test_a_normal_sample_bundle_is_not_ignored() -> None:
    """Confirms the ignore assertions above are not matching everything under sample/.

    Returns:
        None
    """
    completed = subprocess.run(
        ["git", "check-ignore", "-q", "sample/pfl-synthetic/truth.json"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)

    assert completed.returncode == 1, (
        "The tracked synthetic samples must not be ignored, or a fresh clone would "
        "have nothing to benchmark")
