"""
Tests for sample discovery over the bundle layout under sample/

Every preloaded sample is a directory holding exactly one document plus an optional
schema.json and truth.json, and its dropdown label is that directory's path relative
to sample/. These tests run against a tree they build themselves rather than the real
corpus, so they pass on a fresh clone whatever it happens to ship.

The claim numbers and names here are invented, and the bundle layout is what keeps it
easy to hold a sample back from the repository.
"""
import json
import os

import pytest

import sample_handler
import shared.sample_paths
from sample_handler import (
    is_pdf_sample,
    list_sample_documents,
    load_sample_document_and_schema,
    load_sample_schema,
    resolve_sample_path,
    sample_result_directory,
    sample_schema_path,
)
from shared.evaluator import load_truth_data
from shared.sample_paths import (
    SCHEMA_FILENAME,
    TRUTH_FILENAME,
    bundle_document_path,
    sample_document_path,
    sample_truth_path,
)

SCHEMA = {"type": "object", "properties": {"claimNumber": {"type": "string"}}}
TRUTH = {"claimNumber": "SYN-4410772"}

# Bundle labels are paths, so the separator has to be built rather than hard-coded.
PFL_WITH_TRUTH = os.path.join("claims", "PFL", "pfl-with-truth")
PFL_BARE = os.path.join("claims", "PFL", "pfl-bare")
STD_BUNDLE = os.path.join("claims", "STD", "std-bare")


@pytest.fixture
def sample_tree(tmp_path, monkeypatch):
    """
    Build a synthetic bundle tree and point the sample handler at it

    Covers every case discovery has to get right: two top-level image bundles, three
    PDF bundles nested two levels deep for grouping, a bundle carrying a schema and
    ground truth, a directory holding no document at all, and non-document files that
    must be ignored rather than offered as samples.

    Args:
        tmp_path: pytest temporary directory
        monkeypatch: pytest monkeypatch fixture

    Returns:
        Path to the synthetic sample root
    """
    sample_root = tmp_path / "sample"

    bundles = {
        "sheet": "sheet.jpg",
        "chart": "chart.png",
        PFL_WITH_TRUTH: "pfl-with-truth.pdf",
        PFL_BARE: "pfl-bare.pdf",
        STD_BUNDLE: "std-bare.pdf",
    }

    for label, document in bundles.items():
        directory = sample_root / label
        directory.mkdir(parents=True)
        (directory / document).write_bytes(b"%PDF-1.4 fake" if document.endswith(".pdf")
                                          else b"not-a-real-image")

    with_truth = sample_root / PFL_WITH_TRUTH
    (with_truth / SCHEMA_FILENAME).write_text(json.dumps(SCHEMA))
    (with_truth / TRUTH_FILENAME).write_text(json.dumps(TRUTH))

    # Must be ignored: files that are not documents, sitting inside a real bundle.
    (sample_root / "sheet" / "notes.txt").write_text("ignored")
    (sample_root / "sheet" / ".DS_Store").write_bytes(b"ignored")
    (sample_root / STD_BUNDLE / "cover.docx").write_bytes(b"ignored")

    # Must not be a bundle: a directory with no document in it at all. A README beside
    # the samples, and the grouping directories themselves, are exactly this case.
    (sample_root / "docs-only").mkdir()
    (sample_root / "docs-only" / "README.md").write_text("not a sample")
    (sample_root / "README.md").write_text("not a sample either")

    # SAMPLE_DIR is relative, so changing directory is what redirects every lookup -
    # including load_truth_data(), which resolves ground truth through the same module.
    monkeypatch.chdir(tmp_path)

    return sample_root


def test_lists_every_bundle_by_its_path(sample_tree):
    """A label is the bundle's path relative to sample/, at whatever depth"""
    documents = list_sample_documents()

    assert "sheet" in documents
    assert "chart" in documents
    assert PFL_WITH_TRUTH in documents
    assert STD_BUNDLE in documents


def test_images_and_pdfs_are_discovered_by_one_rule(sample_tree):
    """
    Images and PDFs are found the same way, at the same depths

    The old layout scanned sample/images non-recursively for images and sample/source
    recursively for PDFs, so a PDF at the top level or an image inside a group could
    not be selected at all.
    """
    documents = list_sample_documents()

    assert {"sheet", "chart"} <= set(documents), "top-level bundles must be found"
    assert PFL_BARE in documents, "a bundle nested two levels deep must be found too"


def test_ignores_non_document_files(sample_tree):
    """A .txt, a .docx and .DS_Store are not selectable samples"""
    documents = list_sample_documents()

    assert not any(name.endswith((".txt", ".docx", ".DS_Store")) for name in documents)


def test_a_directory_without_a_document_is_not_a_bundle(sample_tree):
    """
    A directory holding no document is skipped rather than offered or rejected

    This is what lets sample/README.md and the grouping directories coexist with the
    samples: "no document here" is a normal answer, not an error.
    """
    documents = list_sample_documents()

    assert "docs-only" not in documents
    assert "claims" not in documents
    assert os.path.join("claims", "PFL") not in documents


def test_finds_every_bundle_exactly_once(sample_tree):
    """Two image bundles plus three PDF bundles, with no duplicate labels"""
    documents = list_sample_documents()

    assert len(documents) == 5
    assert len(set(documents)) == len(documents)


def test_labels_are_sorted(sample_tree):
    """A sorted dropdown groups the nested bundles under their parent directory"""
    documents = list_sample_documents()

    assert documents == sorted(documents)


def test_two_documents_in_one_bundle_fails_loudly(sample_tree):
    """
    A bundle with two documents raises rather than picking one

    There is no way to know which document the bundle's schema.json and truth.json
    describe, and guessing would score a run against the wrong ground truth and report
    the result as though it were meaningful.
    """
    (sample_tree / "sheet" / "extra.png").write_bytes(b"second document")

    with pytest.raises(ValueError, match="holds 2 documents"):
        list_sample_documents()


def test_the_two_document_error_names_the_files(sample_tree):
    """The message has to say which files collided, or the fix is guesswork"""
    (sample_tree / "sheet" / "extra.png").write_bytes(b"second document")

    with pytest.raises(ValueError) as error:
        bundle_document_path(directory=str(sample_tree / "sheet"))

    assert "extra.png" in str(error.value)
    assert "sheet.jpg" in str(error.value)


def test_resolves_a_label_to_the_document_inside_the_bundle(sample_tree):
    """The document is found by looking inside the bundle, not by name"""
    resolved = resolve_sample_path(PFL_WITH_TRUTH)

    assert os.path.exists(resolved)
    assert resolved.endswith("pfl-with-truth.pdf")


def test_every_listed_label_resolves(sample_tree):
    """Nothing the dropdown offers can fail to resolve"""
    for label in list_sample_documents():
        assert os.path.exists(resolve_sample_path(label))


def test_missing_sample_raises(sample_tree):
    """A stale label fails loudly rather than returning None"""
    with pytest.raises(FileNotFoundError, match="Sample not found on disk"):
        resolve_sample_path(os.path.join("claims", "PFL", "does-not-exist"))


def test_an_empty_bundle_directory_raises_on_selection(sample_tree):
    """A directory the dropdown never offered still cannot resolve silently"""
    with pytest.raises(FileNotFoundError, match="Sample not found on disk"):
        resolve_sample_path("docs-only")


def test_parent_path_cannot_select_files_outside_sample_root(sample_tree):
    """A forged dropdown value must not turn into a filesystem path."""
    outside = sample_tree.parent / "outside"
    outside.mkdir()
    (outside / "outside.pdf").write_bytes(b"%PDF-1.4 fake")
    (outside / SCHEMA_FILENAME).write_text(json.dumps(SCHEMA))
    (outside / TRUTH_FILENAME).write_text(json.dumps(TRUTH))
    forged_label = os.path.join("..", "outside")

    assert sample_document_path(sample_name=forged_label) is None
    assert sample_schema_path(forged_label) is None
    assert sample_truth_path(sample_name=forged_label) is None
    assert load_truth_data(forged_label) == ({}, False)


def test_schema_and_truth_live_inside_the_bundle(sample_tree):
    """
    Both sit beside the document rather than in shared directories keyed on filename

    That is what makes one .gitignore line enough to hold a sample's schema and ground
    truth back from the repository along with its document.
    """
    assert sample_schema_path(PFL_WITH_TRUTH) == os.path.join(
        "sample", PFL_WITH_TRUTH, SCHEMA_FILENAME)
    assert sample_truth_path(sample_name=PFL_WITH_TRUTH) == os.path.join(
        "sample", PFL_WITH_TRUTH, TRUTH_FILENAME)


def test_truth_is_found_for_a_nested_bundle(sample_tree):
    """load_truth_data() finds ground truth for a grouped sample from the dropdown"""
    truth_data, truth_exists = load_truth_data(PFL_WITH_TRUTH)

    assert truth_exists is True
    assert truth_data == TRUTH


def test_same_named_documents_in_different_bundles_keep_separate_truth(sample_tree):
    """
    Two samples whose documents share a filename must not share ground truth

    Under the old basename-keyed layout they did, so adding a second scan called
    scan.pdf silently scored it against the first one's truth.
    """
    for label, claim_number in (("first", "SYN-1000001"), ("second", "SYN-2000002")):
        directory = sample_tree / "shared-name" / label
        directory.mkdir(parents=True)
        (directory / "scan.pdf").write_bytes(b"%PDF-1.4 fake")
        (directory / TRUTH_FILENAME).write_text(
            json.dumps({"claimNumber": claim_number}))

    first, _ = load_truth_data(os.path.join("shared-name", "first"))
    second, _ = load_truth_data(os.path.join("shared-name", "second"))

    assert first == {"claimNumber": "SYN-1000001"}
    assert second == {"claimNumber": "SYN-2000002"}


def test_same_named_nested_bundles_get_distinct_result_directories(sample_tree):
    """Batch output keeps the group path instead of collapsing to one basename."""
    for group in ("first-group", "second-group"):
        directory = sample_tree / group / "receipt"
        directory.mkdir(parents=True)
        (directory / "scan.pdf").write_bytes(b"%PDF-1.4 fake")

    run_dir = os.path.join("results", "run-test")
    first = sample_result_directory(
        run_dir=run_dir,
        sample_name=os.path.join("first-group", "receipt"))
    second = sample_result_directory(
        run_dir=run_dir,
        sample_name=os.path.join("second-group", "receipt"))

    assert first != second
    assert first == os.path.join(run_dir, "first-group", "receipt")
    assert second == os.path.join(run_dir, "second-group", "receipt")


def test_truth_is_absent_for_a_bundle_without_it(sample_tree):
    """A sample with no transcribed ground truth reports that, and does not raise"""
    truth_data, truth_exists = load_truth_data(PFL_BARE)

    assert (truth_data, truth_exists) == ({}, False)


def test_an_uploaded_filename_has_no_truth(sample_tree):
    """An uploaded file is not a bundle, so it has no ground truth to find"""
    assert load_truth_data("some-upload.pdf") == ({}, False)


def test_loads_document_path_and_schema_together(sample_tree):
    """Selecting a benchmark PDF yields both its path and its schema text"""
    document_path, schema_text = load_sample_document_and_schema(PFL_WITH_TRUTH)

    assert document_path.endswith("pfl-with-truth.pdf")
    assert json.loads(schema_text) == SCHEMA


def test_bundle_without_schema_returns_none(sample_tree):
    """A sample with no transcribed schema still loads, with no schema"""
    document_path, schema_text = load_sample_document_and_schema(PFL_BARE)

    assert document_path.endswith("pfl-bare.pdf")
    assert schema_text is None


def test_batch_schema_lookup_matches_dropdown_lookup(sample_tree):
    """Batch mode and the dropdown must agree on which schema a sample uses"""
    _document_path, dropdown_schema = load_sample_document_and_schema(PFL_WITH_TRUTH)
    batch_schema = load_sample_schema(PFL_WITH_TRUTH, default_schema="")

    assert json.loads(batch_schema) == json.loads(dropdown_schema)


def test_batch_schema_falls_back_to_the_editor_schema(sample_tree):
    """With no per-sample schema, batch mode uses whatever is in the editor"""
    default_schema = json.dumps({"type": "object", "properties": {}})

    batch_schema = load_sample_schema(PFL_BARE, default_schema=default_schema)

    assert batch_schema == default_schema


def test_pdf_detection_reads_the_document_not_the_label(sample_tree):
    """
    A bundle label carries no extension, so the answer comes from inside the bundle

    Page counting, the multi-page preview and the batch summary all branch on this.
    """
    assert is_pdf_sample(PFL_WITH_TRUTH) is True
    assert is_pdf_sample("sheet") is False


def test_pdf_detection_falls_back_to_an_uploaded_filename(sample_tree):
    """An uploaded file is not a bundle, so its own extension has to classify it"""
    assert is_pdf_sample("upload.pdf") is True
    assert is_pdf_sample("upload.jpg") is False


@pytest.mark.parametrize(
    "filename,expected",
    [("scan.pdf", True), ("SCAN.PDF", True), ("scan.jpg", False), ("scan.PNG", False)],
)
def test_pdf_detection_is_case_insensitive(sample_tree, filename, expected):
    """Extension matching must not depend on the case scanners happen to use"""
    directory = sample_tree / "case-test"
    directory.mkdir()
    (directory / filename).write_bytes(b"fake")

    assert is_pdf_sample("case-test") is expected


def test_no_sample_selected_returns_nothing(sample_tree):
    """An empty dropdown value is not an error"""
    assert load_sample_document_and_schema("") == (None, None)
    assert load_sample_document_and_schema(None) == (None, None)


def test_a_missing_sample_directory_lists_nothing(tmp_path, monkeypatch):
    """
    With no sample/ directory at all, discovery is empty rather than an exception

    The UI reports "no samples" from an empty list; a raise here would take the whole
    interface down at import time.
    """
    monkeypatch.chdir(tmp_path)

    assert list_sample_documents() == []


def test_sample_dir_is_read_at_call_time(sample_tree, monkeypatch):
    """
    Every lookup reads SAMPLE_DIR when called, which is what makes it redirectable

    Capturing it in a default argument would freeze the first value imported and
    quietly ignore the override, so this pins the contract the fixtures rely on.
    """
    other_root = sample_tree.parent / "elsewhere"
    (other_root / "only-bundle").mkdir(parents=True)
    (other_root / "only-bundle" / "doc.png").write_bytes(b"fake")

    monkeypatch.setattr(shared.sample_paths, "SAMPLE_DIR", str(other_root))

    assert list_sample_documents() == ["only-bundle"]


def test_the_handler_and_the_evaluator_share_one_source_of_truth(sample_tree,
                                                                monkeypatch):
    """
    Both resolve against the same sample root, so a sample has one identity

    shared.evaluator cannot import sample_handler - sample_handler imports the engines,
    which would cycle - so both go through shared.sample_paths. Redirecting that one
    module has to move document, schema and ground-truth lookups together; a second
    copy of the rules in sample_handler is how schema and truth drifted apart before.
    """
    other_root = sample_tree.parent / "elsewhere"
    bundle = other_root / "redirected"
    bundle.mkdir(parents=True)
    (bundle / "doc.pdf").write_bytes(b"%PDF-1.4 fake")
    (bundle / TRUTH_FILENAME).write_text(json.dumps({"claimNumber": "SYN-3000003"}))

    monkeypatch.setattr(shared.sample_paths, "SAMPLE_DIR", str(other_root))

    assert sample_handler.resolve_sample_path("redirected") == str(bundle / "doc.pdf")
    assert load_truth_data("redirected") == ({"claimNumber": "SYN-3000003"}, True)
