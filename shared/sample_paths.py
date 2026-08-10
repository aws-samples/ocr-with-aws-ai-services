"""Locate a sample's document, output schema and ground truth.

Every preloaded sample is a *bundle*: one directory under `sample/` holding exactly
one document plus an optional `schema.json` and `truth.json`. A bundle's label - the
string shown in the dropdown - is its path relative to `sample/`, so bundles can be
grouped in sub-directories and two samples whose documents share a filename stay
distinct.

The bundle layout replaced four parallel directories (`sample/images`,
`sample/source`, `sample/schema`, `sample/truth`) keyed on the document's basename.
That had two problems this module exists to avoid:

- images and PDFs were discovered by different rules, so a PDF at the top level or an
  image in a group could not be selected at all; and
- a sample's schema and ground truth lived in directories that have to be tracked in
  git, so ignoring a *document* left the ground truth transcribed from it tracked. In a
  bundle, one ignore rule covers all three files.

This module deliberately imports nothing from the application. `shared.evaluator` and
`sample_handler` both depend on it, and `sample_handler` imports the engines, so any
import in the other direction would cycle.
"""

import os
from typing import Dict, List, Optional

# Resolved relative to the working directory, as the rest of the app's sample and
# results paths are. Tests point it at a temporary tree by monkeypatching this name,
# which only works while every function below reads it at call time - do not capture
# it in a default argument.
SAMPLE_DIR = "sample"

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
PDF_EXTENSION = '.pdf'
DOCUMENT_EXTENSIONS = IMAGE_EXTENSIONS + (PDF_EXTENSION,)

SCHEMA_FILENAME = "schema.json"
TRUTH_FILENAME = "truth.json"


def _is_within(*, path: str, directory: str) -> bool:
    """Return whether path resolves inside directory, including symlink targets."""
    try:
        return os.path.commonpath(
            [os.path.realpath(path), os.path.realpath(directory)]
        ) == os.path.realpath(directory)
    except ValueError:
        # Different Windows drives have no common path. It is not possible for one
        # to be inside the other.
        return False


def bundle_dir(*, sample_name: str) -> Optional[str]:
    """
    Resolve a sample label to a discovered bundle directory

    Args:
        sample_name: Bundle label, i.e. a path relative to SAMPLE_DIR such as
                     "sheet" or "claims/STD/case-77315"

    Returns:
        Path to the bundle directory, or None when the label was not discovered
    """
    if not sample_name:
        return None

    return _sample_bundle_directories().get(sample_name)


def bundle_document_path(*, directory: str) -> Optional[str]:
    """
    Find the single document inside a bundle directory

    Args:
        directory: Path to a candidate bundle directory

    Returns:
        Path to the one document in the directory, or None when the directory holds no
        document at all - which is how a directory that merely groups other bundles,
        or a stray directory such as a schema-only draft, is recognised

    Raises:
        ValueError: If the directory holds more than one document. There is no
                    defensible way to guess which one the bundle's schema.json and
                    truth.json describe, and guessing would silently score a run
                    against the wrong ground truth.
    """
    if not os.path.isdir(directory):
        return None

    documents = sorted(
        entry for entry in os.listdir(directory)
        if entry.lower().endswith(DOCUMENT_EXTENSIONS)
        and os.path.isfile(os.path.join(directory, entry))
        and _is_within(path=os.path.join(directory, entry), directory=directory))

    if not documents:
        return None

    if len(documents) > 1:
        raise ValueError(
            f"Sample bundle '{directory}' holds {len(documents)} documents "
            f"({', '.join(documents)}), but a bundle must hold exactly one so that "
            f"{SCHEMA_FILENAME} and {TRUTH_FILENAME} describe an unambiguous "
            f"document. Split it into one directory per document.")

    return os.path.join(directory, documents[0])


def _sample_bundle_directories() -> Dict[str, str]:
    """Build the label-to-directory registry from the sample tree on disk."""
    if not os.path.isdir(SAMPLE_DIR):
        return {}

    bundles: Dict[str, str] = {}

    for dir_path, dir_names, _file_names in os.walk(SAMPLE_DIR):
        if not _is_within(path=dir_path, directory=SAMPLE_DIR):
            dir_names[:] = []
            continue
        if os.path.normpath(dir_path) == os.path.normpath(SAMPLE_DIR):
            continue
        if bundle_document_path(directory=dir_path) is not None:
            bundles[os.path.relpath(dir_path, SAMPLE_DIR)] = dir_path

    return bundles


def list_sample_bundles() -> List[str]:
    """
    List the label of every bundle under SAMPLE_DIR

    Walks the whole tree, so bundles may be nested to any depth for grouping.

    Returns:
        Sorted list of bundle labels, empty when SAMPLE_DIR does not exist

    Raises:
        ValueError: Propagated from bundle_document_path() for a directory holding
                    more than one document
    """
    return sorted(_sample_bundle_directories())


def sample_document_path(*, sample_name: str) -> Optional[str]:
    """
    Resolve a sample label to the document on disk

    Args:
        sample_name: Bundle label as produced by list_sample_bundles()

    Returns:
        Path to the document, or None when the label is not a bundle - which is the
        case for an uploaded file, whose name reaches this module through
        load_truth_data() and is_pdf_sample()
    """
    if not sample_name:
        return None

    directory = bundle_dir(sample_name=sample_name)
    if directory is None:
        return None

    return bundle_document_path(directory=directory)


def sample_schema_path(*, sample_name: str) -> Optional[str]:
    """
    Build the output-schema path for a sample label

    Args:
        sample_name: Bundle label as produced by list_sample_bundles()

    Returns:
        Path to the bundle's schema.json, or None when the label is not a bundle
    """
    directory = bundle_dir(sample_name=sample_name)
    if directory is None:
        return None

    return os.path.join(directory, SCHEMA_FILENAME)


def sample_truth_path(*, sample_name: str) -> Optional[str]:
    """
    Build the ground-truth path for a sample label

    Args:
        sample_name: Bundle label as produced by list_sample_bundles()

    Returns:
        Path to the bundle's truth.json, or None when the label is not a bundle
    """
    directory = bundle_dir(sample_name=sample_name)
    if directory is None:
        return None

    return os.path.join(directory, TRUTH_FILENAME)


def is_pdf_sample(*, sample_name: str) -> bool:
    """
    Report whether a sample is a PDF rather than a single image

    A bundle label carries no extension, so the answer comes from the document inside
    the bundle. An uploaded file is not a bundle, so its own extension is used - the
    batch summary and the preview both classify uploads as well as samples.

    Args:
        sample_name: Bundle label, or the name of an uploaded file

    Returns:
        True when the resolved document, or the name itself, ends in .pdf
    """
    document_path = sample_document_path(sample_name=sample_name)
    name_to_test = document_path if document_path else (sample_name or "")
    return name_to_test.lower().endswith(PDF_EXTENSION)
