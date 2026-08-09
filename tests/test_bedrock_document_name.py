"""
Tests for the document name attached to every Bedrock PDF request

Converse requires a name on each document content block and restricts it to
alphanumerics, whitespace, hyphens, parentheses and square brackets, with no two
consecutive whitespace characters. A name outside that set is a hard
ValidationException, not a warning, so the whole request fails.

A period is not in the permitted set. `_sanitize_document_name` used to append ".pdf"
after sanitising, which made every name it produced illegal - undetected because
nothing called it until PDFs moved onto Converse. Scanned claim forms arrive with names
like "PFL - 4410772_scan.pdf", so underscores, periods and " - " all have to be handled
rather than assumed absent.
"""

import pytest

from engines.bedrock_engine import BedrockEngine

# Characters Bedrock permits in a document name.
PERMITTED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -()[]")

# Filenames in the shape the claim-form scans use, plus the awkward cases. The names
# are invented; only their character shape matters here.
SAMPLE_NAMES = [
    "PFL - 4410772_scan.pdf",
    "STD_8802455_scan.pdf",
    "7731509 - STD_scan.pdf",
    "claim.form.v2.pdf",
    "form (copy) [final].pdf",
    "réçu#1@2.pdf",
    "___.pdf",
    ".pdf",
    "no-extension",
]


class NamedFile:
    """A stand-in for an uploaded file object, which the engine reads `.name` from"""

    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): Path the object reports as its name.
        """
        self.name = name


@pytest.mark.parametrize("filename", SAMPLE_NAMES)
def test_names_contain_only_permitted_characters(filename: str) -> None:
    """
    Every derived name is one Bedrock will accept

    Args:
        filename (str): Input document filename.
    """
    name = BedrockEngine()._sanitize_document_name(f"/tmp/{filename}")

    illegal = sorted(set(name) - PERMITTED)
    assert not illegal, f"'{name}' contains characters Bedrock rejects: {illegal}"


@pytest.mark.parametrize("filename", SAMPLE_NAMES)
def test_names_carry_no_extension(filename: str) -> None:
    """
    No name carries a ".pdf" suffix

    Regression test for the specific defect: the period in the extension is itself
    illegal, so appending one guaranteed a rejected request.

    Args:
        filename (str): Input document filename.
    """
    name = BedrockEngine()._sanitize_document_name(f"/tmp/{filename}")

    assert "." not in name


@pytest.mark.parametrize("filename", SAMPLE_NAMES)
def test_names_are_never_empty(filename: str) -> None:
    """
    A name is always produced, even from an input that sanitises away to nothing

    Converse rejects an empty name as readily as an illegal one.

    Args:
        filename (str): Input document filename.
    """
    assert BedrockEngine()._sanitize_document_name(f"/tmp/{filename}")


@pytest.mark.parametrize("filename", SAMPLE_NAMES)
def test_no_consecutive_whitespace(filename: str) -> None:
    """
    No name contains two whitespace characters in a row

    Bedrock calls this out as a separate restriction from the character set.

    Args:
        filename (str): Input document filename.
    """
    assert "  " not in BedrockEngine()._sanitize_document_name(f"/tmp/{filename}")


def test_the_real_document_name_is_carried_through() -> None:
    """
    The name is recognisably the document's own, not a placeholder

    The model is shown this name, so replacing every document with "document" would
    withhold a genuine hint about what it is looking at.
    """
    name = BedrockEngine()._sanitize_document_name("/tmp/PFL - 4410772_scan.pdf")

    assert "4410772" in name


def test_a_file_object_is_read_from_its_name_attribute() -> None:
    """Gradio hands the engine a file object rather than a path string"""
    name = BedrockEngine()._sanitize_document_name(
        NamedFile("/var/folders/x/STD_8802455_scan.pdf"))

    assert "8802455" in name
    assert "/" not in name


@pytest.mark.parametrize("value", [None, 42, ""], ids=["none", "int", "empty"])
def test_an_input_with_no_usable_name_falls_back(value) -> None:
    """
    An input carrying no filename still yields a legal name

    An in-memory image has no name at all, and the engine calls this unconditionally.

    Args:
        value: Input with no usable filename.
    """
    assert BedrockEngine()._sanitize_document_name(value) == "document"
