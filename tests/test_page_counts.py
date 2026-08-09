"""Tests for the page counts engines report, and the costs derived from them.

Every engine used to return `"pages": 1` except Textract, so a seven-page PDF was
billed as one page by BDA and reported as one page by the comparison table. These
tests pin the real counts to the payload shapes AWS actually returns - both
verified against a live run on a seven-page claim form - and pin the cost that
follows from them.

No AWS calls: `_page_count` is pure, and the cost functions take the count as an
argument.
"""

from typing import Any, Dict

import pytest

from engines.bda_engine import BDAEngine
from shared.cost_calculator import calculate_bda_cost, describe_bda_cost

# The metadata block a real seven-page BDA standard-output run returned. Trimmed
# to the keys under test; the live payload also carried asset_id, s3_bucket and
# file_type.
SEVEN_PAGE_STANDARD_OUTPUT: Dict[str, Any] = {
    "metadata": {
        "number_of_pages": 7,
        "start_page_index": 0,
        "end_page_index": 6,
    }
}

# What the blueprint path returned for the same document. Note there is no
# `metadata.number_of_pages` here - the page count has to come from the indices.
SEVEN_PAGE_CUSTOM_OUTPUT: Dict[str, Any] = {
    "split_document": {"page_indices": [0, 1, 2, 3, 4, 5, 6]}
}


@pytest.fixture
def engine() -> BDAEngine:
    """Provide a BDA engine for its page-counting helper.

    Constructing the engine does no AWS work; clients are created per call.

    Returns:
        BDAEngine: The engine under test.
    """
    return BDAEngine()


class TestBdaPageCount:
    """BDA's page count comes from the service, not from a hardcoded 1."""

    def test_standard_output_metadata_is_used(self, engine: BDAEngine) -> None:
        """The standard path reports the whole document's page count."""
        assert engine._page_count(
            is_pdf=True,
            custom_output=None,
            standard_output=SEVEN_PAGE_STANDARD_OUTPUT) == 7

    def test_blueprint_page_indices_are_used_when_there_is_no_metadata(
        self, engine: BDAEngine
    ) -> None:
        """The blueprint path can return custom output with no standard output."""
        assert engine._page_count(
            is_pdf=True,
            custom_output=SEVEN_PAGE_CUSTOM_OUTPUT,
            standard_output=None) == 7

    def test_repeated_page_indices_are_counted_once(self, engine: BDAEngine) -> None:
        """A page named twice is still one page, and one page's worth of charge."""
        assert engine._page_count(
            is_pdf=True,
            custom_output={"split_document": {"page_indices": [0, 1, 1, 2]}},
            standard_output=None) == 3

    def test_standard_output_wins_over_page_indices(self, engine: BDAEngine) -> None:
        """Whole-document metadata beats a blueprint's matched subset.

        A blueprint can match a subset of a document's pages, but BDA charges for
        every page it read, so the metadata figure is the billable one.
        """
        assert engine._page_count(
            is_pdf=True,
            custom_output={"split_document": {"page_indices": [0, 1]}},
            standard_output=SEVEN_PAGE_STANDARD_OUTPUT) == 7

    def test_an_image_is_one_unit(self, engine: BDAEngine) -> None:
        """BDA prices images per image, so a single image is always 1."""
        assert engine._page_count(
            is_pdf=False, custom_output=None, standard_output=None) == 1

    def test_a_pdf_with_no_reported_count_raises(self, engine: BDAEngine) -> None:
        """Defaulting to 1 here is exactly the under-billing being fixed."""
        with pytest.raises(ValueError, match="reported no page count"):
            engine._page_count(is_pdf=True, custom_output={}, standard_output={})

    @pytest.mark.parametrize("bad_count", [0, -3, "7", None])
    def test_an_unusable_reported_count_falls_through(
        self, engine: BDAEngine, bad_count: Any
    ) -> None:
        """A count that is not a positive integer is not trusted.

        It falls through to the page indices rather than being used as-is, which
        would produce a zero, negative or non-numeric charge.
        """
        assert engine._page_count(
            is_pdf=True,
            custom_output=SEVEN_PAGE_CUSTOM_OUTPUT,
            standard_output={"metadata": {"number_of_pages": bad_count}}) == 7


class TestBdaCostFollowsThePageCount:
    """The whole point of the count: BDA is billed per page."""

    def test_seven_pages_cost_seven_times_one_page(self) -> None:
        """The defect was a sevenfold under-report on this sample document."""
        _, one_page = calculate_bda_cost(False, 'document', page_count=1)
        _, seven_pages = calculate_bda_cost(False, 'document', page_count=7)
        assert seven_pages == pytest.approx(one_page * 7)

    def test_the_standard_rate_is_a_cent_a_page(self) -> None:
        """Pins the published rate the tooltip quotes."""
        _, cost = calculate_bda_cost(False, 'document', page_count=7)
        assert cost == pytest.approx(0.07)

    def test_the_formula_quotes_the_rate_per_page_not_per_document(self) -> None:
        """"7 pages x $0.010000 per document" read as one charge for the PDF."""
        components = describe_bda_cost(
            use_blueprint=False, document_type='document', page_count=7)

        assert components[0].formula == "7 pages x $0.010000 per page"

    def test_an_image_run_quotes_the_rate_per_image(self) -> None:
        """Images are priced per image, and the formula has to say so."""
        components = describe_bda_cost(
            use_blueprint=False, document_type='image', page_count=3)

        assert "3 images x" in components[0].formula
        assert components[0].formula.endswith("per image")

    def test_the_blueprint_rate_is_higher_than_the_standard_rate(self) -> None:
        """Custom output costs more per page, which is why the flag matters."""
        _, standard = calculate_bda_cost(False, 'document', page_count=7)
        _, custom = calculate_bda_cost(True, 'document', page_count=7)
        assert custom > standard

    def test_extra_blueprint_fields_are_charged_per_page(self) -> None:
        """A 56-field blueprint on 7 pages is charged for 26 extra fields, 7 times."""
        components = describe_bda_cost(
            use_blueprint=True, document_type='document', page_count=7, field_count=56)

        assert len(components) == 2
        extra_field_charge = components[1]
        assert extra_field_charge.amount == pytest.approx(0.0005 * 26 * 7)
        assert "26 extra fields x 7 pages" in extra_field_charge.formula

    def test_a_blueprint_within_the_included_fields_has_no_extra_charge(self) -> None:
        """30 fields are included, so a 30-field blueprint is one component."""
        components = describe_bda_cost(
            use_blueprint=True, document_type='document', page_count=7, field_count=30)
        assert len(components) == 1

    def test_extra_fields_are_not_charged_on_the_standard_path(self) -> None:
        """Standard output has no blueprint, so it has no field count to bill."""
        components = describe_bda_cost(
            use_blueprint=False, document_type='document', page_count=7, field_count=56)
        assert len(components) == 1

    def test_an_unknown_document_type_raises(self) -> None:
        """Silently defaulting to 'document' would misprice an image run."""
        with pytest.raises(ValueError, match="document_type must be"):
            describe_bda_cost(
                use_blueprint=False, document_type='spreadsheet', page_count=1)
