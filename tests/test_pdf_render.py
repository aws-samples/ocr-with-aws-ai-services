"""Tests for rendering PDF pages and drawing bounding boxes on them.

PDF runs used to answer with a 400x600 black rectangle regardless of the
document, so the geometry both engines already had was thrown away. These tests
build small PDFs with PyMuPDF in a temporary directory - no fixture files and no
AWS - and assert the pages come back at the right size, the page cap truncates
loudly rather than silently, and a normalised box lands where the fraction says.
"""

from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import pytest
from PIL import Image

from shared import pdf_render
from shared.pdf_render import (
    DEFAULT_DPI,
    PageBox,
    boxes_from_bda_explainability,
    boxes_from_textract_blocks,
    colour_for_index,
    compose_pdf_visualisation,
    count_pdf_pages,
    draw_boxes,
    render_message_image,
    render_pdf_pages,
    stack_pages_vertically,
)

# US Letter at 72 units per inch, the size the generated PDFs use.
LETTER_WIDTH_POINTS: float = 612
LETTER_HEIGHT_POINTS: float = 792


def make_pdf_bytes(*, page_count: int) -> bytes:
    """Build a PDF with the requested number of US Letter pages.

    Each page carries its own number as text, so a test can tell rendered pages
    apart by their pixel content rather than trusting order alone.

    Args:
        page_count: Number of pages to create.

    Returns:
        bytes: The complete PDF file content.
    """
    document = fitz.open()
    for page_number in range(1, page_count + 1):
        page = document.new_page(
            width=LETTER_WIDTH_POINTS, height=LETTER_HEIGHT_POINTS
        )
        page.insert_text((72, 144), f"Page {page_number}", fontsize=48)
    pdf_bytes: bytes = document.tobytes()
    document.close()
    return pdf_bytes


@pytest.fixture(scope="module")
def three_page_pdf() -> bytes:
    """Provide a three-page PDF.

    Returns:
        bytes: The PDF file content.
    """
    return make_pdf_bytes(page_count=3)


class TestRenderPdfPages:
    """Rendering turns PDF bytes into one image per page."""

    def test_renders_every_page_in_order(self, three_page_pdf: bytes) -> None:
        """All three pages come back, and the total says three."""
        pages, total = render_pdf_pages(pdf_bytes=three_page_pdf)

        assert total == 3
        assert len(pages) == 3
        assert all(isinstance(page, Image.Image) for page in pages)

    def test_page_size_follows_the_requested_dpi(self, three_page_pdf: bytes) -> None:
        """A US Letter page at 120 DPI is 1020x1320 px."""
        pages, _ = render_pdf_pages(pdf_bytes=three_page_pdf, dpi=DEFAULT_DPI)

        expected_width = round(LETTER_WIDTH_POINTS * DEFAULT_DPI / 72)
        expected_height = round(LETTER_HEIGHT_POINTS * DEFAULT_DPI / 72)
        assert pages[0].size == (expected_width, expected_height)

    def test_higher_dpi_produces_a_larger_page(self, three_page_pdf: bytes) -> None:
        """DPI scales the render rather than being ignored."""
        low, _ = render_pdf_pages(pdf_bytes=three_page_pdf, dpi=72, max_pages=1)
        high, _ = render_pdf_pages(pdf_bytes=three_page_pdf, dpi=144, max_pages=1)

        assert high[0].width == pytest.approx(low[0].width * 2, abs=2)

    def test_page_cap_truncates_and_reports_the_real_total(
        self, three_page_pdf: bytes
    ) -> None:
        """The cap limits the images but not the reported page count.

        The caller needs the real total to tell the user what was left out, so a
        truncated render must not look like a shorter document.
        """
        pages, total = render_pdf_pages(pdf_bytes=three_page_pdf, max_pages=2)

        assert len(pages) == 2
        assert total == 3

    def test_cap_truncation_is_logged(
        self, three_page_pdf: bytes, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Dropping pages is logged, so a capped run is never silent."""
        with caplog.at_level("WARNING"):
            render_pdf_pages(pdf_bytes=three_page_pdf, max_pages=1)

        assert "3 pages" in caplog.text

    def test_cap_above_the_page_count_is_not_padded(
        self, three_page_pdf: bytes
    ) -> None:
        """A generous cap returns the pages that exist and no more."""
        pages, total = render_pdf_pages(pdf_bytes=three_page_pdf, max_pages=50)

        assert len(pages) == total == 3


class TestRenderFailsLoudly:
    """Unusable input raises rather than yielding a placeholder image."""

    def test_empty_bytes_raise(self) -> None:
        """No bytes means no document, which is an error not a blank page."""
        with pytest.raises(ValueError, match="empty bytes"):
            render_pdf_pages(pdf_bytes=b"")

    def test_non_pdf_bytes_raise(self) -> None:
        """Content that is not a PDF raises instead of rendering nothing."""
        with pytest.raises(ValueError, match="Could not open PDF"):
            render_pdf_pages(pdf_bytes=b"this is not a PDF at all")

    @pytest.mark.parametrize("dpi", [0, -120])
    def test_non_positive_dpi_raises(self, three_page_pdf: bytes, dpi: int) -> None:
        """A zero or negative DPI cannot produce an image."""
        with pytest.raises(ValueError, match="dpi must be positive"):
            render_pdf_pages(pdf_bytes=three_page_pdf, dpi=dpi)

    def test_non_positive_max_pages_raises(self, three_page_pdf: bytes) -> None:
        """Capping at zero pages would return an empty visualisation."""
        with pytest.raises(ValueError, match="max_pages must be positive"):
            render_pdf_pages(pdf_bytes=three_page_pdf, max_pages=0)


class TestCountPdfPages:
    """Costing needs a page count, and must not pay for a render to get one."""

    @pytest.mark.parametrize("page_count", [1, 3, 7])
    def test_the_count_matches_the_document(self, page_count: int) -> None:
        """Every page is counted, including past the render cap of 10."""
        assert count_pdf_pages(pdf_bytes=make_pdf_bytes(page_count=page_count)) == page_count

    def test_pages_beyond_the_render_cap_are_still_counted(self) -> None:
        """The count is of the document, not of what the visualisation shows.

        `render_pdf_pages` truncates at DEFAULT_MAX_PAGES; billing must not.
        """
        pdf_bytes = make_pdf_bytes(page_count=pdf_render.DEFAULT_MAX_PAGES + 4)
        assert count_pdf_pages(pdf_bytes=pdf_bytes) == pdf_render.DEFAULT_MAX_PAGES + 4

    def test_empty_bytes_raise(self) -> None:
        """Defaulting to 1 page here is exactly the under-billing being fixed."""
        with pytest.raises(ValueError, match="empty bytes"):
            count_pdf_pages(pdf_bytes=b"")

    def test_non_pdf_bytes_raise(self) -> None:
        """An unreadable document has no page count, so it raises."""
        with pytest.raises(ValueError, match="Could not open PDF"):
            count_pdf_pages(pdf_bytes=b"this is not a PDF at all")


class TestDrawBoxes:
    """Normalised geometry is placed by multiplying through the page size."""

    def test_box_is_drawn_at_the_normalised_position(
        self, three_page_pdf: bytes
    ) -> None:
        """A box at 0.25/0.25 marks pixels a quarter of the way in.

        The page starts white apart from one line of text near the top left, so
        a coloured pixel on the box's own edge proves the placement rather than
        finding page content by accident.
        """
        pages, _ = render_pdf_pages(pdf_bytes=three_page_pdf, max_pages=1)
        width, height = pages[0].size

        drawn, skipped = draw_boxes(
            pages=pages,
            boxes=[
                PageBox(
                    page_index=0,
                    left=0.25,
                    top=0.25,
                    width=0.5,
                    height=0.5,
                    colour="#0000FF",
                )
            ],
        )

        assert (drawn, skipped) == (1, 0)
        # Sample the middle of the box's top edge, which the rectangle outline
        # covers, and the middle of the box, which it must leave alone.
        edge = pages[0].getpixel((round(width * 0.5), round(height * 0.25)))
        centre = pages[0].getpixel((round(width * 0.5), round(height * 0.5)))
        assert edge == (0, 0, 255)
        assert centre == (255, 255, 255)

    def test_boxes_land_on_their_own_page(self, three_page_pdf: bytes) -> None:
        """`page_index` selects which rendered page a box is drawn on."""
        pages, _ = render_pdf_pages(pdf_bytes=three_page_pdf)
        width, height = pages[0].size

        draw_boxes(
            pages=pages,
            boxes=[
                PageBox(
                    page_index=2,
                    left=0.25,
                    top=0.25,
                    width=0.5,
                    height=0.5,
                    colour="#0000FF",
                )
            ],
        )

        probe = (round(width * 0.5), round(height * 0.25))
        assert pages[2].getpixel(probe) == (0, 0, 255)
        assert pages[0].getpixel(probe) == (255, 255, 255)
        assert pages[1].getpixel(probe) == (255, 255, 255)

    def test_boxes_on_unrendered_pages_are_counted_not_dropped(
        self, three_page_pdf: bytes, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A box beyond the page cap is reported as skipped and logged."""
        pages, _ = render_pdf_pages(pdf_bytes=three_page_pdf, max_pages=1)

        with caplog.at_level("WARNING"):
            drawn, skipped = draw_boxes(
                pages=pages,
                boxes=[
                    PageBox(page_index=0, left=0.1, top=0.1, width=0.2, height=0.1),
                    PageBox(page_index=2, left=0.1, top=0.1, width=0.2, height=0.1),
                ],
            )

        assert (drawn, skipped) == (1, 1)
        assert "not rendered" in caplog.text

    def test_a_label_does_not_move_the_box(self, three_page_pdf: bytes) -> None:
        """Labelling a box changes nothing about where the rectangle goes."""
        pages, _ = render_pdf_pages(pdf_bytes=three_page_pdf, max_pages=2)
        width, height = pages[0].size
        geometry = dict(left=0.25, top=0.25, width=0.5, height=0.5, colour="#0000FF")

        draw_boxes(pages=pages, boxes=[PageBox(page_index=0, **geometry)])
        draw_boxes(
            pages=pages,
            boxes=[PageBox(page_index=1, label="grossWages (98%)", **geometry)],
        )

        probe = (round(width * 0.5), round(height * 0.25))
        assert pages[0].getpixel(probe) == pages[1].getpixel(probe) == (0, 0, 255)

    def test_label_on_a_box_at_the_top_stays_on_the_page(
        self, three_page_pdf: bytes
    ) -> None:
        """A box flush with the top edge has its label drawn below it instead.

        Drawing above would put the text at a negative y and lose it entirely.
        """
        pages, _ = render_pdf_pages(pdf_bytes=three_page_pdf, max_pages=1)
        width, height = pages[0].size

        draw_boxes(
            pages=pages,
            boxes=[
                PageBox(
                    page_index=0,
                    left=0.25,
                    top=0.0,
                    width=0.5,
                    height=0.05,
                    label="atTheTop",
                    colour="#0000FF",
                )
            ],
        )

        # The strip just under the box has to contain some of the label colour.
        band = pages[0].crop(
            (
                round(width * 0.25),
                round(height * 0.05),
                round(width * 0.75),
                round(height * 0.05) + 18,
            )
        )
        assert any(pixel == (0, 0, 255) for pixel in band.getdata())

    def test_colliding_labels_are_suppressed_but_boxes_are_not(
        self, three_page_pdf: bytes, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Two labels in the same spot draw once; both rectangles still appear.

        A wage table stacks rows a few pixels apart, and drawing every caption
        there made all of them unreadable.
        """
        pages, _ = render_pdf_pages(pdf_bytes=three_page_pdf, max_pages=1)
        overlapping = [
            PageBox(
                page_index=0,
                left=0.2,
                top=0.4 + offset,
                width=0.3,
                height=0.004,
                label="weekNumber (86%)",
                colour="#0000FF",
            )
            for offset in (0.0, 0.006)
        ]

        with caplog.at_level("INFO"):
            drawn, skipped = draw_boxes(pages=pages, boxes=overlapping)

        assert (drawn, skipped) == (2, 0)
        assert "overlapped another label" in caplog.text

    def test_labels_far_apart_are_both_drawn(self, three_page_pdf: bytes) -> None:
        """Suppression only applies to an actual collision."""
        pages, _ = render_pdf_pages(pdf_bytes=three_page_pdf, max_pages=1)
        width, height = pages[0].size
        spaced = [
            PageBox(
                page_index=0,
                left=0.2,
                top=top,
                width=0.3,
                height=0.02,
                label="field",
                colour="#0000FF",
            )
            for top in (0.3, 0.6)
        ]

        draw_boxes(pages=pages, boxes=spaced)

        for top in (0.3, 0.6):
            band = pages[0].crop(
                (
                    round(width * 0.2),
                    round(height * top) - 16,
                    round(width * 0.5),
                    round(height * top) - 1,
                )
            )
            assert any(pixel == (0, 0, 255) for pixel in band.getdata())

    def test_no_boxes_draws_nothing(self, three_page_pdf: bytes) -> None:
        """An empty box list is not an error; some fields have no geometry."""
        pages, _ = render_pdf_pages(pdf_bytes=three_page_pdf, max_pages=1)

        assert draw_boxes(pages=pages, boxes=[]) == (0, 0)


class TestStackPagesVertically:
    """Pages are composed into the one image the Gradio output takes."""

    def test_height_is_the_sum_of_the_pages(self, three_page_pdf: bytes) -> None:
        """Without labels the stack is exactly as tall as its pages."""
        pages, _ = render_pdf_pages(pdf_bytes=three_page_pdf, dpi=72)

        stacked = stack_pages_vertically(pages=pages, max_width=2000)

        assert stacked.width == pages[0].width
        assert stacked.height == sum(page.height for page in pages)

    def test_labels_add_a_strip_per_page(self, three_page_pdf: bytes) -> None:
        """Each caption takes a fixed strip above its page."""
        pages, _ = render_pdf_pages(pdf_bytes=three_page_pdf, dpi=72)
        labels = [f"Page {number}" for number in (1, 2, 3)]

        plain = stack_pages_vertically(pages=pages, max_width=2000)
        labelled = stack_pages_vertically(
            pages=pages, labels=labels, max_width=2000
        )

        extra = labelled.height - plain.height
        assert extra > 0
        assert extra % len(pages) == 0

    def test_wide_pages_are_scaled_down(self, three_page_pdf: bytes) -> None:
        """`max_width` bounds the stack and keeps the aspect ratio."""
        pages, _ = render_pdf_pages(pdf_bytes=three_page_pdf, max_pages=1)
        original_ratio = pages[0].height / pages[0].width

        stacked = stack_pages_vertically(pages=pages, max_width=400)

        assert stacked.width == 400
        assert stacked.height / stacked.width == pytest.approx(
            original_ratio, rel=0.01
        )

    def test_narrow_pages_are_not_scaled_up(self, three_page_pdf: bytes) -> None:
        """A page already inside the width limit is left alone."""
        pages, _ = render_pdf_pages(pdf_bytes=three_page_pdf, dpi=36, max_pages=1)

        stacked = stack_pages_vertically(pages=pages, max_width=2000)

        assert stacked.size == pages[0].size

    def test_boxes_survive_stacking(self, three_page_pdf: bytes) -> None:
        """Drawing then stacking keeps the boxes visible in the composite."""
        pages, _ = render_pdf_pages(pdf_bytes=three_page_pdf, max_pages=2)
        draw_boxes(
            pages=pages,
            boxes=[
                PageBox(
                    page_index=1,
                    left=0.25,
                    top=0.25,
                    width=0.5,
                    height=0.5,
                    colour="#0000FF",
                )
            ],
        )

        stacked = stack_pages_vertically(pages=pages, max_width=4000)

        assert any(pixel == (0, 0, 255) for pixel in stacked.getdata())

    def test_empty_page_list_raises(self) -> None:
        """There is nothing to show, which the caller must not paper over."""
        with pytest.raises(ValueError, match="empty list of pages"):
            stack_pages_vertically(pages=[])

    def test_label_count_mismatch_raises(self, three_page_pdf: bytes) -> None:
        """Mismatched labels would caption the wrong pages."""
        pages, _ = render_pdf_pages(pdf_bytes=three_page_pdf)

        with pytest.raises(ValueError, match="they must match"):
            stack_pages_vertically(pages=pages, labels=["Page 1"])

    def test_non_positive_max_width_raises(self, three_page_pdf: bytes) -> None:
        """A zero width cannot hold a page."""
        pages, _ = render_pdf_pages(pdf_bytes=three_page_pdf, max_pages=1)

        with pytest.raises(ValueError, match="max_width must be positive"):
            stack_pages_vertically(pages=pages, max_width=0)


class TestComposePdfVisualisation:
    """The whole PDF panel both engines return, end to end.

    The captions are drawn as pixels, so they are checked by intercepting the
    labels handed to `stack_pages_vertically` rather than by reading text back
    out of the image.
    """

    def _captured_labels(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> List[List[str]]:
        """Record the labels `compose_pdf_visualisation` captions pages with.

        Args:
            monkeypatch: Fixture used to wrap the stacking call.

        Returns:
            List[List[str]]: A list appended to on each stacking call.
        """
        captured: List[List[str]] = []
        real_stack = pdf_render.stack_pages_vertically

        def spy(*, pages, labels=None, max_width=1000):
            captured.append(list(labels or []))
            return real_stack(pages=pages, labels=labels, max_width=max_width)

        monkeypatch.setattr(pdf_render, "stack_pages_vertically", spy)
        return captured

    def test_every_page_appears_in_one_image(self, three_page_pdf: bytes) -> None:
        """The composite is a single tall image covering all three pages."""
        composed = compose_pdf_visualisation(
            pdf_bytes=three_page_pdf, boxes=[], item_noun="text lines"
        )

        pages, _ = render_pdf_pages(pdf_bytes=three_page_pdf)
        assert composed.height > sum(page.height for page in pages) * 0.9
        assert composed.width <= 1000

    def test_captions_name_the_page_and_the_total(
        self, three_page_pdf: bytes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With boxes present each caption counts the boxes on its own page."""
        captured = self._captured_labels(monkeypatch)

        compose_pdf_visualisation(
            pdf_bytes=three_page_pdf,
            boxes=[
                PageBox(page_index=0, left=0.1, top=0.1, width=0.2, height=0.05),
                PageBox(page_index=2, left=0.1, top=0.2, width=0.2, height=0.05),
                PageBox(page_index=2, left=0.1, top=0.3, width=0.2, height=0.05),
            ],
            item_noun="values located",
        )

        assert captured[0] == [
            "Page 1 of 3  •  1 values located",
            "Page 2 of 3  •  0 values located",
            "Page 3 of 3  •  2 values located",
        ]

    def test_caption_omits_the_count_when_there_is_no_geometry(
        self, three_page_pdf: bytes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run with no geometry says nothing about counts rather than "0"."""
        captured = self._captured_labels(monkeypatch)

        compose_pdf_visualisation(
            pdf_bytes=three_page_pdf, boxes=[], item_noun="values located"
        )

        assert captured[0] == ["Page 1 of 3", "Page 2 of 3", "Page 3 of 3"]

    def test_truncated_pages_are_named_on_the_last_caption(
        self, three_page_pdf: bytes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Boxes on unrendered pages are declared, not silently dropped."""
        captured = self._captured_labels(monkeypatch)

        compose_pdf_visualisation(
            pdf_bytes=three_page_pdf,
            boxes=[
                PageBox(page_index=0, left=0.1, top=0.1, width=0.2, height=0.05),
                PageBox(page_index=2, left=0.1, top=0.2, width=0.2, height=0.05),
            ],
            item_noun="text lines",
            max_pages=2,
        )

        assert len(captured[0]) == 2
        assert captured[0][-1].endswith("1 text lines on pages 3-3 are not shown")

    def test_boxes_are_drawn_on_the_composite(self, three_page_pdf: bytes) -> None:
        """The geometry reaches the returned pixels, not just the captions."""
        composed = compose_pdf_visualisation(
            pdf_bytes=three_page_pdf,
            boxes=[
                PageBox(
                    page_index=1,
                    left=0.25,
                    top=0.25,
                    width=0.5,
                    height=0.5,
                    colour="#0000FF",
                )
            ],
            item_noun="values located",
        )

        # Scaling to the display width resamples the outline, so look for any
        # pixel that is dominantly blue rather than an exact colour match.
        assert any(
            blue > 150 and red < 120 and green < 120
            for red, green, blue in composed.getdata()
        )

    def test_an_unrenderable_pdf_returns_a_message_not_an_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The extraction result still reaches the user, with the cause stated.

        Failing the whole run because a preview could not be drawn would throw
        away the answer the user actually asked for, so this path is loud in the
        log and on screen but not fatal.
        """
        with caplog.at_level("ERROR"):
            composed = compose_pdf_visualisation(
                pdf_bytes=b"this is not a PDF",
                boxes=[],
                item_noun="text lines",
            )

        assert isinstance(composed, Image.Image)
        assert "Could not render the PDF for visualisation" in caplog.text
        # The message image is drawn on a dark red field, which no rendered page
        # ever is - white paper would mean a page slipped through instead.
        assert composed.getpixel((0, 0)) == (38, 20, 22)


class TestBoxesFromTextractBlocks:
    """Textract geometry converts to boxes on the block's own page."""

    def test_line_blocks_become_boxes_on_their_page(self) -> None:
        """`Page` is 1-based in Textract and 0-based in `PageBox`."""
        boxes = boxes_from_textract_blocks(
            blocks=[
                {
                    "BlockType": "LINE",
                    "Page": 3,
                    "Geometry": {
                        "BoundingBox": {
                            "Left": 0.1,
                            "Top": 0.2,
                            "Width": 0.3,
                            "Height": 0.05,
                        }
                    },
                }
            ]
        )

        assert len(boxes) == 1
        assert boxes[0].page_index == 2
        assert (boxes[0].left, boxes[0].top) == (0.1, 0.2)

    def test_missing_page_means_the_first_page(self) -> None:
        """Textract omits `Page` for single-page input, which means page 1."""
        boxes = boxes_from_textract_blocks(
            blocks=[
                {
                    "BlockType": "LINE",
                    "Geometry": {
                        "BoundingBox": {
                            "Left": 0.1,
                            "Top": 0.2,
                            "Width": 0.3,
                            "Height": 0.05,
                        }
                    },
                }
            ]
        )

        assert boxes[0].page_index == 0

    def test_other_block_types_are_ignored(self) -> None:
        """Only the requested block type is converted."""
        blocks: List[dict] = [
            {
                "BlockType": kind,
                "Page": 1,
                "Geometry": {
                    "BoundingBox": {
                        "Left": 0.1,
                        "Top": 0.2,
                        "Width": 0.3,
                        "Height": 0.05,
                    }
                },
            }
            for kind in ("PAGE", "LINE", "WORD")
        ]

        assert len(boxes_from_textract_blocks(blocks=blocks)) == 1
        assert len(boxes_from_textract_blocks(blocks=blocks, block_type="WORD")) == 1

    def test_blocks_without_geometry_are_skipped(self) -> None:
        """A block carrying no bounding box cannot be drawn."""
        boxes = boxes_from_textract_blocks(
            blocks=[{"BlockType": "LINE", "Page": 1, "Text": "no geometry"}]
        )

        assert boxes == []


class TestRenderMessageImage:
    """A failed visualisation shows the reason instead of a blank panel."""

    def test_height_grows_with_the_line_count(self) -> None:
        """Every line gets its own row, so nothing is cut off."""
        one = render_message_image(lines=["only line"])
        three = render_message_image(lines=["a", "b", "c"])

        assert three.height > one.height
        assert three.width == one.width == 900

    def test_message_is_actually_drawn(self) -> None:
        """The image is not left as flat background.

        A blank image would be exactly the placeholder this replaces.
        """
        image = render_message_image(lines=["Could not render the PDF"])
        background = image.getpixel((0, 0))

        assert any(pixel != background for pixel in image.getdata())

    def test_no_lines_raises(self) -> None:
        """An empty message would be indistinguishable from a blank panel."""
        with pytest.raises(ValueError, match="no lines"):
            render_message_image(lines=[])

    def test_non_positive_width_raises(self) -> None:
        """A zero-width image cannot hold text."""
        with pytest.raises(ValueError, match="width must be positive"):
            render_message_image(lines=["text"], width=0)


def explainability_leaf(
    *, page: int, confidence: float = 0.86, value: str = "x"
) -> dict:
    """Build one explainability leaf in the shape BDA returns.

    Copied from a real BDA custom output for a multi-page claim form, captured on
    2026-08-07:
    every leaf carries exactly `success`, `confidence`, `geometry`, `type` and
    `value`, and each geometry entry names its page from 1.

    Args:
        page: One-based page number the value was found on.
        confidence: Extraction confidence, 0.0-1.0.
        value: The extracted value.

    Returns:
        dict: A single explainability leaf.
    """
    return {
        "success": True,
        "confidence": confidence,
        "geometry": [
            {
                "boundingBox": {
                    "top": 0.25,
                    "left": 0.15,
                    "width": 0.3,
                    "height": 0.02,
                },
                "vertices": [{"x": 0.15, "y": 0.25}],
                "page": page,
            }
        ],
        "type": "string",
        "value": value,
    }


class TestBoxesFromBdaExplainability:
    """BDA geometry converts to boxes across groups, tables and pages."""

    def test_group_children_become_boxes_on_their_own_pages(self) -> None:
        """Each leaf in a group is placed on the page its geometry names."""
        boxes = boxes_from_bda_explainability(
            explainability_info=[
                {
                    "pfl1PartA": {
                        "gender": explainability_leaf(page=1),
                        "employeeSignatureDate": explainability_leaf(page=2),
                    }
                }
            ]
        )

        assert [box.page_index for box in boxes] == [0, 1]

    def test_page_numbering_is_converted_from_one_based(self) -> None:
        """BDA counts explainability pages from 1; `PageBox` counts from 0.

        The real capture showed pages 1-7 for a seven-page PDF, alongside
        `split_document.page_indices` of 0-6, so the two differ by one.
        """
        boxes = boxes_from_bda_explainability(
            explainability_info=[
                {"guardianDirectPayEnrollment": {"faxNumber": explainability_leaf(page=7)}}
            ]
        )

        assert boxes[0].page_index == 6

    def test_every_row_of_a_table_is_boxed(self) -> None:
        """A table is a list of row dicts, and each row's cells are boxed."""
        boxes = boxes_from_bda_explainability(
            explainability_info=[
                {
                    "pfl1PartB__grossWages": [
                        {"weekNumber": explainability_leaf(page=3)},
                        {"weekNumber": explainability_leaf(page=3)},
                    ]
                }
            ]
        )

        assert len(boxes) == 2
        assert boxes[0].label == boxes[1].label == "weekNumber (86%)"

    def test_label_is_the_field_name_not_the_whole_path(self) -> None:
        """Only the innermost name is drawn, so table labels stay legible.

        `pfl1PartB.grossWages[1].weekNumber` is wider than the cell it annotates
        and overlapped its neighbours; the group is conveyed by colour instead.
        """
        boxes = boxes_from_bda_explainability(
            explainability_info=[
                {"pfl1PartB__grossWages": [{"weekNumber": explainability_leaf(page=3)}]}
            ]
        )

        assert boxes[0].label is not None
        assert boxes[0].label.startswith("weekNumber")
        assert "pfl1PartB" not in boxes[0].label

    def test_a_top_level_scalar_keeps_its_own_name(self) -> None:
        """A scalar field has no children, so its own name is drawn."""
        boxes = boxes_from_bda_explainability(
            explainability_info=[{"invoiceNumber": explainability_leaf(page=1)}]
        )

        assert boxes[0].label is not None
        assert boxes[0].label.startswith("invoiceNumber")

    def test_confidence_appears_in_the_label(self) -> None:
        """The label carries the confidence, as the image path already does."""
        boxes = boxes_from_bda_explainability(
            explainability_info=[
                {"partA": {"gender": explainability_leaf(page=1, confidence=0.648)}}
            ]
        )

        assert boxes[0].label == "gender (65%)"

    def test_each_top_level_field_gets_its_own_colour(self) -> None:
        """Colouring by field is what makes a dense page readable."""
        boxes = boxes_from_bda_explainability(
            explainability_info=[
                {
                    "partA": {"gender": explainability_leaf(page=1)},
                    "partB": {"title": explainability_leaf(page=1)},
                }
            ]
        )

        assert boxes[0].colour != boxes[1].colour

    def test_children_of_one_field_share_its_colour(self) -> None:
        """Boxes from the same group are visually grouped."""
        boxes = boxes_from_bda_explainability(
            explainability_info=[
                {
                    "partA": {
                        "gender": explainability_leaf(page=1),
                        "race": explainability_leaf(page=1),
                    }
                }
            ]
        )

        assert boxes[0].colour == boxes[1].colour

    def test_only_the_first_box_of_a_value_is_labelled(self) -> None:
        """A value found in two places is boxed twice but captioned once.

        Checkbox fields report the box and the text beside it, so labelling both
        would print the field name twice on top of itself.
        """
        leaf = explainability_leaf(page=1)
        leaf["geometry"].append(dict(leaf["geometry"][0]))

        boxes = boxes_from_bda_explainability(
            explainability_info=[{"partA": {"checkbox": leaf}}]
        )

        assert len(boxes) == 2
        assert boxes[0].label is not None
        assert boxes[1].label is None

    def test_a_leaf_with_no_geometry_yields_no_box(self) -> None:
        """A field BDA could not locate has nothing to draw."""
        boxes = boxes_from_bda_explainability(
            explainability_info=[
                {
                    "partA": {
                        "missing": {
                            "success": False,
                            "confidence": 0.0,
                            "geometry": [],
                            "type": "string",
                            "value": None,
                        }
                    }
                }
            ]
        )

        assert boxes == []

    def test_missing_page_means_the_first_page(self) -> None:
        """Geometry without a page belongs to page 1, as for a single page."""
        leaf = explainability_leaf(page=1)
        del leaf["geometry"][0]["page"]

        boxes = boxes_from_bda_explainability(
            explainability_info=[{"partA": {"gender": leaf}}]
        )

        assert boxes[0].page_index == 0

    def test_empty_explainability_info_yields_no_boxes(self) -> None:
        """A non-blueprint run has no explainability, which is not an error."""
        assert boxes_from_bda_explainability(explainability_info=[]) == []

    def test_real_capture_shape_produces_boxes_on_all_seven_pages(self) -> None:
        """The captured document's 54 values spread across all seven pages.

        This mirrors the real custom output: six top-level keys, of which one is
        a table and the rest are groups.
        """
        info = [
            {
                "pfl1PartA": {
                    f"field{index}": explainability_leaf(page=1 + index % 2)
                    for index in range(6)
                },
                "pfl1PartB": {
                    f"field{index}": explainability_leaf(page=3) for index in range(6)
                },
                "pfl1PartB__grossWages": [
                    {"weekNumber": explainability_leaf(page=4)} for _ in range(4)
                ],
                "pfl1PartB__pflInsuranceCarrier": {
                    "city": explainability_leaf(page=5)
                },
                "pfl2BondingCertification": {
                    "childGender": explainability_leaf(page=6)
                },
                "guardianDirectPayEnrollment": {
                    "faxNumber": explainability_leaf(page=7)
                },
            }
        ]

        boxes = boxes_from_bda_explainability(explainability_info=info)

        assert {box.page_index for box in boxes} == {0, 1, 2, 3, 4, 5, 6}
        assert len(boxes) == 6 + 6 + 4 + 1 + 1 + 1


class TestColourForIndex:
    """Field colours are stable and cycle rather than running out."""

    def test_distinct_colours_for_early_indexes(self) -> None:
        """The first several fields each get their own colour."""
        colours = [colour_for_index(index=index) for index in range(8)]

        assert len(set(colours)) == 8

    def test_colour_is_stable_for_an_index(self) -> None:
        """The same index always yields the same colour."""
        assert colour_for_index(index=4) == colour_for_index(index=4)

    def test_palette_cycles_past_its_length(self) -> None:
        """A large index wraps instead of raising."""
        assert colour_for_index(index=0) == colour_for_index(index=10)


def test_module_imports_without_a_display(tmp_path: Path) -> None:
    """Rendering works headlessly, writing to a temp file like the engines do.

    The engines run inside a Gradio worker with no display, so nothing in the
    render path may need one.
    """
    pages, _ = render_pdf_pages(pdf_bytes=make_pdf_bytes(page_count=1), dpi=72)
    output = tmp_path / "rendered.png"

    stack_pages_vertically(pages=pages, labels=["Page 1"]).save(output)

    assert output.stat().st_size > 0
