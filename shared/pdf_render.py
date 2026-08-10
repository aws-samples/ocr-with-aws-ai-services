"""
Render PDF pages and draw normalised bounding boxes on them.

Both OCR engines used to answer a PDF run with a 400x600 black rectangle and a
line of text, discarding the geometry the service had already returned. This
module renders the real pages so those boxes can be drawn where they belong.

Geometry from Textract and from BDA's `explainability_info` is expressed in
fractions of the page (0.0-1.0), which is resolution independent - so a box can
be placed on a page rendered at any DPI by multiplying through by its pixel
size. `PageBox` carries that fractional geometry plus the page it belongs to.

PyMuPDF is used rather than `pdf2image`, because it is already a pinned
dependency (`preview_handler.convert_pdf_to_image` uses it) and needs no
external `poppler` binary. Rendering happens from bytes rather than a path
because that is what the engines hold at the point they build the visualisation.
"""

import io
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

from shared.config import logger

# Rendering resolution. 120 DPI keeps a US Letter page near 1000x1300 px, which
# is legible in the Gradio image panel without making a multi-page stack huge.
DEFAULT_DPI = 120

# Ceiling on rendered pages. A 200-page PDF stacked vertically would produce an
# image no browser can usefully show, so the tail is dropped - and reported.
DEFAULT_MAX_PAGES = 10

# PDF user space is 72 units per inch, so this converts DPI to a scale factor.
_PDF_UNITS_PER_INCH = 72

# Distinct hues for per-field colouring, chosen to stay legible on white paper.
# Reds and greys are avoided so a field box is never confused with the page.
_FIELD_PALETTE: Tuple[str, ...] = (
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#17BECF",
    "#BCBD22",
    "#7F3FBF",
)

# Height of the banner strip drawn above each page in a stack.
_LABEL_STRIP_HEIGHT = 26


@dataclass
class PageBox:
    """
    One bounding box to draw, in fractions of its page.

    Attributes:
        page_index: Zero-based index of the page the box belongs to.
        left: Distance from the left edge, as a fraction of page width.
        top: Distance from the top edge, as a fraction of page height.
        width: Box width, as a fraction of page width.
        height: Box height, as a fraction of page height.
        label: Text to draw above the box, or None to draw no label.
        colour: Outline colour as a hex string.
    """

    page_index: int
    left: float
    top: float
    width: float
    height: float
    label: Optional[str] = None
    colour: str = "#D62728"


def colour_for_index(*, index: int) -> str:
    """
    Pick a stable outline colour for the nth distinct field.

    Args:
        index: Position of the field among the fields being drawn.

    Returns:
        A hex colour string from the palette, cycling once exhausted.
    """
    return _FIELD_PALETTE[index % len(_FIELD_PALETTE)]


def _load_font(*, size: int) -> ImageFont.ImageFont:
    """
    Load a font for box labels at roughly the requested size.

    Args:
        size: Desired point size.

    Returns:
        A PIL font object. Pillow's built-in bitmap font is used when no
        scalable font is installed; it is only ever used for short labels, so a
        smaller size degrades legibility rather than correctness.
    """
    for candidate in ("DejaVuSans.ttf", "Helvetica.ttc", "Arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def render_pdf_pages(
    *,
    pdf_bytes: bytes,
    dpi: int = DEFAULT_DPI,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> Tuple[List[Image.Image], int]:
    """
    Render the leading pages of a PDF to images.

    Args:
        pdf_bytes: The complete PDF file content.
        dpi: Resolution to render at.
        max_pages: Most pages to render; the rest are skipped and reported
            through the returned total so the caller can say so.

    Returns:
        Tuple of (rendered pages in document order, total pages in the PDF).
        The list is shorter than the total when `max_pages` truncates it.

    Raises:
        ValueError: If `pdf_bytes` is empty, `dpi` is not positive, `max_pages`
            is not positive, or the bytes are not a readable PDF.
    """
    if not pdf_bytes:
        raise ValueError("Cannot render a PDF from empty bytes")
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")
    if max_pages <= 0:
        raise ValueError(f"max_pages must be positive, got {max_pages}")

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as open_error:
        # Re-raised rather than swallowed: a PDF that cannot be opened means the
        # visualisation is wrong, and a black placeholder would hide that.
        raise ValueError(f"Could not open PDF for rendering: {open_error}")

    try:
        total_pages = len(document)
        if total_pages == 0:
            raise ValueError("PDF contains no pages")

        zoom = dpi / _PDF_UNITS_PER_INCH
        matrix = fitz.Matrix(zoom, zoom)

        pages: List[Image.Image] = []
        for page_index in range(min(total_pages, max_pages)):
            pixmap = document[page_index].get_pixmap(matrix=matrix)
            # "ppm" round-trips through PIL without needing PyMuPDF's own PNG
            # encoder, matching what preview_handler already does.
            pages.append(Image.open(io.BytesIO(pixmap.tobytes("ppm"))).convert("RGB"))
    finally:
        document.close()

    if total_pages > len(pages):
        logger.warning(
            f"PDF has {total_pages} pages; rendering only the first {len(pages)} "
            f"for the visualisation"
        )

    return pages, total_pages


def count_pdf_pages(*, pdf_bytes: bytes) -> int:
    """
    Count a PDF's pages without rasterising any of them.

    Engines that need a page count for costing - and for the Pages column - must
    not pay for a render to get it. Opening the document reads only the page
    tree, so this is cheap enough to call on the request path, unlike
    `render_pdf_pages` which draws every page it counts.

    Args:
        pdf_bytes: The complete PDF file content.

    Returns:
        The number of pages in the document.

    Raises:
        ValueError: If `pdf_bytes` is empty, the bytes are not a readable PDF, or
            the document has no pages.
    """
    if not pdf_bytes:
        raise ValueError("Cannot count the pages of a PDF from empty bytes")

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as open_error:
        # Raised rather than defaulted to 1: a wrong page count silently
        # under-bills the run, which is the defect this function exists to fix.
        raise ValueError(f"Could not open PDF to count its pages: {open_error}")

    try:
        total_pages = len(document)
    finally:
        document.close()

    if total_pages == 0:
        raise ValueError("PDF contains no pages")

    return total_pages


def _overlaps(
    *, candidate: Tuple[float, float, float, float], claimed: Iterable[Tuple[float, float, float, float]]
) -> bool:
    """
    Report whether a rectangle intersects any already-claimed rectangle.

    Args:
        candidate: Rectangle as (left, top, right, bottom).
        claimed: Rectangles already occupied by drawn labels.

    Returns:
        True when the candidate overlaps at least one claimed rectangle.
    """
    left, top, right, bottom = candidate
    for other_left, other_top, other_right, other_bottom in claimed:
        if (
            left < other_right
            and right > other_left
            and top < other_bottom
            and bottom > other_top
        ):
            return True
    return False


def draw_boxes(
    *, pages: Sequence[Image.Image], boxes: Iterable[PageBox], outline_width: int = 2
) -> Tuple[int, int]:
    """
    Draw bounding boxes onto rendered pages, in place.

    Every box is drawn. A label is drawn only where it does not collide with one
    already placed: a table packs sixteen values into a few hundred pixels, and
    overlapping captions there turn all of them into noise. The box itself, and
    its per-field colour, still identify the value.

    Args:
        pages: Rendered pages, indexed the way `PageBox.page_index` counts.
        boxes: Boxes to draw.
        outline_width: Rectangle outline thickness in pixels.

    Returns:
        Tuple of (boxes drawn, boxes skipped because their page was not
        rendered). A non-zero skip count means the page cap truncated the
        document, not that the geometry was bad.
    """
    label_font = _load_font(size=13)
    drawn = 0
    skipped = 0
    suppressed_labels = 0
    # Label rectangles already drawn, per page, so collisions can be detected.
    claimed: Dict[int, List[Tuple[float, float, float, float]]] = {}

    for box in boxes:
        if not 0 <= box.page_index < len(pages):
            skipped += 1
            continue

        page = pages[box.page_index]
        canvas = ImageDraw.Draw(page)
        page_width, page_height = page.size

        left = box.left * page_width
        top = box.top * page_height
        right = left + box.width * page_width
        bottom = top + box.height * page_height

        canvas.rectangle(
            [(left, top), (right, bottom)], outline=box.colour, width=outline_width
        )

        if box.label:
            # Labels sit just above the box, or just below it when the box is
            # already at the top of the page and there is no room above.
            label_top = top - 15 if top > 15 else bottom + 2
            label_box = canvas.textbbox((left, label_top), box.label, font=label_font)

            page_claims = claimed.setdefault(box.page_index, [])
            if _overlaps(candidate=label_box, claimed=page_claims):
                suppressed_labels += 1
            else:
                canvas.text(
                    (left, label_top), box.label, fill=box.colour, font=label_font
                )
                page_claims.append(label_box)

        drawn += 1

    if skipped:
        logger.warning(
            f"{skipped} bounding boxes fall on pages that were not rendered and "
            f"are not shown"
        )
    if suppressed_labels:
        logger.info(
            f"{suppressed_labels} box labels were left undrawn because they would "
            f"have overlapped another label"
        )

    return drawn, skipped


def stack_pages_vertically(
    *,
    pages: Sequence[Image.Image],
    labels: Optional[Sequence[str]] = None,
    max_width: int = 1000,
) -> Image.Image:
    """
    Compose rendered pages into the single image the Gradio output expects.

    Args:
        pages: Rendered pages in the order they should appear.
        labels: One caption per page, drawn in a strip above it. None draws no
            strips.
        max_width: Pages wider than this are scaled down, preserving aspect
            ratio, so a stack stays within a sensible width.

    Returns:
        A single RGB image with the pages one under the other.

    Raises:
        ValueError: If `pages` is empty, or `labels` is given but does not have
            one entry per page.
    """
    if not pages:
        raise ValueError("Cannot stack an empty list of pages")
    if labels is not None and len(labels) != len(pages):
        raise ValueError(
            f"Got {len(labels)} labels for {len(pages)} pages; they must match"
        )
    if max_width <= 0:
        raise ValueError(f"max_width must be positive, got {max_width}")

    scaled: List[Image.Image] = []
    for page in pages:
        if page.width > max_width:
            scale = max_width / page.width
            scaled.append(
                page.resize(
                    (max_width, max(1, round(page.height * scale))),
                    Image.LANCZOS,
                )
            )
        else:
            scaled.append(page)

    strip_height = _LABEL_STRIP_HEIGHT if labels is not None else 0
    canvas_width = max(page.width for page in scaled)
    canvas_height = sum(page.height + strip_height for page in scaled)

    stacked = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))
    canvas = ImageDraw.Draw(stacked)
    label_font = _load_font(size=15)

    offset_y = 0
    for position, page in enumerate(scaled):
        if labels is not None:
            canvas.rectangle(
                [(0, offset_y), (canvas_width, offset_y + strip_height)],
                fill=(28, 32, 38),
            )
            canvas.text(
                (8, offset_y + 5),
                labels[position],
                fill=(235, 238, 242),
                font=label_font,
            )
            offset_y += strip_height
        stacked.paste(page, (0, offset_y))
        offset_y += page.height

    return stacked


def compose_pdf_visualisation(
    *,
    pdf_bytes: bytes,
    boxes: Sequence[PageBox],
    item_noun: str,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> Image.Image:
    """
    Render a PDF's pages with boxes drawn on them and a caption per page.

    Both engines need exactly this: render, draw whatever geometry the service
    returned, caption each page with how much of it is on screen, and stack the
    result into the single image the Gradio output takes. They differ only in
    where the boxes come from and what the boxes represent.

    A PDF that cannot be rendered does not fail the run - the extraction result
    is the point - but it does return an image saying so, rather than the blank
    placeholder that used to hide it.

    Args:
        pdf_bytes: The PDF that was processed.
        boxes: Boxes to draw, already converted to page coordinates.
        item_noun: What one box represents, for the page captions, e.g.
            "values located" or "text lines".
        max_pages: Most pages to render.

    Returns:
        The composed visualisation.
    """
    try:
        pages, total_pages = render_pdf_pages(pdf_bytes=pdf_bytes, max_pages=max_pages)
    except ValueError as render_error:
        logger.error(f"Could not render the PDF for visualisation: {render_error}")
        return render_message_image(
            lines=[
                "The document was processed but its pages could not be rendered.",
                str(render_error),
            ]
        )

    drawn, skipped = draw_boxes(pages=pages, boxes=boxes)

    # Count per page so each caption describes what is actually on that page.
    per_page: Dict[int, int] = {}
    for box in boxes:
        per_page[box.page_index] = per_page.get(box.page_index, 0) + 1

    labels: List[str] = []
    for page_index in range(len(pages)):
        label = f"Page {page_index + 1} of {total_pages}"
        if boxes:
            label += f"  •  {per_page.get(page_index, 0)} {item_noun}"
        labels.append(label)

    if skipped:
        # Truncated pages are named on the last visible page rather than being
        # quietly left out of the picture.
        labels[-1] += (
            f"  •  {skipped} {item_noun} on pages "
            f"{len(pages) + 1}-{total_pages} are not shown"
        )

    logger.info(
        f"PDF visualisation: {len(pages)} of {total_pages} pages rendered, "
        f"{drawn} boxes drawn, {skipped} skipped"
    )

    return stack_pages_vertically(pages=pages, labels=labels)


def render_message_image(*, lines: Sequence[str], width: int = 900) -> Image.Image:
    """
    Draw a short message as an image, for when a visualisation cannot be built.

    This exists so a failed render says why rather than showing a blank panel the
    user has to interpret. It is not a fallback that hides the problem: the
    message is the problem, stated where the picture would have been.

    Args:
        lines: Message lines, drawn one per row.
        width: Image width in pixels.

    Returns:
        An RGB image containing the message.

    Raises:
        ValueError: If `lines` is empty or `width` is not positive.
    """
    if not lines:
        raise ValueError("Cannot render a message image with no lines")
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")

    line_height = 24
    padding = 20
    height = padding * 2 + line_height * len(lines)

    image = Image.new("RGB", (width, height), color=(38, 20, 22))
    canvas = ImageDraw.Draw(image)
    font = _load_font(size=15)

    for position, line in enumerate(lines):
        canvas.text(
            (padding, padding + position * line_height),
            line,
            fill=(255, 190, 190),
            font=font,
        )

    return image


def _explainability_leaf_boxes(
    *, label: str, leaf: Dict[str, Any], colour: str
) -> List[PageBox]:
    """
    Turn one extracted value's geometry into boxes.

    A value can be found in several places on the page - a wrapped address, or a
    checkbox plus the text beside it - so one leaf yields one box per geometry
    entry. Only the first is labelled, to keep a dense page readable.

    Args:
        label: Field name to draw beside the box.
        leaf: An explainability leaf, carrying `geometry` and `confidence`.
        colour: Outline colour for every box from this field.

    Returns:
        One `PageBox` per geometry entry that has a bounding box.
    """
    confidence = leaf.get("confidence")
    caption = (
        f"{label} ({confidence * 100:.0f}%)" if confidence is not None else label
    )

    boxes: List[PageBox] = []
    for entry in leaf.get("geometry") or []:
        bounding_box = entry.get("boundingBox")
        if not bounding_box:
            continue

        boxes.append(
            PageBox(
                # BDA numbers explainability pages from 1, while
                # `split_document.page_indices` alongside it counts from 0.
                page_index=entry.get("page", 1) - 1,
                left=bounding_box["left"],
                top=bounding_box["top"],
                width=bounding_box["width"],
                height=bounding_box["height"],
                label=caption if not boxes else None,
                colour=colour,
            )
        )

    return boxes


def _is_explainability_leaf(*, node: Any) -> bool:
    """
    Report whether an explainability node describes one extracted value.

    Args:
        node: A node from `explainability_info`.

    Returns:
        True when the node is a leaf rather than a group or a table.
    """
    return isinstance(node, dict) and "geometry" in node


def _walk_explainability(
    *, label: str, node: Any, colour: str, boxes: List[PageBox]
) -> None:
    """
    Collect boxes from an explainability node, recursing into groups and tables.

    Only the innermost field name is kept as the label. A full dotted path such
    as `pfl1PartB.grossWages[1].weekNumber` is three times as wide as the cell it
    annotates, so on a table it overlaps its neighbours into illegibility - and
    the group it belongs to is already conveyed by the shared box colour.

    Args:
        label: Name to draw for this node; replaced by the child's name as the
            walk descends into a group or a table row.
        node: A leaf, a group (dict of leaves) or a table (list of row dicts).
        colour: Outline colour inherited from the top-level field.
        boxes: Accumulator, appended to in place.
    """
    if _is_explainability_leaf(node=node):
        boxes.extend(_explainability_leaf_boxes(label=label, leaf=node, colour=colour))
    elif isinstance(node, dict):
        for child_name, child in node.items():
            _walk_explainability(
                label=child_name, node=child, colour=colour, boxes=boxes
            )
    elif isinstance(node, list):
        # Rows keep the table's name, since a row has no name of its own and its
        # position on the page already shows which row it is.
        for row in node:
            _walk_explainability(label=label, node=row, colour=colour, boxes=boxes)


def boxes_from_bda_explainability(
    *, explainability_info: Iterable[Any]
) -> List[PageBox]:
    """
    Turn BDA's `explainability_info` into boxes, coloured per top-level field.

    The structure is a list of objects keyed by blueprint field name, where a
    group is a dict of leaves, a table is a list of row dicts, and a leaf carries
    `geometry`, `confidence` and `value`. Each leaf's geometry names the page it
    was found on, so boxes land correctly across a multi-page document.

    Every value under one top-level field shares a colour, which is how a group's
    boxes stay identifiable once the labels are shortened to the field name.

    Args:
        explainability_info: The `explainability_info` list from custom output.

    Returns:
        Boxes for every extracted value that reported geometry.
    """
    boxes: List[PageBox] = []
    colour_index = 0

    for group in explainability_info:
        if not isinstance(group, dict):
            continue

        for key, node in group.items():
            _walk_explainability(
                label=key,
                node=node,
                colour=colour_for_index(index=colour_index),
                boxes=boxes,
            )
            colour_index += 1

    return boxes


def boxes_from_textract_blocks(
    *, blocks: Iterable[Dict[str, Any]], block_type: str = "LINE"
) -> List[PageBox]:
    """
    Turn Textract blocks into boxes, using each block's own page number.

    Textract numbers pages from 1 on multi-page documents and omits `Page`
    entirely for single-page input, so a missing value means page 1.

    Args:
        blocks: The `Blocks` list from a Textract response.
        block_type: Only blocks of this `BlockType` are converted.

    Returns:
        One `PageBox` per matching block that carries geometry.
    """
    boxes: List[PageBox] = []

    for block in blocks:
        if block.get("BlockType") != block_type:
            continue
        bounding_box = block.get("Geometry", {}).get("BoundingBox")
        if not bounding_box:
            continue

        boxes.append(
            PageBox(
                page_index=block.get("Page", 1) - 1,
                left=bounding_box["Left"],
                top=bounding_box["Top"],
                width=bounding_box["Width"],
                height=bounding_box["Height"],
            )
        )

    return boxes
