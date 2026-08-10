"""Generate the tracked synthetic claim-form samples under sample/.

The benchmark needs multi-page documents it is allowed to publish, so this script
produces them: three multi-page leave and disability forms whose every value - names,
employers, carriers, claim numbers, wages, diagnoses - is invented.

Two properties matter and are the reason this is a generator rather than three
checked-in files with no provenance:

- **The output has no text layer.** Each page is drawn with PyMuPDF, rasterised to
  grayscale, lightly speckled and skewed, and then re-inserted as an image. The forms
  these stand in for are fax-grade scans, so an engine reading a text layer instead of
  performing OCR would produce accuracy and cost figures that say nothing about OCR.
- **The ground truth is written from the same table the pages are drawn from**, so it
  is correct by construction rather than by transcription. `truth.json` cannot drift
  from the document the way a hand-typed ground truth can.

Rendering is seeded, so regenerating produces pixel-identical pages and does not churn
the tracked PDFs. `--check` re-renders and compares against what is on disk.

Usage:
    python tools/generate_synthetic_samples.py            # write sample/<id>/
    python tools/generate_synthetic_samples.py --check     # verify, write nothing
"""

import argparse
import hashlib
import io
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

# US Letter at 72 points per inch, the size these forms are printed at.
PAGE_WIDTH: float = 612.0
PAGE_HEIGHT: float = 792.0
MARGIN: float = 44.0
CONTENT_WIDTH: float = PAGE_WIDTH - 2 * MARGIN

# 150 DPI is what a fax-grade scan of a letter page comes in at, and is enough for
# Textract to read 9pt Courier. Higher would inflate the tracked files for no gain.
RENDER_DPI: int = 150
JPEG_QUALITY: int = 60
NOISE_SIGMA: float = 2.6
MAX_SKEW_DEGREES: float = 0.35

FONT_LABEL: str = "helv"
FONT_BOLD: str = "hebo"
FONT_VALUE: str = "cour"

INK: Tuple[float, float, float] = (0.08, 0.08, 0.10)
GREY: Tuple[float, float, float] = (0.42, 0.42, 0.45)
RULE: Tuple[float, float, float] = (0.55, 0.55, 0.58)
BAND: Tuple[float, float, float] = (0.86, 0.86, 0.88)

FIELD_HEIGHT: float = 16.0
FIELD_GAP: float = 11.0


def _money(amount: float) -> str:
    """
    Format a number the way a filled-in form shows a dollar amount

    Args:
        amount: Value in dollars

    Returns:
        The amount with a leading dollar sign and thousands separators
    """
    return f"${amount:,.2f}"


def _us_date(iso_date: Optional[str]) -> str:
    """
    Render an ISO date the way a US claim form shows it

    Ground truth stores dates as ISO 8601 while the form prints MM/DD/YYYY, which is
    deliberate: it is what a real claim form does, so the engines are exercised on
    the same normalisation.

    Args:
        iso_date: Date as YYYY-MM-DD, or None for a field left blank

    Returns:
        The date as MM/DD/YYYY, or an empty string when iso_date is None
    """
    if not iso_date:
        return ""

    year, month, day = iso_date.split("-")
    return f"{month}/{day}/{year}"


def _draw_page_furniture(*, page: fitz.Page, form_number: str, form_title: str,
                         page_number: int, total_pages: int, fax_stamp: str) -> float:
    """
    Draw the header, footer and fax stamp shared by every page of a form

    Args:
        page: Page to draw on
        form_number: Form identifier printed top left, e.g. "PFL-1 (10-25)"
        form_title: Form title printed under the header rule
        page_number: 1-based page number
        total_pages: Total pages in the form
        fax_stamp: Single line imitating the header a fax machine prints

    Returns:
        The y coordinate at which page content can start
    """
    page.insert_text((MARGIN, 22), fax_stamp, fontname=FONT_VALUE, fontsize=6.5,
                     color=GREY)

    page.insert_text((MARGIN, 46), form_number, fontname=FONT_BOLD, fontsize=8,
                     color=INK)
    page.insert_text((MARGIN, 62), form_title, fontname=FONT_BOLD, fontsize=12.5,
                     color=INK)
    page.draw_line(fitz.Point(MARGIN, 70), fitz.Point(PAGE_WIDTH - MARGIN, 70),
                   color=INK, width=1.1)

    footer = f"Page {page_number} of {total_pages}"
    page.insert_text((PAGE_WIDTH - MARGIN - 60, PAGE_HEIGHT - 30), footer,
                     fontname=FONT_LABEL, fontsize=7.5, color=GREY)

    return 92.0


def _draw_section(*, page: fitz.Page, y: float, text: str) -> float:
    """
    Draw a shaded section heading

    Args:
        page: Page to draw on
        y: Top of the heading band
        text: Heading text

    Returns:
        The y coordinate below the heading
    """
    band = fitz.Rect(MARGIN, y, PAGE_WIDTH - MARGIN, y + 15)
    page.draw_rect(band, color=BAND, fill=BAND, width=0)
    page.insert_text((MARGIN + 5, y + 11), text.upper(), fontname=FONT_BOLD,
                     fontsize=8, color=INK)
    return y + 15 + 12


def _draw_field(*, page: fitz.Page, x: float, y: float, width: float, label: str,
                value: str) -> None:
    """
    Draw one labelled field: caption above a ruled line carrying the value

    Args:
        page: Page to draw on
        x: Left edge of the field
        y: Top of the field
        width: Width of the ruled line
        label: Field caption
        value: Value as it appears on the form, possibly empty
    """
    page.insert_text((x, y + 6), label.upper(), fontname=FONT_LABEL, fontsize=6.2,
                     color=GREY)
    page.insert_text((x + 3, y + 15), value, fontname=FONT_VALUE, fontsize=8.6,
                     color=INK)
    page.draw_line(fitz.Point(x, y + FIELD_HEIGHT + 2),
                   fitz.Point(x + width, y + FIELD_HEIGHT + 2), color=RULE, width=0.6)


def _draw_field_row(*, page: fitz.Page, y: float,
                    fields: Sequence[Tuple[str, str]]) -> float:
    """
    Draw between one and three fields side by side across the content width

    Args:
        page: Page to draw on
        y: Top of the row
        fields: (label, value) pairs, laid out left to right in equal columns

    Returns:
        The y coordinate below the row
    """
    gutter = 14.0
    column_width = (CONTENT_WIDTH - gutter * (len(fields) - 1)) / len(fields)

    for index, (label, value) in enumerate(fields):
        _draw_field(page=page, x=MARGIN + index * (column_width + gutter), y=y,
                    width=column_width, label=label, value=value)

    return y + FIELD_HEIGHT + FIELD_GAP


def _draw_checkbox_row(*, page: fitz.Page, y: float, label: str,
                       options: Sequence[Tuple[str, bool]]) -> float:
    """
    Draw a question followed by tick boxes, one of which may be marked

    Args:
        page: Page to draw on
        y: Top of the row
        label: The question text
        options: (option text, whether it is ticked) pairs

    Returns:
        The y coordinate below the row
    """
    page.insert_text((MARGIN, y + 8), label, fontname=FONT_LABEL, fontsize=7.6,
                     color=INK)

    # Measure the options so a long set is pulled left instead of running off the page.
    option_gap = 20.0
    option_widths = [
        9 + 4 + fitz.get_text_length(option_text, fontname=FONT_LABEL, fontsize=7.6)
        for option_text, _ in options]
    total_width = sum(option_widths) + option_gap * (len(options) - 1)

    x = min(MARGIN + 330, PAGE_WIDTH - MARGIN - total_width)
    for (option_text, is_ticked), option_width in zip(options, option_widths):
        box = fitz.Rect(x, y, x + 9, y + 9)
        page.draw_rect(box, color=INK, width=0.7)
        if is_ticked:
            # Two strokes rather than a glyph: a hand-marked box on a fax is a cross.
            page.draw_line(fitz.Point(x + 1.4, y + 1.4), fitz.Point(x + 7.6, y + 7.6),
                           color=INK, width=1.1)
            page.draw_line(fitz.Point(x + 7.6, y + 1.4), fitz.Point(x + 1.4, y + 7.6),
                           color=INK, width=1.1)
        page.insert_text((x + 13, y + 8), option_text, fontname=FONT_LABEL,
                         fontsize=7.6, color=INK)
        x += option_width + option_gap

    return y + 9 + 9


def _draw_table(*, page: fitz.Page, y: float, headers: Sequence[str],
                rows: Sequence[Sequence[str]]) -> float:
    """
    Draw a bordered table with a shaded header row

    Args:
        page: Page to draw on
        y: Top of the table
        headers: Column captions
        rows: Cell text, one sequence per row, same length as headers

    Returns:
        The y coordinate below the table
    """
    row_height = 15.0
    column_width = CONTENT_WIDTH / len(headers)
    total_height = row_height * (len(rows) + 1)

    header_band = fitz.Rect(MARGIN, y, PAGE_WIDTH - MARGIN, y + row_height)
    page.draw_rect(header_band, color=BAND, fill=BAND, width=0)

    for column, caption in enumerate(headers):
        page.insert_text((MARGIN + column * column_width + 4, y + 10.5),
                         caption.upper(), fontname=FONT_BOLD, fontsize=6.4, color=INK)

    for row_index, row in enumerate(rows):
        row_top = y + row_height * (row_index + 1)
        for column, cell in enumerate(row):
            page.insert_text((MARGIN + column * column_width + 4, row_top + 10.5),
                             cell, fontname=FONT_VALUE, fontsize=8, color=INK)

    for line_index in range(len(rows) + 2):
        line_y = y + row_height * line_index
        page.draw_line(fitz.Point(MARGIN, line_y),
                       fitz.Point(PAGE_WIDTH - MARGIN, line_y), color=RULE, width=0.6)

    for column in range(len(headers) + 1):
        line_x = MARGIN + column * column_width
        page.draw_line(fitz.Point(line_x, y), fitz.Point(line_x, y + total_height),
                       color=RULE, width=0.6)

    return y + total_height + FIELD_GAP


def _draw_paragraph(*, page: fitz.Page, y: float, text: str) -> float:
    """
    Draw a block of small print across the content width

    Args:
        page: Page to draw on
        y: Top of the block
        text: The paragraph

    Returns:
        The y coordinate below the block
    """
    height = 11.0 + 7.0 * (len(text) // 118)
    box = fitz.Rect(MARGIN, y, PAGE_WIDTH - MARGIN, y + height + 14)
    page.insert_textbox(box, text, fontname=FONT_LABEL, fontsize=6.8, color=INK)
    return y + height + 16


def _draw_signature(*, page: fitz.Page, y: float, name: str, title: str,
                    signed_on: str) -> float:
    """
    Draw a signature block: a signed name in script-like italics over ruled lines

    Args:
        page: Page to draw on
        y: Top of the block
        name: Name as signed
        title: Signer's title, printed under the second rule
        signed_on: Date of signature, already formatted for the form

    Returns:
        The y coordinate below the block
    """
    page.insert_text((MARGIN + 4, y + 10), name, fontname="tiit", fontsize=12,
                     color=INK)
    page.insert_text((MARGIN + 330, y + 10), signed_on, fontname=FONT_VALUE,
                     fontsize=8.6, color=INK)
    page.draw_line(fitz.Point(MARGIN, y + 14), fitz.Point(MARGIN + 300, y + 14),
                   color=INK, width=0.7)
    page.draw_line(fitz.Point(MARGIN + 326, y + 14),
                   fitz.Point(PAGE_WIDTH - MARGIN, y + 14), color=INK, width=0.7)
    page.insert_text((MARGIN, y + 24), f"SIGNATURE - {title}".upper(),
                     fontname=FONT_LABEL, fontsize=6.2, color=GREY)
    page.insert_text((MARGIN + 326, y + 24), "DATE", fontname=FONT_LABEL,
                     fontsize=6.2, color=GREY)
    return y + 34


def _render_block(*, page: fitz.Page, y: float, block: Tuple[Any, ...]) -> float:
    """
    Draw one layout block, dispatching on its kind

    Args:
        page: Page to draw on
        y: Top of the block
        block: A tuple whose first element names the kind and whose remainder is that
               kind's arguments

    Returns:
        The y coordinate below the block

    Raises:
        ValueError: If the block names a kind this renderer does not know, which would
                    otherwise silently drop content from the generated form
    """
    kind = block[0]

    if kind == "section":
        return _draw_section(page=page, y=y, text=block[1])
    if kind == "fields":
        return _draw_field_row(page=page, y=y, fields=block[1])
    if kind == "checkbox":
        return _draw_checkbox_row(page=page, y=y, label=block[1], options=block[2])
    if kind == "table":
        return _draw_table(page=page, y=y, headers=block[1], rows=block[2])
    if kind == "paragraph":
        return _draw_paragraph(page=page, y=y, text=block[1])
    if kind == "signature":
        return _draw_signature(page=page, y=y, name=block[1], title=block[2],
                               signed_on=block[3])
    if kind == "spacer":
        return y + block[1]

    raise ValueError(f"Unknown layout block kind: {kind!r}")


def _degrade(*, page: fitz.Page, rng: np.random.Generator) -> bytes:
    """
    Rasterise a drawn page and make it look like a faxed scan

    A text-layer PDF would let the engines read the document rather than perform OCR,
    so the crisp page is thrown away and only the degraded raster is kept.

    Args:
        page: The drawn page
        rng: Seeded generator, so the speckle and skew are reproducible

    Returns:
        The page as grayscale JPEG bytes
    """
    pixmap = page.get_pixmap(dpi=RENDER_DPI, colorspace=fitz.csGRAY)
    image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)

    skew = float(rng.uniform(-MAX_SKEW_DEGREES, MAX_SKEW_DEGREES))
    image = image.rotate(skew, resample=Image.BICUBIC, fillcolor=255, expand=False)

    samples = np.asarray(image).astype(np.int16)
    samples += rng.normal(0.0, NOISE_SIGMA, samples.shape).astype(np.int16)
    image = Image.fromarray(np.clip(samples, 0, 255).astype(np.uint8), mode="L")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buffer.getvalue()


def build_pdf(*, form: Dict[str, Any], seed: int) -> bytes:
    """
    Draw, degrade and assemble one form into an image-only PDF

    Args:
        form: A form specification with "form_number", "title", "fax_stamp" and
              "pages", where each page is a sequence of layout blocks
        seed: Seed for the speckle and skew, so output is reproducible

    Returns:
        The finished PDF as bytes
    """
    rng = np.random.default_rng(seed)
    pages: List[Tuple[str, Any]] = form["pages"]

    scanned = fitz.open()

    for page_index, blocks in enumerate(pages):
        drawing = fitz.open()
        drawn_page = drawing.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)

        y = _draw_page_furniture(
            page=drawn_page, form_number=form["form_number"],
            form_title=form["title"], page_number=page_index + 1,
            total_pages=len(pages), fax_stamp=form["fax_stamp"])

        for block in blocks:
            y = _render_block(page=drawn_page, y=y, block=block)

        jpeg = _degrade(page=drawn_page, rng=rng)
        drawing.close()

        scanned_page = scanned.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        scanned_page.insert_image(
            fitz.Rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT), stream=jpeg)

    # Cleared so that the bytes do not carry a creation timestamp, which would make
    # every regeneration a diff even when the pages are identical.
    scanned.set_metadata({})
    scanned.del_xml_metadata()
    pdf_bytes: bytes = scanned.tobytes(garbage=4, deflate=True)
    scanned.close()
    return pdf_bytes


def page_fingerprints(*, pdf_bytes: bytes) -> List[str]:
    """
    Hash each page's pixels, so two PDFs can be compared by what they show

    Comparing raw file bytes would report a difference for a PDF that renders
    identically, since the container carries ids and object ordering that are not part
    of the document.

    Args:
        pdf_bytes: A PDF

    Returns:
        One hex digest per page, in page order
    """
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    digests: List[str] = []

    for page in document:
        pixmap = page.get_pixmap(dpi=72, colorspace=fitz.csGRAY)
        digests.append(hashlib.sha256(pixmap.samples).hexdigest())

    document.close()
    return digests


def _pfl_form() -> Dict[str, Any]:
    """
    Build the paid family leave bonding request

    Returns:
        A form specification with "id", "pages", "schema" and "truth"
    """
    truth: Dict[str, Any] = {
        "requestPartA": {
            "employeeName": "Marisol Ndiaye-Craft",
            "claimNumber": "SYN-4410772",
            "employeeCity": "Rensselaer",
            "employeeState": "NY",
            "employeeZipCode": "12144",
            "gender": "F",
            "preferredLanguage": "English",
            "race": "Black or African American",
            "reasonForRequest": "Bond with child",
            "familyMemberRelationship": "Child",
            "leaveScheduleType": "Continuous",
            "leaveStartDate": "2025-06-02",
            "leaveEndDate": "2025-08-24",
            "datesAreEstimated": True,
            "employeeDateOfHire": "2022-04-11",
            "hasMoreThanOneEmployer": "No",
            "takingLeaveFromOtherEmployer": "No",
            "receivingWorkersCompensation": "No",
            "employeeSignatureDate": "2025-05-19",
        },
        "requestPartB": {
            "employerName": "Halberd Ceramics Co-operative",
            "employerFein": "27-3319845",
            "employerDateOfHire": "2022-04-18",
            "occupationCode": "43-4051",
            # numberOfDaysWorked is left blank for one week on the form, which is why
            # the schema declares it ["number", "null"] - the same optional column a
            # real wage table carries.
            "grossWages": [
                {"weekNumber": 1, "weekEndingDate": "2025-05-11",
                 "numberOfDaysWorked": 5, "grossAmountPaid": 1412.55},
                {"weekNumber": 2, "weekEndingDate": "2025-05-04",
                 "numberOfDaysWorked": 5, "grossAmountPaid": 1388.20},
                {"weekNumber": 3, "weekEndingDate": "2025-04-27",
                 "numberOfDaysWorked": 5, "grossAmountPaid": 1502.90},
                {"weekNumber": 4, "weekEndingDate": "2025-04-20",
                 "numberOfDaysWorked": 4, "grossAmountPaid": 1344.75},
                {"weekNumber": 5, "weekEndingDate": "2025-04-13",
                 "numberOfDaysWorked": 5, "grossAmountPaid": 1476.30},
                {"weekNumber": 6, "weekEndingDate": "2025-04-06",
                 "numberOfDaysWorked": 5, "grossAmountPaid": 1412.55},
                {"weekNumber": 7, "weekEndingDate": "2025-03-30",
                 "numberOfDaysWorked": None, "grossAmountPaid": 1399.05},
                {"weekNumber": 8, "weekEndingDate": "2025-03-23",
                 "numberOfDaysWorked": 5, "grossAmountPaid": 1455.40},
            ],
            # Mean of the eight weeks above, rounded to the cent as the form's own
            # instructions require.
            "calculatedAverageGrossWeeklyWage": 1423.96,
            "requestingReimbursement": "No",
            "leaveTakenInPreceding52Weeks": "None",
            "takingFmlaConcurrently": "Yes",
            "employerSignatureDate": "2025-05-22",
            "employerTitle": "Benefits Administrator",
            # A group inside a group: the carrier block is printed within Part B, so it
            # is modelled where it appears rather than promoted to the top level.
            "insuranceCarrier": {
                "name": "Pemberton Mutual Assurance",
                "streetAddress": "PO Box 88214",
                "city": "Watervliet",
                "state": "NY",
                "zipCode": "12189-0214",
            },
        },
        "bondingCertification": {
            "childName": "Amaia Ndiaye-Craft",
            "childDateOfBirth": "2025-05-31",
            "childGender": "F",
            "childLivesWithEmployee": "Yes",
            "relationshipToChild": "Biological child",
            "documentationAttached": "Health care provider certification of birth",
            "employeeSignatureDate": "2025-05-19",
        },
        "paymentEnrollment": {
            "policyNumber": "PMA-NY-660184",
            "carrierTelephoneNumber": "(800) 447-2210",
            "paymentMethod": "Direct deposit",
            "accountType": "Checking",
            "bankRoutingNumberLastFour": "0119",
        },
    }

    part_a = truth["requestPartA"]
    part_b = truth["requestPartB"]
    bonding = truth["bondingCertification"]
    carrier = part_b["insuranceCarrier"]
    payment = truth["paymentEnrollment"]

    pages = [
        [
            ("section", "Part A - to be completed by the employee"),
            ("fields", [("Employee name (last, first, middle)",
                         part_a["employeeName"]),
                        ("Claim number", part_a["claimNumber"])]),
            ("fields", [("City", part_a["employeeCity"]),
                        ("State", part_a["employeeState"]),
                        ("ZIP code", part_a["employeeZipCode"])]),
            ("fields", [("Gender", part_a["gender"]),
                        ("Preferred language", part_a["preferredLanguage"]),
                        ("Race", part_a["race"])]),
            ("spacer", 6),
            ("section", "Reason for this request"),
            ("checkbox", "I am requesting paid family leave to:",
             [("Bond with child", True), ("Care for relative", False),
              ("Military event", False)]),
            ("fields", [("Family member relationship",
                         part_a["familyMemberRelationship"]),
                        ("Leave schedule type", part_a["leaveScheduleType"])]),
            ("fields", [("First day of leave requested",
                         _us_date(part_a["leaveStartDate"])),
                        ("Last day of leave requested",
                         _us_date(part_a["leaveEndDate"]))]),
            ("checkbox", "The dates above are estimated:",
             [("Yes", True), ("No", False)]),
            ("fields", [("Date of hire", _us_date(part_a["employeeDateOfHire"]))]),
            ("checkbox", "Do you work for more than one employer?",
             [("Yes", False), ("No", True)]),
            ("checkbox", "Are you taking family leave from another employer?",
             [("Yes", False), ("No", True)]),
            ("checkbox", "Are you receiving workers' compensation benefits?",
             [("Yes", False), ("No", True)]),
            ("spacer", 8),
            ("paragraph",
             "I certify that the statements above are true to the best of my "
             "knowledge. I understand that a false statement made to obtain "
             "benefits is a violation of law and may result in the denial of this "
             "request and recovery of any amounts already paid to me."),
            ("signature", part_a["employeeName"], "Employee",
             _us_date(part_a["employeeSignatureDate"])),
        ],
        [
            ("section", "Part B - to be completed by the employer"),
            ("fields", [("Employer name", part_b["employerName"]),
                        ("Federal employer id number", part_b["employerFein"])]),
            ("fields", [("Date of hire", _us_date(part_b["employerDateOfHire"])),
                        ("Occupation code", part_b["occupationCode"])]),
            ("spacer", 6),
            ("section", "Gross wages for the eight weeks preceding the leave"),
            ("table", ["Week", "Week ending", "Days worked", "Gross amount paid"],
             [[str(week["weekNumber"]), _us_date(week["weekEndingDate"]),
               "" if week["numberOfDaysWorked"] is None
               else str(week["numberOfDaysWorked"]),
               _money(week["grossAmountPaid"])]
              for week in part_b["grossWages"]]),
            ("fields", [("Calculated average gross weekly wage",
                         _money(part_b["calculatedAverageGrossWeeklyWage"]))]),
            ("checkbox", "Is the employer requesting reimbursement?",
             [("Yes", False), ("No", True)]),
            ("fields", [("Family leave taken in the preceding 52 weeks",
                         part_b["leaveTakenInPreceding52Weeks"])]),
            ("checkbox", "Is FMLA leave being taken concurrently?",
             [("Yes", True), ("No", False)]),
            ("spacer", 6),
            ("section", "Insurance carrier providing this coverage"),
            ("fields", [("Carrier name", carrier["name"]),
                        ("Street address", carrier["streetAddress"])]),
            ("fields", [("City", carrier["city"]),
                        ("State", carrier["state"]),
                        ("ZIP code", carrier["zipCode"])]),
            ("spacer", 8),
            ("paragraph",
             "The employer affirms that the wage information reported above is "
             "drawn from its payroll records and that the employee named in Part A "
             "is eligible for paid family leave on the dates requested."),
            ("signature", "Delphine Okonjo-Reyes", part_b["employerTitle"],
             _us_date(part_b["employerSignatureDate"])),
        ],
        [
            ("section", "Bonding certification"),
            ("fields", [("Name of child", bonding["childName"]),
                        ("Date of birth", _us_date(bonding["childDateOfBirth"])),
                        ("Gender", bonding["childGender"])]),
            ("checkbox", "Does the child live with you?",
             [("Yes", True), ("No", False)]),
            ("fields", [("Relationship to child",
                         bonding["relationshipToChild"])]),
            ("fields", [("Documentation attached",
                         bonding["documentationAttached"])]),
            ("spacer", 10),
            ("paragraph",
             "Attach one of the following: a birth certificate, a certification "
             "from a health care provider stating the date of birth, a court "
             "document showing placement for adoption, or a foster care placement "
             "letter issued by the county department of social services."),
            ("spacer", 10),
            ("signature", part_a["employeeName"], "Employee",
             _us_date(bonding["employeeSignatureDate"])),
        ],
        [
            ("section", "Payment enrollment"),
            ("fields", [("Policy number", payment["policyNumber"]),
                        ("Carrier telephone",
                         payment["carrierTelephoneNumber"])]),
            ("spacer", 6),
            ("section", "How benefits will be paid"),
            ("checkbox", "Select one payment method:",
             [("Direct deposit", True), ("Paper check", False)]),
            ("fields", [("Account type", payment["accountType"]),
                        ("Routing number - last four digits",
                         payment["bankRoutingNumberLastFour"])]),
            ("spacer", 10),
            ("paragraph",
             "Return the completed request to the carrier at the address above "
             "within thirty days of the first day of leave. Keep a copy for your "
             "records. Benefits cannot be paid until Part B has been completed by "
             "the employer."),
        ],
    ]

    schema = {
        "type": "object",
        "properties": {
            "requestPartA": {
                "type": "object",
                "description": "Part A, completed by the employee",
                "properties": {
                    "employeeName": {"type": "string"},
                    "claimNumber": {"type": "string"},
                    "employeeCity": {"type": "string"},
                    "employeeState": {"type": "string"},
                    "employeeZipCode": {"type": "string"},
                    "gender": {"type": "string"},
                    "preferredLanguage": {"type": "string"},
                    "race": {"type": "string"},
                    "reasonForRequest": {"type": "string"},
                    "familyMemberRelationship": {"type": "string"},
                    "leaveScheduleType": {"type": "string"},
                    "leaveStartDate": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "leaveEndDate": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "datesAreEstimated": {"type": "boolean"},
                    "employeeDateOfHire": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "hasMoreThanOneEmployer": {
                        "type": "string", "enum": ["Yes", "No"]},
                    "takingLeaveFromOtherEmployer": {
                        "type": "string", "enum": ["Yes", "No"]},
                    "receivingWorkersCompensation": {
                        "type": "string", "enum": ["Yes", "No"]},
                    "employeeSignatureDate": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                },
            },
            "requestPartB": {
                "type": "object",
                "description": "Part B, completed by the employer",
                "properties": {
                    "employerName": {"type": "string"},
                    "employerFein": {"type": "string"},
                    "employerDateOfHire": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "occupationCode": {"type": "string"},
                    "grossWages": {
                        "type": "array",
                        "description": "One entry per week of reported wages",
                        "items": {
                            "type": "object",
                            "properties": {
                                "weekNumber": {"type": "number"},
                                "weekEndingDate": {
                                    "type": "string",
                                    "description": "ISO 8601 date (YYYY-MM-DD)"},
                                "numberOfDaysWorked": {
                                    "type": ["number", "null"],
                                    "description": "Blank where the week was not "
                                                   "reported"},
                                "grossAmountPaid": {"type": "number"},
                            },
                        },
                    },
                    "calculatedAverageGrossWeeklyWage": {"type": "number"},
                    "requestingReimbursement": {
                        "type": "string", "enum": ["Yes", "No"]},
                    "leaveTakenInPreceding52Weeks": {"type": "string"},
                    "takingFmlaConcurrently": {
                        "type": "string", "enum": ["Yes", "No"]},
                    "employerSignatureDate": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "employerTitle": {"type": "string"},
                    "insuranceCarrier": {
                        "type": "object",
                        "description": "Carrier providing the coverage",
                        "properties": {
                            "name": {"type": "string"},
                            "streetAddress": {"type": "string"},
                            "city": {"type": "string"},
                            "state": {"type": "string"},
                            "zipCode": {"type": "string"},
                        },
                    },
                },
            },
            "bondingCertification": {
                "type": "object",
                "description": "Certification of the birth or placement of a child",
                "properties": {
                    "childName": {"type": "string"},
                    "childDateOfBirth": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "childGender": {"type": "string"},
                    "childLivesWithEmployee": {
                        "type": "string", "enum": ["Yes", "No"]},
                    "relationshipToChild": {"type": "string"},
                    "documentationAttached": {"type": "string"},
                    "employeeSignatureDate": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                },
            },
            "paymentEnrollment": {
                "type": "object",
                "description": "How benefits are paid",
                "properties": {
                    "policyNumber": {"type": "string"},
                    "carrierTelephoneNumber": {"type": "string"},
                    "paymentMethod": {"type": "string"},
                    "accountType": {"type": "string"},
                    "bankRoutingNumberLastFour": {"type": "string"},
                },
            },
        },
    }

    return {
        "id": "pfl-synthetic",
        "form_number": "SYN-PFL-1 (rev 10-25)",
        "title": "Request For Paid Family Leave - Bonding",
        "fax_stamp": "FAX 05/22/2025 09:41  HALBERD CERAMICS HR  518-555-0142  P.001",
        "pages": pages,
        "schema": schema,
        "truth": truth,
        "seed": 20260808,
    }


def _pml_form() -> Dict[str, Any]:
    """
    Build the paid medical leave claim

    Returns:
        A form specification with "id", "pages", "schema" and "truth"
    """
    truth: Dict[str, Any] = {
        "claimantInformation": {
            "claimantName": "Errol Vantassel",
            "claimNumber": "SYN-8802455",
            "dateOfBirth": "1984-11-02",
            "workState": "NJ",
            "occupation": "Warehouse team lead",
            "lastDayWorked": "2025-03-14",
            "firstDayOfDisability": "2025-03-17",
            "expectedReturnToWork": "2025-06-09",
            "disabilityDueToEmployment": "No",
            "disabilityDueToPregnancy": "No",
            "otherIncomeReceived": "None",
            "claimantSignatureDate": "2025-03-21",
        },
        "employerStatement": {
            "employerName": "Kestrel Freight Handling LLC",
            "employerFein": "22-4180976",
            "supervisorName": "Roshanne Iyer-Bowe",
            "weeklyHours": 40,
            "hourlyRate": 31.5,
            "annualSalary": 65520.0,
            "payFrequency": "Biweekly",
            "employeeStillEmployed": "Yes",
            "lastDayWorkedPerPayroll": "2025-03-14",
            "employerSignatureDate": "2025-03-25",
        },
        "attendingPhysicianStatement": {
            "physicianName": "Dr. Ines Rakotomalala",
            "physicianLicenseNumber": "NJ-MD-118093",
            "practiceName": "Passaic Valley Orthopaedics",
            "diagnosis": "Lumbar disc herniation with radiculopathy",
            "icd10Code": "M51.16",
            "dateOfFirstVisit": "2025-03-17",
            "dateSymptomsFirstAppeared": "2025-03-10",
            "datesOfTreatment": ["2025-03-17", "2025-04-07", "2025-05-12"],
            "surgeryPerformed": "No",
            "totallyDisabledFrom": "2025-03-17",
            "totallyDisabledThrough": "2025-06-08",
            "canReturnOnLimitedBasis": "Yes",
            "limitationsRestrictions": "No lifting over 15 lbs, no repeated bending",
            "physicianSignatureDate": "2025-03-24",
        },
    }

    claimant = truth["claimantInformation"]
    employer = truth["employerStatement"]
    physician = truth["attendingPhysicianStatement"]

    pages = [
        [
            ("section", "Section 1 - claimant information"),
            ("fields", [("Claimant name", claimant["claimantName"]),
                        ("Claim number", claimant["claimNumber"])]),
            ("fields", [("Date of birth", _us_date(claimant["dateOfBirth"])),
                        ("State where you work", claimant["workState"]),
                        ("Occupation", claimant["occupation"])]),
            ("fields", [("Last day worked", _us_date(claimant["lastDayWorked"])),
                        ("First day of disability",
                         _us_date(claimant["firstDayOfDisability"]))]),
            ("fields", [("Date you expect to return to work",
                         _us_date(claimant["expectedReturnToWork"]))]),
            ("checkbox", "Is this disability related to your employment?",
             [("Yes", False), ("No", True)]),
            ("checkbox", "Is this disability related to pregnancy?",
             [("Yes", False), ("No", True)]),
            ("fields", [("Other income received during this period",
                         claimant["otherIncomeReceived"])]),
            ("spacer", 10),
            ("paragraph",
             "I authorise any physician, hospital or other institution to release "
             "to the division any information concerning my medical history that "
             "may be needed to determine my eligibility for benefits. A photocopy "
             "of this authorisation is as valid as the original."),
            ("signature", claimant["claimantName"], "Claimant",
             _us_date(claimant["claimantSignatureDate"])),
        ],
        [
            ("section", "Section 2 - employer statement"),
            ("fields", [("Employer name", employer["employerName"]),
                        ("Federal employer id number", employer["employerFein"])]),
            ("fields", [("Supervisor name", employer["supervisorName"]),
                        ("Scheduled hours per week", str(employer["weeklyHours"]))]),
            ("fields", [("Hourly rate", _money(employer["hourlyRate"])),
                        ("Annual salary", _money(employer["annualSalary"])),
                        ("Pay frequency", employer["payFrequency"])]),
            ("checkbox", "Is the employee still on your payroll?",
             [("Yes", True), ("No", False)]),
            ("fields", [("Last day worked per payroll records",
                         _us_date(employer["lastDayWorkedPerPayroll"]))]),
            ("spacer", 10),
            ("paragraph",
             "Complete and return this section within ten days of receipt. Wage "
             "figures must be taken from payroll records rather than from the "
             "employee's statement, and must exclude any bonus paid outside the "
             "base period."),
            ("signature", employer["supervisorName"], "Authorised representative",
             _us_date(employer["employerSignatureDate"])),
        ],
        [
            ("section", "Section 3 - attending physician statement"),
            ("fields", [("Physician name", physician["physicianName"]),
                        ("License number",
                         physician["physicianLicenseNumber"])]),
            ("fields", [("Practice name", physician["practiceName"])]),
            ("fields", [("Diagnosis", physician["diagnosis"]),
                        ("ICD-10 code", physician["icd10Code"])]),
            ("fields", [("Date of first visit",
                         _us_date(physician["dateOfFirstVisit"])),
                        ("Date symptoms first appeared",
                         _us_date(physician["dateSymptomsFirstAppeared"]))]),
            ("table", ["Visit", "Date of treatment"],
             [[str(index + 1), _us_date(date)]
              for index, date in enumerate(physician["datesOfTreatment"])]),
            ("checkbox", "Was surgery performed?", [("Yes", False), ("No", True)]),
            ("fields", [("Totally disabled from",
                         _us_date(physician["totallyDisabledFrom"])),
                        ("Totally disabled through",
                         _us_date(physician["totallyDisabledThrough"]))]),
            ("checkbox", "Can the patient return to work on a limited basis?",
             [("Yes", True), ("No", False)]),
            ("fields", [("Limitations and restrictions",
                         physician["limitationsRestrictions"])]),
            ("signature", physician["physicianName"], "Attending physician",
             _us_date(physician["physicianSignatureDate"])),
        ],
    ]

    schema = {
        "type": "object",
        "properties": {
            "claimantInformation": {
                "type": "object",
                "description": "Section 1, completed by the claimant",
                "properties": {
                    "claimantName": {"type": "string"},
                    "claimNumber": {"type": "string"},
                    "dateOfBirth": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "workState": {"type": "string"},
                    "occupation": {"type": "string"},
                    "lastDayWorked": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "firstDayOfDisability": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "expectedReturnToWork": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "disabilityDueToEmployment": {
                        "type": "string", "enum": ["Yes", "No"]},
                    "disabilityDueToPregnancy": {
                        "type": "string", "enum": ["Yes", "No"]},
                    "otherIncomeReceived": {"type": "string"},
                    "claimantSignatureDate": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                },
            },
            "employerStatement": {
                "type": "object",
                "description": "Section 2, completed by the employer",
                "properties": {
                    "employerName": {"type": "string"},
                    "employerFein": {"type": "string"},
                    "supervisorName": {"type": "string"},
                    "weeklyHours": {"type": "number"},
                    "hourlyRate": {"type": "number"},
                    "annualSalary": {"type": "number"},
                    "payFrequency": {"type": "string"},
                    "employeeStillEmployed": {
                        "type": "string", "enum": ["Yes", "No"]},
                    "lastDayWorkedPerPayroll": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "employerSignatureDate": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                },
            },
            "attendingPhysicianStatement": {
                "type": "object",
                "description": "Section 3, completed by the attending physician",
                "properties": {
                    "physicianName": {"type": "string"},
                    "physicianLicenseNumber": {"type": "string"},
                    "practiceName": {"type": "string"},
                    "diagnosis": {"type": "string"},
                    "icd10Code": {"type": "string"},
                    "dateOfFirstVisit": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "dateSymptomsFirstAppeared": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "datesOfTreatment": {
                        "type": "array",
                        "description": "ISO 8601 dates (YYYY-MM-DD)",
                        "items": {"type": "string"},
                    },
                    "surgeryPerformed": {"type": "string", "enum": ["Yes", "No"]},
                    "totallyDisabledFrom": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "totallyDisabledThrough": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "canReturnOnLimitedBasis": {
                        "type": "string", "enum": ["Yes", "No"]},
                    "limitationsRestrictions": {"type": "string"},
                    "physicianSignatureDate": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                },
            },
        },
    }

    return {
        "id": "pml-synthetic",
        "form_number": "SYN-PML-30 (rev 01-25)",
        "title": "Claim For Paid Medical Leave Benefits",
        "fax_stamp": "FAX 03/25/2025 14:07  KESTREL FREIGHT  973-555-0188  P.001",
        "pages": pages,
        "schema": schema,
        "truth": truth,
        "seed": 20260809,
    }


def _std_form() -> Dict[str, Any]:
    """
    Build the short-term disability claim

    Returns:
        A form specification with "id", "pages", "schema" and "truth"
    """
    truth: Dict[str, Any] = {
        "employeeSection": {
            "employeeName": "Priya Lindqvist-Amara",
            "memberId": "SYN-WA-77315",
            "workState": "WA",
            "gender": "FEMALE",
            "jobTitle": "Dental hygienist",
            "disabilityDueToEmployment": "NO",
            "disabilityDueToAccident": "NO",
            "disabilityDueToMilitaryService": "NO",
            "dateSymptomsFirstAppeared": "2025-01-27",
            "lastDayWorked": "2025-02-03",
            "returnToWorkDate": "2025-05-05",
            "returnToWorkDateType": "ESTIMATED",
            "federalTaxWithholdingPerWeek": 25,
            "employeeSignatureDate": "2025-02-10",
        },
        "physicianSection": {
            "physicianName": "Dr. Tobias Wrenfield",
            "diagnosis": "Left rotator cuff tear, post-operative",
            "icd10Code": "M75.102",
            "dateOfFirstVisit": "2025-01-29",
            "dateOfSurgery": "2025-02-14",
            "surgicalProcedure": "Arthroscopic rotator cuff repair",
            "hospitalizedFrom": "2025-02-14",
            "hospitalizedThrough": "2025-02-15",
            "totallyDisabledFrom": "2025-02-04",
            "totallyDisabledThrough": "2025-05-04",
            "dateOfNextAppointment": "2025-04-21",
            "canReturnOnLimitedBasis": "YES",
            "limitationsRestrictions": "No overhead reaching, 10 lb lifting limit",
            "referredPatientToAnotherPhysician": "YES",
            "physicianSignatureDate": "2025-02-18",
        },
        "employerSection": {
            "employerName": "Cascadia Family Dental PLLC",
            "employerGroupNumber": "GRP-441902",
            "employeeDateOfHire": "2019-08-26",
            "scheduledHoursPerWeek": 36,
            "grossWeeklyEarnings": 1585.0,
            "salaryContinuationPaid": "Yes",
            "salaryContinuationThrough": "2025-02-28",
            "employerSignatureDate": "2025-02-21",
        },
        "authorization": {
            "authorizationSigned": True,
            "authorizationDate": "2025-02-10",
            "recordsReleaseScope": "All records relating to the treated condition",
            "revocationNoticeAddress": "Claims Intake, PO Box 20551, Tukwila WA 98188",
        },
    }

    employee = truth["employeeSection"]
    physician = truth["physicianSection"]
    employer = truth["employerSection"]
    authorization = truth["authorization"]

    pages = [
        [
            ("section", "Employee statement"),
            ("fields", [("Employee name", employee["employeeName"]),
                        ("Member id", employee["memberId"])]),
            ("fields", [("Work state", employee["workState"]),
                        ("Gender", employee["gender"]),
                        ("Job title", employee["jobTitle"])]),
            ("checkbox", "Is the disability due to your employment?",
             [("Yes", False), ("No", True)]),
            ("checkbox", "Is the disability the result of an accident?",
             [("Yes", False), ("No", True)]),
            ("checkbox", "Is the disability due to military service?",
             [("Yes", False), ("No", True)]),
            ("fields", [("Date symptoms first appeared",
                         _us_date(employee["dateSymptomsFirstAppeared"])),
                        ("Last day worked",
                         _us_date(employee["lastDayWorked"]))]),
            ("fields", [("Return to work date",
                         _us_date(employee["returnToWorkDate"])),
                        ("Is that date actual or estimated?",
                         employee["returnToWorkDateType"])]),
            ("fields", [("Federal tax to withhold per week",
                         _money(employee["federalTaxWithholdingPerWeek"]))]),
            ("spacer", 8),
            ("paragraph",
             "Benefits are payable only for the period during which you are "
             "disabled and under the continuing care of a physician. Notify the "
             "claims office within five days of any return to work, whether full "
             "time or on a limited basis."),
            ("signature", employee["employeeName"], "Employee",
             _us_date(employee["employeeSignatureDate"])),
        ],
        [
            ("section", "Attending physician statement"),
            ("fields", [("Physician name", physician["physicianName"]),
                        ("ICD-10 code", physician["icd10Code"])]),
            ("fields", [("Diagnosis", physician["diagnosis"])]),
            ("fields", [("Date of first visit",
                         _us_date(physician["dateOfFirstVisit"])),
                        ("Date of next appointment",
                         _us_date(physician["dateOfNextAppointment"]))]),
            ("spacer", 6),
            ("section", "Surgery and hospitalisation"),
            ("fields", [("Date of surgery",
                         _us_date(physician["dateOfSurgery"]))]),
            ("fields", [("Procedure performed",
                         physician["surgicalProcedure"])]),
            ("fields", [("Hospitalised from",
                         _us_date(physician["hospitalizedFrom"])),
                        ("Hospitalised through",
                         _us_date(physician["hospitalizedThrough"]))]),
            ("spacer", 6),
            ("section", "Period of disability"),
            ("fields", [("Totally disabled from",
                         _us_date(physician["totallyDisabledFrom"])),
                        ("Totally disabled through",
                         _us_date(physician["totallyDisabledThrough"]))]),
            ("checkbox", "Can the patient work on a limited basis?",
             [("Yes", True), ("No", False)]),
            ("fields", [("Limitations and restrictions",
                         physician["limitationsRestrictions"])]),
            ("checkbox", "Have you referred the patient to another physician?",
             [("Yes", True), ("No", False)]),
            ("spacer", 8),
            ("signature", physician["physicianName"], "Attending physician",
             _us_date(physician["physicianSignatureDate"])),
        ],
        [
            ("section", "Employer statement"),
            ("fields", [("Employer name", employer["employerName"]),
                        ("Group number", employer["employerGroupNumber"])]),
            ("fields", [("Employee date of hire",
                         _us_date(employer["employeeDateOfHire"])),
                        ("Scheduled hours per week",
                         str(employer["scheduledHoursPerWeek"]))]),
            ("fields", [("Gross weekly earnings",
                         _money(employer["grossWeeklyEarnings"]))]),
            ("checkbox", "Is salary continuation being paid?",
             [("Yes", True), ("No", False)]),
            ("fields", [("Salary continuation paid through",
                         _us_date(employer["salaryContinuationThrough"]))]),
            ("spacer", 10),
            ("paragraph",
             "Report gross earnings before deductions, excluding overtime and any "
             "bonus not paid at regular intervals. Where salary continuation is "
             "paid, benefits are offset for the same period and the employer is "
             "reimbursed directly."),
            ("signature", "Marguerite Osei-Duffy", "Plan administrator",
             _us_date(employer["employerSignatureDate"])),
        ],
        [
            ("section", "Authorisation to release medical records"),
            ("fields", [("Scope of records released",
                         authorization["recordsReleaseScope"])]),
            ("fields", [("Send written revocation to",
                         authorization["revocationNoticeAddress"])]),
            ("checkbox", "I have read and agree to this authorisation:",
             [("Yes", authorization["authorizationSigned"]), ("No", False)]),
            ("spacer", 10),
            ("paragraph",
             "This authorisation is valid for twenty-four months from the date "
             "signed. I may revoke it at any time by writing to the address above, "
             "except to the extent that action has already been taken in reliance "
             "on it. I understand that refusing to sign may mean the claim cannot "
             "be evaluated."),
            ("spacer", 10),
            ("signature", employee["employeeName"], "Employee",
             _us_date(authorization["authorizationDate"])),
        ],
    ]

    schema = {
        "type": "object",
        "properties": {
            "employeeSection": {
                "type": "object",
                "description": "Completed by the employee",
                "properties": {
                    "employeeName": {"type": "string"},
                    "memberId": {"type": "string"},
                    "workState": {"type": "string"},
                    "gender": {"type": "string"},
                    "jobTitle": {"type": "string"},
                    "disabilityDueToEmployment": {
                        "type": "string", "enum": ["YES", "NO"]},
                    "disabilityDueToAccident": {
                        "type": "string", "enum": ["YES", "NO"]},
                    "disabilityDueToMilitaryService": {
                        "type": "string", "enum": ["YES", "NO"]},
                    "dateSymptomsFirstAppeared": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "lastDayWorked": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "returnToWorkDate": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "returnToWorkDateType": {
                        "type": "string", "enum": ["ACTUAL", "ESTIMATED"]},
                    "federalTaxWithholdingPerWeek": {"type": "number"},
                    "employeeSignatureDate": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                },
            },
            "physicianSection": {
                "type": "object",
                "description": "Completed by the attending physician",
                "properties": {
                    "physicianName": {"type": "string"},
                    "diagnosis": {"type": "string"},
                    "icd10Code": {"type": "string"},
                    "dateOfFirstVisit": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "dateOfSurgery": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "surgicalProcedure": {"type": "string"},
                    "hospitalizedFrom": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "hospitalizedThrough": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "totallyDisabledFrom": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "totallyDisabledThrough": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "dateOfNextAppointment": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "canReturnOnLimitedBasis": {
                        "type": "string", "enum": ["YES", "NO"]},
                    "limitationsRestrictions": {"type": "string"},
                    "referredPatientToAnotherPhysician": {
                        "type": "string", "enum": ["YES", "NO"]},
                    "physicianSignatureDate": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                },
            },
            "employerSection": {
                "type": "object",
                "description": "Completed by the employer",
                "properties": {
                    "employerName": {"type": "string"},
                    "employerGroupNumber": {"type": "string"},
                    "employeeDateOfHire": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "scheduledHoursPerWeek": {"type": "number"},
                    "grossWeeklyEarnings": {"type": "number"},
                    "salaryContinuationPaid": {
                        "type": "string", "enum": ["Yes", "No"]},
                    "salaryContinuationThrough": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "employerSignatureDate": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                },
            },
            "authorization": {
                "type": "object",
                "description": "Authorisation to release medical records",
                "properties": {
                    "authorizationSigned": {"type": "boolean"},
                    "authorizationDate": {
                        "type": "string",
                        "description": "ISO 8601 date (YYYY-MM-DD)"},
                    "recordsReleaseScope": {"type": "string"},
                    "revocationNoticeAddress": {"type": "string"},
                },
            },
        },
    }

    return {
        "id": "std-synthetic",
        "form_number": "SYN-STD-450 (rev 07-25)",
        "title": "Short Term Disability Claim Form",
        "fax_stamp": "FAX 02/21/2025 11:16  CASCADIA FAMILY DENTAL  206-555-0173  P.001",
        "pages": pages,
        "schema": schema,
        "truth": truth,
        "seed": 20260810,
    }


FORM_BUILDERS: Tuple[Callable[[], Dict[str, Any]], ...] = (
    _pfl_form, _pml_form, _std_form)


def write_sample(*, form: Dict[str, Any], sample_root: str) -> str:
    """
    Write one form's bundle - document, schema and ground truth

    Args:
        form: A form specification as returned by one of the FORM_BUILDERS
        sample_root: Directory the bundles live in, normally "sample"

    Returns:
        Path to the bundle directory that was written
    """
    bundle_dir = os.path.join(sample_root, form["id"])
    os.makedirs(bundle_dir, exist_ok=True)

    pdf_path = os.path.join(bundle_dir, f"{form['id']}.pdf")
    with open(pdf_path, "wb") as pdf_file:
        pdf_file.write(build_pdf(form=form, seed=form["seed"]))

    with open(os.path.join(bundle_dir, "schema.json"), "w", encoding="utf-8") as f:
        json.dump(form["schema"], f, indent=2)
        f.write("\n")

    with open(os.path.join(bundle_dir, "truth.json"), "w", encoding="utf-8") as f:
        json.dump(form["truth"], f, indent=2)
        f.write("\n")

    return bundle_dir


def check_sample(*, form: Dict[str, Any], sample_root: str) -> List[str]:
    """
    Report how a form's tracked bundle differs from what the generator produces now

    Args:
        form: A form specification as returned by one of the FORM_BUILDERS
        sample_root: Directory the bundles live in, normally "sample"

    Returns:
        One message per difference, empty when the bundle is up to date
    """
    bundle_dir = os.path.join(sample_root, form["id"])
    differences: List[str] = []

    pdf_path = os.path.join(bundle_dir, f"{form['id']}.pdf")
    if not os.path.exists(pdf_path):
        return [f"{pdf_path} is missing"]

    with open(pdf_path, "rb") as pdf_file:
        tracked_pages = page_fingerprints(pdf_bytes=pdf_file.read())

    expected_pages = page_fingerprints(pdf_bytes=build_pdf(form=form,
                                                           seed=form["seed"]))

    if tracked_pages != expected_pages:
        differences.append(
            f"{pdf_path} renders differently from the generator "
            f"({len(tracked_pages)} tracked pages, {len(expected_pages)} expected)")

    for filename, expected in (("schema.json", form["schema"]),
                               ("truth.json", form["truth"])):
        path = os.path.join(bundle_dir, filename)
        if not os.path.exists(path):
            differences.append(f"{path} is missing")
            continue
        with open(path, encoding="utf-8") as handle:
            if json.load(handle) != expected:
                differences.append(f"{path} differs from the generator")

    return differences


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Write, or verify, every synthetic sample bundle

    Args:
        argv: Command-line arguments, defaulting to sys.argv[1:]

    Returns:
        Process exit status: 0 on success, 1 when --check found a difference
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample-root", default="sample",
                        help="directory the sample bundles live in")
    parser.add_argument("--check", action="store_true",
                        help="verify the tracked bundles without writing anything")
    arguments = parser.parse_args(argv)

    forms = [builder() for builder in FORM_BUILDERS]

    if arguments.check:
        all_differences: List[str] = []
        for form in forms:
            all_differences.extend(
                check_sample(form=form, sample_root=arguments.sample_root))

        if all_differences:
            for difference in all_differences:
                print(f"stale: {difference}")
            return 1

        print(f"{len(forms)} synthetic sample bundles are up to date")
        return 0

    for form in forms:
        bundle_dir = write_sample(form=form, sample_root=arguments.sample_root)
        print(f"wrote {bundle_dir} ({len(form['pages'])} pages)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
