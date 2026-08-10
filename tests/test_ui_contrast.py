"""
Tests that every colour the app chooses stays legible in both Gradio themes

VS Code's built-in browser follows the editor theme, so it serves Gradio the dark
palette. Every readability defect this suite guards against had one shape: a colour
picked for a light background, rendered on a dark one. The Compare tab was the worst
case - pale green and pink row tints with near-white inherited text.

Screenshots diagnosed that bug but make a poor regression test: they need a human eye
and they only ever cover one theme. So this suite works numerically instead. It

  1. resolves the real body-text and background colours out of CUSTOM_THEME, for light
     AND dark, following Gradio's `*name` reference syntax;
  2. composites every translucent tint in APP_CSS over both backgrounds; and
  3. asserts WCAG AA contrast against the theme's own text colour on each result.

What it deliberately does not prove is that Gradio applied the stylesheet at all - only
that the colour arithmetic is sound. That last step is a look at the running app.

This file replaces tests/test_diff_table_contrast.py, which asserted on an inline
<style> block inside the rendered diff HTML. That block no longer exists: the rules
moved into APP_CSS, injected once. Its intent - "no rule sets a background without a
text colour to go with it" - is carried forward below, and strengthened, because the
old version could not have caught the bug that actually shipped.
"""

import re
from typing import Dict, Tuple

import pytest

from shared.comparison_utils import create_diff_view
from shared.ui_theme import (
    APP_CSS,
    CUSTOM_THEME,
    STATUS_TONES,
    banner,
    note,
    page_readout,
)

# WCAG 2.1 AA: 4.5:1 for body text, 3:1 for large text and non-text (borders, marks).
AA_TEXT = 4.5
AA_LARGE = 3.0

RGBA = re.compile(r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d.]+)\s*\)")
CSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")

TRUTH = {
    "pfl1PartA": {
        "employeeDateOfHire": "2023-09-06",
        "employeeSignatureDate": "2025-03-05",
    },
    "pfl1PartB": {"grossWages": [{"weekNumber": 1, "grossAmountPaid": "2400.00"}]},
}

# One match, one mismatch and one missing field, so every coloured row class and the
# .missing marker all appear in the rendered output.
EXTRACTED = {
    "pfl1PartA": {"employeeSignatureDate": "2025-03-25"},
    "pfl1PartB": {"grossWages": [{"weekNumber": 1, "grossAmountPaid": "2400.00"}]},
}


def resolve_theme_colour(*, attribute: str) -> str:
    """
    Read one colour off CUSTOM_THEME, following Gradio's reference syntax

    Gradio stores a theme value either as a literal ("white", "#1e293b") or as a
    reference to a named step on the palette, written "*neutral_800". A reference has to
    be dereferenced against the theme to get a hex value.

    Args:
        attribute (str): Theme attribute name, e.g. "body_text_color_dark".

    Returns:
        str: The colour as "#rrggbb".

    Raises:
        AttributeError: If the attribute, or the palette step it points at, is absent -
                        which is how a Gradio upgrade that renames a token surfaces,
                        rather than as a silently skipped assertion.
        ValueError: If the resolved value is neither a hex colour nor a keyword this
                    function knows.
    """
    value = getattr(CUSTOM_THEME, attribute)

    if value.startswith("*"):
        value = getattr(CUSTOM_THEME, value[1:])

    if value.startswith("#"):
        return value

    # Gradio writes the light theme's primary background as the CSS keyword rather than
    # a palette step.
    keywords = {"white": "#ffffff", "black": "#000000"}
    if value in keywords:
        return keywords[value]

    raise ValueError(f"Cannot resolve theme colour {attribute}={value!r} to hex")


def parse_hex(*, hex_colour: str) -> Tuple[int, int, int]:
    """
    Split "#rrggbb" into 8-bit channel values

    Args:
        hex_colour (str): Colour as "#rrggbb".

    Returns:
        Tuple[int, int, int]: Red, green and blue, each 0-255.
    """
    return (
        int(hex_colour[1:3], 16),
        int(hex_colour[3:5], 16),
        int(hex_colour[5:7], 16),
    )


def composite(
    *, foreground: Tuple[int, int, int], alpha: float, background: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    """
    Flatten a translucent colour onto an opaque one, as a browser would

    Args:
        foreground (Tuple[int, int, int]): The translucent colour's RGB.
        alpha (float): Its opacity, 0-1.
        background (Tuple[int, int, int]): The opaque colour underneath.

    Returns:
        Tuple[int, int, int]: The resulting opaque RGB.
    """
    return tuple(
        round(front * alpha + back * (1 - alpha))
        for front, back in zip(foreground, background)
    )


def relative_luminance(*, rgb: Tuple[int, int, int]) -> float:
    """
    Compute WCAG relative luminance

    Args:
        rgb (Tuple[int, int, int]): Channel values, 0-255.

    Returns:
        float: Luminance between 0 and 1.
    """
    channels = []
    for value in rgb:
        srgb = value / 255
        # The WCAG transfer function: linear below the knee, gamma above it.
        channels.append(srgb / 12.92 if srgb <= 0.03928 else ((srgb + 0.055) / 1.055) ** 2.4)

    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(
    *, first: Tuple[int, int, int], second: Tuple[int, int, int]
) -> float:
    """
    Compute the WCAG contrast ratio between two opaque colours

    Args:
        first (Tuple[int, int, int]): One colour's RGB.
        second (Tuple[int, int, int]): The other colour's RGB.

    Returns:
        float: A ratio from 1.0 (identical) to 21.0 (black on white).
    """
    lighter = max(
        relative_luminance(rgb=first), relative_luminance(rgb=second)
    )
    darker = min(relative_luminance(rgb=first), relative_luminance(rgb=second))
    return (lighter + 0.05) / (darker + 0.05)


def theme_pair(*, theme: str) -> Dict[str, Tuple[int, int, int]]:
    """
    Resolve the text and background colours for one theme

    Args:
        theme (str): "light" or "dark".

    Returns:
        Dict[str, Tuple[int, int, int]]: RGB for "text", "primary" and "secondary".

    Raises:
        ValueError: If the theme name is not "light" or "dark".
    """
    if theme not in {"light", "dark"}:
        raise ValueError(f"Expected 'light' or 'dark', got {theme!r}")

    suffix = "_dark" if theme == "dark" else ""

    return {
        "text": parse_hex(
            hex_colour=resolve_theme_colour(attribute=f"body_text_color{suffix}")
        ),
        "primary": parse_hex(
            hex_colour=resolve_theme_colour(
                attribute=f"background_fill_primary{suffix}"
            )
        ),
        "secondary": parse_hex(
            hex_colour=resolve_theme_colour(
                attribute=f"background_fill_secondary{suffix}"
            )
        ),
    }


def css_rules() -> list[tuple[str, str]]:
    """
    Parse APP_CSS into (selector, declarations) pairs

    Comments are stripped first: they sit between rules, so otherwise they land in the
    following rule's selector capture.

    Returns:
        List of (selector, declarations), both lowercased and stripped.
    """
    body = CSS_COMMENT.sub("", APP_CSS)
    return [
        (selector.strip().lower(), declarations.strip().lower())
        for selector, declarations in RULE.findall(body)
    ]


THEMES = ["light", "dark"]
SURFACES = ["primary", "secondary"]


class TestThemeResolution:
    """The theme tokens this suite depends on exist and resolve to real colours"""

    @pytest.mark.parametrize("theme", THEMES)
    def test_text_and_backgrounds_resolve(self, theme: str) -> None:
        colours = theme_pair(theme=theme)
        assert set(colours) == {"text", "primary", "secondary"}

    @pytest.mark.parametrize("theme", THEMES)
    @pytest.mark.parametrize("surface", SURFACES)
    def test_theme_text_on_its_own_background_already_passes(
        self, theme: str, surface: str
    ) -> None:
        """
        Gradio's own pairing is AA-compliant

        This is the premise the whole approach rests on: if the theme's text on the
        theme's background is fine, then a tint that barely moves the background keeps
        it fine. If Gradio ever ships a palette that fails here, the tints below are no
        longer safe either, and this is the test that says so.
        """
        colours = theme_pair(theme=theme)
        ratio = contrast_ratio(first=colours["text"], second=colours[surface])

        assert ratio >= AA_TEXT, (
            f"{theme} theme's own body text on its {surface} background is only "
            f"{ratio:.1f}:1"
        )


class TestTintsStayLegible:
    """Every translucent tint in APP_CSS keeps AA contrast, in both themes"""

    @pytest.mark.parametrize("theme", THEMES)
    @pytest.mark.parametrize("surface", SURFACES)
    def test_every_rgba_tint_in_the_stylesheet(self, theme: str, surface: str) -> None:
        """
        Composite each rgba() in APP_CSS over the theme background and check the text

        Driven off the stylesheet text rather than off a hand-written list, so a tint
        added later is covered without anyone remembering to add it here.
        """
        tints = RGBA.findall(APP_CSS)
        assert tints, "no rgba() tints found in APP_CSS - has the stylesheet moved?"

        colours = theme_pair(theme=theme)
        failures = []

        for red, green, blue, alpha in tints:
            tinted = composite(
                foreground=(int(red), int(green), int(blue)),
                alpha=float(alpha),
                background=colours[surface],
            )
            ratio = contrast_ratio(first=colours["text"], second=tinted)
            if ratio < AA_TEXT:
                failures.append(
                    f"rgba({red},{green},{blue},{alpha}) -> {ratio:.2f}:1"
                )

        assert failures == [], (
            f"On the {theme} theme's {surface} surface these tints drop body text "
            f"below {AA_TEXT}:1 - {failures}"
        )

    @pytest.mark.parametrize("tone,colour", sorted(STATUS_TONES.items()))
    @pytest.mark.parametrize("theme", THEMES)
    def test_tone_accent_is_visible_as_a_border(
        self, tone: str, colour: str, theme: str
    ) -> None:
        """
        Each tone's saturated accent meets the 3:1 non-text threshold

        The accent carries the semantic signal as a 3px left rule, so it has to be
        distinguishable from the surface behind it even though it never has text on it.
        """
        colours = theme_pair(theme=theme)
        ratio = contrast_ratio(
            first=parse_hex(hex_colour=colour), second=colours["primary"]
        )

        assert ratio >= AA_LARGE, (
            f"The {tone} accent {colour} is only {ratio:.1f}:1 against the {theme} "
            f"background, so its left rule is hard to see"
        )


class TestNoRuleAssertsATextColourWithoutABackground:
    """
    The structural invariant, stated as its own test

    The bug that shipped was subtler than "a rule set a background but no colour": the
    previous fix DID set a colour - on the <tr>. A colour on a row only reaches a cell
    by inheritance, and inheritance loses to any rule matching the cell directly, which
    Gradio's stylesheet has. So the background applied and the colour did not.
    """

    def test_diff_table_sets_colour_on_cells_not_only_rows(self) -> None:
        """
        The cells carry the text colour

        This is the assertion that would have caught the shipped bug, and no
        colour-value assertion could have.
        """
        # Scoped to .diff-table on purpose. An earlier draft of this test asked only
        # for "some rule mentioning td that sets a colour", and .sub-table's rule
        # satisfied that - so it passed even with the diff table reverted to the broken
        # row-level form. The selector has to name the table being asserted about.
        cell_rules = [
            declarations
            for selector, declarations in css_rules()
            if ".diff-table" in selector
            and re.search(r"\btd\b", selector)
            and "color:" in declarations
        ]

        assert cell_rules, (
            "No rule targets .diff-table's cells with a text colour. A colour set on "
            "the row inherits, and inheritance loses to Gradio's own td rule."
        )

    def test_row_tints_are_applied_through_the_cells(self) -> None:
        """
        The match/mismatch tints target `tr.match > td`, not `tr.match`

        A background on the <tr> paints behind cells that have their own background,
        and it was the pairing of a row background with a row colour that failed.
        """
        tinted_row_selectors = [
            selector
            for selector, declarations in css_rules()
            if ("tr.match" in selector or "tr.mismatch" in selector)
            and "background" in declarations
        ]

        assert tinted_row_selectors, "match/mismatch tints are absent from APP_CSS"
        for selector in tinted_row_selectors:
            assert selector.endswith("td"), (
                f"{selector!r} tints the row itself; it should tint the cells so the "
                f"tint and the text colour land on the same element"
            )

    def test_no_rule_hardcodes_a_body_text_colour(self) -> None:
        """
        Text colours come from the theme, never from a literal

        A literal text colour is exactly the defect being removed: it cannot follow the
        theme, so it is wrong in one of the two. Note the negative lookbehind - a
        literal is still fine on `border-left-color`, which is not text.
        """
        offenders = [
            (selector, declarations)
            for selector, declarations in css_rules()
            if re.search(r"(?<!-)\bcolor:\s*#", declarations)
        ]

        assert offenders == [], (
            f"These rules set a literal text colour instead of var(--body-text-color), "
            f"so they cannot follow the theme: {offenders}"
        )


class TestRenderedMarkupUsesTheStylesheet:
    """The HTML the app emits carries the classes APP_CSS styles - no inline colours"""

    def test_diff_view_carries_no_style_block(self) -> None:
        """
        The comparison view no longer ships its own <style>

        It used to inject one on every render, which is both duplicated markup and a
        second place a colour decision could be made.
        """
        assert "<style>" not in create_diff_view(TRUTH, EXTRACTED)

    def test_diff_view_applies_the_row_classes(self) -> None:
        """Both row classes reach real rows, not merely the stylesheet"""
        diff_html = create_diff_view(TRUTH, EXTRACTED)

        assert "class='match'" in diff_html
        assert "class='mismatch'" in diff_html

    def test_values_are_wrapped_so_the_height_cap_applies(self) -> None:
        """
        Values sit in a .value-box div

        max-height is ignored on a <td>, so the previous cap - set on the cell - never
        did anything, and long claim-form values made the tab thousands of pixels tall.
        """
        assert "class='value-box'" in create_diff_view(TRUTH, EXTRACTED)

    @pytest.mark.parametrize("tone", sorted(STATUS_TONES))
    def test_banner_markup_is_class_only(self, tone: str) -> None:
        """A banner names its tone and carries no inline style"""
        markup = banner(tone=tone, text="something happened")

        assert f"ocr-banner--{tone}" in markup
        assert "style=" not in markup
        # The tone is written out as text, so a banner is still readable in greyscale
        # and to a colour-blind user.
        assert f">{tone}<" in markup

    def test_banner_rejects_an_unknown_tone(self) -> None:
        """An unrecognised tone fails loudly rather than rendering unstyled"""
        with pytest.raises(ValueError, match="Unknown banner tone"):
            banner(tone="critical", text="something happened")

    def test_note_and_page_readout_carry_no_inline_style(self) -> None:
        """The two other builders are class-only too"""
        assert "style=" not in note(text="nothing yet")
        assert "style=" not in note(text="nothing yet", tall=True)
        assert "style=" not in page_readout(current_page=0, total_pages=3)

    def test_page_readout_is_one_indexed_for_the_reader(self) -> None:
        """Page numbers are zero-indexed internally and shown one-indexed"""
        assert "Page 1 of 3" in page_readout(current_page=0, total_pages=3)
        assert "Page 3 of 3" in page_readout(current_page=2, total_pages=3)
