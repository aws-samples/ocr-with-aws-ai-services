"""
The app's visual language: one theme, one stylesheet, one way to build a status banner

Every readability defect this module replaces had the same shape: a hardcoded colour
in an inline `style` attribute that assumed the background it would land on. VS Code's
built-in browser follows the editor theme, so it serves Gradio its dark palette, and
a `color: #666` chosen for white then renders near-invisible.

The rule here is therefore: **never assert a background.**

- Text colour is always `var(--body-text-color)`, which Gradio sets correctly per
  theme, so it is right in both without this app knowing which is active.
- Semantic colour arrives as a translucent tint plus a saturated left rule. A tint at
  13-14% alpha barely moves the background's luminance, so text on it keeps
  essentially the contrast the theme already guarantees - again in both themes.

Callers use banner() and note() rather than writing HTML, so there is one place a
colour decision can be made and one place a test can check it.
tests/test_ui_contrast.py resolves the real theme colours from CUSTOM_THEME, composites
these tints over both, and asserts WCAG AA.
"""

from typing import Dict

import gradio as gr

# Semantic accents, in the app's amber-on-slate palette.
#
# These appear only as a 3px left rule and as the alpha component of a background
# tint - never as a text colour, and never as an opaque fill behind text. A saturated
# colour needs a matching foreground to stay readable, which is precisely the coupling
# that broke here before.
#
# Each is deliberately mid-toned so it clears the 3:1 non-text threshold against BOTH
# theme backgrounds - white and #0a0f1e. A brighter amber (#e08a1e was the first choice)
# reaches only 2.7:1 on white and its left rule all but disappears there.
#
# None of them is used as a text colour, and that is not a stylistic preference: for
# 4.5:1 against white a colour needs relative luminance <= 0.1833, and for 4.5:1 against
# #0a0f1e it needs >= 0.1975. No colour satisfies both, so a single saturated accent
# CANNOT be AA-legible text in both themes. Text is therefore always
# var(--body-text-color); tone is conveyed by the tint, the left rule, and the tone
# word itself. tests/test_ui_contrast.py asserts all of this.
STATUS_TONES: Dict[str, str] = {
    "info": "#3b7dd8",
    "ok": "#2a8f47",
    "warn": "#b8730f",
    "error": "#cf4040",
}

# Alpha used for every semantic tint. Low enough that the composited background stays
# within a hair of the theme's own, which is what keeps text contrast intact in both
# light and dark; high enough to read as a colour at a glance.
TINT_ALPHA = 0.14

# Cap on the height of a single value cell in the comparison table. Claim-form values
# include multi-page prose, and an uncapped cell made the Compare tab thousands of
# pixels tall. The previous cap was set on the <td> itself, where max-height has no
# effect, so it never applied.
VALUE_BOX_MAX_HEIGHT_PX = 200


def _tint(*, hex_colour: str, alpha: float = TINT_ALPHA) -> str:
    """
    Build an rgba() tint from a #rrggbb colour

    rgba is used in preference to color-mix() so the stylesheet carries no dependency
    on a recent browser engine - the tint has to render in whatever VS Code embeds.

    Args:
        hex_colour (str): Colour as "#rrggbb".
        alpha (float): Opacity between 0 and 1.

    Returns:
        str: A CSS rgba() colour.

    Raises:
        ValueError: If the colour is not a 7-character #rrggbb string.
    """
    if not (len(hex_colour) == 7 and hex_colour.startswith("#")):
        raise ValueError(f"Expected a #rrggbb colour, got {hex_colour!r}")

    red = int(hex_colour[1:3], 16)
    green = int(hex_colour[3:5], 16)
    blue = int(hex_colour[5:7], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _tone_rules() -> str:
    """
    Generate the per-tone banner modifier rules

    Written out from STATUS_TONES rather than by hand so a tone cannot exist in Python
    without a matching rule in the stylesheet.

    Returns:
        str: CSS rules, one block per tone.
    """
    return "\n".join(
        f"""
.ocr-banner--{tone} {{
    background: {_tint(hex_colour=colour)};
    border-left-color: {colour};
}}"""
        for tone, colour in STATUS_TONES.items()
    )


# The full stylesheet, injected once via gr.Blocks(css=...).
#
# Everything here targets a class this app owns. Gradio's own component internals are
# deliberately left alone: restyling framework classes by name is what breaks on
# upgrade.
APP_CSS = f"""
/* --- status banners ------------------------------------------------------- */

.ocr-banner {{
    display: flex;
    align-items: baseline;
    gap: 10px;
    padding: 10px 14px;
    margin: 2px 0;
    border-radius: 6px;
    border-left: 3px solid transparent;
    color: var(--body-text-color);
    font-size: 14px;
    line-height: 1.45;
}}

/* The tone name doubles as the only uppercase text in the app, which makes a banner
   scannable without relying on colour alone - it still reads for a colour-blind user
   and in a greyscale screenshot.

   It is NOT set in the tone's accent colour. At 11px it is small text needing 4.5:1,
   and no single saturated colour reaches that on both theme backgrounds - see the note
   on STATUS_TONES. The theme's own colour plus the tint and left rule carry it. */
.ocr-banner__tone {{
    flex: 0 0 auto;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--body-text-color);
    opacity: 0.8;
}}

.ocr-banner__text {{ flex: 1 1 auto; }}
.ocr-banner__text b {{ font-weight: 600; }}

/* Figures are the point of this app, so they are set in the mono face wherever they
   appear in prose. */
.ocr-banner__text code,
.ocr-metric {{
    font-family: var(--font-mono);
    font-size: 0.95em;
}}
{_tone_rules()}

/* --- muted secondary text ------------------------------------------------- */

/* Replaces every `color: #666` placeholder. Muted here means "the theme's own text,
   dimmed", which cannot invert when the theme does. */
.ocr-note {{
    color: var(--body-text-color);
    opacity: 0.65;
    font-size: 14px;
    text-align: center;
    padding: 14px 10px;
}}

.ocr-note--tall {{ padding: 44px 10px; }}

/* --- info cards ----------------------------------------------------------- */

.ocr-card {{
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    padding: 14px 16px;
    color: var(--body-text-color);
}}

.ocr-card__title {{
    margin: 0 0 8px 0;
    font-size: 14px;
    font-weight: 600;
}}

.ocr-card__row {{
    margin: 3px 0;
    font-size: 13px;
    display: flex;
    justify-content: space-between;
    gap: 12px;
}}

.ocr-card__row span:last-child {{
    font-family: var(--font-mono);
    opacity: 0.85;
}}

.ocr-card__footer {{
    margin: 10px 0 0 0;
    font-size: 12px;
    opacity: 0.65;
}}

/* --- page-navigation readout --------------------------------------------- */

.ocr-pageinfo {{
    text-align: center;
    padding: 6px 0;
    color: var(--body-text-color);
    font-family: var(--font-mono);
    font-size: 13px;
    opacity: 0.8;
}}

/* --- field-by-field comparison ------------------------------------------- */

.diff-container {{ color: var(--body-text-color); }}

.diff-summary {{
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
    align-items: baseline;
    padding: 10px 14px;
    margin-bottom: 12px;
    border-radius: 6px;
    border-left: 3px solid var(--color-accent);
    background: var(--background-fill-secondary);
}}

.diff-summary__figure {{
    font-family: var(--font-mono);
    font-size: 20px;
    font-weight: 600;
}}

.diff-summary__label {{
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    opacity: 0.65;
}}

.diff-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}

/* Colour is set on the CELLS, not on the row.
   A colour on <tr> only reaches a <td> by inheritance, and inheritance loses to any
   rule matching the cell directly - which Gradio's stylesheet has. That is why the
   previous fix left pale rows with near-white text: the background applied and the
   colour did not. */
.diff-table th,
.diff-table td {{
    border: 1px solid var(--border-color-primary);
    padding: 7px 9px;
    text-align: left;
    vertical-align: top;
    color: var(--body-text-color);
}}

.diff-table th {{
    background: var(--background-fill-secondary);
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    position: sticky;
    top: 0;
}}

.diff-table tr.match > td {{ background: {_tint(hex_colour=STATUS_TONES["ok"], alpha=0.13)}; }}
.diff-table tr.mismatch > td {{ background: {_tint(hex_colour=STATUS_TONES["error"], alpha=0.13)}; }}

/* With every engine in the table at once there is no row-level verdict to colour:
   Textract can match a field that BDA misses. So the tint goes on each engine's cell,
   leaving the field and expected columns plain. */
.diff-table td.cell-match {{ background: {_tint(hex_colour=STATUS_TONES["ok"], alpha=0.13)}; }}
.diff-table td.cell-mismatch {{ background: {_tint(hex_colour=STATUS_TONES["error"], alpha=0.13)}; }}

/* The glyph repeats what the tint says, so the verdict does not depend on separating
   two low-alpha tints - same reasoning as .missing's border. */
.diff-table .cell-mark {{
    font-weight: 700;
    margin-right: 6px;
    opacity: 0.75;
}}

/* Five columns of monospaced claim-form values overflow an auto-laid-out table, and
   the field name is the column that needs the least room. */
.diff-table--multi {{ table-layout: fixed; }}

.diff-table--multi th:first-child,
.diff-table--multi td:first-child {{ width: 14%; }}

.diff-table td.parent-path {{
    background: var(--background-fill-secondary);
    font-family: var(--font-mono);
    font-weight: 600;
    letter-spacing: 0.02em;
}}

.diff-table td.mark {{
    text-align: center;
    width: 40px;
    font-weight: 700;
}}

/* max-height on a <td> is ignored, so the cap goes on a wrapper inside it. Without
   this the Compare tab renders every value at full height. */
.value-box {{
    font-family: var(--font-mono);
    white-space: pre-wrap;
    word-break: break-word;
    max-height: {VALUE_BOX_MAX_HEIGHT_PX}px;
    overflow-y: auto;
}}

.value-box pre {{ margin: 0; font-size: 12px; }}

/* MISSING has to read on both the green and the pink tint, in both themes. Rather
   than hunt for one colour that satisfies all four, it uses the theme's own text
   colour and is marked out by weight and a border instead. */
.missing {{
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    padding: 1px 6px;
    border: 1px solid currentColor;
    border-radius: 3px;
    opacity: 0.75;
}}

.sub-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}}

.sub-table th,
.sub-table td {{
    border: 1px solid var(--border-color-primary);
    padding: 3px 6px;
    color: var(--body-text-color);
}}

.sub-table th {{ font-weight: 600; opacity: 0.8; }}

/* --- layout -------------------------------------------------------------- */

/* The three checkboxes that decide what actually runs used to sit unlabelled between
   the preview and five sections of configuration. */
#engine-select {{
    border: 1px solid var(--border-color-primary);
    border-left: 3px solid var(--color-accent);
    border-radius: 6px;
    padding: 10px 14px 4px 14px;
    margin: 6px 0 10px 0;
}}

/* --- comparison results table -------------------------------------------- */

/* Hand-rolled markup rather than gr.Dataframe, because a cost cell needs a
   `title` attribute to explain the figure it shows and a DataFrame cell cannot
   carry one. Widths are left to the browser: eight columns of short numbers fit
   without fixed percentages, and the previous fixed widths pushed Accuracy - the
   column the table exists for - off the right edge. */
table.results-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 12px;
}}

table.results-table th,
table.results-table td {{
    border: 1px solid var(--border-color-primary);
    padding: 5px 8px;
    color: var(--body-text-color);
}}

table.results-table th {{
    font-weight: 600;
    text-align: left;
    opacity: 0.85;
    /* Every header carries a tooltip, so every header gets the affordance. */
    cursor: help;
}}

/* Figures compare far more easily right-aligned than centred. */
table.results-table td.numeric {{ text-align: right; }}
table.results-table td.engine-name {{ font-weight: 600; }}

/* The one thing a static table cannot advertise is that hovering does something,
   so the cells carrying a cost formula are marked as interactive. */
table.results-table td.has-formula {{
    cursor: help;
    text-decoration: underline dotted;
    text-decoration-color: var(--color-accent);
    text-underline-offset: 3px;
}}

.results-table-note {{
    margin: 6px 0 0 0;
    font-size: 11px;
    opacity: 0.7;
    color: var(--body-text-color);
}}

.results-table-empty {{
    padding: 12px;
    font-size: 12px;
    opacity: 0.7;
    color: var(--body-text-color);
}}
"""


def banner(*, tone: str, text: str) -> str:
    """
    Build a status banner

    Args:
        tone (str): One of STATUS_TONES - "info", "ok", "warn" or "err".
        text (str): Banner text. Trusted as HTML so callers can emphasise a figure;
                    every current caller composes it from its own strings.

    Returns:
        str: An HTML div.

    Raises:
        ValueError: If the tone is not known. An unrecognised tone would otherwise
                    render as an unstyled div, which looks like a layout glitch rather
                    than like the typo it is.
    """
    if tone not in STATUS_TONES:
        raise ValueError(
            f"Unknown banner tone {tone!r}; expected one of {sorted(STATUS_TONES)}"
        )

    return (
        f"<div class='ocr-banner ocr-banner--{tone}'>"
        f"<span class='ocr-banner__tone'>{tone}</span>"
        f"<span class='ocr-banner__text'>{text}</span>"
        f"</div>"
    )


def note(*, text: str, tall: bool = False) -> str:
    """
    Build a muted secondary note, used for placeholders and empty states

    Args:
        text (str): Note text.
        tall (bool): Pad generously, for a note standing in for absent content such
                     as a preview that has not been loaded yet.

    Returns:
        str: An HTML div.
    """
    modifier = " ocr-note--tall" if tall else ""
    return f"<div class='ocr-note{modifier}'>{text}</div>"


def page_readout(*, current_page: int, total_pages: int) -> str:
    """
    Build the "Page N of M" readout shown above a multi-page PDF preview

    Args:
        current_page (int): Zero-indexed page currently displayed.
        total_pages (int): Total page count.

    Returns:
        str: An HTML div.
    """
    return (
        f"<div class='ocr-pageinfo'>Page {current_page + 1} of {total_pages}</div>"
    )


# The app's Gradio theme.
#
# IBM Plex is drawn for technical interfaces - unambiguous digits, a slashed zero -
# and almost every string in this UI is a figure, a filename, a JSON key or a block of
# OCR output. The mono face carries all of those. GoogleFont fetches from
# fonts.googleapis.com; the fallbacks in each list cover an offline machine.
#
# Amber is already Gradio's active-tab accent, so taking it as the primary hue means
# the accent is the one the framework was going to use regardless. Slate gives the dark
# theme a cooler background for document imagery than the default grey.
CUSTOM_THEME = gr.themes.Default(
    primary_hue="amber",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
)
