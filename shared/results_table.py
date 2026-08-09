"""
Build the Comparison Results table: one row per engine, with cost provenance.

The table previously reported "Samples Processed: 1" for a seven-page PDF, which
read as "one page was processed" when it meant "one document". It also showed a
single dollar figure per engine with no indication of where that figure came
from - and the three engines are priced three different ways, two of them by two
services at once.

This module is the one place those numbers are computed. `build_run_rows` derives
every metric once, and the two renderers only format: so the DataFrame a caller
persists and the HTML the browser shows cannot disagree about what the run cost.

The table is rendered as HTML rather than through `gr.Dataframe` because a hover
tooltip needs a `title` attribute on the cell, which the DataFrame component
cannot carry.
"""

from dataclasses import dataclass, field
from html import escape
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from shared.cost_calculator import CostComponent

# Column order, and the tooltip explaining what each column means. These are the
# headers the persisted DataFrame uses too, so renaming one changes both.
#
# "Documents" replaces the old "Samples Processed": it counted documents, and
# calling it samples next to no page count at all is what caused a seven-page run
# to be read as a one-page run.
COLUMN_TOOLTIPS: Dict[str, str] = {
    "Engine": "The OCR engine that produced this row.",
    "Documents": (
        "Documents processed, not pages. One PDF is one document however many "
        "pages it has."
    ),
    "Pages": (
        "Pages the engine actually read, as reported by the service itself. "
        "Every engine reads the whole document."
    ),
    "Total Time (s)": (
        "Wall-clock time for the whole document, including upload, polling and "
        "any JSON structuring step."
    ),
    "Time / Page (s)": "Total time divided by pages read.",
    "Total Cost ($)": (
        "Estimated charge for the whole document. Hover for the formula and the "
        "published rates it uses."
    ),
    "Cost / Page ($)": (
        "Total cost divided by pages read. Hover for the formula behind the total."
    ),
    "Accuracy (%)": (
        "Share of ground-truth fields matched across the whole document. This is "
        "not divided by pages: it is already a proportion, and a fraction of a "
        "fraction would mean nothing."
    ),
}

COLUMNS: List[str] = list(COLUMN_TOOLTIPS)

# Columns whose cells carry a per-engine cost tooltip rather than the generic
# column one, and which therefore get the hover affordance in the CSS.
COST_COLUMNS = ("Total Cost ($)", "Cost / Page ($)")

# Shown where a per-page figure cannot be derived. An engine that failed reports
# 0 pages, and dividing by it would either raise or invent a number.
UNKNOWN = "—"


@dataclass(frozen=True)
class RunRow:
    """
    One engine's measurements from a single run.

    Attributes:
        engine (str): Engine name, as shown in the first column.
        documents (int): Documents processed by this engine.
        pages (int): Pages read, or 0 when the engine failed or could not tell.
        total_time_s (float): Wall-clock seconds for the whole document.
        total_cost_usd (float): Estimated charge for the whole document.
        accuracy_pct (float): Ground-truth match rate over the whole document.
        cost_breakdown (List[CostComponent]): The charges making up the total.
    """

    engine: str
    documents: int
    pages: int
    total_time_s: float
    total_cost_usd: float
    accuracy_pct: float
    cost_breakdown: List[CostComponent] = field(default_factory=list)

    @property
    def time_per_page_s(self) -> Optional[float]:
        """Seconds per page, or None when the page count is unknown.

        Returns:
            Optional[float]: The per-page time, or None if pages is not positive.
        """
        return self.total_time_s / self.pages if self.pages > 0 else None

    @property
    def cost_per_page_usd(self) -> Optional[float]:
        """Cost per page, or None when the page count is unknown.

        Returns:
            Optional[float]: The per-page cost, or None if pages is not positive.
        """
        return self.total_cost_usd / self.pages if self.pages > 0 else None

    def cost_tooltip(self) -> str:
        """
        Render the arithmetic behind this row's cost as plain text.

        Written for a `title` attribute, so it is newline-separated plain text
        rather than HTML. Every figure quoted is this run's own, not a generic
        description of the pricing model, so a number that looks wrong can be
        checked line by line against the cited price.

        Returns:
            str: The tooltip text, or a note that nothing was billed.
        """
        if not self.cost_breakdown:
            return (
                "No charge was recorded for this run. An engine that failed is "
                "not billed an estimate."
            )

        lines = [f"How ${self.total_cost_usd:.6f} was calculated:", ""]
        for component in self.cost_breakdown:
            lines.append(f"{component.label}")
            lines.append(f"    {component.formula}")
            lines.append(f"    = ${component.amount:.6f}")
        lines.append("")
        lines.append(f"Total: ${self.total_cost_usd:.6f}")
        if self.pages > 0:
            lines.append(
                f"Per page: ${self.total_cost_usd:.6f} / {self.pages} pages "
                f"= ${self.total_cost_usd / self.pages:.6f}")

        # One line per distinct source, in first-seen order, so a two-service
        # charge cites both pricing pages without repeating either.
        sources: List[str] = []
        for component in self.cost_breakdown:
            if component.source not in sources:
                sources.append(component.source)
        lines.append("")
        lines.append("Rates published at: " + ", ".join(sources))

        return "\n".join(lines)


def build_run_rows(
    *, engine_results: Dict[str, Dict[str, Any]], document_count: int = 1
) -> List[RunRow]:
    """
    Turn the per-engine result dicts into rows, deriving every metric once.

    Args:
        engine_results: Processed results keyed by engine name, as
            `process_engine_result` returns them.
        document_count: Documents this run covered. One for a single file; the
            batch path passes its own count.

    Returns:
        List[RunRow]: One row per engine, in the order the results were given.
    """
    return [
        RunRow(
            engine=engine_name,
            documents=document_count,
            pages=data.get("pages", 0),
            total_time_s=data.get("time", 0.0),
            total_cost_usd=data.get("cost", 0.0),
            accuracy_pct=data.get("accuracy", 0.0),
            cost_breakdown=list(data.get("cost_breakdown", [])),
        )
        for engine_name, data in engine_results.items()
    ]


def _format_cells(*, row: RunRow) -> Dict[str, str]:
    """
    Format one row's values for display, column by column.

    Args:
        row: The row to format.

    Returns:
        Dict[str, str]: Display strings keyed by column name, covering COLUMNS.
    """
    time_per_page = row.time_per_page_s
    cost_per_page = row.cost_per_page_usd

    return {
        "Engine": row.engine,
        "Documents": str(row.documents),
        # 0 pages means the engine failed or could not tell, not that it read none.
        "Pages": str(row.pages) if row.pages > 0 else UNKNOWN,
        "Total Time (s)": f"{row.total_time_s:.3f}",
        "Time / Page (s)": f"{time_per_page:.3f}" if time_per_page is not None else UNKNOWN,
        "Total Cost ($)": f"{row.total_cost_usd:.6f}",
        "Cost / Page ($)": f"{cost_per_page:.6f}" if cost_per_page is not None else UNKNOWN,
        "Accuracy (%)": f"{row.accuracy_pct:.2f}",
    }


def rows_to_dataframe(*, rows: Sequence[RunRow]) -> pd.DataFrame:
    """
    Render the rows as a DataFrame, for persisting and for programmatic use.

    Args:
        rows: The rows to render. May be empty, which yields an empty frame that
            still has every column, so a caller can rely on the shape.

    Returns:
        pd.DataFrame: One record per row, with COLUMNS in order.
    """
    return pd.DataFrame(
        [_format_cells(row=row) for row in rows], columns=COLUMNS)


def rows_to_html(*, rows: Sequence[RunRow]) -> str:
    """
    Render the rows as an HTML table with tooltips on the headers and cost cells.

    Args:
        rows: The rows to render. Empty yields a placeholder rather than a bare
            header, so the panel does not look broken before the first run.

    Returns:
        str: The table markup, safe to hand to a Gradio HTML component.
    """
    if not rows:
        return (
            '<div class="results-table-empty" id="results-table">'
            'Run a document through one or more engines to compare them.'
            '</div>'
        )

    header_cells = "".join(
        f'<th title="{escape(COLUMN_TOOLTIPS[column])}">{escape(column)}</th>'
        for column in COLUMNS
    )

    body_rows = []
    for row in rows:
        cells = _format_cells(row=row)
        tooltip = row.cost_tooltip()

        rendered_cells = []
        for column in COLUMNS:
            # Cost cells explain this engine's own arithmetic; every other cell
            # falls back to the column's general meaning.
            is_cost = column in COST_COLUMNS
            title = tooltip if is_cost else COLUMN_TOOLTIPS[column]
            # Everything but the engine name is a number, and numbers compare far
            # more easily right-aligned in a monospace column.
            classes = ["numeric"] if column != "Engine" else ["engine-name"]
            if is_cost:
                classes.append("has-formula")
            rendered_cells.append(
                f'<td class="{" ".join(classes)}" title="{escape(title)}">'
                f'{escape(cells[column])}</td>'
            )
        body_rows.append(f"<tr>{''.join(rendered_cells)}</tr>")

    return (
        f'<table class="results-table" id="results-table">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        f"</table>"
        f'<p class="results-table-note">'
        f"Hover a cost for the formula and the published rates behind it. "
        f"Costs are estimates from the rates in <code>shared/config.py</code>, "
        f"not billed amounts."
        f"</p>"
    )
