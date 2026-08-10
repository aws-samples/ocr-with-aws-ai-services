"""
Persist each run's measurements so runs can be compared after the fact.

Until now a run existed only on screen: the Comparison Results table was rebuilt
on every click and the previous run's figures were gone. Comparing two engines, or
the same engine before and after a configuration change, meant reading numbers off
a screenshot.

Every run now writes two things under `results/`:

- `<timestamp>-<document>.json` - the full record, including each engine's cost
  breakdown and the configuration that produced it. This is the file to read when
  asking "why did that cost that much?".
- `history.jsonl` - one flat line per engine per run, appended. This is the file to
  read when asking "how has BDA's accuracy moved?", because a line-per-observation
  file needs no traversal to filter:

      jq -r 'select(.engine == "BDA") | [.recorded_at, .pages, .accuracy_pct] | @tsv' \\
        results/history.jsonl

The figures come from shared.results_table.RunRow, the same objects the on-screen
table is rendered from, so the record and the screen cannot disagree about what a
run cost.

`recorded_at` is a parameter rather than being read from the clock here, so a test
can assert an exact filename and an exact record.
"""

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from shared.config import logger
from shared.results_table import RunRow

# Where records are written. Relative to the working directory, like the rest of the
# app's paths (sample/, results/run_*), and already covered by .gitignore.
DEFAULT_RESULTS_DIR = Path("results")

# Appended to, never rewritten: the history of every run is the point of the file.
HISTORY_FILENAME = "history.jsonl"

# Used in the record filename. Anything outside this set becomes a hyphen, which
# both keeps the name portable and stops a document name such as
# "claims/PFL/form - 4410772" writing outside the results directory.
_UNSAFE_IN_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class RunRecord:
    """
    Where a run was written, so the caller can tell the user.

    Attributes:
        record_path (Path): The run's own JSON file.
        history_path (Path): The append-only history the run was added to.
    """

    record_path: Path
    history_path: Path


def _filename_slug(*, document_name: str) -> str:
    """
    Reduce a document name to something safe to put in a filename.

    Args:
        document_name: Document name as shown in the UI, which for a sample PDF is
            a path such as "claims/PFL/form - 4410772".

    Returns:
        str: The document's base name with unsafe runs collapsed to hyphens.

    Raises:
        ValueError: If nothing usable is left. A record whose name says nothing
            about the document it describes is not worth writing.
    """
    # Grouping labels remain in the record body, but never become directories in
    # the record filename.
    base_name = os.path.splitext(os.path.basename(document_name))[0]
    slug = _UNSAFE_IN_FILENAME.sub("-", base_name).strip("-")

    if not slug:
        raise ValueError(
            f"Cannot build a record filename from document name {document_name!r}")

    return slug


def _describe_row(*, row: RunRow) -> Dict[str, Any]:
    """
    Render one engine's row as the record's JSON object.

    Per-page figures are null rather than 0 when the page count is unknown, so a
    reader cannot mistake "we could not tell" for "it was free".

    Args:
        row: The engine's row, as built for the on-screen table.

    Returns:
        Dict[str, Any]: The engine's metrics and its cost breakdown.
    """
    return {
        "engine": row.engine,
        "documents": row.documents,
        "pages": row.pages,
        "total_time_s": row.total_time_s,
        "time_per_page_s": row.time_per_page_s,
        "total_cost_usd": row.total_cost_usd,
        "cost_per_page_usd": row.cost_per_page_usd,
        "accuracy_pct": row.accuracy_pct,
        "cost_breakdown": [
            {
                "label": component.label,
                "formula": component.formula,
                "amount_usd": component.amount,
                "source": component.source,
            }
            for component in row.cost_breakdown
        ],
    }


def record_run(
    *,
    document_name: str,
    rows: Sequence[RunRow],
    recorded_at: datetime,
    total_time_s: float,
    configuration: Optional[Dict[str, Any]] = None,
    ground_truth_available: bool = False,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> RunRecord:
    """
    Write a run's measurements to `results/`, both as a record and as history.

    Args:
        document_name: The document that was processed, as shown in the UI.
        rows: One row per engine that ran, as built for the comparison table.
        recorded_at: When the run finished. Passed in rather than read from the
            clock so the filename and the record are reproducible in a test.
        total_time_s: Wall-clock seconds for the whole run, which is less than the
            sum of the engines' times because they run in parallel.
        configuration: The settings that produced the run - engines, model,
            document type, Textract features. Recorded because two runs over the
            same document are only comparable if the configuration is known.
        ground_truth_available: Whether accuracy could be scored at all. Without
            this, an accuracy of 0.0 is ambiguous.
        results_dir: Directory to write into. Created if it does not exist.

    Returns:
        RunRecord: The paths written.

    Raises:
        ValueError: If no engine rows were given, or the document name yields no
            usable filename. Both mean the caller has nothing to record, and
            writing an empty record would hide that.
        OSError: If either file cannot be written. Raised rather than swallowed so
            the caller can tell the user their run was not saved.
    """
    if not rows:
        raise ValueError(
            "Refusing to record a run with no engine results: there would be "
            "nothing in the record to compare against a later run")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Sorts chronologically as text, so `ls results/` is in run order.
    timestamp = recorded_at.strftime("%Y%m%d-%H%M%S")
    record_path = results_dir / f"{timestamp}-{_filename_slug(document_name=document_name)}.json"
    history_path = results_dir / HISTORY_FILENAME

    recorded_at_iso = recorded_at.isoformat(timespec="seconds")
    engine_records = [_describe_row(row=row) for row in rows]

    record: Dict[str, Any] = {
        "recorded_at": recorded_at_iso,
        "document": document_name,
        "total_time_s": total_time_s,
        # The run's total, which is what the status banner reports.
        "total_cost_usd": sum(row.total_cost_usd for row in rows),
        "ground_truth_available": ground_truth_available,
        "configuration": configuration or {},
        "engines": engine_records,
    }

    record_path.write_text(json.dumps(record, indent=2, default=str))

    # One line per engine per run: a flat observation is what makes the file
    # usable from jq without traversing into a nested structure.
    history_lines: List[str] = []
    for engine_record in engine_records:
        history_lines.append(json.dumps({
            "recorded_at": recorded_at_iso,
            "document": document_name,
            **{key: value for key, value in engine_record.items()
               if key != "cost_breakdown"},
            "record": record_path.name,
        }, default=str))

    with history_path.open("a") as history_file:
        history_file.write("\n".join(history_lines) + "\n")

    logger.info(
        f"Recorded run of {document_name} to {record_path} "
        f"and appended {len(history_lines)} line(s) to {history_path}")

    return RunRecord(record_path=record_path, history_path=history_path)


def save_run_record(
    *,
    document_name: str,
    rows: Sequence[RunRow],
    total_time_s: float,
    configuration: Dict[str, Any],
    ground_truth_available: bool,
    recorded_at: Optional[datetime] = None,
) -> str:
    """
    Persist the run and describe the outcome for the status banner.

    A run that is not saved is not silently lost: the failure is logged at ERROR
    and said plainly in the banner, because the figures on screen are gone the
    moment the next run starts and the user is the only one who can act on it.

    Lives here rather than in processor.py so both the single-document path and the
    batch path in sample_handler.py can record through it without one importing the
    other.

    Args:
        document_name: The document processed, as shown in the UI.
        rows: One row per engine that ran.
        total_time_s: Wall-clock seconds for the whole run.
        configuration: The settings that produced the run.
        ground_truth_available: Whether accuracy could be scored.
        recorded_at: When the run finished. Defaults to now; injected by tests so a
            record's filename is predictable.

    Returns:
        str: A fragment for the completion banner naming the file written, or
            saying why nothing was.
    """
    try:
        run_record = record_run(
            document_name=document_name,
            rows=rows,
            recorded_at=recorded_at or datetime.now(),
            total_time_s=total_time_s,
            configuration=configuration,
            ground_truth_available=ground_truth_available)
    except (OSError, ValueError) as record_error:
        logger.error(f"Could not save this run's results: {record_error}")
        return f" · <b>not saved</b>: {record_error}"

    return f" · saved to <code>{run_record.record_path}</code>"
