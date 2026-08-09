"""
Combining per-chunk LLM structuring results into one schema-shaped object

Textract's asynchronous PDF path produces text page by page. Structuring each page
into its own object and returning them under a "pages" wrapper produces JSON whose
top level is page numbers, while ground truth and the output schema are keyed by
document field. Every field then reads as missing, so a perfect extraction scores
0%.

These helpers group page text into as few LLM calls as will fit and merge the
results back into a single object shaped like the schema.
"""

import json
from typing import Any, Dict, List, Tuple

from shared.config import logger

# Values that count as "nothing extracted" and may be overwritten by a later
# chunk. Mirrors the set shared.evaluator skips when walking ground truth, so a
# value the evaluator ignores is not treated here as a real conflict.
EMPTY_VALUES = (None, "", "null", "None")


def is_empty_value(value: Any) -> bool:
    """
    Report whether a value carries no extracted information

    Args:
        value: Any JSON-compatible value

    Returns:
        True if the value is null-like or an empty container
    """
    if isinstance(value, (dict, list)):
        return len(value) == 0
    # Compared by identity-or-equality against a tuple rather than with `in` on a
    # set, because False == 0 and both are legitimate extracted values.
    if isinstance(value, bool):
        return False
    return any(value is empty or value == empty for empty in EMPTY_VALUES)


def group_pages_for_structuring(
    page_texts: Dict[int, str], char_budget: int
) -> List[Tuple[List[int], str]]:
    """
    Group consecutive page texts into chunks that each fit a character budget

    Pages are kept in order and never split, so a field spanning a page boundary
    stays inside one chunk whenever the budget allows. The common case for a claim
    form is a single chunk containing the whole document, which needs no merging
    and lets the model see every page at once.

    Args:
        page_texts: Mapping of page number to that page's extracted text
        char_budget: Maximum characters of document text per chunk; must be positive

    Returns:
        List of (page numbers, combined text) tuples in page order

    Raises:
        ValueError: If char_budget is not positive
    """
    if char_budget <= 0:
        raise ValueError(f"char_budget must be positive, got {char_budget}")

    chunks: List[Tuple[List[int], str]] = []
    current_pages: List[int] = []
    current_parts: List[str] = []
    current_length = 0

    for page_num in sorted(page_texts):
        text = page_texts[page_num]
        if not text.strip():
            continue

        # Start a new chunk when adding this page would overflow, unless the
        # current chunk is empty - a single page over budget has to go somewhere,
        # and truncating it would silently discard extracted text.
        if current_pages and current_length + len(text) > char_budget:
            chunks.append((current_pages, "\n".join(current_parts)))
            current_pages, current_parts, current_length = [], [], 0

        if not current_pages and len(text) > char_budget:
            logger.warning(
                f"Page {page_num} is {len(text)} characters, over the "
                f"{char_budget}-character chunk budget. Sending it whole rather than "
                f"truncating; the model may reject it as too long."
            )

        current_pages.append(page_num)
        current_parts.append(f"--- Page {page_num} ---\n{text}")
        current_length += len(text)

    if current_pages:
        chunks.append((current_pages, "\n".join(current_parts)))

    return chunks


def merge_structured_results(
    results: List[Dict[str, Any]], labels: List[str]
) -> Dict[str, Any]:
    """
    Deep-merge structured results from several chunks into one object

    Merging is needed because each chunk is structured against the full schema, so
    every result is a mostly-empty copy of the whole schema holding only the fields
    its own pages contained. Stripping a wrapper would not be enough: a section
    split across a chunk boundary has to have its fields united.

    Args:
        results: Structured objects, one per chunk, in chunk order
        labels: Human-readable description of each chunk, used in conflict
                warnings; must be the same length as results

    Returns:
        A single merged object shaped like the output schema

    Raises:
        ValueError: If results and labels differ in length, or if any result is not
                    a JSON object. Merging a list or scalar into a schema-shaped
                    object is undefined, and guessing would corrupt accuracy
                    numbers rather than reporting a problem.
    """
    if len(results) != len(labels):
        raise ValueError(
            f"results and labels must be the same length, got {len(results)} and {len(labels)}"
        )

    merged: Dict[str, Any] = {}
    for result, label in zip(results, labels):
        if not isinstance(result, dict):
            raise ValueError(
                f"Structured result for {label} is a {type(result).__name__}, not a "
                f"JSON object, so it cannot be merged into the schema shape"
            )
        _merge_into(target=merged, source=result, path="", label=label)

    return merged


def _merge_into(target: Dict[str, Any], source: Dict[str, Any], path: str, label: str) -> None:
    """
    Merge one source object into a target dict in place

    Args:
        target: Accumulating merged object, modified in place
        source: Object to merge in
        path: Dotted path to the current position, for conflict warnings
        label: Description of the chunk the source came from
    """
    for key, incoming in source.items():
        current_path = f"{path}.{key}" if path else key

        if key not in target or is_empty_value(target[key]):
            target[key] = incoming
            continue

        if is_empty_value(incoming):
            continue

        existing = target[key]

        if isinstance(existing, dict) and isinstance(incoming, dict):
            _merge_into(target=existing, source=incoming, path=current_path, label=label)
        elif isinstance(existing, list) and isinstance(incoming, list):
            target[key] = _merge_lists(existing=existing, incoming=incoming)
        elif existing != incoming:
            # Two chunks disagree on a scalar. Keeping the first is arbitrary, so
            # say so rather than quietly picking a winner - on a benchmark the
            # discarded value may be the correct one.
            logger.warning(
                f"Conflicting values for '{current_path}' while merging {label}: "
                f"keeping {existing!r}, discarding {incoming!r}"
            )


def _merge_lists(existing: List[Any], incoming: List[Any]) -> List[Any]:
    """
    Concatenate two lists, skipping items already present

    Table rows continuing across a chunk boundary should accumulate, but a repeated
    header row or a field the model restated on every page should not be
    duplicated. Items are compared by their JSON encoding so that dicts and lists
    compare structurally without needing to be hashable.

    Args:
        existing: Items already merged
        incoming: Items to append

    Returns:
        A new list containing existing items followed by the unseen incoming ones
    """
    def fingerprint(item: Any) -> str:
        """
        Build a stable structural key for an item

        Args:
            item: Any JSON-compatible value

        Returns:
            A canonical JSON encoding, or repr() for anything not serialisable
        """
        try:
            return json.dumps(item, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return repr(item)

    seen = {fingerprint(item) for item in existing}
    merged = list(existing)
    for item in incoming:
        key = fingerprint(item)
        if key not in seen:
            seen.add(key)
            merged.append(item)

    return merged
