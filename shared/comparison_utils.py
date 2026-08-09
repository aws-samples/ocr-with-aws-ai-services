import json
import html
from typing import Dict, Any, List, Optional, Tuple
from shared.evaluator import get_detailed_accuracy
from shared.ui_theme import note

# The engines, in the order their columns appear in the comparison table.
#
# Defined here rather than in ui.py because four places need to agree on them: the
# Compare tab's filter dropdown, the value processor.py writes into that dropdown,
# the handler that reads it, and the tests. They have drifted before - the guard
# payloads in processor.py once wrote an HTML string into this dropdown, which
# Gradio rejects as "not in the list of choices".
ENGINE_NAMES: Tuple[str, ...] = ("Textract", "Bedrock", "BDA")

# The default: ground truth against every engine that ran, side by side. Picking a
# single engine narrows the table rather than being the only way to see anything.
ALL_ENGINES_LABEL = "All engines"

ENGINE_FILTER_CHOICES: Tuple[str, ...] = (ALL_ENGINES_LABEL,) + ENGINE_NAMES

def format_complex_value(value):
    """Format a complex value (dict or list) as collapsible HTML structure"""
    if isinstance(value, list) and all(isinstance(item, dict) for item in value) and len(value) > 0:
        # Format as a table if it's a list of similar objects (likely table data)
        return format_as_table(value)
    else:
        # Format as pretty JSON
        formatted = json.dumps(value, indent=2)
        return f"<pre>{html.escape(formatted)}</pre>"

def format_as_table(data_list):
    """Format a list of objects as an HTML table showing ALL rows"""
    if not data_list:
        return "<div>(Empty array)</div>"
    
    # Get all possible keys from all objects
    all_keys = set()
    for item in data_list:
        if isinstance(item, dict):
            all_keys.update(item.keys())
    
    # Create header row
    result = "<table class='sub-table'>"
    result += "<tr>"
    for key in sorted(all_keys):
        result += f"<th>{html.escape(str(key))}</th>"
    result += "</tr>"
    
    # Add ALL data rows (no limit)
    for item in data_list:
        result += "<tr>"
        for key in sorted(all_keys):
            if key in item:
                cell_value = str(item[key])
                result += f"<td>{html.escape(cell_value)}</td>"
            else:
                result += "<td>-</td>"
        result += "</tr>"
    
    result += "</table>"
    return result


def compare_complex_structures(val1, val2):
    """Compare complex structures more intelligently than string equality"""
    # For lists of dicts (table-like data), compare by contents rather than order
    if isinstance(val1, list) and isinstance(val2, list) and all(isinstance(i, dict) for i in val1 + val2):
        # Count items with matching key-value pairs
        matches = 0
        for item1 in val1:
            for item2 in val2:
                if all(k in item2 and item2[k] == v for k, v in item1.items()):
                    matches += 1
                    break
        
        # If at least 80% match, consider it a match
        return matches >= len(val1) * 0.8
    
    # For other types, do a normalized comparison
    norm1 = json.dumps(val1, sort_keys=True)
    norm2 = json.dumps(val2, sort_keys=True)
    return norm1 == norm2

def create_diff_view(truth_data_or_result, extracted_data=None, *, column_label: str = "Extracted"):
    """
    Generate HTML highlighting differences between truth and extracted data

    One engine, so one verdict per row - which is why this renderer tints the whole
    <tr>. The multi-engine table cannot do that (two engines can disagree about the
    same field) and tints each engine's cell instead.

    Args:
        truth_data_or_result: Either the truth data or the complete evaluation result
        extracted_data: Extracted JSON data (optional if first arg is evaluation result)
        column_label (str): Heading for the extracted-value column. The Compare tab
            passes the engine's name so a filtered table says whose values it shows.

    Returns:
        HTML string with formatted comparison
    """
    # Check if the first argument is an evaluation result or truth data
    if extracted_data is None and isinstance(truth_data_or_result, dict) and "field_details" in truth_data_or_result:
        # New format: using the evaluation result directly
        evaluation_result = truth_data_or_result
    else:
        # Old format: calculate the evaluation result from truth and extracted data
        evaluation_result = get_detailed_accuracy(extracted_data, truth_data_or_result)
    
    field_details = evaluation_result.get("field_details", [])

    html_output = "<div class='diff-container'>"
    html_output += _summary_bar(evaluation_result=evaluation_result)
    html_output += "<table class='diff-table'>"
    html_output += (
        f"<tr><th>Field</th><th>Expected</th>"
        f"<th>{html.escape(column_label)}</th><th>Match</th></tr>"
    )
    
    # Group fields by their parent path for better organization
    grouped_fields = {}
    
    for field_info in field_details:
        field_path = field_info["field"]
        
        # Split path into components
        path_parts = field_path.split(".")
        
        # Get parent path and field name
        if len(path_parts) > 1:
            parent_path = ".".join(path_parts[:-1])
            field_name = path_parts[-1]
        else:
            parent_path = ""
            field_name = field_path
            
        # Add to grouped fields
        if parent_path not in grouped_fields:
            grouped_fields[parent_path] = []
            
        field_info["field_name"] = field_name
        grouped_fields[parent_path].append(field_info)
    
    # Sort parent paths for consistent display
    parent_paths = sorted(grouped_fields.keys())
    
    # Process each parent path
    for parent_path in parent_paths:
        fields = grouped_fields[parent_path]
        
        # Add parent path header if it's not root
        if parent_path:
            html_output += f"<tr><td colspan='4' class='parent-path'><b>{parent_path}</b></td></tr>"
        
        # Process fields in this group
        for field_info in fields:
            field_name = field_info["field_name"]
            expected = field_info["expected"]
            extracted = field_info["extracted"]
            is_match = field_info["match"]
            
            row_class = "match" if is_match else "mismatch"
            
            # Format values based on their types
            if isinstance(expected, (dict, list)):
                formatted_expected = format_complex_value(expected)
            else:
                formatted_expected = html.escape(str(expected))
                
            if isinstance(extracted, (dict, list)):
                formatted_extracted = format_complex_value(extracted)
            elif extracted is None:
                formatted_extracted = "<span class='missing'>MISSING</span>"
                row_class = "mismatch"
            else:
                formatted_extracted = html.escape(str(extracted))
            
            # Generate row. Both value cells wrap their content in .value-box: the
            # height cap has to live on an element inside the <td>, because max-height
            # is not honoured on a table cell.
            html_output += f"<tr class='{row_class}'>"
            html_output += f"<td>{html.escape(field_name)}</td>"
            html_output += f"<td><div class='value-box'>{formatted_expected}</div></td>"
            html_output += f"<td><div class='value-box'>{formatted_extracted}</div></td>"
            html_output += f"<td class='mark'>{'✓' if is_match else '✗'}</td></tr>"

    html_output += "</table></div>"

    return html_output


def _summary_bar(*, evaluation_result: Dict[str, Any]) -> str:
    """
    Build the headline figures shown above the field-by-field table

    The tab previously opened straight into a table hundreds of rows long with no
    total anywhere, so the one number a reader wants first was the one number absent.

    Args:
        evaluation_result (Dict[str, Any]): Result from get_detailed_accuracy, read for
                                            'total_accuracy', 'matches' and 'total'.

    Returns:
        str: An HTML div.
    """
    accuracy = evaluation_result.get("total_accuracy", 0)
    matches = evaluation_result.get("matches", 0)
    total = evaluation_result.get("total", 0)

    return _figures_bar(figures=[
        (f"{accuracy:.1f}%", "accuracy"),
        (f"{matches}/{total}", "fields matched"),
        (f"{max(total - matches, 0)}", "fields differing"),
    ])


def _figures_bar(*, figures: List[Tuple[str, str]]) -> str:
    """
    Render a row of figure/label pairs as the bar above a comparison table

    Args:
        figures (List[Tuple[str, str]]): (figure, label) pairs, already formatted for
            display. Labels are escaped; figures are not, so a caller may emphasise
            part of one.

    Returns:
        str: An HTML div.
    """
    cells = "".join(
        f"<span><span class='diff-summary__figure'>{value}</span> "
        f"<span class='diff-summary__label'>{html.escape(label)}</span></span>"
        for value, label in figures
    )
    return f"<div class='diff-summary'>{cells}</div>"


def create_comparison_view(*, truth_data: Optional[Dict[str, Any]],
                           engine_json_by_name: Dict[str, Any],
                           engine_filter: str = ALL_ENGINES_LABEL) -> str:
    """
    Build the body of the Compare tab

    The single entry point for both callers - the processing generator, which renders
    the table as results arrive, and the filter dropdown's change handler.

    Args:
        truth_data (Optional[Dict[str, Any]]): Ground truth for the current document,
            or None/empty when the document has none.
        engine_json_by_name (Dict[str, Any]): Each engine's extracted JSON, keyed by
            engine name. Entries whose value is falsy are treated as "did not run" and
            get no column, so selecting two engines does not produce three columns.
        engine_filter (str): One of ENGINE_FILTER_CHOICES. ALL_ENGINES_LABEL shows
            every engine that produced a result; an engine name narrows the table to
            that engine.

    Returns:
        str: HTML for the comparison view, or a note explaining why there is nothing
            to compare.

    Raises:
        ValueError: If engine_filter is not one of ENGINE_FILTER_CHOICES. A filter
            value that silently fell through to "all engines" would misreport whose
            numbers are on screen.
    """
    if engine_filter not in ENGINE_FILTER_CHOICES:
        raise ValueError(
            f"Unknown engine filter {engine_filter!r}; expected one of "
            f"{', '.join(ENGINE_FILTER_CHOICES)}"
        )

    if not truth_data:
        return note(
            text="No ground truth is available for this document, so there is nothing "
                 "to compare against",
            tall=True)

    # Ordered by ENGINE_NAMES rather than by the caller's dict, so the columns keep the
    # same left-to-right order whichever engine happens to finish first.
    available = {
        name: engine_json_by_name[name]
        for name in ENGINE_NAMES
        if engine_json_by_name.get(name)
    }

    if engine_filter != ALL_ENGINES_LABEL:
        if engine_filter not in available:
            return note(
                text=f"<b>{html.escape(engine_filter)}</b> has no result to compare "
                     f"yet — select it as an engine and process the document",
                tall=True)
        available = {engine_filter: available[engine_filter]}

    if not available:
        return note(
            text="No engine results available for comparison yet", tall=True)

    if len(available) == 1:
        engine_name, extracted = next(iter(available.items()))
        return create_diff_view(truth_data, extracted, column_label=engine_name)

    return create_multi_engine_diff_view(
        truth_data=truth_data, engine_json_by_name=available)


def create_multi_engine_diff_view(*, truth_data: Dict[str, Any],
                                  engine_json_by_name: Dict[str, Any]) -> str:
    """
    Render every engine's extraction against ground truth in one table

    One row per ground-truth field, one column per engine. The verdict is per cell
    rather than per row because engines disagree: Textract can match a field that BDA
    misses, so a row-level match/mismatch does not exist here.

    Args:
        truth_data (Dict[str, Any]): Ground truth for the document.
        engine_json_by_name (Dict[str, Any]): Each engine's extracted JSON, keyed by
            engine name. Column order follows this dict's order.

    Returns:
        str: HTML for the comparison table.
    """
    evaluations = {
        name: get_detailed_accuracy(extracted, truth_data)
        for name, extracted in engine_json_by_name.items()
    }
    engines = list(engine_json_by_name)

    details_by_engine = {
        name: {detail["field"]: detail
               for detail in evaluation.get("field_details", [])}
        for name, evaluation in evaluations.items()
    }

    field_paths = _ordered_field_paths(evaluations=evaluations, engines=engines)
    column_count = 2 + len(engines)

    html_output = "<div class='diff-container'>"
    html_output += _engine_summary_bar(evaluations=evaluations, engines=engines)
    html_output += "<table class='diff-table diff-table--multi'>"
    html_output += "<tr><th>Field</th><th>Expected</th>"
    html_output += "".join(f"<th>{html.escape(name)}</th>" for name in engines)
    html_output += "</tr>"

    for parent_path, paths in _group_paths_by_parent(field_paths=field_paths).items():
        if parent_path:
            html_output += (
                f"<tr><td colspan='{column_count}' class='parent-path'>"
                f"<b>{html.escape(parent_path)}</b></td></tr>"
            )

        for field_path in paths:
            field_name = field_path.split(".")[-1]
            expected = _expected_value(
                field_path=field_path, details_by_engine=details_by_engine,
                engines=engines)

            html_output += "<tr>"
            html_output += f"<td>{html.escape(field_name)}</td>"
            html_output += (
                f"<td><div class='value-box'>{_format_value(value=expected)}"
                f"</div></td>"
            )
            for name in engines:
                html_output += _engine_cell(
                    detail=details_by_engine[name].get(field_path))
            html_output += "</tr>"

    html_output += "</table></div>"

    return html_output


def _ordered_field_paths(*, evaluations: Dict[str, Dict[str, Any]],
                         engines: List[str]) -> List[str]:
    """
    Collect the field paths to render, in first-seen order across the engines

    get_detailed_accuracy() walks the ground truth, so the engines normally report the
    same paths - but not always: compare_lists() walks the extracted list, so an engine
    that returned fewer array items reports fewer paths. Taking the union rather than
    the first engine's list keeps those fields on screen; a field dropped from the
    table would read as agreement.

    Args:
        evaluations (Dict[str, Dict[str, Any]]): get_detailed_accuracy() result per
            engine name.
        engines (List[str]): Engine names, in column order.

    Returns:
        List[str]: Field paths, each appearing once.
    """
    ordered: List[str] = []
    seen = set()

    for name in engines:
        for detail in evaluations[name].get("field_details", []):
            field_path = detail["field"]
            if field_path not in seen:
                seen.add(field_path)
                ordered.append(field_path)

    return ordered


def _group_paths_by_parent(*, field_paths: List[str]) -> Dict[str, List[str]]:
    """
    Group field paths under their parent path, parents sorted

    Args:
        field_paths (List[str]): Dotted field paths.

    Returns:
        Dict[str, List[str]]: Parent path (empty string for root) to its field paths,
            iterated in sorted parent order to match the single-engine table.
    """
    grouped: Dict[str, List[str]] = {}

    for field_path in field_paths:
        path_parts = field_path.split(".")
        parent_path = ".".join(path_parts[:-1]) if len(path_parts) > 1 else ""
        grouped.setdefault(parent_path, []).append(field_path)

    return {parent: grouped[parent] for parent in sorted(grouped)}


def _expected_value(*, field_path: str,
                    details_by_engine: Dict[str, Dict[str, Dict[str, Any]]],
                    engines: List[str]) -> Any:
    """
    Read the expected value for a field from whichever engine reported it

    Every engine is scored against the same ground truth, so the expected value is the
    same wherever it comes from; this only has to find an engine that has the path.

    Args:
        field_path (str): Dotted path of the field.
        details_by_engine (Dict[str, Dict[str, Dict[str, Any]]]): Field detail keyed by
            path, per engine name.
        engines (List[str]): Engine names, in column order.

    Returns:
        Any: The ground-truth value, or None if no engine reported the path (which
            cannot happen for a path that came out of _ordered_field_paths).
    """
    for name in engines:
        detail = details_by_engine[name].get(field_path)
        if detail is not None:
            return detail["expected"]

    return None


def _format_value(*, value: Any) -> str:
    """
    Render one value for a table cell

    Args:
        value (Any): A scalar, dict or list from either side of the comparison.

    Returns:
        str: Escaped text for a scalar, or nested HTML for a dict or list.
    """
    if isinstance(value, (dict, list)):
        return format_complex_value(value)

    return html.escape(str(value))


def _engine_cell(*, detail: Optional[Dict[str, Any]]) -> str:
    """
    Render one engine's cell for one field

    Args:
        detail (Optional[Dict[str, Any]]): The engine's field detail - 'extracted' and
            'match' - or None when this engine reported nothing for the field.

    Returns:
        str: A <td> carrying the per-cell tint class and the verdict glyph.
    """
    if detail is None:
        # Not the same thing as MISSING: the engine's evaluation never covered this
        # field, which happens when it returned a shorter array than another engine.
        return ("<td class='cell-mismatch'><div class='value-box'>"
                "<span class='missing'>NOT REPORTED</span></div></td>")

    extracted = detail["extracted"]
    is_match = bool(detail["match"])

    if extracted is None:
        body = "<span class='missing'>MISSING</span>"
        is_match = False
    else:
        body = _format_value(value=extracted)

    cell_class = "cell-match" if is_match else "cell-mismatch"
    mark = "✓" if is_match else "✗"

    # The glyph repeats what the tint says, for a reader who cannot separate the two
    # tints - the same reason .missing is marked out by a border rather than a colour.
    return (f"<td class='{cell_class}'><div class='value-box'>"
            f"<span class='cell-mark'>{mark}</span>{body}</div></td>")


def _engine_summary_bar(*, evaluations: Dict[str, Dict[str, Any]],
                        engines: List[str]) -> str:
    """
    Build the per-engine headline figures above the multi-engine table

    Args:
        evaluations (Dict[str, Dict[str, Any]]): get_detailed_accuracy() result per
            engine name.
        engines (List[str]): Engine names, in column order.

    Returns:
        str: An HTML div, one figure per engine.
    """
    figures: List[Tuple[str, str]] = []

    for name in engines:
        evaluation = evaluations[name]
        accuracy = evaluation.get("total_accuracy", 0)
        matches = evaluation.get("matches", 0)
        total = evaluation.get("total", 0)
        figures.append((f"{accuracy:.1f}%", f"{name} · {matches}/{total} matched"))

    return _figures_bar(figures=figures)
