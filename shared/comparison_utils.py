import json, html
from typing import Dict, Any, List, Tuple
from shared.config import logger
from shared.evaluator import get_detailed_accuracy

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

def create_diff_view(truth_data_or_result, extracted_data=None):
    """
    Generate HTML highlighting differences between truth and extracted data
    
    Args:
        truth_data_or_result: Either the truth data or the complete evaluation result
        extracted_data: Extracted JSON data (optional if first arg is evaluation result)
        
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
    
    html_output = "<div class='diff-container'><h3>Field-by-Field Comparison</h3><table class='diff-table'>"
    html_output += "<tr><th>Field</th><th>Expected</th><th>Extracted</th><th>Match</th></tr>"
    
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
            
            # Generate row
            html_output += f"<tr class='{row_class}'>"
            html_output += f"<td>{html.escape(field_name)}</td>"
            html_output += f"<td class='value-cell'>{formatted_expected}</td>"
            html_output += f"<td class='value-cell'>{formatted_extracted}</td>"
            html_output += f"<td>{'✓' if is_match else '✗'}</td></tr>"
    
    html_output += "</table></div>"
    
    html_output = f"""
    <style>
    .diff-container {{ font-family: Arial, sans-serif; margin: 10px; }}
    .diff-table {{ width: 100%; border-collapse: collapse; }}
    .diff-table th, .diff-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    .diff-table th {{ background-color: #f2f2f2; }}
    .match {{ background-color: #e6ffe6; }} /* Light green */
    .mismatch {{ background-color: #ffe6e6; }} /* Light red */
    .parent-path {{ background-color: #f0f0f0; font-weight: bold; }}
    .value-cell {{ font-family: monospace; white-space: pre-wrap; max-height: 200px; overflow-y: auto; }}
    .missing {{ color: red; font-style: italic; }}
    .sub-table {{ width: 100%; border-collapse: collapse; }}
    .sub-table th, .sub-table td {{ border: 1px solid #ccc; padding: 2px 4px; }}
    </style>
    {html_output}
    """
    
    return html_output
