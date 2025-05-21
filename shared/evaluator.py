# evaluator.py
import os
import json
from typing import Dict, Any, Union, List, Tuple
from shared.config import logger

def load_truth_data(image_name: str) -> Tuple[Dict[str, Any], bool]:
    """
    Load ground truth data for a given image name
    
    Args:
        image_name: Name of the image file
        
    Returns:
        Tuple of (truth data as dictionary, whether truth exists)
    """
    if not image_name:
        return {}, False
    
    # Extract base name without extension
    base_name = os.path.splitext(os.path.basename(image_name))[0]
    
    # Check for truth file in sample/truth directory
    truth_path = os.path.join("sample/truth", f"{base_name}.json")
    
    if os.path.exists(truth_path):
        try:
            with open(truth_path, "r") as f:
                truth_data = json.load(f)
                logger.info(f"Loaded truth data from {truth_path}")
                return truth_data, True
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding truth data from {truth_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading truth data from {truth_path}: {e}")
    else:
        logger.info(f"No truth data found for {base_name} at {truth_path}")
    
    return {}, False

def calculate_accuracy(extracted_json, truth_json):
    """
    Calculate accuracy between extracted and truth JSONs with improved matching
    
    Args:
        extracted_json: Extracted data from OCR
        truth_json: Ground truth data
        
    Returns:
        Dictionary with accuracy metrics and detailed field comparison results
    """
    detailed_results = calculate_enhanced_accuracy(extracted_json, truth_json)
    
    # Return the full detailed results instead of just the accuracy number
    return detailed_results["total_accuracy"]

def get_detailed_accuracy(extracted_json, truth_json):
    """
    Get detailed accuracy results with field-by-field comparison
    
    Args:
        extracted_json: Extracted data from OCR
        truth_json: Ground truth data
        
    Returns:
        Dictionary with accuracy metrics and detailed comparison
    """
    return calculate_enhanced_accuracy(extracted_json, truth_json)

def calculate_enhanced_accuracy(extracted_json, truth_json):
    """
    Calculate accuracy between extracted and truth JSONs with improved matching algorithm
    that works across document types and handles nested structures and arrays intelligently
    
    Args:
        extracted_json: Extracted data from OCR
        truth_json: Ground truth data
        
    Returns:
        Dictionary with accuracy metrics and detailed scoring
    """
    if not truth_json or not extracted_json:
        logger.info("Missing data for accuracy calculation")
        return {"total_accuracy": 0, "field_accuracy": 0, "field_details": []}
    
    # Track metrics
    result = {
        "total_accuracy": 0,
        "field_accuracy": 0,
        "matches": 0,
        "total": 0,
        "field_details": []  # Will contain detailed field-by-field comparison
    }
    
    # Process the entire JSON structure recursively
    compare_json_recursive(truth_json, extracted_json, "", result)
    
    # Calculate accuracy percentages
    if result["total"] > 0:
        result["field_accuracy"] = round(100 * result["matches"] / result["total"], 2)
        result["total_accuracy"] = result["field_accuracy"]
    
    return result

def compare_json_recursive(truth_obj, extracted_obj, path, result):
    """
    Recursively compare JSON objects based on the truth object structure
    and add detailed field-by-field results
    
    Args:
        truth_obj: Truth/expected JSON object
        extracted_obj: Extracted JSON object to compare against
        path: Current path in the JSON
        result: Result dictionary to update with matches and field details
    """
    # Handle different types
    if isinstance(truth_obj, dict):
        # Process dictionary fields
        for key, truth_val in truth_obj.items():
            current_path = f"{path}.{key}" if path else key
            
            # Skip null values in truth
            if truth_val in [None, "", "null", "None"]:
                continue
            
            # Check if field exists in extracted
            extracted_val = extracted_obj.get(key) if isinstance(extracted_obj, dict) else None
            
            # Process based on value type
            if isinstance(truth_val, dict):
                # Nested object case
                if isinstance(extracted_val, dict):
                    compare_json_recursive(truth_val, extracted_val, current_path, result)
                else:
                    # Missing or wrong type - add all nested fields as missing
                    add_missing_fields(truth_val, current_path, result)
                    
            elif isinstance(truth_val, list):
                # List case
                if isinstance(extracted_val, list):
                    compare_lists(truth_val, extracted_val, current_path, result)
                else:
                    # Missing or wrong type - add as missing
                    add_missing_fields(truth_val, current_path, result)
                    
            else:
                # Simple value case
                result["total"] += 1
                
                if extracted_val is not None:
                    if compare_values(truth_val, extracted_val):
                        result["matches"] += 1
                        result["field_details"].append({
                            "field": current_path,
                            "expected": truth_val,
                            "extracted": extracted_val,
                            "match": True
                        })
                        logger.debug(f"Match: {current_path} = {truth_val}")
                    else:
                        result["field_details"].append({
                            "field": current_path,
                            "expected": truth_val,
                            "extracted": extracted_val,
                            "match": False
                        })
                        logger.debug(f"Mismatch: {current_path}, Truth: {truth_val}, Extracted: {extracted_val}")
                else:
                    result["field_details"].append({
                        "field": current_path,
                        "expected": truth_val,
                        "extracted": None,
                        "match": False
                    })
                    logger.debug(f"Missing field: {current_path}, Truth: {truth_val}")
    
    # Handle list at top level
    elif isinstance(truth_obj, list):
        if isinstance(extracted_obj, list):
            compare_lists(truth_obj, extracted_obj, path, result)
        else:
            add_missing_fields(truth_obj, path, result)
    
    # Handle scalar at top level
    elif path:  # Only process if we have a path (not at the very top level)
        result["total"] += 1
        
        if extracted_obj is not None:
            if compare_values(truth_obj, extracted_obj):
                result["matches"] += 1
                result["field_details"].append({
                    "field": path,
                    "expected": truth_obj,
                    "extracted": extracted_obj,
                    "match": True
                })
            else:
                result["field_details"].append({
                    "field": path,
                    "expected": truth_obj,
                    "extracted": extracted_obj,
                    "match": False
                })
        else:
            result["field_details"].append({
                "field": path,
                "expected": truth_obj,
                "extracted": None,
                "match": False
            })

def compare_lists(truth_list, extracted_list, path, result):
    """
    Compare lists of values or objects and update result with field-by-field details
    
    Args:
        truth_list: List from truth data
        extracted_list: List from extracted data
        path: Current path in the JSON
        result: Result dictionary to update
    """
    # Empty list case
    if not truth_list:
        return
    
    # Special handling for lists of objects
    if all(isinstance(item, dict) for item in truth_list):
        # For each truth item, find best match in extracted list
        for i, truth_item in enumerate(truth_list):
            item_path = f"{path}[{i}]"
            
            best_match = None
            best_score = -1
            best_idx = -1
            
            for j, extracted_item in enumerate(extracted_list):
                if not isinstance(extracted_item, dict):
                    continue
                    
                # Calculate similarity score between items
                temp_result = {"matches": 0, "total": 0, "field_details": []}
                compare_json_recursive(truth_item, extracted_item, "", temp_result)
                
                score = temp_result["matches"] / max(1, temp_result["total"])
                
                if score > best_score:
                    best_score = score
                    best_match = extracted_item
                    best_idx = j
            
            # If we found a reasonable match
            if best_match and best_score > 0.3:  # Threshold for considering a match
                # Compare fields properly with the best match
                compare_json_recursive(truth_item, best_match, item_path, result)
            else:
                # No match, add all fields as missing
                add_missing_fields(truth_item, item_path, result)
    
    # Simple list of values
    else:
        for i, truth_item in enumerate(truth_list):
            if truth_item in [None, "", "null", "None"]:
                continue
                
            item_path = f"{path}[{i}]"
            result["total"] += 1
            
            # Check if item exists in extracted list
            match_found = False
            matched_value = None
            
            for ext_item in extracted_list:
                if compare_values(truth_item, ext_item):
                    match_found = True
                    matched_value = ext_item
                    break
            
            if match_found:
                result["matches"] += 1
                result["field_details"].append({
                    "field": item_path,
                    "expected": truth_item,
                    "extracted": matched_value,
                    "match": True
                })
            else:
                result["field_details"].append({
                    "field": item_path,
                    "expected": truth_item,
                    "extracted": None,
                    "match": False
                })

def add_missing_fields(obj, base_path, result):
    """
    Add all fields in a nested object/list as missing fields in the result
    
    Args:
        obj: Object whose fields should be marked as missing
        base_path: Base path for these fields
        result: Result dictionary to update
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if value in [None, "", "null", "None"]:
                continue
                
            current_path = f"{base_path}.{key}"
            
            if isinstance(value, (dict, list)):
                add_missing_fields(value, current_path, result)
            else:
                result["total"] += 1
                result["field_details"].append({
                    "field": current_path,
                    "expected": value,
                    "extracted": None,
                    "match": False
                })
                
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if item in [None, "", "null", "None"]:
                continue
                
            current_path = f"{base_path}[{i}]"
            
            if isinstance(item, (dict, list)):
                add_missing_fields(item, current_path, result)
            else:
                result["total"] += 1
                result["field_details"].append({
                    "field": current_path,
                    "expected": item,
                    "extracted": None,
                    "match": False
                })


def compare_values(truth_val, extracted_val):
    """
    Compare two scalar values with type handling
    
    Args:
        truth_val: Expected value
        extracted_val: Extracted value to compare
        
    Returns:
        Boolean indicating match status
    """
    # Handle None
    if truth_val is None:
        return extracted_val is None
    
    # Handle numeric types
    if isinstance(truth_val, (int, float)) and isinstance(extracted_val, (int, float)):
        return abs(float(truth_val) - float(extracted_val)) < 0.001
    
    # String comparison (case-insensitive)
    return str(truth_val).lower() == str(extracted_val).lower()

def count_fields(obj):
    """
    Count the number of scalar fields in a JSON structure
    
    Args:
        obj: JSON object (dict, list, or scalar)
        
    Returns:
        Number of scalar fields
    """
    count = 0
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            if value in [None, "", "null", "None"]:
                continue
                
            if isinstance(value, (dict, list)):
                count += count_fields(value)
            else:
                count += 1
                
    elif isinstance(obj, list):
        for item in obj:
            count += count_fields(item)
    else:
        count = 1
        
    return count
