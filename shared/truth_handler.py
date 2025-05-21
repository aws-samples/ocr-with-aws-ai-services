# truth_handler.py

import os
import json
import logging
from typing import Dict, Any, Tuple, Optional, List

from shared.config import logger
from shared.evaluator import load_truth_data

def on_sample_selected_truth(sample_filename):
    """
    Handle sample selection and load truth data
    
    Args:
        sample_filename: Name of the selected sample file
        
    Returns:
        Tuple of (truth_data, truth_status_html)
    """
    # Load truth data if available
    truth_data, truth_exists = load_truth_data(sample_filename)
    
    # Create truth status HTML based on whether truth data exists
    if truth_exists:
        truth_status_html = f"""<div style='padding: 10px; background-color: #2e7d32; color: white; 
                                border-radius: 5px; font-weight: bold;'>Ground truth data available for {sample_filename}</div>"""
    else:
        truth_status_html = f"""<div style='padding: 10px; background-color: #ed6c02; color: white; 
                                border-radius: 5px; font-weight: bold;'>No ground truth data available for {sample_filename}</div>"""
    
    return truth_data, truth_status_html

def add_accuracy_column_to_results(results_by_engine):
    """
    Add accuracy column to results DataFrame
    
    Args:
        results_by_engine: Dictionary of results by engine
        
    Returns:
        List of dictionaries with result data including accuracy
    """
    final_results = []
    for engine, data in results_by_engine.items():
        result_row = {
            "Engine": engine,
            "Samples Processed": data["count"],
            "Avg. Processing Time (s)": f"{data["total_time"] / data["count"]:.3f}",
            "Avg. Cost ($)": round(data["total_cost"] / data["count"], 6),
            "Total Cost ($)": round(data["total_cost"], 6)
        }
        
        # Add accuracy if available
        if "accuracy" in data:
            result_row["Accuracy (%)"] = data["accuracy"]
        else:
            result_row["Accuracy (%)"] = 0.0
        
        final_results.append(result_row)
    
    return final_results