import concurrent.futures
import os
import json
import time
import re
import pandas as pd
import datetime
import numpy as np
from PIL import Image
from engines.textract_engine import TextractEngine
from engines.bedrock_engine import BedrockEngine
from engines.bda_engine import BDAEngine
from shared.config import logger, BEDROCK_MODELS
from shared.cost_calculator import calculate_full_textract_cost
from shared.evaluator import load_truth_data, calculate_accuracy, get_detailed_accuracy

def list_sample_images():
    """List all sample images from the sample/images directory"""
    samples = []
    sample_dir = "sample/images"
    
    if os.path.exists(sample_dir):
        for file in os.listdir(sample_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                samples.append(file)
    else:
        logger.warning(f"Sample directory not found: {sample_dir}")
    
    logger.info(f"Found {len(samples)} sample images")
    return samples

def load_sample_image_and_schema(sample_filename):
    """Load a sample image and its corresponding schema"""
    if not sample_filename:
        return None, None
    
    # Load the image
    image_path = os.path.join("sample/images", sample_filename)
    if not os.path.exists(image_path):
        logger.error(f"Sample image not found: {image_path}")
        return None, None
    
    image = Image.open(image_path)
    image.name = sample_filename
    logger.info(f"Loaded sample image: {image_path}")
    
    # Load the schema if available
    schema = None
    schema_path = os.path.join("sample/schema", os.path.splitext(sample_filename)[0] + ".json")
    
    if os.path.exists(schema_path):
        try:
            with open(schema_path, "r") as f:
                schema = f.read()
                # Validate JSON
                json.loads(schema)
                logger.info(f"Loaded schema: {schema_path}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON schema: {schema_path}")
        except Exception as e:
            logger.error(f"Error loading schema: {str(e)}")
    else:
        logger.info(f"No schema found for sample: {schema_path}")
    
    return image, schema

def on_sample_selected(sample_filename):
    """
    Handle sample selection and load the image and schema
    
    Args:
        sample_filename: Name of the selected sample file
        
    Returns:
        Tuple of (image, schema, truth_data, truth_status_html)
    """
    image, schema = load_sample_image_and_schema(sample_filename)

    truth_data, truth_exists = load_truth_data(sample_filename)
    
    if truth_exists:
        truth_status_html = f"""<div style='padding: 10px; background-color: #2e7d32; color: white; 
                                border-radius: 5px; font-weight: bold;'>Ground truth data available for {sample_filename}</div>"""
    else:
        truth_status_html = f"""<div style='padding: 10px; background-color: #ed6c02; color: white; 
                                border-radius: 5px; font-weight: bold;'>No ground truth data available for {sample_filename}</div>"""
    
    return image, schema, truth_data, truth_status_html


def process_all_samples(use_textract, use_bedrock, use_bda,
                     bedrock_model_name, bda_s3_bucket="",
                     document_type="generic", output_schema="",
                     use_bda_blueprint=False):
    """Process all sample images with parallel engine processing"""
    
    # Get list of all sample images
    samples = list_sample_images()
    if not samples:
        return "<div class='status-error'>No sample images found</div>", pd.DataFrame()
    
    # Initialize results tracking by engine
    results_by_engine = {
        "Textract": {"count": 0, "total_time": 0, "total_cost": 0, "accuracy_values": []},
        "Bedrock": {"count": 0, "total_time": 0, "total_cost": 0, "accuracy_values": []},
        "BDA": {"count": 0, "total_time": 0, "total_cost": 0, "accuracy_values": []}
    }
    
    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logger.info(f"Created results directory: {results_dir}")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f"run_{timestamp}")
    os.makedirs(run_dir)
    logger.info(f"Created directory for this run: {run_dir}")
    
    total_start = time.time()
    
    # Initialize engines
    textract_engine = TextractEngine()
    bedrock_engine = BedrockEngine()
    bda_engine = BDAEngine()
    
    # Get bedrock model ID if needed
    model_id = BEDROCK_MODELS.get(bedrock_model_name, "") if use_bedrock else ""
    
    # Process each sample
    for i, sample_name in enumerate(samples):
        status_html = f"<div class='status-processing'>Processing sample {i+1}/{len(samples)}: {sample_name}</div>"
        
        # Create current results dataframe for display
        current_results = create_current_results(results_by_engine)
        yield status_html, pd.DataFrame(current_results)
        
        try:
            # Setup sample directory and load image
            sample_base_name = os.path.splitext(sample_name)[0]
            sample_dir = os.path.join(run_dir, sample_base_name)
            os.makedirs(sample_dir, exist_ok=True)
            
            # Load the sample image
            sample_path = os.path.join("sample/images", sample_name)
            image = Image.open(sample_path)
            image.name = sample_name  # Set image name for proper truth data loading
            
            # Save the original image
            save_original_image(image, os.path.join(sample_dir, "original.jpg"))
            
            # Load truth data for accuracy calculation
            truth_data, truth_exists = load_truth_data(sample_name)
            
            # Load sample-specific schema (image-specific schema takes precedence over general schema)
            image_output_schema = load_sample_schema(sample_name, output_schema)

            # Process with selected engines in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {}
                
                # Submit tasks for each enabled engine
                if use_textract:
                    futures['Textract'] = executor.submit(
                        textract_engine.process_image,
                        image, {"output_schema": image_output_schema}
                    )
                
                if use_bedrock:
                    futures['Bedrock'] = executor.submit(
                        bedrock_engine.process_image,
                        image, {
                            'model_id': model_id,
                            'document_type': document_type,
                            'output_schema': image_output_schema if image_output_schema else None
                        }
                    )
                
                if use_bda:
                    futures['BDA'] = executor.submit(
                        bda_engine.process_image,
                        image, {
                            's3_bucket': bda_s3_bucket,
                            'document_type': document_type,
                            'output_schema': image_output_schema if image_output_schema else None,
                            'use_blueprint': use_bda_blueprint
                        }
                    )
                
                # Process results as they complete
                for engine_name, future in futures.items():
                    try:
                        # Get the direct engine result
                        result = future.result()
                        
                        if result:
                            # Use process_engine_result for consistent accuracy calculation
                            from processor import process_engine_result
                            processed_result = process_engine_result(engine_name, result, truth_data, truth_exists)
                            
                            # Extract fields from processed result
                            process_time = processed_result.get('time', 0)
                            extracted_text = processed_result.get('text', '')
                            json_data = processed_result.get('json', {})
                            image_data = processed_result.get('image')
                            accuracy = processed_result.get('accuracy', 0)
                            cost = processed_result.get('cost', 0)
                            
                            # Log debug information about structure comparison
                            if truth_exists and json_data:
                                log_structure_comparison(sample_name, engine_name, truth_data, json_data, accuracy)
                            
                            # Save results to disk
                            engine_dir = os.path.join(sample_dir, engine_name.lower())
                            os.makedirs(engine_dir, exist_ok=True)
                            
                            # Save extracted text
                            save_text_result(extracted_text, os.path.join(engine_dir, "text.txt"))
                            
                            # Save JSON result
                            save_json_result(json_data, engine_name, sample_name, os.path.join(engine_dir, "result.json"))
                            
                            # Save visualization image
                            save_visualization_image(image_data, os.path.join(engine_dir, "visualization.jpg"))
                            
                            # Save metadata
                            save_metadata(engine_name, result, process_time, cost, accuracy, 
                                         os.path.join(engine_dir, "metadata.json"))
                            
                            # Update engine results
                            results_by_engine[engine_name]["count"] += 1
                            results_by_engine[engine_name]["total_time"] += process_time
                            results_by_engine[engine_name]["total_cost"] += cost
                            results_by_engine[engine_name]["accuracy_values"].append(accuracy)
                            
                            # Update UI with current progress
                            current_results = create_current_results(results_by_engine)
                            intermediate_status = f"<div class='status-processing'>Processing sample {i+1}/{len(samples)}: {sample_name} - {engine_name} completed</div>"
                            yield intermediate_status, pd.DataFrame(current_results)
                            
                    except Exception as e:
                        handle_engine_error(engine_name, sample_name, e)
        
        except Exception as e:
            handle_sample_error(sample_name, e, run_dir)
    
    # Create summary at the end
    create_summary(results_by_engine, len(samples), total_start, run_dir, 
                  use_textract, use_bedrock, bedrock_model_name, use_bda, use_bda_blueprint)
    
    status_html = f"<div class='status-completed'>All {len(samples)} samples processed in {time.time() - total_start:.2f} seconds. Results saved in {run_dir}</div>"
    
    # Create final results dataframe
    final_results = create_current_results(results_by_engine)
    return status_html, pd.DataFrame(final_results)


def load_sample_schema(sample_name, default_schema=""):
    """Load sample-specific schema if available"""
    sample_schema = None
    schema_path = os.path.join("sample/schema", os.path.splitext(sample_name)[0] + ".json")
    
    if os.path.exists(schema_path):
        try:
            with open(schema_path, "r") as f:
                sample_schema = f.read()
                json.loads(sample_schema)  # Validate JSON
                logger.info(f"Loaded schema for batch processing: {schema_path}")
        except Exception as e:
            logger.error(f"Error loading schema: {str(e)}")
    
    return sample_schema if sample_schema else default_schema


def save_original_image(image, output_path):
    """Save the original image with proper format conversion"""
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  
        background.save(output_path, format='JPEG')
    else:
        rgb_image = image.convert('RGB')  
        rgb_image.save(output_path, format='JPEG')


def log_structure_comparison(sample_name, engine_name, truth_data, json_data, accuracy):
    """Log structure comparison between truth data and result JSON"""
    logger.info(f"======= DEBUG FOR {sample_name} / {engine_name} =======")
    logger.info(f"Truth data keys: {list(truth_data.keys())}")
    logger.info(f"JSON result keys: {list(json_data.keys())}")
    common_keys = set(truth_data.keys()).intersection(set(json_data.keys()))
    logger.info(f"Common keys: {common_keys}")
    logger.info(f"Final accuracy: {accuracy}%")
    logger.info(f"======= END DEBUG FOR {sample_name} / {engine_name} =======")


def save_text_result(text, output_path):
    """Save extracted text to file"""
    with open(output_path, "w") as f:
        f.write(text)


def save_json_result(json_data, engine_name, sample_name, output_path):
    """Save JSON result to file"""
    if json_data:
        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"Saved {engine_name} JSON result for {sample_name}")
    else:
        logger.warning(f"Empty JSON result for {engine_name} on {sample_name}")


def save_visualization_image(image_data, output_path):
    """Save visualization image if available"""
    if image_data is not None:
        if isinstance(image_data, Image.Image):
            image_to_save = image_data
        elif isinstance(image_data, np.ndarray):
            image_to_save = Image.fromarray(image_data)
        else:
            return
            
        if image_to_save:
            if image_to_save.mode == 'RGBA':
                background = Image.new('RGB', image_to_save.size, (255, 255, 255))
                background.paste(image_to_save, mask=image_to_save.split()[3])
                image_to_save = background
            image_to_save.save(output_path)


def save_metadata(engine_name, result, process_time, cost, accuracy, output_path):
    """Save metadata including engine-specific information"""
    metadata = {
        "process_time": process_time,
        "cost": cost,
        "accuracy": accuracy,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Add engine-specific metadata
    if engine_name == "Bedrock":
        metadata["token_usage"] = result.get("token_usage")
        metadata["model_id"] = result.get("model_id", "")
    elif engine_name == "BDA":
        metadata["use_blueprint"] = result.get("use_blueprint", False)
        metadata["field_count"] = result.get("field_count", 0)
        if result.get("token_usage"):
            metadata["token_usage"] = result.get("token_usage")
    
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)


def create_current_results(results_by_engine):
    """Create current results list for dataframe display"""
    current_results = []
    for engine, data in results_by_engine.items():
        if data["count"] > 0:
            # Calculate average accuracy
            avg_accuracy = 0
            if data["accuracy_values"]:
                avg_accuracy = sum(data["accuracy_values"]) / len(data["accuracy_values"])
            
            current_results.append({
                "Engine": engine,
                "Samples Processed": data["count"],
                "Avg. Processing Time (s)": round(data["total_time"] / data["count"], 3),
                "Avg. Cost ($)": round(data["total_cost"] / data["count"], 6),
                "Total Cost ($)": round(data["total_cost"], 6),
                "Accuracy (%)": round(avg_accuracy, 2)
            })
    return current_results


def handle_engine_error(engine_name, sample_name, error):
    """Handle errors during engine processing"""
    logger.error(f"Error getting result for {engine_name} on {sample_name}: {str(error)}")
    import traceback
    logger.error(f"Stack trace: {traceback.format_exc()}")


def handle_sample_error(sample_name, error, run_dir):
    """Handle errors during sample processing"""
    logger.error(f"Error processing sample {sample_name}: {str(error)}")
    sample_dir = os.path.join(run_dir, os.path.splitext(sample_name)[0])
    os.makedirs(sample_dir, exist_ok=True)
    error_file = os.path.join(sample_dir, "error.txt")
    with open(error_file, "w") as f:
        f.write(f"Error processing {sample_name}: {str(error)}")


def create_summary(results_by_engine, samples_count, start_time, run_dir, 
                  use_textract, use_bedrock, bedrock_model_name, use_bda, use_bda_blueprint):
    """Create and save summary of processing results"""
    total_time = time.time() - start_time
    
    summary = {
        "total_samples": samples_count,
        "total_time": total_time,
        "engines_used": {
            "textract": use_textract,
            "bedrock": use_bedrock,
            "bedrock_model": bedrock_model_name if use_bedrock else None,
            "bda": use_bda,
            "bda_blueprint": use_bda_blueprint if use_bda else None
        },
        "results": {}
    }
    
    # Add engine-specific results to summary
    for engine, data in results_by_engine.items():
        if data["count"] > 0:
            avg_accuracy = 0
            if data["accuracy_values"]:
                avg_accuracy = sum(data["accuracy_values"]) / len(data["accuracy_values"])
                
            summary["results"][engine] = {
                "samples_processed": data["count"],
                "total_time": data["total_time"],
                "avg_time": data["total_time"] / data["count"],
                "total_cost": data["total_cost"],
                "avg_cost": data["total_cost"] / data["count"],
                "avg_accuracy": avg_accuracy
            }
    
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
