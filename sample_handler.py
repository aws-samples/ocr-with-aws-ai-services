import concurrent.futures
import hashlib
import os
import json
import time
import datetime
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
from engines.textract_engine import TextractEngine
from engines.bedrock_engine import BedrockEngine
from engines.bda_engine import BDAEngine
from preview_handler import convert_pdf_to_image, get_pdf_page_count
from shared.config import (
    logger,
    BEDROCK_MODELS,
    DEFAULT_BDA_S3_BUCKET,
    DEFAULT_S3_BUCKET,
)
from shared.cost_calculator import merge_cost_components
from shared.evaluator import load_truth_data
from shared.results_table import RunRow, rows_to_html
from shared.run_recorder import save_run_record
import shared.sample_paths
from shared.sample_paths import SAMPLE_DIR, list_sample_bundles
from shared.sample_paths import bundle_dir
from shared.sample_paths import is_pdf_sample as bundle_is_pdf
from shared.sample_paths import sample_document_path
from shared.sample_paths import sample_schema_path as bundle_schema_path
from shared.truth_handler import truth_status_banner
from shared.ui_theme import banner

# Path resolution for samples lives in shared.sample_paths, which shared.evaluator
# also imports. The thin wrappers below keep sample_handler's existing call signatures
# for the UI and the tests, which import them from here.


def is_pdf_sample(sample_name: str) -> bool:
    """
    Report whether a dropdown label refers to a PDF rather than an image

    Args:
        sample_name: Dropdown label, e.g. "sheet" or "PFL/ny-4410772"

    Returns:
        True when the bundle's document is a PDF
    """
    return bundle_is_pdf(sample_name=sample_name)


def list_sample_documents() -> List[str]:
    """
    List every selectable sample document

    One label per bundle under sample/, each being the bundle's path relative to
    sample/. Images and PDFs are listed by the same rule, and a bundle nested in a
    grouping directory keeps that group visible in the dropdown.

    Returns:
        Sorted list of dropdown labels
    """
    samples = list_sample_bundles()

    if not samples:
        logger.warning(f"No sample bundles found under {SAMPLE_DIR}")
    else:
        logger.info(f"Found {len(samples)} sample documents")

    return samples


def resolve_sample_path(sample_name: str) -> str:
    """
    Map a dropdown label back to a path on disk

    Args:
        sample_name: Dropdown label as produced by list_sample_documents()

    Returns:
        Path to the sample's document

    Raises:
        FileNotFoundError: If the label does not resolve. A label that no longer
                           resolves means the sample tree changed under the running
                           app; returning None here would surface much later as an
                           unrelated engine error.
    """
    document_path = sample_document_path(sample_name=sample_name)

    if document_path is None:
        raise FileNotFoundError(
            f"Sample not found on disk: no bundle registered as {sample_name!r}")

    return document_path


def sample_schema_path(sample_name: str) -> Optional[str]:
    """
    Build the schema path for a sample label

    The schema lives inside the bundle, so "claims/STD/case-77315" pairs with
    "sample/claims/STD/case-77315/schema.json". Ground truth is resolved the same
    way by shared.evaluator.load_truth_data(), so a sample has one identity across
    both, and neither can be picked up by a same-named file belonging to a different
    sample.

    Args:
        sample_name: Dropdown label as produced by list_sample_documents()

    Returns:
        Path to the schema file, or None when the label is not a sample bundle
    """
    return bundle_schema_path(sample_name=sample_name)


def sample_result_directory(*, run_dir: str, sample_name: str) -> str:
    """Return a collision-free result directory for a discovered sample bundle."""
    source_directory = bundle_dir(sample_name=sample_name)
    if source_directory is None:
        raise FileNotFoundError(
            f"Cannot create a result directory for unknown sample {sample_name!r}")

    relative_directory = os.path.relpath(
        source_directory, shared.sample_paths.SAMPLE_DIR)
    result_directory = os.path.join(run_dir, relative_directory)

    # The relative path comes from the discovered directory, not from sample_name.
    # Keep an explicit containment check as defense in depth for future changes.
    run_root = os.path.abspath(run_dir)
    if os.path.commonpath(
        [run_root, os.path.abspath(result_directory)]
    ) != run_root:
        raise ValueError(
            f"Sample result path resolves outside the run directory: {sample_name!r}")

    return result_directory


def load_sample_document_and_schema(sample_filename):
    """Load a sample document path and its corresponding schema"""
    if not sample_filename:
        return None, None

    try:
        document_path = resolve_sample_path(sample_filename)
    except FileNotFoundError as resolve_error:
        logger.error(str(resolve_error))
        return None, None

    # Return the file path rather than a PIL Image: the Gradio File component takes
    # a path, and the engines need the original bytes for PDFs.
    logger.info(f"Found sample document: {document_path}")

    # Load the schema if available
    schema = None
    schema_path = sample_schema_path(sample_filename)

    if schema_path and os.path.exists(schema_path):
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
        logger.info(f"No schema found for sample: {sample_filename}")

    return document_path, schema

def on_sample_selected(sample_filename):
    """
    Handle sample selection and load the document, schema and ground truth

    Args:
        sample_filename: Dropdown label of the selected sample, which is a bare
                         filename for an image and a sub-folder-qualified relative
                         path for a PDF

    Returns:
        Tuple of (document path, schema, truth_data, truth_status_html)
    """
    document_path, schema = load_sample_document_and_schema(sample_filename)

    truth_data, truth_exists = load_truth_data(sample_filename)

    truth_status_html = truth_status_banner(
        sample_name=sample_filename, truth_exists=truth_exists)

    return document_path, schema, truth_data, truth_status_html


def process_all_samples(use_textract, use_bedrock, use_bda,
                     bedrock_model_name, bda_s3_bucket=DEFAULT_BDA_S3_BUCKET,
                     s3_bucket=DEFAULT_S3_BUCKET,
                     document_type="generic", enable_structured_output=True, output_schema="",
                     use_bda_blueprint=False, textract_features=None, textract_queries=""):
    """Process all sample images with parallel engine processing

    Args:
        textract_features: Textract feature types to request, any of FORMS, TABLES,
            QUERIES, SIGNATURES, LAYOUT. Empty or None means text detection only.
        textract_queries: Newline-separated questions, used only when QUERIES is
            among textract_features.
    """
    
    # Get list of all sample documents (images plus the source PDFs)
    samples = list_sample_documents()
    if not samples:
        # yield, not return: this is a generator, so a returned payload is
        # discarded and the UI would never show this message.
        yield "<div class='status-error'>No sample documents found</div>", rows_to_html(rows=[])
        return

    # Multi-page PDFs make a batch run far more expensive than the original
    # image-only set, so state the size of the run before any API call is made.
    log_batch_scope(samples)

    # Initialize results tracking by engine.
    #
    # total_pages and cost_components are tracked alongside the totals so a batch row
    # can show the same per-page figures and the same cost formulas as a single-document
    # run: a batch over the claim-form PDFs covers dozens of pages, and a figure
    # averaged per document says nothing about what a page costs.
    results_by_engine = {
        engine_name: {
            "count": 0,
            "total_pages": 0,
            "total_time": 0,
            "total_cost": 0,
            "accuracy_values": [],
            "cost_components": [],
            "failure_count": 0,
        }
        for engine_name in ("Textract", "Bedrock", "BDA")
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
    selected_engines = [
        name for name, selected in (
            ("Textract", use_textract),
            ("Bedrock", use_bedrock),
            ("BDA", use_bda),
        ) if selected
    ]
    
    # Process each sample
    for i, sample_name in enumerate(samples):
        settled_engines = set()
        status_html = f"<div class='status-processing'>Processing sample {i+1}/{len(samples)}: {sample_name}</div>"
        
        yield status_html, rows_to_html(
            rows=build_batch_rows(results_by_engine=results_by_engine))
        
        try:
            # Mirror the validated bundle hierarchy under this run. Using only the
            # basename made a/receipt and b/receipt overwrite each other.
            sample_dir = sample_result_directory(
                run_dir=run_dir, sample_name=sample_name)
            os.makedirs(sample_dir, exist_ok=True)

            # Load the sample. PDFs are handed to the engines as a path - all three
            # detect a .pdf path and switch to their document APIs - while images are
            # handed over as a PIL Image, as before.
            engine_input, preview_image = load_sample_for_processing(sample_name)

            # Save the original image. For a PDF this is page 1 only, and may be None
            # when PyMuPDF is unavailable.
            if preview_image is not None:
                save_original_image(preview_image, os.path.join(sample_dir, "original.jpg"))
            else:
                logger.warning(f"No preview render available for {sample_name}; skipping original.jpg")

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
                        engine_input, {
                            "output_schema": image_output_schema if enable_structured_output else None,
                            "s3_bucket": s3_bucket,
                            "feature_types": textract_features,
                            "textract_queries": textract_queries
                        }
                    )
                
                if use_bedrock:
                    futures['Bedrock'] = executor.submit(
                        bedrock_engine.process_image,
                        engine_input, {
                            'model_id': model_id,
                            'document_type': document_type,
                            'output_schema': image_output_schema if enable_structured_output and image_output_schema else None
                        }
                    )
                
                if use_bda:
                    futures['BDA'] = executor.submit(
                        bda_engine.process_image,
                        engine_input, {
                            's3_bucket': bda_s3_bucket,
                            'document_type': document_type,
                            'output_schema': image_output_schema if enable_structured_output and image_output_schema else None,
                            'use_blueprint': use_bda_blueprint
                        }
                    )
                
                # Process results as they complete
                for engine_name, future in futures.items():
                    try:
                        # Get the direct engine result
                        result = future.result()
                        
                        # Use process_engine_result for consistent error detection,
                        # accuracy calculation and costing.
                        from processor import process_engine_result
                        processed_result = process_engine_result(
                            engine_name, result, truth_data, truth_exists)
                        settled_engines.add(engine_name)

                        if processed_result.get("succeeded") is not True:
                            results_by_engine[engine_name]["failure_count"] += 1
                            handle_engine_error(
                                engine_name, sample_name,
                                RuntimeError(processed_result.get(
                                    "text", "engine reported an error")))
                            intermediate_status = (
                                f"<div class='status-processing'>Processing sample "
                                f"{i+1}/{len(samples)}: {sample_name} - "
                                f"{engine_name} failed</div>")
                            yield intermediate_status, rows_to_html(
                                rows=build_batch_rows(
                                    results_by_engine=results_by_engine))
                            continue
                            
                        # Extract fields from processed result
                        process_time = processed_result.get('time', 0)
                        extracted_text = processed_result.get('text', '')
                        json_data = processed_result.get('json', {})
                        image_data = processed_result.get('image')
                        accuracy = processed_result.get('accuracy', 0)
                        cost = processed_result.get('cost', 0)
                        page_count = processed_result.get('pages', 0)
                        cost_breakdown = processed_result.get('cost_breakdown', [])
                            
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
                                     os.path.join(engine_dir, "metadata.json"),
                                     page_count=page_count)

                        # Update engine results
                        results_by_engine[engine_name]["count"] += 1
                        results_by_engine[engine_name]["total_pages"] += page_count
                        results_by_engine[engine_name]["total_time"] += process_time
                        results_by_engine[engine_name]["total_cost"] += cost
                        results_by_engine[engine_name]["accuracy_values"].append(accuracy)
                        results_by_engine[engine_name]["cost_components"].extend(cost_breakdown)

                        # Update UI with current progress
                        intermediate_status = f"<div class='status-processing'>Processing sample {i+1}/{len(samples)}: {sample_name} - {engine_name} completed</div>"
                        yield intermediate_status, rows_to_html(
                            rows=build_batch_rows(results_by_engine=results_by_engine))

                    except Exception as e:
                        settled_engines.add(engine_name)
                        results_by_engine[engine_name]["failure_count"] += 1
                        handle_engine_error(engine_name, sample_name, e)
        
        except Exception as e:
            for engine_name in selected_engines:
                if engine_name not in settled_engines:
                    results_by_engine[engine_name]["failure_count"] += 1
            handle_sample_error(sample_name, e, run_dir)
    
    # Create summary at the end
    create_summary(results_by_engine, len(samples), total_start, run_dir,
                  use_textract, use_bedrock, bedrock_model_name, use_bda, use_bda_blueprint)

    total_time = time.time() - total_start
    batch_rows = build_batch_rows(results_by_engine=results_by_engine)

    # Recorded through the same path a single-document run uses, so a batch appears
    # in results/history.jsonl alongside single runs rather than only inside its own
    # run_ directory. Without this, "run it repeatedly and compare" would not cover
    # the button that does the most work.
    successful_attempts = sum(
        data["count"] for data in results_by_engine.values())
    failed_attempts = sum(
        data["failure_count"] for data in results_by_engine.values())
    total_attempts = successful_attempts + failed_attempts

    saved_note = ""
    if batch_rows:
        saved_note = save_run_record(
            document_name=f"all-samples-{len(samples)}",
            rows=batch_rows,
            total_time_s=total_time,
            configuration={
                "batch": True,
                "samples": len(samples),
                "engines": sorted(selected_engines),
                "successful_attempts": successful_attempts,
                "failed_attempts": failed_attempts,
                "bedrock_model": bedrock_model_name if use_bedrock else None,
                "document_type": document_type,
                "structured_output": enable_structured_output,
                "bda_blueprint": use_bda_blueprint if use_bda else None,
                "textract_features": sorted(textract_features or []) if use_textract else None,
                "textract_queries": bool(textract_queries) if use_textract else None,
                "run_directory": run_dir,
            },
            ground_truth_available=any(
                data["accuracy_values"] for data in results_by_engine.values()))

    if failed_attempts == 0:
        status_html = banner(
            tone="ok",
            text=f"All <b>{total_attempts}</b> engine attempts completed in "
                 f"<code>{total_time:.2f}s</code> · results in "
                 f"<code>{run_dir}</code>{saved_note}")
    elif successful_attempts:
        status_html = banner(
            tone="warn",
            text=f"<b>{successful_attempts}/{total_attempts}</b> engine attempts "
                 f"completed in <code>{total_time:.2f}s</code> · results in "
                 f"<code>{run_dir}</code>{saved_note}")
    else:
        status_html = banner(
            tone="error",
            text=f"All <b>{failed_attempts}</b> engine attempts failed after "
                 f"<code>{total_time:.2f}s</code> · no benchmark result was saved")

    # yield, not return: a generator's return value never reaches Gradio, so
    # returning here left the last per-sample progress message on screen and the
    # completion summary was never displayed.
    yield status_html, rows_to_html(rows=batch_rows)


def log_batch_scope(samples: List[str]) -> None:
    """
    Log how much work a batch run represents before any API call is made

    Counts pages for PDF samples so a run over the multi-page claim forms cannot
    look like a run over seven single-page images.

    Args:
        samples: Dropdown labels about to be processed

    Returns:
        None
    """
    pdf_samples = [name for name in samples if is_pdf_sample(name)]
    image_samples = [name for name in samples if not is_pdf_sample(name)]

    pdf_pages = 0
    for sample_name in pdf_samples:
        try:
            pdf_pages += get_pdf_page_count(resolve_sample_path(sample_name))
        except FileNotFoundError as resolve_error:
            logger.error(str(resolve_error))

    logger.info(
        f"Batch scope: {len(image_samples)} image(s) + {len(pdf_samples)} PDF(s) "
        f"spanning {pdf_pages} page(s) = {len(image_samples) + pdf_pages} billable page(s) per engine"
    )


def load_sample_for_processing(sample_name: str) -> Tuple[Union[str, Image.Image], Optional[Image.Image]]:
    """
    Load a sample in the form each engine expects, plus an image for the run record

    Args:
        sample_name: Dropdown label as produced by list_sample_documents()

    Returns:
        Tuple of (engine input, preview image). The engine input is a file path for a
        PDF and a PIL Image with .name set for an image. The preview image is page 1
        for a PDF - None when PyMuPDF is unavailable - and the image itself otherwise.

    Raises:
        FileNotFoundError: If the label does not resolve on disk
    """
    sample_path = resolve_sample_path(sample_name)

    if is_pdf_sample(sample_name):
        # The engines read PDF bytes from the path themselves; the render is only for
        # the saved original.jpg.
        return sample_path, convert_pdf_to_image(sample_path, page_num=0)

    image = Image.open(sample_path)
    image.name = sample_name  # Set image name for proper truth data loading
    return image, image


def load_sample_schema(sample_name, default_schema=""):
    """Load sample-specific schema if available"""
    sample_schema = None
    schema_path = sample_schema_path(sample_name)

    if schema_path and os.path.exists(schema_path):
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


def save_metadata(engine_name, result, process_time, cost, accuracy, output_path,
                  page_count=0):
    """Save metadata including engine-specific information

    Args:
        engine_name: Engine that produced the result.
        result: The engine's own raw result dictionary.
        process_time: Wall-clock seconds for the document.
        cost: Estimated charge for the document, in USD.
        accuracy: Ground-truth match rate, as a percentage.
        output_path: Where to write the metadata JSON.
        page_count: Pages the engine read. Recorded because cost only means
            something next to the number of pages it covered.
    """
    metadata = {
        "process_time": process_time,
        "pages": page_count,
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


def build_batch_rows(*, results_by_engine: Dict[str, Dict[str, Any]]) -> List[RunRow]:
    """
    Turn the running batch totals into rows for the comparison table.

    A batch row means the same thing as a single-document row with more in it: the
    time and cost columns are the batch's totals, the per-page columns divide those
    by every page the engine read, and accuracy is the mean over the documents.
    Averaging per document - as this previously did - made the figures incomparable
    between a run over single-page images and one over multi-page PDFs.

    Engines with no completed document are left out entirely rather than shown as
    zeros, which would read as "it ran and cost nothing".

    Args:
        results_by_engine: Running totals keyed by engine name, holding count,
            total_pages, total_time, total_cost, accuracy_values and
            cost_components.

    Returns:
        List[RunRow]: One row per engine that completed at least one document.
    """
    rows: List[RunRow] = []

    for engine_name, data in results_by_engine.items():
        if data["count"] == 0:
            continue

        accuracy_values = data["accuracy_values"]
        mean_accuracy = (
            sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0.0)

        rows.append(RunRow(
            engine=engine_name,
            documents=data["count"],
            pages=data["total_pages"],
            total_time_s=data["total_time"],
            total_cost_usd=data["total_cost"],
            accuracy_pct=mean_accuracy,
            # The same charges recur once per document, so they are merged into one
            # line each rather than listing the batch's every individual formula.
            cost_breakdown=merge_cost_components(
                components=data["cost_components"], document_count=data["count"]),
        ))

    return rows


def handle_engine_error(engine_name, sample_name, error):
    """Handle errors during engine processing"""
    logger.error(f"Error getting result for {engine_name} on {sample_name}: {str(error)}")
    import traceback
    stack_trace = traceback.format_exc()
    if stack_trace.strip() != "NoneType: None":
        logger.error(f"Stack trace: {stack_trace}")


def handle_sample_error(sample_name, error, run_dir):
    """Handle errors during sample processing"""
    logger.error(f"Error processing sample {sample_name}: {str(error)}")
    try:
        sample_dir = sample_result_directory(
            run_dir=run_dir, sample_name=sample_name)
    except (FileNotFoundError, ValueError):
        sample_id = hashlib.sha256(sample_name.encode("utf-8")).hexdigest()[:12]
        sample_dir = os.path.join(run_dir, "errors", sample_id)
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
        if data["count"] > 0 or data.get("failure_count", 0) > 0:
            avg_accuracy = 0
            if data["accuracy_values"]:
                avg_accuracy = sum(data["accuracy_values"]) / len(data["accuracy_values"])
                
            # Per-page figures are omitted rather than guessed when no engine
            # reported a page count, so a null here means "not known", not "zero".
            total_pages = data["total_pages"]

            summary["results"][engine] = {
                "documents_processed": data["count"],
                "documents_failed": data.get("failure_count", 0),
                "total_pages": total_pages,
                "total_time": data["total_time"],
                "avg_time_per_document": (
                    data["total_time"] / data["count"] if data["count"] else None),
                "avg_time_per_page": (
                    data["total_time"] / total_pages if total_pages > 0 else None),
                "total_cost": data["total_cost"],
                "avg_cost_per_document": (
                    data["total_cost"] / data["count"] if data["count"] else None),
                "avg_cost_per_page": (
                    data["total_cost"] / total_pages if total_pages > 0 else None),
                "avg_accuracy": avg_accuracy
            }
    
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
