import concurrent.futures
import time
from typing import List

import gradio as gr

# Import from reorganized modules
from shared.config import (
    logger,
    BEDROCK_MODELS,
    STATUS_HTML,
    POSTPROCESSING_MODEL,
    DEFAULT_BDA_S3_BUCKET,
    DEFAULT_S3_BUCKET,
)
from engines.textract_engine import TextractEngine
from engines.bedrock_engine import BedrockEngine
from engines.bda_engine import BDAEngine
from shared.cost_calculator import (
    calculate_bedrock_cost,
    calculate_bda_cost,
    describe_bda_cost,
    describe_bedrock_cost,
    describe_textract_cost,
)
from shared.evaluator import load_truth_data, calculate_accuracy
from shared.comparison_utils import (
    ALL_ENGINES_LABEL,
    ENGINE_NAMES,
    create_comparison_view,
)
from shared.results_table import RunRow, build_run_rows, rows_to_html
from shared.run_recorder import save_run_record
from shared.truth_handler import truth_status_banner

def initialize_processing(image, image_name=None):
    """Initialize data for image processing"""
    # Determine image name - use provided name or extract from image if available
    if image_name is None:
        if hasattr(image, 'name'):
            image_name = image.name
        else:
            logger.warning("No image name provided and image has no name attribute")
            image_name = "unknown_image"

    logger.info(f"Processing image: {image_name}")
    
    # Load truth data using the determined image name
    truth_data, truth_exists = load_truth_data(image_name)
    
    if truth_exists:
        logger.debug(f"Truth data loaded successfully. Keys: {list(truth_data.keys())}")
        logger.debug(f"Truth data first level values: {str(truth_data)[:200]}...")
        logger.info(f"Loaded truth data for {image_name}")
    else:
        logger.info(f"No truth data found for {image_name}")

    truth_status_html = truth_status_banner(
        sample_name=image_name, truth_exists=truth_exists)


    return image_name, truth_data, truth_exists, truth_status_html

def _error_message(*, text: str) -> str:
    """Reduce an engine's error text to the underlying cause.

    Every engine prefixes its failure message with its own name — "BDA Error: ",
    "Textract Error: ", "Amazon Bedrock Error: ". STATUS_HTML["error"] already
    names the engine, so the prefix is dropped to avoid repeating it.

    Args:
        text (str): The engine's `text` field from a result whose operation_type
            is "error".

    Returns:
        str: The message without the engine-name prefix, or a stand-in when the
            engine gave no message at all.
    """
    if not text:
        return "engine reported an error but gave no message"
    # Split on the first occurrence only: the message itself may contain "Error:".
    _, separator, remainder = text.partition("Error:")
    return remainder.strip() if separator else text.strip()


def _failed_engine_result(*, engine_name: str, message: str, process_time: float = 0):
    """Build the processed-result shape for an engine that did not complete."""
    return {
        "text": message,
        "json": None,
        "image": None,
        "time": process_time,
        "status_html": STATUS_HTML["error"](engine_name, process_time, message),
        "accuracy": 0.0,
        "token_usage": None,
        "cost": 0.0,
        "cost_html": "<div></div>",
        "pages": 0,
        "cost_breakdown": [],
        "succeeded": False,
    }


def process_engine_result(engine_name, result, truth_data, truth_exists):
    """Process result from an OCR engine"""
    # Default values
    text = ""
    json_data = None
    image_data = None
    process_time = 0
    status_html = ""
    accuracy = 0.0
    token_usage = None
    cost = 0.0
    cost_html = "<div></div>"
    
    if not isinstance(result, dict):
        return {
            "text": str(result), 
            "json": None, 
            "image": None, 
            "time": 0, 
            "status_html": STATUS_HTML["error"](engine_name, 0, "Invalid result format"),
            "accuracy": 0.0,
            "token_usage": None,
            "cost": 0.0,
            "cost_html": "<div></div>",
            "pages": 0,
            "cost_breakdown": [],
            "succeeded": False,
        }
    
    # Extract common fields from result
    text = result.get('text', '')
    json_data = result.get('json', {})
    image_data = result.get('image')
    process_time = result.get('process_time', 0)
    token_usage = result.get('token_usage')
    # Every engine reports the pages it actually read. This drives the Pages
    # column, the per-page statistics, and BDA's per-page charge. A failed run
    # reports 0, which reads downstream as "no per-page figure", not as free.
    page_count = result.get('pages', 0)

    # Engines report failure by returning this dict with operation_type "error"
    # rather than by raising, so nothing downstream sees an exception. Without this
    # check every branch below ends at STATUS_HTML["completed"] unconditionally, so
    # a failed run rendered as a green "completed" banner with a cost attached, and
    # the missing accuracy figure was the only hint anything had gone wrong. The
    # error message was already in `text` — shown in the text panel, contradicted
    # by the banner above it.
    if result.get('operation_type') == 'error':
        return {
            "text": text,
            "json": None,
            "image": image_data,
            "time": process_time,
            "status_html": STATUS_HTML["error"](
                engine_name, process_time, _error_message(text=text)),
            "accuracy": 0.0,
            "token_usage": token_usage,
            # A run that returned nothing must not report a charge it cannot
            # substantiate, whatever the engine's per-page rate would have been.
            "cost": 0.0,
            "cost_html": "<div></div>",
            # A failed run has no per-page statistic to report. 0 is read as
            # "unknown" by the results table, which leaves those cells blank
            # rather than dividing by zero or implying one page was read.
            "pages": 0,
            "cost_breakdown": [],
            "succeeded": False,
        }

    # Every engine's charge is now assembled from CostComponents rather than from
    # a bare float, so the total, the status banner's breakdown and the comparison
    # table's tooltip are all derived from one description of the arithmetic and
    # cannot drift apart.
    cost_breakdown = []

    if engine_name == "Textract":
        # Textract's own charge is per page, from the operation type and the
        # features that were actually requested - not the text-detection rate,
        # which would understate an AnalyzeDocument run by up to 30x.
        cost_breakdown += describe_textract_cost(
            operation_type=result.get('operation_type', 'textract_detect'),
            page_count=page_count,
            feature_types=result.get('feature_types'))

        # Textract does the OCR alone; no LLM reads the document. This second
        # charge is the separate step that turns its text into schema-shaped JSON,
        # and it is usually the larger of the two.
        cost_breakdown += describe_bedrock_cost(
            model_id=POSTPROCESSING_MODEL,
            token_usage=token_usage,
            purpose="JSON structuring")

        cost = sum(component.amount for component in cost_breakdown)

        # The engine computes its own Textract charge from the same three fields,
        # so a mismatch means one of the two paths is wrong and worth saying so.
        reported = result.get('textract_cost')
        derived_textract = sum(
            component.amount for component in cost_breakdown
            if component.label.startswith("Textract"))
        if reported is not None and abs(reported - derived_textract) > 1e-9:
            logger.warning(
                f"Textract reported a cost of ${reported:.6f} but the same inputs "
                f"derive ${derived_textract:.6f}")

    elif engine_name == "Bedrock":
        # One Converse call does OCR and structuring together, so there is a
        # single per-token charge and no per-page component at all.
        cost_breakdown += describe_bedrock_cost(
            model_id=result.get('model_id', ''),
            token_usage=token_usage,
            purpose="Bedrock extraction")
        cost = sum(component.amount for component in cost_breakdown)
        cost_html, _ = calculate_bedrock_cost(result.get('model_id', ''), token_usage)

    elif engine_name == "BDA":
        field_count = result.get('field_count', 0)
        use_blueprint = result.get('use_blueprint', False)
        bda_document_type = (
            'image' if result.get('file_type') == 'image' else 'document'
        )

        # BDA is priced per page, and this used to pass page_count=1 regardless,
        # so a seven-page document was billed as one. The engine now reports the
        # count BDA itself gave, from standard_output.metadata.number_of_pages.
        if page_count <= 0:
            raise ValueError(
                f"BDA completed but reported {page_count} pages, so its per-page "
                f"charge cannot be calculated")

        cost_breakdown += describe_bda_cost(
            use_blueprint=use_blueprint,
            document_type=bda_document_type,
            page_count=page_count,
            field_count=field_count)

        # Only the standard-output path needs an LLM afterwards: the blueprint
        # path returns schema-shaped JSON directly, so nothing is billed here.
        if not use_blueprint:
            cost_breakdown += describe_bedrock_cost(
                model_id=POSTPROCESSING_MODEL,
                token_usage=token_usage,
                purpose="JSON structuring")

        cost = sum(component.amount for component in cost_breakdown)
        cost_html, _ = calculate_bda_cost(
            use_blueprint, bda_document_type,
            page_count=page_count, field_count=field_count)

    # One banner for every engine, itemised when more than one service was billed.
    if len(cost_breakdown) > 1:
        cost_detail = " (" + ", ".join(
            f"{component.label.split(' (')[0]}: ${component.amount:.6f}"
            for component in cost_breakdown) + ")"
        status_html = STATUS_HTML["completed"](
            engine_name, process_time, cost, cost_detail)
    else:
        status_html = STATUS_HTML["completed"](engine_name, process_time, cost)


    # Calculate accuracy if truth data is available
    if truth_exists and json_data:
        logger.debug(f"Calculating accuracy for {engine_name}")
        accuracy_result = calculate_accuracy(json_data, truth_data)
        accuracy = accuracy_result["total_accuracy"] if isinstance(accuracy_result, dict) else accuracy_result
        logger.info(f"{engine_name} accuracy: {accuracy}%")
        
        # Add accuracy to status
        status_html = status_html.replace("</div>", f" | Accuracy: {accuracy}%</div>")
    
    return {
        "text": text,
        "json": json_data,
        "image": image_data,
        "time": process_time,
        "status_html": status_html,
        "accuracy": accuracy,
        "token_usage": token_usage,
        "cost": cost,
        "cost_html": cost_html,
        "pages": page_count,
        "cost_breakdown": cost_breakdown,
        "succeeded": True,
    }

def create_comparison_view_for_engines(truth_data, truth_exists, engine_results):
    """Create comparison view based on available engine results

    Renders every engine that produced JSON in one table rather than picking a
    preferred one, so the Compare tab opens on the side-by-side view. Engines that
    were not selected, or that failed, have a falsy 'json' and get no column.

    Args:
        truth_data: Ground truth for the document, or None.
        truth_exists (bool): Whether ground truth was found for this document.
        engine_results (dict): Per-engine result dicts keyed by engine name.

    Returns:
        str: HTML for the Compare tab.
    """
    return create_comparison_view(
        truth_data=truth_data if truth_exists else None,
        engine_json_by_name={
            engine_name: engine_results.get(engine_name, {}).get("json")
            for engine_name in ENGINE_NAMES
        },
        engine_filter=ALL_ENGINES_LABEL)

def create_results_table_html(engine_results):
    """Render the Comparison Results table for the engines that ran.

    Returns HTML rather than a DataFrame so each cost cell can carry a `title`
    with the formula behind it; `gr.Dataframe` cannot hold a per-cell tooltip.
    The numbers themselves come from shared.results_table, which is also what
    the run recorder persists, so the saved record and the screen agree.

    Args:
        engine_results (dict): Per-engine result dicts keyed by engine name.

    Returns:
        str: The table markup.
    """
    return rows_to_html(rows=build_run_rows(engine_results=engine_results))




def process_image_with_engines(image, use_textract, use_bedrock, use_bda,
                             bedrock_model_name, bda_s3_bucket=DEFAULT_BDA_S3_BUCKET,
                             s3_bucket=DEFAULT_S3_BUCKET,
                             document_type="generic", enable_structured_output=True, output_schema="",
                             use_bda_blueprint=False, image_name=None,
                             textract_features=None, textract_queries=""):
    """Process image with selected OCR engines in parallel

    Args:
        textract_features: Textract feature types to request, any of FORMS, TABLES,
            QUERIES, SIGNATURES, LAYOUT. Empty or None means text detection only.
        textract_queries: Newline-separated questions, used only when QUERIES is
            among textract_features.
    """
    total_start = time.time()
    default_result = {
        "text": "", "json": None, "image": None, "time": 0,
        "accuracy": 0, "cost": 0, "succeeded": None,
    }
    default_bedrock_result = {**default_result, "token_usage": None, "cost_html": "<div></div>"}

    engine_results = {
        "Textract": default_result.copy() if use_textract else default_result.copy(),
        "Bedrock": default_bedrock_result.copy() if use_bedrock else default_bedrock_result.copy(),
        "BDA": default_result.copy() if use_bda else default_result.copy()
    }    

    # Empty table for error cases. Rendering the same component the successful
    # path renders keeps the panel's shape stable across a failed run.
    empty_table = rows_to_html(rows=[])

    # Check for image and selected engines.
    #
    # These guards must yield, not return. This is a generator function, so a
    # `return value` raises StopIteration(value) and the payload is discarded -
    # Gradio then substitutes None for all 20 outputs and the UI sits on the
    # spinner forever with no error shown. tests/test_generator_contract.py pins
    # this for every guard clause in both streaming handlers.
    if image is None:
        yield [
            STATUS_HTML["error"]("Upload", 0, "No image uploaded"),
            "<div></div>", "", None, None,
            "<div></div>", "", None, None,
            "<div></div>", None,
            "<div></div>", "", None, None,
            "<div></div>", None,
            # Output 17 is the Compare tab's filter Dropdown, whose only valid values
            # are ENGINE_FILTER_CHOICES. It used to receive an HTML string here,
            # which Gradio rejects as "not in the list of choices" - harmless only
            # for as long as this payload was never actually emitted.
            gr.update(), "<div>No comparison available</div>",
            empty_table
        ]
        return

    if not any([use_textract, use_bedrock, use_bda]):
        yield [
            STATUS_HTML["error"]("Selection", 0, "Please select at least one OCR engine"),
            "<div></div>", "", None, None,
            "<div></div>", "", None, None,
            "<div></div>", None,
            "<div></div>", "", None, None,
            "<div></div>", None,
            # Output 17 is the Compare tab's filter Dropdown, whose only valid values
            # are ENGINE_FILTER_CHOICES. It used to receive an HTML string here,
            # which Gradio rejects as "not in the list of choices" - harmless only
            # for as long as this payload was never actually emitted.
            gr.update(), "<div>No comparison available</div>",
            empty_table
        ]
        return

    # Initialize processing data
    image_name, truth_data, truth_exists, truth_status_html = initialize_processing(image, image_name)
    
    # Get bedrock model ID if needed
    model_id = BEDROCK_MODELS.get(bedrock_model_name, "") if use_bedrock else ""
    
    # Initialize engine status and results
    global_status_html = STATUS_HTML["global_processing"]()
    engine_status = {
        "Textract": STATUS_HTML["processing"]("Textract") if use_textract else "<div></div>",
        "Bedrock": STATUS_HTML["processing"]("Bedrock") if use_bedrock else "<div></div>",
        "BDA": STATUS_HTML["processing"]("BDA") if use_bda else "<div></div>"
    }

    # The Compare tab's filter resets to the side-by-side view on every run.
    diff_engine_value = ALL_ENGINES_LABEL
    # Initial UI update
    yield [
        global_status_html,
        engine_status.get("Textract", "<div></div>"), 
        engine_results.get("Textract", {}).get("text", ""), 
        engine_results.get("Textract", {}).get("json"), 
        engine_results.get("Textract", {}).get("image"),
        
        engine_status.get("Bedrock", "<div></div>"), 
        engine_results.get("Bedrock", {}).get("text", ""), 
        engine_results.get("Bedrock", {}).get("json"), 
        engine_results.get("Bedrock", {}).get("image"),
        
        "<div></div>" if not use_bedrock else engine_results.get("Bedrock", {}).get("cost_html", "<div></div>"), 
        None if not use_bedrock else engine_results.get("Bedrock", {}).get("token_usage"),
        
        engine_status.get("BDA", "<div></div>"), 
        engine_results.get("BDA", {}).get("text", ""), 
        engine_results.get("BDA", {}).get("json"), 
        engine_results.get("BDA", {}).get("image"),
        
        truth_status_html, 
        truth_data,
        
        diff_engine_value, 
        "<div>Processing results... comparison will be available when completed</div>",
        
        empty_table
    ]
    
    # Create engine instances
    textract_engine = TextractEngine()
    bedrock_engine = BedrockEngine()
    bda_engine = BDAEngine()

    # Process with selected engines in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        
        if use_textract:
            futures['Textract'] = executor.submit(
                textract_engine.process_image, 
                image, 
                {
                    'output_schema': output_schema if enable_structured_output else None,
                    's3_bucket': s3_bucket,
                    'feature_types': textract_features,
                    'textract_queries': textract_queries
                }
            )
        
        if use_bedrock:
            futures['Bedrock'] = executor.submit(
                bedrock_engine.process_image, 
                image,
                {
                    'model_id': model_id,
                    'document_type': document_type,
                    'output_schema': (
                        output_schema
                        if enable_structured_output and output_schema
                        else None
                    )
                }
            )
        
        if use_bda:
            futures['BDA'] = executor.submit(
                bda_engine.process_image,
                image, 
                {
                    's3_bucket': bda_s3_bucket,
                    'document_type': document_type,
                    'output_schema': output_schema if enable_structured_output and output_schema else None,
                    'use_blueprint': use_bda_blueprint
                }
            )
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures.values()):
            # Find which engine this future belongs to
            engine_name = None
            for name, engine_future in futures.items():
                if future == engine_future:
                    engine_name = name
                    break
                    
            try:
                # Process result for this engine
                result = future.result()
                processed_result = process_engine_result(engine_name, result, truth_data, truth_exists)
                
                # Update engine results
                engine_results[engine_name] = processed_result
                engine_status[engine_name] = processed_result["status_html"]
                
                # Create comparison view with available results
                comparison_html = create_comparison_view_for_engines(truth_data, truth_exists, engine_results)
                
                # Create results table
                results_table_html = create_results_table_html({
                    name: data for name, data in engine_results.items() 
                    if name in futures and data.get("succeeded") is True
                })
                
                # Individual engine panels show progress as each future finishes.
                # The global banner stays in progress until every future has settled.
                global_status_html = STATUS_HTML["global_processing"]()
                
                # Update UI
                yield [
                    global_status_html,
                    engine_status.get("Textract", "<div></div>"), 
                    engine_results.get("Textract", {}).get("text", ""), 
                    engine_results.get("Textract", {}).get("json"), 
                    engine_results.get("Textract", {}).get("image"),
                    
                    engine_status.get("Bedrock", "<div></div>"), 
                    engine_results.get("Bedrock", {}).get("text", ""), 
                    engine_results.get("Bedrock", {}).get("json"), 
                    engine_results.get("Bedrock", {}).get("image"),
                    
                    "<div></div>" if not use_bedrock else engine_results.get("Bedrock", {}).get("cost_html", "<div></div>"), 
                    None if not use_bedrock else engine_results.get("Bedrock", {}).get("token_usage"),
                    
                    engine_status.get("BDA", "<div></div>"), 
                    engine_results.get("BDA", {}).get("text", ""), 
                    engine_results.get("BDA", {}).get("json"), 
                    engine_results.get("BDA", {}).get("image"),
                    
                    truth_status_html, 
                    truth_data,
                    
                    diff_engine_value, 
                    comparison_html,
                    
                    results_table_html
                ]
                
            except Exception as e:
                logger.error(f"Error in {engine_name} processing: {str(e)}")
                
                # Normalize raised exceptions to the same result shape used when an
                # engine returns operation_type="error".
                engine_results[engine_name] = _failed_engine_result(
                    engine_name=engine_name, message=str(e))
                engine_status[engine_name] = engine_results[engine_name]["status_html"]
                
                # Create comparison with available results
                comparison_html = create_comparison_view_for_engines(truth_data, truth_exists, engine_results)
                
                # Create results table
                results_table_html = create_results_table_html({
                    name: data for name, data in engine_results.items() 
                    if name in futures and data.get("succeeded") is True
                })
                
                global_status_html = STATUS_HTML["global_processing"]()
                
                # Update UI
                yield [
                    global_status_html,
                    engine_status.get("Textract", "<div></div>"), 
                    engine_results.get("Textract", {}).get("text", ""), 
                    engine_results.get("Textract", {}).get("json"), 
                    engine_results.get("Textract", {}).get("image"),
                    
                    engine_status.get("Bedrock", "<div></div>"), 
                    engine_results.get("Bedrock", {}).get("text", ""), 
                    engine_results.get("Bedrock", {}).get("json"), 
                    engine_results.get("Bedrock", {}).get("image"),
                    
                    "<div></div>" if not use_bedrock else engine_results.get("Bedrock", {}).get("cost_html", "<div></div>"), 
                    None if not use_bedrock else engine_results.get("Bedrock", {}).get("token_usage"),
                    
                    engine_status.get("BDA", "<div></div>"), 
                    engine_results.get("BDA", {}).get("text", ""), 
                    engine_results.get("BDA", {}).get("json"), 
                    engine_results.get("BDA", {}).get("image"),
                    
                    truth_status_html, 
                    truth_data,
                    
                    diff_engine_value, 
                    comparison_html,
                    
                    results_table_html
                ]
    
    # Final comparison view
    comparison_html = create_comparison_view_for_engines(truth_data, truth_exists, engine_results)

    # Final results. The rows are built once and used twice - rendered to the table
    # and written to the record - so the saved figures are the figures on screen.
    selected_results = {
        name: data for name, data in engine_results.items()
        if name in futures and data.get("succeeded") is True
    }
    final_rows: List[RunRow] = build_run_rows(engine_results=selected_results)
    results_table_html = rows_to_html(rows=final_rows)

    # Final status
    total_time = time.time() - total_start
    total_cost = sum(data["cost"] for data in selected_results.values())
    failed_engines = sorted(
        name for name in futures
        if engine_results[name].get("succeeded") is not True)

    # Persist the run so it can be compared with later ones. Only here, at the end:
    # the intermediate yields describe a run still in progress.
    saved_note = ""
    if final_rows:
        saved_note = save_run_record(
            document_name=image_name,
            rows=final_rows,
            total_time_s=total_time,
            configuration={
                "engines": sorted(futures.keys()),
                "failed_engines": failed_engines,
                "bedrock_model": bedrock_model_name if use_bedrock else None,
                "bedrock_model_id": model_id or None,
                "document_type": document_type,
                "structured_output": enable_structured_output,
                "bda_blueprint": use_bda_blueprint if use_bda else None,
                "textract_features": sorted(textract_features or []) if use_textract else None,
                "textract_queries": bool(textract_queries) if use_textract else None,
            },
            ground_truth_available=truth_exists)

    success_count = len(selected_results)
    if success_count == len(futures):
        global_status_html = STATUS_HTML["global_completed"](
            total_time, total_cost, saved_note)
    elif success_count:
        global_status_html = STATUS_HTML["global_partial"](
            success_count, len(futures), total_time, total_cost, saved_note)
    else:
        global_status_html = STATUS_HTML["global_failed"](
            len(futures), total_time)


    # Emit the final UI update. Again this must be a yield: a generator's return
    # value never reaches Gradio.
    yield [
        global_status_html,
        engine_status.get("Textract", "<div></div>"), 
        engine_results.get("Textract", {}).get("text", ""), 
        engine_results.get("Textract", {}).get("json"), 
        engine_results.get("Textract", {}).get("image"),
        
        engine_status.get("Bedrock", "<div></div>"), 
        engine_results.get("Bedrock", {}).get("text", ""), 
        engine_results.get("Bedrock", {}).get("json"), 
        engine_results.get("Bedrock", {}).get("image"),
        
        "<div></div>" if not use_bedrock else engine_results.get("Bedrock", {}).get("cost_html", "<div></div>"), 
        None if not use_bedrock else engine_results.get("Bedrock", {}).get("token_usage"),
        
        engine_status.get("BDA", "<div></div>"), 
        engine_results.get("BDA", {}).get("text", ""), 
        engine_results.get("BDA", {}).get("json"), 
        engine_results.get("BDA", {}).get("image"),
        
        truth_status_html, 
        truth_data,
        
        diff_engine_value, 
        comparison_html,
        
        results_table_html
    ]
