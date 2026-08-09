import json
import os

import gradio as gr
from sample_handler import list_sample_documents, on_sample_selected, process_all_samples
from processor import process_image_with_engines
from shared.comparison_utils import create_comparison_view
from shared.config import logger
from shared.ui_theme import banner, page_readout
from preview_handler import handle_file_preview, navigate_pdf_page


def load_schema_from_file(schema_file):
    """
    Read and validate an uploaded JSON schema file

    Args:
        schema_file: Gradio File value - an object with a .name path, a path
                     string, or None when the upload is cleared

    Returns:
        Tuple of (schema text for the editor, status HTML)

    Raises:
        gr.Error: If the file cannot be read or does not contain valid JSON. A
                  broken schema must not be swallowed - extracting against the
                  previously loaded schema instead would corrupt the accuracy
                  numbers without any visible sign.
    """
    if not schema_file:
        return gr.update(), "<div></div>"

    schema_path = getattr(schema_file, 'name', schema_file)

    try:
        with open(schema_path, 'r', encoding='utf-8') as handle:
            schema_text = handle.read()
    except OSError as read_error:
        logger.error(f"Failed to read schema file {schema_path}: {read_error}")
        raise gr.Error(f"Could not read schema file: {read_error}")

    try:
        schema = json.loads(schema_text)
    except json.JSONDecodeError as parse_error:
        logger.error(f"Invalid JSON in schema file {schema_path}: {parse_error}")
        raise gr.Error(
            f"Schema file is not valid JSON (line {parse_error.lineno}, "
            f"column {parse_error.colno}): {parse_error.msg}"
        )

    if not isinstance(schema, dict):
        raise gr.Error("Schema file must contain a JSON object at the top level")

    file_name = os.path.basename(schema_path)
    property_count = len(schema.get('properties', {}))
    logger.info(f"Loaded output schema from {file_name} ({property_count} top-level properties)")

    status_html = banner(
        tone="ok",
        text=f"Loaded <b>{file_name}</b> — {property_count} top-level "
             f"propert{'y' if property_count == 1 else 'ies'}"
    )

    return schema_text, status_html


def handle_sample_selection(sample):
    """
    Load the selected sample's path, schema and ground truth

    Assigning the resolved path to input_image triggers input_image.change, which is
    what renders the preview - including PDF page navigation. Previewing is
    deliberately not duplicated here.

    A sample whose bundle holds no schema.json leaves the schema editor alone rather
    than clearing it: gr.Code cannot take a None value, and batch processing already
    treats a missing per-sample schema as "fall back to the editor" via
    load_sample_schema(). Clearing it here would silently discard an uploaded schema.

    Args:
        sample: Dropdown label of the selected sample

    Returns:
        Tuple of (sample label, document path, schema update, truth data,
        truth status HTML)
    """
    sample_result = on_sample_selected(sample)
    if sample_result and len(sample_result) >= 4:
        document_path, schema, truth_data, truth_status = sample_result
        schema_update = schema if schema is not None else gr.update()
        return sample, document_path, schema_update, truth_data, truth_status

    return sample, None, gr.update(), None, None


def setup_event_handlers(
    use_textract, use_bedrock, use_bda,
    sample_dropdown, input_image, s3_bucket, enable_structured_output, output_schema,
    refresh_samples, process_file_button, process_all_samples_button,
    bedrock_model, document_type, bda_s3_bucket,
    input_components, output_components, use_bda_blueprint,
    results_table, image_preview, pdf_preview, pdf_controls,
    prev_page_btn, page_info, next_page_btn, current_page, total_pages, current_pdf_path,
    textract_features, textract_queries, schema_upload, schema_status):
    """Setup all event handlers for the UI"""

    # Get global_status from input_components
    global_status = input_components.get("global_status", output_components[0])
    
    # Create state to track current sample name
    current_sample_name = gr.State("")
    
    
    # Get truth components
    truth_status = input_components.get("truth_status")
    truth_json = input_components.get("truth_json")
    
    # Get comparison components
    diff_engine = input_components.get("diff_engine")
    comparison_view = input_components.get("comparison_view")
    
    # Get JSON outputs for comparison
    textract_json = input_components.get("textract_json")
    bedrock_json = input_components.get("bedrock_json")
    bda_json = input_components.get("bda_json")
    
    sample_dropdown.change(
        fn=handle_sample_selection,
        inputs=sample_dropdown,
        outputs=[current_sample_name, input_image, output_schema, truth_json, truth_status]
    )

    refresh_samples.click(
        fn=lambda: gr.Dropdown(choices=list_sample_documents()),
        outputs=sample_dropdown
    )
    
    # Handle output schema upload
    schema_upload.change(
        fn=load_schema_from_file,
        inputs=schema_upload,
        outputs=[output_schema, schema_status]
    )

    # Handle file upload preview
    def handle_upload_preview(file):
        preview_result = handle_file_preview(file)
        image_prev, pdf_prev, controls_visible, curr_page, tot_pages, pdf_path = preview_result
        
        page_info_html = page_readout(current_page=curr_page, total_pages=tot_pages)

        return (image_prev, pdf_prev, gr.Column(visible=controls_visible),
               page_info_html, curr_page, tot_pages, pdf_path)
    
    input_image.change(
        fn=handle_upload_preview,
        inputs=input_image,
        outputs=[image_preview, pdf_preview, pdf_controls, page_info, current_page, total_pages, current_pdf_path]
    )
    
    # Handle PDF page navigation
    def go_to_prev_page(curr_page, tot_pages, pdf_path):
        new_page = max(0, curr_page - 1)
        image, info_html, page_info_html = navigate_pdf_page(pdf_path, new_page, tot_pages)
        return image, info_html, page_info_html, new_page
    
    def go_to_next_page(curr_page, tot_pages, pdf_path):
        new_page = min(tot_pages - 1, curr_page + 1)
        image, info_html, page_info_html = navigate_pdf_page(pdf_path, new_page, tot_pages)
        return image, info_html, page_info_html, new_page
    
    prev_page_btn.click(
        fn=go_to_prev_page,
        inputs=[current_page, total_pages, current_pdf_path],
        outputs=[image_preview, pdf_preview, page_info, current_page]
    )
    
    next_page_btn.click(
        fn=go_to_next_page,
        inputs=[current_page, total_pages, current_pdf_path],
        outputs=[image_preview, pdf_preview, page_info, current_page]
    )
    
    # Process single file - modified to include current_sample_name
    process_file_button.click(
        fn=process_image_with_engines,
        inputs=[
            input_image, use_textract, use_bedrock, use_bda,
            bedrock_model, bda_s3_bucket, s3_bucket,
            document_type, enable_structured_output, output_schema, use_bda_blueprint,
            current_sample_name,  # Pass the current sample name
            textract_features, textract_queries
        ],
        outputs=output_components + [results_table]
    )
    
    # Process all samples
    process_all_samples_button.click(
        fn=process_all_samples,
        inputs=[
            use_textract, use_bedrock, use_bda,
            bedrock_model, bda_s3_bucket, s3_bucket,
            document_type, enable_structured_output, output_schema, use_bda_blueprint,
            textract_features, textract_queries
        ],
        outputs=[global_status, results_table]
    )
    
    # Narrow or widen the comparison table without re-processing anything: every
    # engine's JSON is already on the page, so the filter is a pure re-render.
    def filter_comparison_view(engine_filter, truth, textract, bedrock, bda):
        """
        Re-render the Compare tab for the selected filter

        Args:
            engine_filter: Selected value of the filter dropdown, one of
                ENGINE_FILTER_CHOICES.
            truth: Ground truth JSON currently held by the Truth tab, or None.
            textract: Textract's extracted JSON, or None if it did not run.
            bedrock: Bedrock's extracted JSON, or None if it did not run.
            bda: BDA's extracted JSON, or None if it did not run.

        Returns:
            str: HTML for the comparison view.
        """
        return create_comparison_view(
            truth_data=truth,
            # Named rather than zipped against ENGINE_NAMES: create_comparison_view
            # orders the columns itself, so nothing here depends on that order.
            engine_json_by_name={
                "Textract": textract, "Bedrock": bedrock, "BDA": bda},
            engine_filter=engine_filter)

    diff_engine.change(
        fn=filter_comparison_view,
        inputs=[diff_engine, truth_json, textract_json, bedrock_json, bda_json],
        outputs=comparison_view
    )
    
    logger.info("Event handlers setup completed")
    
    # Return the state component to make it accessible in the app
    return current_sample_name
