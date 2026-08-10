import gradio as gr
from shared.config import (
    BEDROCK_MODELS,
    TEXTRACT_FEATURE_TYPES,
    DEFAULT_S3_BUCKET,
    DEFAULT_BDA_S3_BUCKET,
    POSTPROCESSING_MODEL,
)
from shared.results_table import rows_to_html
from shared.ui_theme import note, page_readout
from shared.comparison_utils import ALL_ENGINES_LABEL, ENGINE_FILTER_CHOICES
from sample_handler import list_sample_documents

# Default schema offered in the editor before a sample or a file is loaded
DEFAULT_OUTPUT_SCHEMA = (
    '{\n  "type": "object",\n  "properties": {\n    "text": {\n      "type": "string"\n    }\n  }\n}'
)

def create_input_panel():
    """Create the input panel with sample selection and image upload"""
    with gr.Column() as panel:
        with gr.Row():
            sample_dropdown = gr.Dropdown(
                choices=list_sample_documents(),
                label="Sample Documents",
                info="Every sample bundle under sample/, labelled by its path "
                     "relative to sample/. Or upload your own document above.",
                scale=4
            )
            refresh_samples = gr.Button("Refresh", scale=1)
            
        with gr.Row():
            # Left column for file upload
            with gr.Column(scale=1):
                input_image = gr.File(
                    file_types=["image", ".pdf"], 
                    label="Input Image or PDF"
                )
            
            # Right column for preview
            with gr.Column(scale=1):
                gr.Markdown("### 👁️ Preview")
                
                # PDF page navigation controls
                with gr.Column(visible=False) as pdf_controls:
                    page_info = gr.HTML(page_readout(current_page=0, total_pages=1))
                    with gr.Row():
                        prev_page_btn = gr.Button("◀ Previous", variant="secondary", size="sm", scale=1)
                        next_page_btn = gr.Button("Next ▶", variant="secondary", size="sm", scale=1)
                
                image_preview = gr.Image(
                    label="Image Preview",
                    show_label=False,
                    height=400,
                    visible=True
                )
                pdf_preview = gr.HTML(
                    label="PDF Preview",
                    value=note(text="Upload a file to see a preview", tall=True),
                    visible=False
                )

        # Engine selection and the two Process buttons sit at panel level, spanning the
        # full width, rather than inside the narrow preview column where the buttons
        # used to be squeezed. The order is the order the user works in: pick a
        # document, see it, choose engines, run. Configuration comes after all of it,
        # because its defaults are usually already right.
        #
        # The three checkboxes were previously unlabelled and stranded between the
        # preview and five sections of settings, so the one choice every run needs was
        # the least visible thing on the page.
        with gr.Group(elem_id="engine-select"):
            gr.Markdown("**Engines to run**")
            with gr.Row():
                use_textract = gr.Checkbox(value=True, label="Textract")
                use_bedrock = gr.Checkbox(value=False, label="Bedrock")
                use_bda = gr.Checkbox(value=False, label="BDA")

        with gr.Row():
            process_file_button = gr.Button("🚀 Process File", variant="primary", scale=2)
            process_all_button = gr.Button("📁 Process All Samples", variant="secondary", scale=1)

        # Hidden state for PDF navigation
        current_page = gr.State(0)
        total_pages = gr.State(1)
        current_pdf_path = gr.State(None)

    return (panel, sample_dropdown, input_image, refresh_samples, image_preview, pdf_preview,
            pdf_controls, prev_page_btn, page_info, next_page_btn, current_page, total_pages, current_pdf_path,
            process_file_button, process_all_button,
            use_textract, use_bedrock, use_bda)

def create_results_table():
    """Create the panel that displays comparative performance metrics.

    An HTML table rather than gr.Dataframe: each cost cell carries a `title` with
    the formula behind that number, which a DataFrame cell cannot hold. The markup
    comes from shared.results_table, and the widths are set in CSS rather than
    per-column here so all eight columns stay visible at any window size.

    Returns:
        gr.HTML: The component the processing generator writes the table into.
    """
    return gr.HTML(
        value=rows_to_html(rows=[]),
        label="Comparison Results",
        elem_id="results-table-panel"
    )

def create_common_options_panel():
    """Create common options panel for all engines"""
    # Five stacked Markdown headings previously put roughly two screens of settings
    # between the Process buttons and anything else, and every one of them was expanded
    # whether or not the selected engines used it. Each section is now a collapsible
    # accordion: Basic is open because it applies to every run, and the four
    # engine-specific ones start closed - their defaults are read from the environment
    # and are usually correct, so they are worth opening only when overriding something.
    with gr.Column() as panel:
        with gr.Accordion("🔧 Basic configuration", open=True):
            with gr.Row():
                document_type = gr.Dropdown(
                    choices=["generic", "form", "receipt", "table", "handwritten"],
                    value="generic",
                    label="Document Type",
                    info="Select the type of document to optimize prompt selection",
                    scale=1
                )

                enable_structured_output = gr.Checkbox(
                    label="Enable Structured Output",
                    value=True,
                    info="Enable structured JSON output processing (uses additional Bedrock API calls)",
                    scale=1
                )

        with gr.Accordion("🪣 S3 buckets", open=False):
            with gr.Row():
                s3_bucket = gr.Textbox(
                    label="S3 Bucket for Processing",
                    value=DEFAULT_S3_BUCKET,
                    placeholder="Enter your S3 bucket name",
                    info="Required for Textract PDFs; image calls use bytes directly. "
                         "Must be in the same account and region as your credentials. "
                         "Set OCR_S3_BUCKET to change the default.",
                    scale=2
                )

                bda_s3_bucket = gr.Textbox(
                    label="S3 Bucket for BDA Processing",
                    value=DEFAULT_BDA_S3_BUCKET,
                    placeholder="Enter your S3 bucket name for BDA",
                    info="Required for BDA input and output. "
                         "Set OCR_BDA_S3_BUCKET to change the default.",
                    scale=2
                )

        with gr.Accordion("📄 Textract features and queries", open=False):
            gr.Markdown(
                "*Leave the feature list empty for text detection only "
                "(DetectDocumentText / StartDocumentTextDetection, $0.0015 per page). "
                "Selecting one or more features switches to AnalyzeDocument / "
                "StartDocumentAnalysis, which costs more per page but returns form "
                "fields, tables, query answers and signatures.*"
            )

            with gr.Row():
                textract_features = gr.CheckboxGroup(
                    choices=TEXTRACT_FEATURE_TYPES,
                    value=[],
                    label="Textract Analysis Features",
                    info="Any combination is valid. OCR text is always returned. "
                         "Checkbox states need FORMS or TABLES. LAYOUT is free with TABLES.",
                    scale=2
                )

            with gr.Row():
                textract_queries = gr.Textbox(
                    label="Textract Queries",
                    placeholder="What is the diagnosis code?\nWhat is the date of hire?",
                    lines=3,
                    info="One question per line. Only used when QUERIES is selected above; "
                         "selecting QUERIES without any question raises an error.",
                    scale=2
                )

        with gr.Accordion("🤖 Bedrock model", open=False):
            bedrock_model = gr.Dropdown(
                choices=list(BEDROCK_MODELS.keys()),
                value="Claude Sonnet 5",
                label="Bedrock Model",
                info="Select an Amazon Bedrock model for processing. "
                     "See the Bedrock models table in README.md for prices."
            )

        with gr.Accordion("BDA options", open=False):
            use_bda_blueprint = gr.Checkbox(
                label="Use Custom Blueprint",
                value=False,
                # Names POSTPROCESSING_MODEL rather than hardcoding it, so the
                # tooltip cannot drift from the model actually used again.
                info=(
                    f"Enabled: BDA extracts against a custom blueprint built from "
                    f"the output schema. Disabled: BDA returns text, which "
                    f"{POSTPROCESSING_MODEL} then structures. Either way an output "
                    f"schema is required."
                )
            )

        with gr.Accordion("📋 Output schema", open=False):
            gr.Markdown("*Define the JSON schema for structured output, or upload one from a file*")

            # A multi-page bundle's schema.json runs to several kilobytes, and a real
            # claim form's is larger, which is not practical to paste into the editor
            # below.
            schema_upload = gr.File(
                file_types=[".json"],
                file_count="single",
                label="Upload Schema (.json)"
            )
            schema_status = gr.HTML("<div></div>", label="Schema Status")

            output_schema = gr.Code(
                language="json",
                label="Output Schema",
                value=DEFAULT_OUTPUT_SCHEMA
            )

    return (panel, s3_bucket, document_type, enable_structured_output, output_schema,
            bedrock_model, bda_s3_bucket, use_bda_blueprint, textract_features,
            textract_queries, schema_upload, schema_status)


def create_results_panel():
    """Create the results panel with tabs for each engine"""
    with gr.Column() as panel:
        with gr.Tabs():
            with gr.TabItem("Textract"):
                textract_status = gr.HTML("<div></div>", label="Status")
                textract_extracted_text = gr.Textbox(label="Extracted Text", lines=10, interactive=False)
                textract_json = gr.JSON(label="Raw JSON Output")
                textract_image = gr.Image(label="Visualization", interactive=False)
            
            with gr.TabItem("Bedrock"):
                bedrock_status = gr.HTML("<div></div>", label="Status")
                bedrock_extracted_text = gr.Textbox(label="Extracted Text", lines=10, interactive=False)
                bedrock_json = gr.JSON(label="Raw JSON Output")
                bedrock_image = gr.Image(label="Visualization", interactive=False)
                
                bedrock_cost = gr.HTML("<div></div>", label="API Cost")
                bedrock_token_usage = gr.JSON(label="Token Usage", visible=False)
            
            with gr.TabItem("BDA"):
                bda_status = gr.HTML("<div></div>", label="Status")
                bda_extracted_text = gr.Textbox(label="Extracted Text", lines=10, interactive=False)
                bda_json = gr.JSON(label="Raw JSON Output")
                bda_image = gr.Image(label="Visualization", interactive=False)
            
            with gr.TabItem("Truth"):
                truth_status = gr.HTML("<div></div>", label="Status")
                truth_json = gr.JSON(label="Ground Truth Data")
            
            with gr.TabItem("Compare"):
                # A filter, not a prerequisite: the table shows ground truth against
                # every engine that ran unless narrowed to one.
                diff_engine = gr.Dropdown(
                    choices=list(ENGINE_FILTER_CHOICES),
                    label="Columns to show",
                    value=ALL_ENGINES_LABEL,
                    info="Ground truth is compared against every engine that ran. "
                         "Pick one engine to narrow the table to its column."
                )
                comparison_view = gr.HTML(
                    note(text="Process a document with ground truth to see a "
                              "field-by-field comparison", tall=True)
                )
    
    # Organize components for easier access
    input_components = {
        "textract_status": textract_status,
        "textract_text": textract_extracted_text,
        "textract_json": textract_json,
        "textract_image": textract_image,
        "bedrock_status": bedrock_status,
        "bedrock_text": bedrock_extracted_text,
        "bedrock_json": bedrock_json,
        "bedrock_image": bedrock_image,
        "bedrock_cost": bedrock_cost,
        "bedrock_token_usage": bedrock_token_usage,
        "bda_status": bda_status,
        "bda_text": bda_extracted_text,
        "bda_json": bda_json,
        "bda_image": bda_image,
        "truth_status": truth_status,
        "truth_json": truth_json,
        "diff_engine": diff_engine,
        "comparison_view": comparison_view
    }
    
    output_components = [
        textract_status, textract_extracted_text, textract_json, textract_image,
        bedrock_status, bedrock_extracted_text, bedrock_json, bedrock_image,
        bedrock_cost, bedrock_token_usage,
        bda_status, bda_extracted_text, bda_json, bda_image,
        truth_status, truth_json,
        diff_engine, comparison_view
    ]
    
    return panel, input_components, output_components
