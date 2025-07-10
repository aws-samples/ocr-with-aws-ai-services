import gradio as gr
import pandas as pd
from shared.config import CUSTOM_THEME, BEDROCK_MODELS, STATUS_HTML
from sample_handler import list_sample_images

def create_input_panel():
    """Create the input panel with sample selection and image upload"""
    with gr.Column() as panel:
        with gr.Row():
            sample_dropdown = gr.Dropdown(
                choices=list_sample_images(),
                label="Sample Images",
                info="Select a sample image or upload your own",
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
                    page_info = gr.HTML(
                        "<div style='text-align: center; padding: 4px 0; font-weight: 500; color: #666; font-size: 14px;'>Page 1 of 1</div>"
                    )
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
                    value="<div style='text-align: center; padding: 50px; color: #666;'>Upload a PDF to see preview</div>",
                    visible=False
                )
                
                # Process buttons moved here, right under preview
                gr.Markdown("---")
                with gr.Row():
                    process_file_button = gr.Button("🚀 Process File", variant="primary", scale=2)
                    process_all_button = gr.Button("📁 Process All Samples", variant="secondary", scale=1)
                
                # Hidden state for PDF navigation
                current_page = gr.State(0)
                total_pages = gr.State(1)
                current_pdf_path = gr.State(None)
    
    return (panel, sample_dropdown, input_image, refresh_samples, image_preview, pdf_preview, 
            pdf_controls, prev_page_btn, page_info, next_page_btn, current_page, total_pages, current_pdf_path,
            process_file_button, process_all_button)

def create_results_table():
    """Create a table to display comparative performance metrics with dark mode support"""
    results_df = pd.DataFrame({
        "Engine": [],
        "Samples Processed": [],
        "Avg. Processing Time (s)": [],
        "Avg. Cost ($)": [],
        "Total Cost ($)": [],
        "Accuracy (%)" : []
    })
    
    results_table = gr.Dataframe(
        value=results_df,
        label="Comparison Results",
        interactive=False,
        wrap=True,
        column_widths=["130px", "130px", "150px", "110px", "110px", "110px"],
        elem_id="results-dataframe"
    )
    
    return results_table

def create_common_options_panel():
    """Create common options panel for all engines"""
    with gr.Column() as panel:
        # Basic Configuration Section
        gr.Markdown("### 🔧 Basic Configuration")
        
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
        
        # S3 Configuration Section
        gr.Markdown("### 🪣 S3 Configuration")
        
        with gr.Row():
            s3_bucket = gr.Textbox(
                label="S3 Bucket for Processing",
                value="ocr-with-ai-services-demo-bucket",
                placeholder="Enter your S3 bucket name",
                info="S3 bucket for uploading files for processing (required for all engines)",
                scale=2
            )
            
            bda_s3_bucket = gr.Textbox(
                label="S3 Bucket for BDA Processing",
                value="my-bda-demo-bucket",
                placeholder="Enter your S3 bucket name for BDA",
                info="S3 bucket specifically for BDA processing",
                scale=2
            )
        
        # Bedrock Configuration Section
        gr.Markdown("### 🤖 Bedrock Configuration")
        
        with gr.Row():
            bedrock_model = gr.Dropdown(
                choices=list(BEDROCK_MODELS.keys()),
                value="Claude Sonnet 4",
                label="Bedrock Model",
                info="Select an Amazon Bedrock model for processing",
                scale=2
            )
            
            use_bda_blueprint = gr.Checkbox(
                label="Use Custom Blueprint (BDA)",
                value=False,
                info="When enabled, creates a custom blueprint based on the output schema. When disabled, uses default extraction with Claude Haiku post-processing.",
                scale=1
            )
        
        # Output Schema Section
        gr.Markdown("### 📋 Output Schema Configuration")
        gr.Markdown("*Define the JSON schema for structured output*")
        output_schema = gr.Code(
            language="json",
            label="Output Schema",
            value="{\n  \"type\": \"object\",\n  \"properties\": {\n    \"text\": {\n      \"type\": \"string\"\n    }\n  }\n}"
        )
    
    return panel, s3_bucket, document_type, enable_structured_output, output_schema, bedrock_model, bda_s3_bucket, use_bda_blueprint


def create_results_panel():
    """Create the results panel with tabs for each engine"""
    with gr.Column() as panel:
        with gr.Tabs() as tabs:
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
                diff_engine = gr.Dropdown(
                    choices=["Textract", "Bedrock", "BDA"], 
                    label="Select engine to compare with ground truth",
                    value="Bedrock"
                )
                comparison_view = gr.HTML("<div>Select an engine and process an image to see comparison</div>")
    
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
