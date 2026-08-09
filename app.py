import gradio as gr
from shared.ui_theme import APP_CSS, CUSTOM_THEME, note
from shared.aws_client import log_aws_identity
from ui import create_input_panel, create_common_options_panel, create_results_panel, create_results_table
from event_handler import setup_event_handlers

def create_ocr_app():
    """Create the OCR application with all components"""
    # APP_CSS is injected here, once, and every colour the app chooses lives in it.
    # Nothing below writes an inline style attribute.
    with gr.Blocks(theme=CUSTOM_THEME, css=APP_CSS, title="Multi-Engine OCR") as app:
        gr.Markdown("# 📝 Multi-Engine OCR Application\n\nUpload an image containing text and select your preferred processing engines.")

        # A ticking clock used to live here, refreshed with every=1. It served no
        # purpose, and it re-rendered the page every second - which in the VS Code
        # browser reads as the UI flickering.

        with gr.Row():
            # Left column for inputs
            with gr.Column(scale=1):
                # Engine selection and the Process buttons are built by
                # create_input_panel() so they can sit directly under the preview, above
                # configuration, in the order the user works in.
                (input_panel, sample_dropdown, input_image, refresh_samples, image_preview, pdf_preview,
                 pdf_controls, prev_page_btn, page_info, next_page_btn, current_page, total_pages, current_pdf_path,
                 process_file_button, process_all_samples_button,
                 use_textract, use_bedrock, use_bda) = create_input_panel()

                # Create common options panel
                (common_options, s3_bucket, document_type, enable_structured_output, output_schema,
                 bedrock_model, bda_s3_bucket, use_bda_blueprint, textract_features,
                 textract_queries, schema_upload, schema_status) = create_common_options_panel()

            # Right column for results
            with gr.Column(scale=2):
                # Global status for all processing. This used to carry a class named
                # status-ready that no stylesheet ever defined.
                global_status = gr.HTML(note(text="Ready to process"), label="Status")
                results_table = create_results_table()
                
                # Results panel with tabs for each engine
                results_panel, input_components, output_components = create_results_panel()
        
        # Insert global status at the beginning of output components
        output_components.insert(0, global_status)
        
        # Setup event handlers
        setup_event_handlers(
            use_textract, use_bedrock, use_bda,
            sample_dropdown, input_image, s3_bucket, enable_structured_output, output_schema,
            refresh_samples, process_file_button, process_all_samples_button,
            bedrock_model, document_type, bda_s3_bucket,
            input_components, output_components, use_bda_blueprint,
            results_table, image_preview, pdf_preview, pdf_controls,
            prev_page_btn, page_info, next_page_btn, current_page, total_pages, current_pdf_path,
            textract_features, textract_queries, schema_upload, schema_status
        )
    
    return app


if __name__ == "__main__":
    # Log which profile/account the app will authenticate as before serving any
    # requests, so a wrong-profile misconfiguration is visible immediately rather
    # than as an opaque AccessDenied on the first Process File click.
    log_aws_identity()
    demo = create_ocr_app()
    demo.launch(share=False)
