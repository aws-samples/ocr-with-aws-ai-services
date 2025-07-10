import os
import gradio as gr
import datetime
from shared.config import CUSTOM_THEME, logger
from ui import create_input_panel, create_common_options_panel, create_results_panel, create_results_table
from event_handler import setup_event_handlers

def create_ocr_app():
    """Create the OCR application with all components"""
    with gr.Blocks(theme=CUSTOM_THEME) as app:
        gr.Markdown("# 📝 Multi-Engine OCR Application\n\nUpload an image containing text and select your preferred processing engines.")
        
        # Current timestamp display
        timestamp_html = gr.HTML(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            label="Current Time",
            every=1
        )
        
        with gr.Row():
            # Left column for inputs
            with gr.Column(scale=1):
                # Create input panel
                (input_panel, sample_dropdown, input_image, refresh_samples, image_preview, pdf_preview,
                 pdf_controls, prev_page_btn, page_info, next_page_btn, current_page, total_pages, current_pdf_path,
                 process_file_button, process_all_samples_button) = create_input_panel()
                
                # Engine selection
                with gr.Row():
                    use_textract = gr.Checkbox(value=True, label="Use Textract")
                    use_bedrock = gr.Checkbox(value=False, label="Use Bedrock")
                    use_bda = gr.Checkbox(value=False, label="Use BDA")
                
                # Create common options panel
                common_options, s3_bucket, document_type, enable_structured_output, output_schema, bedrock_model, bda_s3_bucket, use_bda_blueprint = create_common_options_panel()
                
                # These buttons are now in the input panel under preview
            
            # Right column for results
            with gr.Column(scale=2):
                # Global status for all processing
                global_status = gr.HTML("<div class='status-ready'>Ready for processing</div>", label="Status")
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
            prev_page_btn, page_info, next_page_btn, current_page, total_pages, current_pdf_path
        )
    
    return app


if __name__ == "__main__":
    demo = create_ocr_app()
    demo.launch(share=False)
