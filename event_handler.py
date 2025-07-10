import gradio as gr
from sample_handler import list_sample_images, on_sample_selected, process_all_samples
from processor import process_image_with_engines
from shared.comparison_utils import create_diff_view
from shared.config import logger
from preview_handler import handle_file_preview, handle_sample_preview, navigate_pdf_page

def setup_event_handlers(
    use_textract, use_bedrock, use_bda,
    sample_dropdown, input_image, s3_bucket, enable_structured_output, output_schema,
    refresh_samples, process_file_button, process_all_samples_button,
    bedrock_model, document_type, bda_s3_bucket,
    input_components, output_components, use_bda_blueprint,
    results_table, image_preview, pdf_preview, pdf_controls, 
    prev_page_btn, page_info, next_page_btn, current_page, total_pages, current_pdf_path):
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
    
    # Modified to capture and store the selected sample name, and handle preview
    def handle_sample_selection(sample):
        sample_result = on_sample_selected(sample)
        if sample_result and len(sample_result) >= 4:
            image_path, schema, truth_data, truth_status = sample_result
            # Handle preview for the selected sample
            preview_result = handle_sample_preview(image_path)
            return (sample, image_path, schema, truth_data, truth_status, 
                   preview_result[0], preview_result[1])
        else:
            return (sample, None, None, None, None, None, 
                   "<div style='text-align: center; padding: 50px; color: #666;'>No sample selected</div>")
    
    sample_dropdown.change(
        fn=handle_sample_selection,
        inputs=sample_dropdown,
        outputs=[current_sample_name, input_image, output_schema, truth_json, truth_status, 
                image_preview, pdf_preview]
    )
    
    refresh_samples.click(
        fn=lambda: gr.Dropdown(choices=list_sample_images()),
        outputs=sample_dropdown
    )
    
    # Handle file upload preview
    def handle_upload_preview(file):
        preview_result = handle_file_preview(file)
        image_prev, pdf_prev, controls_visible, curr_page, tot_pages, pdf_path = preview_result
        
        # Update page info display
        page_info_html = f"<div style='text-align: center; padding: 8px;'>Page {curr_page + 1} of {tot_pages}</div>"
        
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
            current_sample_name  # Pass the current sample name
        ],
        outputs=output_components + [results_table]
    )
    
    # Process all samples
    process_all_samples_button.click(
        fn=process_all_samples,
        inputs=[
            use_textract, use_bedrock, use_bda,
            bedrock_model, bda_s3_bucket, s3_bucket,
            document_type, enable_structured_output, output_schema, use_bda_blueprint
        ],
        outputs=[global_status, results_table]
    )
    
    # Add event handler for comparison view updates
    diff_engine.change(
        fn=lambda engine, truth, textract, bedrock, bda: create_diff_view(
            truth or {}, 
            {"Textract": textract, "Bedrock": bedrock, "BDA": bda}.get(engine, {}) or {}
        ),
        inputs=[diff_engine, truth_json, textract_json, bedrock_json, bda_json],
        outputs=comparison_view
    )
    
    logger.info("Event handlers setup completed")
    
    # Return the state component to make it accessible in the app
    return current_sample_name
