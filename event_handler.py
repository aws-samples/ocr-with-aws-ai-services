import gradio as gr
from sample_handler import list_sample_images, on_sample_selected, process_all_samples
from processor import process_image_with_engines
from shared.comparison_utils import create_diff_view
from shared.config import logger

def setup_event_handlers(
    use_textract, use_bedrock, bedrock_options, 
    use_bda, bda_options,
    sample_dropdown, input_image, output_schema,
    refresh_samples, process_sample_button, process_all_button,
    bedrock_model, document_type, bda_s3_bucket,
    input_components, output_components, use_bda_blueprint,
    results_table):
    """Setup all event handlers for the UI"""
    
    # Get global_status from input_components
    global_status = input_components.get("global_status", output_components[0])
    
    # Create state to track current sample name
    current_sample_name = gr.State("")
    
    # Existing event handlers
    use_bedrock.change(
        fn=lambda checked: gr.Column(visible=checked),
        inputs=use_bedrock,
        outputs=bedrock_options
    )
    
    use_bda.change(
        fn=lambda checked: gr.Column(visible=checked),
        inputs=use_bda,
        outputs=bda_options
    )
    
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
    
    # Modified to capture and store the selected sample name
    sample_dropdown.change(
        fn=lambda sample: (sample, *on_sample_selected(sample)),
        inputs=sample_dropdown,
        outputs=[current_sample_name, input_image, output_schema, truth_json, truth_status]
    )
    
    refresh_samples.click(
        fn=lambda: gr.Dropdown(choices=list_sample_images()),
        outputs=sample_dropdown
    )
    
    # Process single sample - modified to include current_sample_name
    process_sample_button.click(
        fn=process_image_with_engines,
        inputs=[
            input_image, use_textract, use_bedrock, use_bda,
            bedrock_model, bda_s3_bucket,
            document_type, output_schema, use_bda_blueprint,
            current_sample_name  # Pass the current sample name
        ],
        outputs=output_components + [results_table]
    )
    
    # Process all samples
    process_all_button.click(
        fn=process_all_samples,
        inputs=[
            use_textract, use_bedrock, use_bda,
            bedrock_model, bda_s3_bucket,
            document_type, output_schema, use_bda_blueprint
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
