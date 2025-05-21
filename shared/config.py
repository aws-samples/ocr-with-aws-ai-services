import logging
import gradio as gr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Image size constants
MAX_IMAGE_SIZE = 5 * 1024 * 1024 - 100000  # 5MB minus buffer for Bedrock

# Available Bedrock models
BEDROCK_MODELS = {
    "Claude 3.7 Sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "Claude 3.5 Sonnet": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "Amazon Nova Premier": "us.amazon.nova-premier-v1:0",
    "Amazon Nova Pro": "us.amazon.nova-pro-v1:0"
}

# Default model for post-processing
# POSTPROCESSING_MODEL = "anthropic.claude-3-5-haiku-20241022-v1:0"
POSTPROCESSING_MODEL = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

# API cost information
API_COSTS = {
    'textract_detect': 1.50 / 1000,  # Detect Document Text API: $1.50 per 1,000 Pages
    'textract_analyze_tables_forms': 65.00 / 1000,  # Tables + Forms: $65.00 per 1,000 Pages
    'bedrock': {
        # Claude models
        'us.anthropic.claude-3-7-sonnet-20250219-v1:0': {
            'input': 0.003 / 1000,   # $0.003 per 1,000 input tokens
            'output': 0.015 / 1000   # $0.015 per 1,000 output tokens
        },
        'us.anthropic.claude-3-5-sonnet-20240620-v1:0': {
            'input': 0.003 / 1000,   # $0.003 per 1,000 input tokens
            'output': 0.015 / 1000   # $0.015 per 1,000 output tokens
        },
        # Nova models
        'us.amazon.nova-premier-v1:0': {
            'input': 0.0025 / 1000,  # $0.0025 per 1,000 input tokens
            'output': 0.0125 / 1000  # $0.0125 per 1,000 output tokens
        },
        'us.amazon.nova-pro-v1:0': {
            'input': 0.0008 / 1000,  # $0.0008 per 1,000 input tokens
            'output': 0.0032 / 1000  # $0.0032 per 1,000 output tokens
        },
        'anthropic.claude-3-5-haiku-20241022-v1:0': {
            'input': 0.0008 / 1000,   # $0.0008 per 1,000 input tokens
            'output': 0.004 / 1000    # $0.004 per 1,000 output tokens
        }
    },
    'bda': {
        'standard': {
            'document': 0.010,  # $0.010 per page
            'image': 0.003      # $0.003 per image
        },
        'custom': {
            'document': 0.040,  # $0.040 per page
            'image': 0.005,     # $0.005 per image
            'extra_field': 0.0005  # $0.0005 per additional field (beyond 30)
        }
    }
}



# Status HTML templates
STATUS_HTML = {
    "processing": lambda engine: f"""<div style='padding: 10px; background-color: #3b5998; color: white; 
                                     border-radius: 5px; font-weight: bold;'>Processing with {engine}...</div>""",
    "completed": lambda engine, time, cost, token_info="": f"""<div style='padding: 10px; background-color: #2e7d32; color: white; 
                                         border-radius: 5px; font-weight: bold;'> {engine} completed in {time:.3f} seconds (Est. cost: ${cost:.6f}){token_info}</div>""",
    "error": lambda engine, time, error: f"""<div style='padding: 10px; background-color: #c62828; color: white; 
                                           border-radius: 5px; font-weight: bold;'> {engine} error ({time:.3f}s): {error}</div>""",
    "global_processing": lambda: """<div style='padding: 10px; background-color: #3b5998; color: white; 
                                    border-radius: 5px; font-weight: bold;'>Processing with selected engines...</div>""",
    "global_completed": lambda time, cost: f"""<div style='padding: 10px; background-color: #2e7d32; color: white; 
                                        border-radius: 5px; font-weight: bold;'>All processing completed in {time:.3f} seconds (Total est. cost: ${cost:.6f})</div>""",
    "global_partial": lambda success, total, time, cost: f"""<div style='padding: 10px; background-color: #ed6c02; color: white; 
                                                     border-radius: 5px; font-weight: bold;'>{success}/{total} engines completed in {time:.3f} seconds (Total est. cost: ${cost:.6f})</div>"""
}

# Use a simpler theme approach that works across Gradio versions
CUSTOM_THEME = gr.themes.Default()