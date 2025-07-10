import time
import json
import base64
import os
import tempfile
import shutil
import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, Any, Tuple, Optional

from engines.base import OCREngine
from shared.aws_client import get_aws_client
from shared.image_utils import convert_to_bytes
from shared.config import logger, API_COSTS, MAX_IMAGE_SIZE
from shared.prompt_manager import get_prompt_for_document_type, get_json_formatting_instructions, OCR_SYSTEM_PROMPT

class BedrockEngine(OCREngine):
    """
    Implementation of OCR engine using Amazon Bedrock
    """
    
    def __init__(self):
        super().__init__("Bedrock")
    
    def process_image(self, image, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an image or PDF with Amazon Bedrock using converse API
        
        Args:
            image: PIL Image, numpy array, path to image, or PDF file
            options: Dictionary of options including:
                - model_id: Bedrock model ID
                - document_type: Type of document (generic, form, receipt, table, handwritten)
                - output_schema: JSON schema for structuring the output
                
        Returns:
            Dictionary containing results including:
            - text: Extracted text
            - image: Annotated image
            - process_time: Processing time
            - token_usage: Token usage information
            - model_id: Model ID used
        """

        options = options or {}
        model_id = options.get('model_id', '')
        document_type = options.get('document_type', 'generic')
        output_schema = options.get('output_schema')
        
        overall_start_time = time.time()
        # Set up timing context manager
        timing_ctx = self.get_timing_wrapper()
        
        # Check if input is a PDF file
        is_pdf = self._is_pdf_input(image)
        
        if is_pdf:
            # Handle PDF files - copy to temp with clean name
            temp_pdf_path = None
            try:
                # Get original file content
                if hasattr(image, 'name') and image.name:
                    with open(image.name, 'rb') as f:
                        file_bytes = f.read()
                elif isinstance(image, str):
                    with open(image, 'rb') as f:
                        file_bytes = f.read()
                else:
                    raise ValueError("PDF input must be a file path or file object")
                
                logger.info(f"PDF file size before API call: {len(file_bytes) / 1024:.2f}KB")
                
                # Create temporary PDF with clean name
                temp_pdf_path = self._create_temp_pdf(file_bytes)
                logger.info(f"Created temporary PDF: {temp_pdf_path}")
                
                img_pil = None  # No PIL image for PDF
                
            except Exception as e:
                logger.error(f"Error handling PDF file: {str(e)}")
                if temp_pdf_path and os.path.exists(temp_pdf_path):
                    os.unlink(temp_pdf_path)
                raise
        else:
            # Convert image to bytes OUTSIDE the timing context
            image_bytes, img_pil = convert_to_bytes(image, MAX_IMAGE_SIZE)
            logger.info(f"Image bytes size before API call: {len(image_bytes) / 1024:.2f}KB")
        
        # Start timing for the actual processing
        with timing_ctx:
            
            try:
                # Create Bedrock Runtime client
                bedrock_runtime = get_aws_client('bedrock-runtime')
                
                # Get appropriate prompt based on document type
                prompt = get_prompt_for_document_type(document_type)
                prompt += get_json_formatting_instructions(output_schema)
                system_prompt = OCR_SYSTEM_PROMPT
                    
                # Create request payload based on file type
                if is_pdf:
                    # For PDF files, use invoke_model API
                    pdf_base64 = base64.b64encode(file_bytes).decode('utf-8')
                    
                    # Create request body for invoke_model (no citations, no cache)
                    request_body = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 4000,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "document",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "application/pdf",
                                            "data": pdf_base64
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ]
                            }
                        ]
                    }
                    
                    # Add system prompt if provided
                    if system_prompt:
                        request_body["system"] = system_prompt
                    
                    # Use invoke_model for PDF
                    response = bedrock_runtime.invoke_model(
                        modelId=model_id,
                        body=json.dumps(request_body),
                        contentType='application/json',
                        accept='application/json'
                    )
                    
                    # Parse invoke_model response
                    response_body = json.loads(response['body'].read())
                    
                    # Extract text from invoke_model response
                    extracted_text = ""
                    for block in response_body.get('content', []):
                        if block.get('type') == 'text':
                            extracted_text += block.get('text', '')
                    
                    # Extract token usage for invoke_model
                    usage = response_body.get('usage', {})
                    token_usage = {
                        'inputTokens': usage.get('input_tokens', 0),
                        'outputTokens': usage.get('output_tokens', 0),
                        'totalTokens': usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
                    }
                    
                    logger.info(f"PDF processed with invoke_model - Input: {token_usage['inputTokens']}, Output: {token_usage['outputTokens']}")
                    
                else:
                    # For images, use converse API
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "text": prompt
                                },
                                {
                                    "image": {
                                        "format": "jpeg",
                                        "source": {
                                            "bytes": image_bytes
                                        }
                                    }
                                }
                            ]
                        }
                    ]
                    
                    # Call the converse API with system messages (no citations or cache)
                    converse_args = {
                        "modelId": model_id,
                        "messages": messages,
                        "system": [{"text": system_prompt}]
                    }
                    
                    response = bedrock_runtime.converse(**converse_args)
                    
                    # Extract text from converse response
                    extracted_text = ""
                    
                    # Extract token usage information
                    token_usage = {
                        'inputTokens': response.get('usage', {}).get('inputTokens', 0),
                        'outputTokens': response.get('usage', {}).get('outputTokens', 0),
                        'totalTokens': response.get('usage', {}).get('totalTokens', 0)
                    }
                    
                    logger.info(f"Token usage - Input: {token_usage['inputTokens']}, Output: {token_usage['outputTokens']}, Total: {token_usage['totalTokens']}")
                    
                    # Process response according to the provided format
                    if 'output' in response and 'message' in response['output']:
                        message = response['output']['message']
                        if 'content' in message:
                            for content_item in message['content']:
                                if 'text' in content_item:
                                    text = content_item['text']
                                    # Remove any markdown code block wrapping
                                    text = text.strip()
                                    if text.startswith("```json"):
                                        text = text[7:]
                                    if text.startswith("```"):
                                        text = text[3:]
                                    if text.endswith("```"):
                                        text = text[:-3]
                                    extracted_text += text.strip()
                
                # Create visual annotation based on file type
                if is_pdf:
                    # For PDF files, create a simple placeholder image
                    annotated_image = np.zeros((400, 600, 3), dtype=np.uint8)
                    from PIL import Image as PILImage, ImageDraw as PILImageDraw
                    pil_img = PILImage.fromarray(annotated_image)
                    draw = PILImageDraw.Draw(pil_img)
                    model_name = model_id.split(':')[0].split('.')[-1].upper()
                    draw.text((20, 20), f"PDF Processed with {model_name}", fill=(0, 204, 255))
                    draw.text((20, 50), f"Document Type: {document_type}", fill=(0, 204, 255))
                    annotated_image = np.array(pil_img)
                else:
                    # Create a visual indicator on the image
                    annotated_img_copy = img_pil.copy()
                    draw = ImageDraw.Draw(annotated_img_copy)
                    width, height = annotated_img_copy.size
                    
                    # Draw border
                    border_width = 10
                    draw.rectangle(
                        [(0, 0), (width, height)],
                        outline='#00CCFF',
                        width=border_width
                    )
                    
                    # Add model info text
                    model_name = model_id.split(':')[0].split('.')[-1].upper()
                    draw.text(
                        (20, 20),
                        f"Processed with {model_name} ({width}x{height})",
                        fill='#00CCFF'
                    )
                    
                    # Convert to numpy array
                    annotated_image = np.array(annotated_img_copy)
                
                # Try to parse the JSON
                structured_json = None
                try:
                    structured_json = json.loads(extracted_text)
                except json.JSONDecodeError:
                    structured_json = {"text": extracted_text}
                
                logger.info(f"Bedrock processing completed in {timing_ctx.process_time:.2f} seconds")
                overall_process_time = time.time() - overall_start_time
                logger.info(f"Bedrock total processing time: {overall_process_time:.2f} seconds")

                # Clean up temporary PDF file if created
                if is_pdf and temp_pdf_path and os.path.exists(temp_pdf_path):
                    try:
                        os.unlink(temp_pdf_path)
                        logger.info(f"Cleaned up temporary PDF: {temp_pdf_path}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to clean up temporary PDF: {cleanup_error}")

                # Return dictionary with all necessary information
                return {
                    "text": extracted_text,
                    "json": structured_json,
                    "image": annotated_image,
                    "process_time": overall_process_time,
                    "token_usage": token_usage,
                    "model_id": model_id,
                    "pages": 1,
                    "operation_type": "bedrock",
                    "file_type": "pdf" if is_pdf else "image"
                }
                
            except Exception as e:
                logger.error(f"Error in Bedrock processing: {str(e)}")
                overall_process_time = time.time() - overall_start_time
                logger.info(f"Bedrock error processing time: {overall_process_time:.2f} seconds")
                
                # Clean up temporary PDF file if created
                if is_pdf and temp_pdf_path and os.path.exists(temp_pdf_path):
                    try:
                        os.unlink(temp_pdf_path)
                        logger.info(f"Cleaned up temporary PDF after error: {temp_pdf_path}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to clean up temporary PDF after error: {cleanup_error}")
                
                return {
                    "text": f"Amazon Bedrock Error: {str(e)}",
                    "json": None,
                    "image": None,
                    "process_time": overall_process_time,
                    "token_usage": {'inputTokens': 0, 'outputTokens': 0, 'totalTokens': 0},
                    "model_id": model_id,
                    "operation_type": "error",
                    "pages": 0
                }
    
    def get_cost(self, result: Dict[str, Any]) -> Tuple[str, float]:
        """
        Calculate the cost for Bedrock processing
        
        Args:
            result: Result dictionary from process_image
            
        Returns:
            Tuple of (HTML representation of cost, actual cost value)
        """
        token_usage = result.get('token_usage')
        model_id = result.get('model_id', '')
        
        if not token_usage or model_id not in API_COSTS.get('bedrock', {}):
            return '<div class="cost-none">No cost data available</div>', 0.0
            
        # Get cost per token for the model from the correct structure
        model_costs = API_COSTS['bedrock'][model_id]
        cost_per_1k_input = model_costs['input']
        cost_per_1k_output = model_costs['output']
        
        # Calculate cost
        input_tokens = token_usage.get('inputTokens', 0)
        output_tokens = token_usage.get('outputTokens', 0)
        
        input_cost = (input_tokens / 1000) * cost_per_1k_input
        output_cost = (output_tokens / 1000) * cost_per_1k_output
        total_cost = input_cost + output_cost
        
        # Format HTML output
        html = f'''
        <div class="cost-container">
            <div class="cost-total">${total_cost:.6f} total</div>
            <div class="cost-breakdown">
                <span>${input_cost:.6f} for {input_tokens} input tokens (${cost_per_1k_input:.6f}/1K tokens)</span><br>
                <span>${output_cost:.6f} for {output_tokens} output tokens (${cost_per_1k_output:.6f}/1K tokens)</span>
            </div>
        </div>
        '''
        
        # Return both the HTML and the actual cost value
        return html, total_cost
    
    def _is_pdf_input(self, image):
        """Check if input is a PDF file"""
        if hasattr(image, 'name') and image.name and image.name.lower().endswith('.pdf'):
            return True
        elif isinstance(image, str) and image.lower().endswith('.pdf'):
            return True
        return False
    
    def _create_temp_pdf(self, file_bytes):
        """
        Create a temporary PDF file with clean name
        
        Args:
            file_bytes: PDF file content as bytes
            
        Returns:
            str: Path to temporary PDF file
        """
        # Create temporary file with clean name
        temp_dir = tempfile.gettempdir()
        temp_filename = f"bedrock_temp_{int(time.time())}_{os.getpid()}.pdf"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        try:
            with open(temp_path, 'wb') as temp_file:
                temp_file.write(file_bytes)
            logger.info(f"Created temporary PDF file: {temp_path}")
            return temp_path
        except Exception as e:
            logger.error(f"Failed to create temporary PDF: {str(e)}")
            raise Exception(f"Failed to create temporary PDF: {str(e)}")
    
    def _sanitize_document_name(self, image):
        """
        Sanitize document name to meet Bedrock requirements:
        - Only alphanumeric characters, whitespace, hyphens, parentheses, and square brackets
        - No more than one consecutive whitespace character
        """
        import re
        import os
        
        # Get original filename
        original_name = None
        if hasattr(image, 'name') and image.name:
            original_name = os.path.basename(image.name)
            logger.info(f"Original filename from image.name: {original_name}")
        elif isinstance(image, str) and image:
            original_name = os.path.basename(image)
            logger.info(f"Original filename from string: {original_name}")
        
        # If no valid filename found, use default
        if not original_name or not original_name.strip():
            logger.info("No valid filename found, using default")
            return "document.pdf"
        
        # Remove file extension for processing
        name_without_ext = os.path.splitext(original_name)[0]
        logger.info(f"Name without extension: '{name_without_ext}'")
        
        # If name without extension is empty, use default
        if not name_without_ext or not name_without_ext.strip():
            logger.info("Name without extension is empty, using default")
            return "document.pdf"
        
        # Replace invalid characters with spaces or hyphens
        # Keep only alphanumeric, whitespace, hyphens, parentheses, and square brackets
        # Convert underscores and dots to hyphens, other invalid chars to spaces
        sanitized = re.sub(r'[_\.]', '-', name_without_ext)  # Convert _ and . to -
        sanitized = re.sub(r'[^a-zA-Z0-9\s\-\(\)\[\]]', ' ', sanitized)  # Convert other invalid chars to space
        
        # Replace multiple consecutive whitespace with single space
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Replace multiple consecutive hyphens with single hyphen
        sanitized = re.sub(r'-+', '-', sanitized)
        
        # Trim whitespace and hyphens from start and end
        sanitized = sanitized.strip(' -')
        
        # If name is empty after sanitization, use default
        if not sanitized:
            logger.info("Name is empty after sanitization, using default")
            sanitized = "document"
        
        # Add .pdf extension back
        final_name = f"{sanitized}.pdf"
        logger.info(f"Final sanitized document name: '{final_name}'")
        return final_name