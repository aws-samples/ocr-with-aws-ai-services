import time
import json
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
        Process an image with Amazon Bedrock using converse API
        
        Args:
            image: PIL Image, numpy array, or path to image
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
                    
                # Create request payload with binary image data
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
                
                # Add system prompt
                system_messages = [{"text": system_prompt}]
                logger.info(f"Using document type: {document_type}")
                
                logger.info(f"Calling Bedrock with model: {model_id}")
                
                # Call the converse API with system messages
                converse_args = {
                    "modelId": model_id,
                    "messages": messages,
                    "system": system_messages
                }
                    
                response = bedrock_runtime.converse(**converse_args)
                
                # Extract text from response based on the correct format
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

                # Return dictionary with all necessary information
                return {
                    "text": extracted_text,
                    "json": structured_json,
                    "image": annotated_image,
                    "process_time": overall_process_time,
                    "token_usage": token_usage,
                    "model_id": model_id,
                    "pages": 1,
                    "operation_type": "bedrock"
                }
                
            except Exception as e:
                logger.error(f"Error in Bedrock processing: {str(e)}")
                overall_process_time = time.time() - overall_start_time
                logger.info(f"Bedrock error processing time: {overall_process_time:.2f} seconds")
                logger.error(f"Error in Bedrock processing: {str(e)}")
                
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