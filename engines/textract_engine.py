import json
import time
import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, Any, Tuple, List, Optional
import boto3

from engines.base import OCREngine
from shared.aws_client import get_aws_client
from shared.image_utils import convert_to_bytes
from shared.config import logger, API_COSTS, POSTPROCESSING_MODEL

class TextractEngine(OCREngine):
    """
    Implementation of OCR engine using Amazon Textract
    """
    
    def __init__(self):
        super().__init__("Textract")
        
    def process_image(self, image, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an image with Amazon Textract
        
        Args:
            image: PIL Image, numpy array, or path to image
            options: Dictionary of options including:
                - output_schema: JSON schema for structuring the output
                
        Returns:
            Dictionary containing results including:
            - text: Extracted text
            - json: Structured JSON (if applicable)
            - image: Annotated image
            - process_time: Processing time
        """
        options = options or {}
        output_schema = options.get('output_schema')
        
        overall_start_time = time.time()
        # Set up timing context manager for accurate timing
        timing_ctx = self.get_timing_wrapper()
        
        image_bytes, img_pil = convert_to_bytes(image)
        
        # Start timing for all processing (including LLM)
        with timing_ctx:
            try:
                # Get Textract client - try both methods for backward compatibility
                try:
                    textract = get_aws_client('textract')
                except Exception as client_error:
                    logger.warning(f"Failed to get client via get_aws_client: {str(client_error)}")
                    # Fall back to direct boto3 client creation
                    textract = boto3.client('textract')
                
                # Call Textract's DetectDocumentText API
                operation_type = 'textract_detect'  # Track operation type for cost calculation
                response = textract.detect_document_text(Document={'Bytes': image_bytes})
                
                # Extract text and collect bounding boxes
                extracted_text = ""
                blocks_count = 0
                
                # Create a copy of the image for annotation
                annotated_img = img_pil.copy()
                draw = ImageDraw.Draw(annotated_img)
                width, height = annotated_img.size
                
                # Draw border and title
                draw.rectangle(
                    [(0, 0), (width, height)],
                    outline='#FF0000',
                    width=3
                )
                draw.text(
                    (20, 20),
                    f"Processed with Textract ({width}x{height})",
                    fill='#FF0000'
                )
                
                # Process blocks and draw bounding boxes
                for item in response["Blocks"]:
                    if item["BlockType"] == "LINE":
                        extracted_text += item["Text"] + "\n"
                        blocks_count += 1
                        
                        # Draw bounding box
                        if "Geometry" in item and "BoundingBox" in item["Geometry"]:
                            box = item["Geometry"]["BoundingBox"]
                            left = width * box["Left"]
                            top = height * box["Top"]
                            box_width = width * box["Width"]
                            box_height = height * box["Height"]
                            
                            draw.rectangle(
                                [(left, top), (left + box_width, top + box_height)],
                                outline='#FF0000',
                                width=2
                            )
                
                # Convert to numpy array for display
                annotated_image = np.array(annotated_img)
                
                # Process with LLM if needed - INSIDE the timing context to capture full processing time
                structured_json = None
                token_usage = None
                
                if extracted_text and output_schema:
                    try:
                        from shared.prompt_manager import process_text_with_llm
                        structured_json, token_usage = process_text_with_llm(extracted_text, output_schema)
                        logger.info("Successfully structured text with LLM")
                    except Exception as llm_error:
                        logger.error(f"Error in LLM JSON structuring: {str(llm_error)}")
                        structured_json = {"error": str(llm_error), "raw_text": extracted_text}
                
                # Calculate textract cost
                textract_cost = API_COSTS['textract_detect'] * 1  # 1 page
                
                logger.info(f"Textract processing completed in {timing_ctx.process_time:.2f} seconds")
                
                overall_process_time = time.time() - overall_start_time
                logger.info(f"Bedrock total processing time: {overall_process_time:.2f} seconds")

                # Return results as dictionary
                return {
                    "text": extracted_text,
                    "json": structured_json,
                    "image": annotated_image,
                    "process_time": overall_process_time,
                    "operation_type": operation_type,
                    "pages": 1,  # Assume one page by default
                    "blocks_count": blocks_count,
                    "token_usage": token_usage,
                    "textract_cost": textract_cost
                }
                
            except Exception as e:
                logger.error(f"Textract Error: {str(e)}")
                
                return {
                    "text": f"Textract Error: {str(e)}",
                    "json": None,
                    "image": None,
                    "process_time": overall_process_time, 
                    "operation_type": "error",
                    "pages": 0
                }

    
    def get_cost(self, result: Dict[str, Any]) -> Tuple[str, float]:
        """
        Calculate the cost of Textract processing
        
        Args:
            result: Result dictionary from process_image
            
        Returns:
            Tuple of (HTML representation of cost, actual cost value)
        """
        pages = result.get("pages", 1)
        operation_type = result.get("operation_type", "textract_detect")
        
        # Get base textract cost
        cost_per_page = API_COSTS[operation_type]
        textract_base_cost = cost_per_page * pages
        
        # Add LLM cost if applicable
        total_cost = textract_base_cost
        token_usage = result.get("token_usage")
        
        # Format HTML output
        if token_usage:
            from shared.cost_calculator import calculate_bedrock_cost
            llm_cost_html, llm_cost = calculate_bedrock_cost(POSTPROCESSING_MODEL, token_usage)
            total_cost += llm_cost
            
            html = f'''
            <div class="cost-container">
                <div class="cost-total">${total_cost:.6f} total</div>
                <div class="cost-breakdown">
                    <span>${textract_base_cost:.6f} for Textract ({cost_per_page:.6f} per page u00d7 {pages} pages)</span><br>
                    <span>${llm_cost:.6f} for LLM post-processing</span>
                </div>
            </div>
            '''
        else:
            html = f'''
            <div class="cost-container">
                <div class="cost-total">${total_cost:.6f} total</div>
                <div class="cost-breakdown">
                    <span>${cost_per_page:.6f} per page u00d7 {pages} pages</span>
                </div>
            </div>
            '''
        
        return html, total_cost