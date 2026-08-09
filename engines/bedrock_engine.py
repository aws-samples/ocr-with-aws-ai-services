import time
import json
import os
import tempfile
import numpy as np
from PIL import ImageDraw
from typing import Dict, Any, Tuple, Optional

from engines.base import OCREngine
from shared.aws_client import get_aws_client, describe_credential_error
from shared.image_utils import convert_to_bytes
from shared.config import (
    logger,
    API_COSTS,
    MAX_IMAGE_SIZE,
    LLM_MAX_OUTPUT_TOKENS,
    MANTLE_MODEL_IDS,
)
from shared.mantle_client import invoke_mantle_responses
from shared.pdf_render import compose_pdf_visualisation, count_pdf_pages
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
                # The bedrock-mantle models are not a boto3 service at all - botocore
                # ships no service model for that endpoint - so no client is built for
                # them and the call goes through shared.mantle_client instead.
                is_mantle_model = model_id in MANTLE_MODEL_IDS

                # Create Bedrock Runtime client
                bedrock_runtime = None if is_mantle_model else get_aws_client('bedrock-runtime')

                # Get appropriate prompt based on document type
                prompt = get_prompt_for_document_type(document_type)
                prompt += get_json_formatting_instructions(output_schema)
                system_prompt = OCR_SYSTEM_PROMPT

                # Create request payload based on file type
                if is_mantle_model:
                    # One request shape covers both PDFs and images here, because the
                    # Responses API distinguishes them by content-block type rather
                    # than by API operation the way bedrock-runtime does.
                    mantle_result = invoke_mantle_responses(
                        model_id=model_id,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
                        pdf_bytes=file_bytes if is_pdf else None,
                        # Unlike Converse, the Responses API wants an actual filename,
                        # so the extension is added back on here.
                        pdf_filename=f"{self._sanitize_document_name(image)}.pdf",
                        image_bytes=None if is_pdf else image_bytes,
                    )

                    extracted_text = self._strip_json_code_fence(
                        text=mantle_result["text"])
                    token_usage = mantle_result["token_usage"]

                    # Reasoning tokens are charged against max_output_tokens, so
                    # these models can exhaust the budget before emitting any text.
                    # Truncation is reported as an incomplete status rather than a
                    # stop reason; mantle_client normalises it to this field.
                    self._raise_if_truncated(stop_reason=mantle_result["stop_reason"])

                else:
                    # Converse for both PDFs and images.
                    #
                    # PDFs used to go through invoke_model with a native Anthropic
                    # body ("anthropic_version", a base64 "document" content block).
                    # That request is only valid for the Claude models: Nova rejects
                    # it outright with "extraneous key [type] is not permitted", so
                    # every non-Claude model in BEDROCK_MODELS failed on any PDF -
                    # which is the whole sample set in this repo. Converse takes one
                    # request shape for every model on bedrock-runtime, so the
                    # document block below works for all of them.
                    if is_pdf:
                        # Converse takes raw bytes, not base64, and requires a name.
                        # Bedrock restricts that name to alphanumerics, whitespace,
                        # hyphens, parentheses and brackets, hence the sanitising.
                        document_block = {
                            "document": {
                                "format": "pdf",
                                "name": self._sanitize_document_name(image),
                                "source": {"bytes": file_bytes}
                            }
                        }
                    else:
                        # convert_to_bytes() always encodes JPEG.
                        document_block = {
                            "image": {
                                "format": "jpeg",
                                "source": {"bytes": image_bytes}
                            }
                        }

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"text": prompt},
                                document_block
                            ]
                        }
                    ]

                    # Call the converse API with system messages (no citations or cache)
                    converse_args = {
                        "modelId": model_id,
                        "messages": messages,
                        "system": [{"text": system_prompt}],
                        # Converse caps output at 4096 tokens when inferenceConfig is
                        # omitted and truncates mid-string without raising. This one
                        # call does OCR and JSON structuring for the whole document,
                        # so its output is the largest the app produces.
                        "inferenceConfig": {"maxTokens": LLM_MAX_OUTPUT_TOKENS}
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

                    # Converse spells the same signal in camelCase.
                    self._raise_if_truncated(stop_reason=response.get('stopReason'))
                    
                    # Process response according to the provided format
                    if 'output' in response and 'message' in response['output']:
                        message = response['output']['message']
                        if 'content' in message:
                            for content_item in message['content']:
                                if 'text' in content_item:
                                    extracted_text += self._strip_json_code_fence(
                                        text=content_item['text'])
                
                # Create visual annotation based on file type
                if is_pdf:
                    # Render the real pages rather than the 400x600 black rectangle
                    # this used to draw. No boxes are passed because Converse
                    # returns none: its response content is text and tool-use
                    # only, with no geometry field anywhere in the API. Passing an
                    # empty list makes the captions read "Page n of N" with no box
                    # count, so nothing implies boxes were looked for and missed.
                    try:
                        page_count = count_pdf_pages(pdf_bytes=file_bytes)
                    except ValueError as count_error:
                        # The extraction has already succeeded and been paid for, so
                        # an unreadable page tree must not discard it. 0 means
                        # "unknown" to the results table, which leaves the per-page
                        # cells blank rather than claiming one page was read.
                        # Bedrock is billed per token, so no charge is misstated.
                        logger.error(
                            f"Could not count the PDF's pages, so per-page figures "
                            f"will be blank: {count_error}")
                        page_count = 0
                    annotated_image = np.array(
                        compose_pdf_visualisation(
                            pdf_bytes=file_bytes, boxes=[], item_noun="text lines"
                        )
                    )
                else:
                    page_count = 1
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
                    "pages": page_count,
                    "operation_type": "bedrock",
                    "file_type": "pdf" if is_pdf else "image"
                }
                
            except Exception as e:
                # Credential failures are reported by botocore without naming the
                # profile at fault, which is the one thing the user needs to know.
                error_message = describe_credential_error(e) or str(e)
                logger.error(f"Error in Bedrock processing: {error_message}")
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
                    "text": f"Amazon Bedrock Error: {error_message}",
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
    
    @staticmethod
    def _strip_json_code_fence(*, text: str) -> str:
        """
        Remove a markdown code fence wrapping a model's JSON response

        Models routinely return JSON inside a ```json ... ``` block despite being
        asked for JSON only, and json.loads() rejects the fence.

        Args:
            text (str): One text block from a model response.

        Returns:
            str: The block with any surrounding fence and whitespace removed.
        """
        stripped = text.strip()
        if stripped.startswith("```json"):
            stripped = stripped[7:]
        if stripped.startswith("```"):
            stripped = stripped[3:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        return stripped.strip()

    # The two APIs spell a hit output limit differently: Converse reports 'max_tokens'
    # as 'stopReason', and the Mantle Responses API reports 'max_output_tokens' as
    # incomplete_details.reason.
    TRUNCATED_STOP_REASONS = frozenset({'max_tokens', 'max_output_tokens'})

    def _raise_if_truncated(self, stop_reason: Optional[str]) -> None:
        """
        Fail loudly when the model's response was cut off at the output limit

        A truncated response is not valid JSON, but the only symptom downstream is a
        parse error on an unterminated string, which reads as a model quality problem
        rather than a configuration one.

        Args:
            stop_reason: The API's stop reason, or None when it did not report one

        Returns:
            None

        Raises:
            ValueError: If the response was truncated at the output limit
        """
        if stop_reason not in self.TRUNCATED_STOP_REASONS:
            return

        raise ValueError(
            f"The model hit the {LLM_MAX_OUTPUT_TOKENS}-token output limit and its "
            f"response was truncated, so the extracted text is incomplete and any "
            f"JSON in it is invalid. Raise OCR_LLM_MAX_OUTPUT_TOKENS, or process "
            f"fewer pages at a time."
        )

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
    
    def _sanitize_document_name(self, image) -> str:
        """
        Derive a Bedrock-legal document name from the input file's own name

        Converse requires a name on every document block and restricts it to
        alphanumerics, whitespace, hyphens, parentheses and square brackets, with no
        two consecutive whitespace characters. The name is not a filename and carries
        no extension: a period is not in the permitted set, so returning
        "claim-form.pdf" is rejected outright. This method used to append ".pdf" and
        was never called, so that was never discovered.

        The model is shown this name, so it is derived from the document's real name
        rather than being a fixed placeholder.

        Args:
            image: The engine's input - a file object with a `.name`, a path string,
                or anything else, in which case a generic name is used.

        Returns:
            str: A name Bedrock accepts, never empty.
        """
        import re
        import os

        # Get original filename
        original_name = None
        if hasattr(image, 'name') and image.name:
            original_name = os.path.basename(image.name)
        elif isinstance(image, str) and image:
            original_name = os.path.basename(image)

        # Strip the extension: its period is not a permitted character, and the model
        # has the document itself, so the format adds nothing.
        name_without_ext = os.path.splitext(original_name or "")[0]
        if not name_without_ext.strip():
            return "document"

        # Convert underscores and periods to hyphens, and anything else outside the
        # permitted set to a space.
        sanitized = re.sub(r'[_\.]', '-', name_without_ext)
        sanitized = re.sub(r'[^a-zA-Z0-9\s\-\(\)\[\]]', ' ', sanitized)

        # Collapse runs of whitespace - two in a row is rejected - and of hyphens.
        sanitized = re.sub(r'\s+', ' ', sanitized)
        sanitized = re.sub(r'-+', '-', sanitized)

        # A leading or trailing separator is legal but reads as a truncated name.
        sanitized = sanitized.strip(' -')

        if not sanitized:
            return "document"

        logger.info(f"Bedrock document name: '{sanitized}'")
        return sanitized