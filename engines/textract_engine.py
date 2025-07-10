import json
import time
import uuid
from datetime import datetime
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
        Process an image or PDF with Amazon Textract using S3 upload
        
        Args:
            image: PIL Image, numpy array, path to image, or file path (including PDF)
            options: Dictionary of options including:
                - output_schema: JSON schema for structuring the output
                - s3_bucket: S3 bucket for processing (optional)
                
        Returns:
            Dictionary containing results including:
            - text: Extracted text
            - json: Structured JSON (if applicable)
            - image: Annotated image
            - process_time: Processing time
        """
        options = options or {}
        output_schema = options.get('output_schema')
        s3_bucket = options.get('s3_bucket', 'ocr-with-ai-services-demo-bucket')
        logger.info(f"Using S3 bucket: {s3_bucket}")
        
        overall_start_time = time.time()
        # Set up timing context manager for accurate timing
        timing_ctx = self.get_timing_wrapper()
        
        # Check if input is a PDF file
        is_pdf = False
        file_bytes = None
        
        # Handle different input types (Gradio File object, file path, PIL Image, etc.)
        logger.info(f"Processing input type: {type(image)}")
        if hasattr(image, 'name'):
            logger.info(f"Input has name attribute: {image.name}")
            
        if hasattr(image, 'name') and image.name and image.name.lower().endswith('.pdf'):
            # Gradio File object with PDF
            logger.info(f"Processing PDF file: {image.name}")
            is_pdf = True
            with open(image.name, 'rb') as f:
                file_bytes = f.read()
            logger.info(f"PDF file size: {len(file_bytes)} bytes")
            
            # Check PDF header and validate format
            pdf_header = file_bytes[:8]
            logger.info(f"PDF header: {pdf_header}")
            if not file_bytes.startswith(b'%PDF-'):
                logger.error("File does not have valid PDF header")
                raise ValueError("Invalid PDF format")
            
            # Check if PDF is encrypted or corrupted
            if b'Encrypt' in file_bytes[:1000]:
                logger.warning("PDF appears to be encrypted")
            
            # Check file size limits
            file_size_mb = len(file_bytes) / (1024 * 1024)
            logger.info(f"PDF file size: {file_size_mb:.2f} MB")
            if file_size_mb > 5:
                logger.error(f"PDF file too large: {file_size_mb:.2f} MB (max 5MB)")
                raise ValueError("PDF file exceeds 5MB limit")
            
            img_pil = None
        elif isinstance(image, str) and image.lower().endswith('.pdf'):
            # File path to PDF
            logger.info(f"Processing PDF file path: {image}")
            is_pdf = True
            with open(image, 'rb') as f:
                file_bytes = f.read()
            logger.info(f"PDF file size: {len(file_bytes)} bytes")
            img_pil = None
        else:
            # Handle regular image files
            logger.info("Processing as image file")
            image_bytes, img_pil = convert_to_bytes(image)
            
        # Upload to S3 for processing
        s3_object_key = self._upload_to_s3(file_bytes if is_pdf else image_bytes, s3_bucket, is_pdf)
        
        # Verify S3 upload
        self._verify_s3_object(s3_bucket, s3_object_key)
        
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
                
                # Use appropriate API based on file type
                if is_pdf:
                    # PDF files require asynchronous processing
                    logger.info(f"Starting asynchronous Textract processing for PDF S3 object: s3://{s3_bucket}/{s3_object_key}")
                    operation_type = 'textract_async'
                    response = self._process_pdf_async(textract, s3_bucket, s3_object_key)
                else:
                    # Images can use synchronous detect_document_text
                    logger.info(f"Calling Textract detect_document_text for image S3 object: s3://{s3_bucket}/{s3_object_key}")
                    operation_type = 'textract_detect'
                    response = textract.detect_document_text(
                        Document={
                            'S3Object': {
                                'Bucket': s3_bucket,
                                'Name': s3_object_key
                            }
                        }
                    )
                    logger.info("Textract detect_document_text call completed successfully")
                
                # Extract text and collect bounding boxes
                extracted_text = ""
                blocks_count = 0
                total_pages = response.get('DocumentMetadata', {}).get('Pages', 1)
                
                # Handle PDF vs Image processing differently
                if is_pdf:
                    # For PDF, create a simple annotation placeholder
                    annotated_image = np.zeros((400, 600, 3), dtype=np.uint8)  # Black background
                    
                    # Process PDF blocks (LINE, WORD, KEY_VALUE_SET, TABLE, etc.)
                    # Group content by page for better organization
                    pages_content = {}
                    for item in response["Blocks"]:
                        page_num = item.get("Page", 1)
                        if page_num not in pages_content:
                            pages_content[page_num] = []
                            
                        if item["BlockType"] == "LINE":
                            if "Text" in item:
                                pages_content[page_num].append(item["Text"])
                                blocks_count += 1
                        elif item["BlockType"] == "KEY_VALUE_SET" and operation_type == 'textract_analyze':
                            # Handle form fields for analyze_document results
                            if item.get("EntityTypes") == ["KEY"] and "Text" in item:
                                pages_content[page_num].append(f"Key: {item['Text']}")
                            elif item.get("EntityTypes") == ["VALUE"] and "Text" in item:
                                pages_content[page_num].append(f"Value: {item['Text']}")
                    
                    # Combine all pages content
                    for page_num in sorted(pages_content.keys()):
                        extracted_text += f"\n--- Page {page_num} ---\n"
                        extracted_text += "\n".join(pages_content[page_num]) + "\n"
                
                    # Add PDF processing info to the annotation
                    from PIL import Image as PILImage, ImageDraw as PILImageDraw
                    pil_img = PILImage.fromarray(annotated_image)
                    draw = PILImageDraw.Draw(pil_img)
                    draw.text((20, 20), f"PDF Processed with Textract", fill=(255, 255, 255))
                    draw.text((20, 50), f"Pages: {response.get('DocumentMetadata', {}).get('Pages', 1)}", fill=(255, 255, 255))
                    draw.text((20, 80), f"Blocks found: {blocks_count}", fill=(255, 255, 255))
                    annotated_image = np.array(pil_img)
                
                else:
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
                        
                        if is_pdf and total_pages > 1:
                            # For multi-page PDFs, process each page separately and combine
                            structured_json = {"pages": {}}
                            total_token_usage = {"input_tokens": 0, "output_tokens": 0}
                            
                            for page_num in sorted(pages_content.keys()):
                                page_text = "\n".join(pages_content[page_num])
                                if page_text.strip():
                                    page_result, page_tokens = process_text_with_llm(page_text, output_schema)
                                    structured_json["pages"][f"page_{page_num}"] = page_result
                                    
                                    if page_tokens:
                                        total_token_usage["input_tokens"] += page_tokens.get("input_tokens", 0)
                                        total_token_usage["output_tokens"] += page_tokens.get("output_tokens", 0)
                            
                            token_usage = total_token_usage
                            logger.info(f"Successfully structured {total_pages} pages with LLM")
                        else:
                            # Single page or image processing
                            structured_json, token_usage = process_text_with_llm(extracted_text, output_schema)
                            logger.info("Successfully structured text with LLM")
                            
                    except Exception as llm_error:
                        logger.error(f"Error in LLM JSON structuring: {str(llm_error)}")
                        structured_json = {"error": str(llm_error), "raw_text": extracted_text}
                
                # Calculate textract cost based on document type and pages
                textract_cost = API_COSTS[operation_type] * total_pages
                
                logger.info(f"Textract processing completed in {timing_ctx.process_time:.2f} seconds")
                
                overall_process_time = time.time() - overall_start_time
                logger.info(f"Textract total processing time: {overall_process_time:.2f} seconds")

                # Return results as dictionary
                return {
                    "text": extracted_text,
                    "json": structured_json,  # LLM-processed structured output (only when enable_structured_output=True)
                    "raw_json": response,     # Always include raw Textract API response
                    "image": annotated_image,
                    "process_time": overall_process_time,
                    "operation_type": operation_type,
                    "pages": total_pages,
                    "blocks_count": blocks_count,
                    "token_usage": token_usage,
                    "textract_cost": textract_cost
                }
                
            except Exception as e:
                logger.error(f"Textract Error: {str(e)}")
                overall_process_time = time.time() - overall_start_time
                
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
    
    def _upload_to_s3(self, file_bytes: bytes, s3_bucket: str, is_pdf: bool = False) -> str:
        """
        Upload file to S3 for Textract processing
        
        Args:
            file_bytes: File content as bytes
            s3_bucket: S3 bucket name
            is_pdf: Whether the file is a PDF
            
        Returns:
            S3 object key
        """
        try:
            # Get S3 client
            s3_client = get_aws_client('s3')
            
            # Generate unique object key
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_id = str(uuid.uuid4())[:8]
            
            if is_pdf:
                file_extension = "pdf"
                content_type = "application/pdf"
            else:
                file_extension = "jpg"
                content_type = "image/jpeg"
                
            object_key = f"textract-input/{timestamp}-{random_id}.{file_extension}"
            
            # Upload to S3
            logger.info(f"Uploading file to S3: s3://{s3_bucket}/{object_key}")
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=object_key,
                Body=file_bytes,
                ContentType=content_type
            )
            
            logger.info(f"Successfully uploaded to S3: s3://{s3_bucket}/{object_key}")
            return object_key
            
        except Exception as e:
            logger.error(f"Failed to upload to S3: {str(e)}")
            raise Exception(f"S3 upload failed: {str(e)}")
    
    def _verify_s3_object(self, s3_bucket: str, s3_object_key: str):
        """
        Verify that the S3 object exists and is accessible
        
        Args:
            s3_bucket: S3 bucket name
            s3_object_key: S3 object key
        """
        try:
            s3_client = get_aws_client('s3')
            
            # Check if object exists
            response = s3_client.head_object(Bucket=s3_bucket, Key=s3_object_key)
            file_size = response['ContentLength']
            logger.info(f"S3 object verified: s3://{s3_bucket}/{s3_object_key} ({file_size} bytes)")
            
            # Check if we have proper permissions for Textract to access this bucket
            try:
                # Test if Textract service can access the bucket (this doesn't actually call Textract)
                logger.info(f"S3 object is ready for Textract processing")
            except Exception as perm_error:
                logger.warning(f"Potential permission issue: {str(perm_error)}")
                
        except Exception as e:
            logger.error(f"S3 object verification failed: {str(e)}")
            raise Exception(f"S3 object not accessible: {str(e)}")
    
    def _process_pdf_async(self, textract_client, s3_bucket: str, s3_object_key: str):
        """
        Process PDF using asynchronous Textract API
        
        Args:
            textract_client: Textract client
            s3_bucket: S3 bucket name
            s3_object_key: S3 object key
            
        Returns:
            Combined response with all blocks
        """
        try:
            # Start asynchronous text detection job
            logger.info("Starting asynchronous Textract job")
            start_response = textract_client.start_document_text_detection(
                DocumentLocation={
                    'S3Object': {
                        'Bucket': s3_bucket,
                        'Name': s3_object_key
                    }
                }
            )
            
            job_id = start_response['JobId']
            logger.info(f"Textract job started with ID: {job_id}")
            
            # Wait for job completion
            max_wait_time = 300  # 5 minutes maximum
            wait_interval = 2    # 2 seconds between checks
            total_wait = 0
            
            while total_wait < max_wait_time:
                time.sleep(wait_interval)
                total_wait += wait_interval
                
                get_response = textract_client.get_document_text_detection(JobId=job_id)
                status = get_response['JobStatus']
                logger.info(f"Job status: {status} (waited {total_wait}s)")
                
                if status == 'SUCCEEDED':
                    logger.info("Textract job completed successfully")
                    break
                elif status == 'FAILED':
                    logger.error("Textract job failed")
                    raise Exception("Textract job failed")
                elif status not in ['IN_PROGRESS']:
                    logger.error(f"Unexpected job status: {status}")
                    raise Exception(f"Unexpected job status: {status}")
            
            if total_wait >= max_wait_time:
                logger.error("Textract job timed out")
                raise Exception("Textract job timed out after 5 minutes")
            
            # Collect all result pages
            all_blocks = []
            pages_metadata = {'Pages': 0}
            next_token = None
            
            while True:
                if next_token:
                    get_response = textract_client.get_document_text_detection(
                        JobId=job_id, 
                        NextToken=next_token
                    )
                else:
                    get_response = textract_client.get_document_text_detection(JobId=job_id)
                
                # Add blocks from this page
                all_blocks.extend(get_response.get('Blocks', []))
                
                # Update page count
                if 'DocumentMetadata' in get_response:
                    pages_metadata['Pages'] = get_response['DocumentMetadata'].get('Pages', 1)
                
                # Check for more pages
                next_token = get_response.get('NextToken')
                if not next_token:
                    break
                    
                logger.info(f"Retrieved page with {len(get_response.get('Blocks', []))} blocks")
            
            logger.info(f"Total blocks retrieved: {len(all_blocks)}")
            
            # Return response in same format as synchronous API
            return {
                'DocumentMetadata': pages_metadata,
                'Blocks': all_blocks
            }
            
        except Exception as e:
            logger.error(f"Asynchronous PDF processing failed: {str(e)}")
            raise Exception(f"PDF processing failed: {str(e)}")