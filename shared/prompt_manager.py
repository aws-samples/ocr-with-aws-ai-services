
# prompt_manager.py
import json
import time
import logging

from .aws_client import get_aws_client
from .config import POSTPROCESSING_MODEL, logger

OCR_SYSTEM_PROMPT = """
You are an expert OCR and document analysis system. Extract all text and information from the image 
with high accuracy. Organize the results clearly and preserve the structure of the document. 
Pay special attention to tables, forms, and formatted data. Use the provided schema to structure 
your response if one is specified.
"""

DOCUMENT_TYPE_PROMPT_MAP = {
    "generic": """
Extract all the text from the provided image. Preserve the original structure and formatting as much as possible.
For tables, maintain the rows and columns relationships. For forms, associate labels with their corresponding values.
For multi-column layouts, process each column in order from left to right.
""",
    "form": """
This image contains a form or receipt. Please extract all fields and their values, preserving the 
relationships between labels and data. Format the output as key-value pairs.
""",
    "receipt": """
This image contains a form or receipt. Please extract all fields and their values, preserving the 
relationships between labels and data. Format the output as key-value pairs.
""",
    "table": """
This image contains tabular data. Extract the complete table including headers and all cell values.
Preserve the row and column structure precisely.
""",
    "handwritten": """
This image contains handwritten text. Try to extract all the handwritten content with high accuracy.
If parts are illegible, indicate that with [illegible].
"""
}

JSON_SYSTEM_PROMPT = """
You are an AI assistant specialized in structuring extracted document text into JSON format.
Your task is to analyze the document content and create the most appropriate JSON structure.
Focus on capturing all relevant information in a logical hierarchy.
Ensure the JSON is valid and represents the information accurately.
IMPORTANT: When returning JSON, do not use code blocks, backticks or markdown formatting.
"""

JSON_TEMPLATE_NO_SCHEMA = """
Please analyze the content and create an appropriate JSON schema based on the document type.
Follow these guidelines:

1. If it's an invoice or receipt:
   - Extract date, invoice number, total amount, tax, vendor details
   - Itemize line items with quantity, description, unit price and subtotal
   
2. If it's a form or application:
   - Identify all field names and their corresponding values
   - Group related fields into logical sections
   
3. If it's a table:
   - Preserve the tabular structure with rows and columns
   - Include column headers as keys
   
4. If it's an ID or card:
   - Extract all personal information fields
   - Include issuer, ID number, date fields, and other relevant data
   
5. For general documents:
   - Extract titles, subtitles, and section headers
   - Organize content into a logical hierarchy
   - Include metadata such as dates, reference numbers, etc.

IMPORTANT: Return ONLY the JSON data without any markdown code blocks, backticks or formatting. Ensure the output is valid JSON.
"""

def get_json_formatting_instructions(output_schema=None):
    """
    Get JSON formatting instructions based on whether a schema is provided
    
    Args:
        output_schema: Optional JSON schema to conform to
        
    Returns:
        str: The appropriate JSON formatting instructions
    """
    if output_schema:
        return f"\n\nPlease format the output according to this JSON schema: {output_schema}\nIMPORTANT: Return ONLY the JSON data without any markdown code blocks, backticks or formatting. Ensure all quotes and special characters are properly escaped."
    else:
        return "\n\n" + JSON_TEMPLATE_NO_SCHEMA

def get_prompt_for_document_type(document_type="generic"):
    """
    Get appropriate user prompt based on document type
    
    Args:
        document_type: Type of document (generic, form, receipt, table, handwritten)
        
    Returns:
        str: The appropriate user prompt
    """
    return DOCUMENT_TYPE_PROMPT_MAP.get(document_type.lower(), DOCUMENT_TYPE_PROMPT_MAP["generic"])

def process_text_with_llm(text, output_schema=None):
    """
    Process extracted text with LLM to structure it as JSON
    
    Args:
        text: Extracted text to process
        output_schema: Optional JSON schema to conform to
        
    Returns:
        Tuple of (structured JSON, token usage)
    """
    # Start timing
    start_time = time.time()
    
    # Create Bedrock Runtime client
    bedrock_runtime = get_aws_client('bedrock-runtime')
    
    try:
        # Prepare the prompt for JSON conversion
        prompt = f"""Please convert the following extracted text into structured JSON format:

{text}
"""
        if output_schema:
            prompt += f"\n\nPlease format the output according to this JSON schema: {output_schema}"
            prompt += "\nIMPORTANT: Return ONLY the JSON data without any markdown code blocks, backticks or formatting. Ensure all quotes and special characters are properly escaped."
        else:
            prompt += "\n\n" + JSON_TEMPLATE_NO_SCHEMA
        
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        system_messages = [{"text": JSON_SYSTEM_PROMPT}]
        
        logger.info(f"Calling Bedrock with model: {POSTPROCESSING_MODEL}")
        
        response = bedrock_runtime.converse(
            modelId=POSTPROCESSING_MODEL,
            messages=messages,
            system=system_messages
        )
        
        # Extract text and token usage
        structured_text = ""
        token_usage = {
            'inputTokens': response.get('usage', {}).get('inputTokens', 0),
            'outputTokens': response.get('usage', {}).get('outputTokens', 0),
            'totalTokens': response.get('usage', {}).get('totalTokens', 0)
        }
        
        logger.info(f"Token usage - Input: {token_usage['inputTokens']}, Output: {token_usage['outputTokens']}")
        
        if 'output' in response and 'message' in response['output']:
            message = response['output']['message']
            if 'content' in message:
                for content_item in message['content']:
                    if 'text' in content_item:
                        text = content_item['text'].strip()
                        if text.startswith("```json"):
                            text = text[7:]
                        elif text.startswith("```"):
                            text = text[3:]
                        if text.endswith("```"):
                            text = text[:-3]
                        structured_text += text.strip()
        
        try:
            structured_json = json.loads(structured_text)
            json_process_time = time.time() - start_time
            logger.info(f"JSON conversion completed in {json_process_time:.2f} seconds")
            return structured_json, token_usage
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            
            error_pos = e.pos
            context_range = 20
            start_pos = max(0, error_pos - context_range)
            end_pos = min(len(structured_text), error_pos + context_range)
            error_context = structured_text[start_pos:end_pos]
            logger.error(f"Context around error: '...{error_context}...'")
            
            json_process_time = time.time() - start_time
            logger.info(f"JSON conversion failed in {json_process_time:.2f} seconds")
            return {"error": "Failed to parse JSON", "raw_text": structured_text}, token_usage
            
    except Exception as e:
        logger.error(f"Error in LLM processing: {str(e)}")
        json_process_time = time.time() - start_time
        logger.info(f"JSON conversion failed in {json_process_time:.2f} seconds")
        raise
