import json
import time
import uuid
from datetime import datetime
import numpy as np
from PIL import ImageDraw
from typing import Dict, Any, Tuple, List, Optional

from engines.base import OCREngine
from shared.aws_client import get_aws_client, describe_credential_error
from shared.image_utils import convert_to_bytes
from shared.config import (
    logger,
    POSTPROCESSING_MODEL,
    TEXTRACT_FEATURE_TYPES,
    DEFAULT_S3_BUCKET,
    TEXTRACT_ASYNC_TIMEOUT_SECONDS,
    TEXTRACT_ASYNC_POLL_INITIAL_SECONDS,
    TEXTRACT_ASYNC_POLL_MAX_SECONDS,
    TEXTRACT_ASYNC_POLL_BACKOFF,
    LLM_STRUCTURING_CHAR_BUDGET,
)
from shared.json_merge import group_pages_for_structuring, merge_structured_results
from shared.cost_calculator import calculate_textract_cost
from shared.pdf_render import boxes_from_textract_blocks, compose_pdf_visualisation

class TextractEngine(OCREngine):
    """
    Implementation of OCR engine using Amazon Textract
    """

    def __init__(self):
        super().__init__("Textract")

    def process_image(self, image, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an image or PDF with Amazon Textract

        With no feature types selected this calls the text-detection APIs
        (DetectDocumentText for images, StartDocumentTextDetection for PDFs). With
        one or more feature types selected it calls the document-analysis APIs
        instead (AnalyzeDocument / StartDocumentAnalysis), which return the same
        LINE and WORD blocks plus the blocks specific to each requested feature.

        Args:
            image: PIL Image, numpy array, path to image, or file path (including PDF)
            options: Dictionary of options including:
                - output_schema: JSON schema for structuring the output
                - s3_bucket: S3 bucket required for PDF processing
                - feature_types: List of Textract feature types to request, any of
                  FORMS, TABLES, QUERIES, SIGNATURES, LAYOUT. Empty or absent means
                  text detection only.
                - textract_queries: List of natural-language questions, required
                  when QUERIES is among the feature types

        Returns:
            Dictionary containing results including:
            - text: Extracted text
            - json: Structured JSON (if applicable)
            - image: Annotated image
            - process_time: Processing time
            - operation_type: Which Textract API family was used
            - feature_types: The feature types that were requested
        """
        options = options or {}
        output_schema = options.get('output_schema')
        s3_bucket = options.get('s3_bucket', DEFAULT_S3_BUCKET)
        feature_types = self._validate_feature_types(options.get('feature_types'))
        queries = self._normalize_queries(options.get('textract_queries'))
        queries_config = self._build_queries_config(feature_types, queries)
        logger.info(f"Using S3 bucket: {s3_bucket}")
        if feature_types:
            logger.info(f"Requesting Textract features: {feature_types}")
        else:
            logger.info("No Textract features selected - using text detection only")

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
            
            # PDFs go through the asynchronous APIs, whose limit is 500 MB and
            # 3,000 pages - not the 5 MB ceiling that applies to the synchronous
            # image APIs. The previous 5 MB check would have rejected several of
            # the multi-page claim forms this was benchmarked against for no
            # reason, and it only fired on this branch rather than on the str
            # path below.
            # https://docs.aws.amazon.com/textract/latest/dg/limits-document.html
            file_size_mb = len(file_bytes) / (1024 * 1024)
            logger.info(f"PDF file size: {file_size_mb:.2f} MB")
            if file_size_mb > 500:
                raise ValueError(
                    f"PDF is {file_size_mb:.1f} MB; Textract asynchronous "
                    "operations accept at most 500 MB"
                )

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
            
        # Only the asynchronous PDF APIs require S3. Synchronous image APIs accept
        # document bytes directly, avoiding an unnecessary upload and bucket
        # dependency for every image run.
        s3_object_key = None
        if is_pdf:
            if not s3_bucket:
                raise ValueError(
                    "Textract PDF processing requires an S3 bucket in the same "
                    "account and region. Set OCR_S3_BUCKET or enter one in the UI.")
            s3_object_key = self._upload_to_s3(
                file_bytes, s3_bucket, is_pdf=True)
            self._verify_s3_object(s3_bucket, s3_object_key)
        
        # Start timing for all processing (including LLM)
        with timing_ctx:
            try:
                # Get Textract client.
                #
                # There is deliberately no boto3.client('textract') fallback here. It
                # used to catch any failure of get_aws_client() and build an
                # unconfigured client instead, which ignores the selected profile and
                # authenticates against whatever the default credential chain resolves
                # to - a different account. A failure here is a configuration problem
                # and must surface.
                textract = get_aws_client('textract')
                
                # Use appropriate API based on file type and requested features
                if is_pdf:
                    # PDF files require asynchronous processing
                    operation_type = 'textract_analyze_async' if feature_types else 'textract_async'
                    logger.info(
                        f"Starting asynchronous Textract {operation_type} for PDF S3 object: "
                        f"s3://{s3_bucket}/{s3_object_key}"
                    )
                    response = self._process_pdf_async(
                        textract_client=textract,
                        s3_bucket=s3_bucket,
                        s3_object_key=s3_object_key,
                        feature_types=feature_types,
                        queries_config=queries_config
                    )
                elif feature_types:
                    # Images with feature types selected use synchronous analyze_document
                    logger.info(
                        f"Calling Textract analyze_document ({feature_types}) "
                        "with image bytes"
                    )
                    operation_type = 'textract_analyze'
                    analyze_kwargs = {
                        'Document': {'Bytes': image_bytes},
                        'FeatureTypes': feature_types,
                    }
                    if queries_config:
                        analyze_kwargs['QueriesConfig'] = queries_config
                    response = textract.analyze_document(**analyze_kwargs)
                    logger.info("Textract analyze_document call completed successfully")
                else:
                    # Images with no feature types use synchronous detect_document_text
                    logger.info(
                        "Calling Textract detect_document_text with image bytes")
                    operation_type = 'textract_detect'
                    response = textract.detect_document_text(
                        Document={'Bytes': image_bytes})
                    logger.info("Textract detect_document_text call completed successfully")

                # Blocks for the requested features (forms, tables, queries,
                # signatures) carry no text of their own - it lives on their WORD
                # and SELECTION_ELEMENT children - so serialising them needs a
                # lookup from block id to block.
                feature_sections = self._serialize_feature_blocks(
                    blocks=response.get("Blocks", []),
                    feature_types=feature_types
                )

                # Extract text and collect bounding boxes
                extracted_text = ""
                blocks_count = 0
                total_pages = response.get('DocumentMetadata', {}).get('Pages', 1)
                
                # Handle PDF vs Image processing differently
                if is_pdf:
                    # Group the raw OCR lines by page for readable output
                    pages_content = {}
                    for item in response["Blocks"]:
                        page_num = item.get("Page", 1)
                        if page_num not in pages_content:
                            pages_content[page_num] = []

                        if item["BlockType"] == "LINE" and "Text" in item:
                            pages_content[page_num].append(item["Text"])
                            blocks_count += 1

                    # Make sure pages that only produced feature blocks still appear
                    for page_num in feature_sections:
                        pages_content.setdefault(page_num, [])

                    # Combine all pages content, appending each page's feature
                    # sections after its raw lines
                    for page_num in sorted(pages_content.keys()):
                        extracted_text += f"\n--- Page {page_num} ---\n"
                        extracted_text += "\n".join(pages_content[page_num]) + "\n"
                        if page_num in feature_sections:
                            extracted_text += "\n".join(feature_sections[page_num]) + "\n"

                    annotated_image = self._visualise_pdf(
                        pdf_bytes=file_bytes, blocks=response["Blocks"]
                    )

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
                    
                    # Append the feature sections after the raw lines
                    for page_num in sorted(feature_sections.keys()):
                        extracted_text += "\n" + "\n".join(feature_sections[page_num]) + "\n"

                    # Convert to numpy array for display
                    annotated_image = np.array(annotated_img)

                # Process with LLM if needed - INSIDE the timing context to capture full processing time
                structured_json = None
                token_usage = None
                
                if extracted_text and output_schema:
                    try:
                        from shared.prompt_manager import process_text_with_llm
                        
                        if is_pdf and total_pages > 1:
                            structured_json, token_usage = self._structure_pdf_pages(
                                pages_content=pages_content,
                                output_schema=output_schema
                            )
                        else:
                            # Single page or image processing
                            structured_json, token_usage = process_text_with_llm(extracted_text, output_schema)
                            logger.info("Successfully structured text with LLM")
                            
                    except Exception as llm_error:
                        logger.error(f"Error in LLM JSON structuring: {str(llm_error)}")
                        structured_json = {"error": str(llm_error), "raw_text": extracted_text}
                
                # Calculate textract cost from the operation type, the requested
                # features and the real page count
                _, textract_cost = calculate_textract_cost(
                    operation_type=operation_type,
                    page_count=total_pages,
                    feature_types=feature_types
                )

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
                    "feature_types": feature_types,
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
                    "feature_types": feature_types,
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
        feature_types = result.get("feature_types") or []

        # Get base textract cost - per-feature for the analyze operations, flat
        # per page for the text-detection ones
        _, textract_base_cost = calculate_textract_cost(
            operation_type=operation_type,
            page_count=pages,
            feature_types=feature_types
        )
        cost_per_page = textract_base_cost / pages if pages else 0.0

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
    
    def _validate_feature_types(self, feature_types: Optional[List[str]]) -> List[str]:
        """
        Normalise and validate the requested Textract feature types

        Args:
            feature_types: Raw feature list from the caller, possibly None or empty

        Returns:
            De-duplicated, upper-cased feature list in the canonical order defined
            by TEXTRACT_FEATURE_TYPES; empty means text detection only

        Raises:
            ValueError: If any feature name is not a Textract feature type
        """
        if not feature_types:
            return []

        requested = {str(feature).strip().upper() for feature in feature_types if str(feature).strip()}
        unknown = sorted(requested - set(TEXTRACT_FEATURE_TYPES))
        if unknown:
            raise ValueError(
                f"Unsupported Textract feature type(s): {unknown}. "
                f"Supported types are {TEXTRACT_FEATURE_TYPES}"
            )

        # Keep the canonical order so that logs and cost breakdowns are stable
        return [feature for feature in TEXTRACT_FEATURE_TYPES if feature in requested]

    def _normalize_queries(self, queries: Any) -> List[str]:
        """
        Turn the caller's query input into a clean list of question strings

        Args:
            queries: Either a list of questions or a newline-separated string, as
                     typed into the UI textbox

        Returns:
            List of non-empty question strings with surrounding whitespace removed
        """
        if not queries:
            return []

        if isinstance(queries, str):
            candidates = queries.splitlines()
        else:
            candidates = list(queries)

        return [str(query).strip() for query in candidates if str(query).strip()]

    def _build_queries_config(self, feature_types: List[str], queries: List[str]) -> Optional[Dict[str, Any]]:
        """
        Build the QueriesConfig payload required by the QUERIES feature type

        Each query gets an Alias derived from its text so the QUERY_RESULT blocks
        come back keyed by something readable rather than by position.

        Args:
            feature_types: Validated Textract feature types
            queries: Cleaned list of question strings

        Returns:
            QueriesConfig dictionary, or None when QUERIES was not requested

        Raises:
            ValueError: If QUERIES is requested with no queries, or a query exceeds
                        the 200-character API limit
        """
        if 'QUERIES' not in feature_types:
            if queries:
                logger.warning(
                    f"{len(queries)} Textract queries were supplied but the QUERIES "
                    "feature is not selected - they will be ignored"
                )
            return None

        if not queries:
            # Textract rejects QUERIES without QueriesConfig. Silently falling back
            # to text-only would change what is being measured without saying so.
            raise ValueError(
                "The QUERIES feature type requires at least one query. "
                "Enter one question per line in the Textract Queries box."
            )

        too_long = [query for query in queries if len(query) > 200]
        if too_long:
            raise ValueError(
                f"Textract queries are limited to 200 characters; "
                f"{len(too_long)} query/queries exceed that limit"
            )

        return {
            'Queries': [
                {'Text': query, 'Alias': self._query_alias(query)}
                for query in queries
            ]
        }

    def _query_alias(self, query: str) -> str:
        """
        Derive a Textract query alias from the question text

        Aliases may only contain alphanumerics, underscores and hyphens, and are
        capped at 200 characters by the API.

        Args:
            query: The question text

        Returns:
            An alias safe to send in a QueriesConfig entry
        """
        alias = "".join(char if char.isalnum() else "_" for char in query.lower())
        alias = "_".join(part for part in alias.split("_") if part)
        return alias[:200] or "query"

    def _build_block_map(self, blocks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Index Textract blocks by their Id so relationships can be resolved

        Args:
            blocks: The Blocks list from a Textract response

        Returns:
            Mapping of block Id to block
        """
        return {block['Id']: block for block in blocks if 'Id' in block}

    def _get_child_text(self, block: Dict[str, Any], block_map: Dict[str, Dict[str, Any]]) -> str:
        """
        Resolve the text of a block from its CHILD relationships

        KEY_VALUE_SET, CELL and QUERY_RESULT blocks carry no Text of their own -
        the text lives on their WORD children, and checkbox state lives on their
        SELECTION_ELEMENT children as SELECTED / NOT_SELECTED.

        Args:
            block: The parent block whose text is wanted
            block_map: Mapping of block Id to block for the whole response

        Returns:
            Space-joined text of the block's children, empty string if it has none
        """
        # QUERY_RESULT blocks are the exception: they do carry their own Text.
        if block.get('BlockType') == 'QUERY_RESULT' and 'Text' in block:
            return block['Text']

        words: List[str] = []
        for relationship in block.get('Relationships', []):
            if relationship.get('Type') != 'CHILD':
                continue
            for child_id in relationship.get('Ids', []):
                child = block_map.get(child_id)
                if not child:
                    continue
                if child.get('BlockType') == 'WORD':
                    words.append(child.get('Text', ''))
                elif child.get('BlockType') == 'SELECTION_ELEMENT':
                    words.append(child.get('SelectionStatus', 'NOT_SELECTED'))

        return " ".join(word for word in words if word).strip()

    def _extract_form_fields(self, blocks: List[Dict[str, Any]],
                             block_map: Dict[str, Dict[str, Any]]) -> Dict[int, List[str]]:
        """
        Serialise FORMS key-value pairs, grouped by page

        Args:
            blocks: The Blocks list from a Textract response
            block_map: Mapping of block Id to block

        Returns:
            Mapping of page number to "key: value" lines
        """
        by_page: Dict[int, List[str]] = {}

        for block in blocks:
            if block.get('BlockType') != 'KEY_VALUE_SET':
                continue
            if 'KEY' not in block.get('EntityTypes', []):
                continue

            key_text = self._get_child_text(block, block_map)

            # The VALUE half is reached through a VALUE relationship on the KEY.
            value_text = ""
            for relationship in block.get('Relationships', []):
                if relationship.get('Type') != 'VALUE':
                    continue
                for value_id in relationship.get('Ids', []):
                    value_block = block_map.get(value_id)
                    if value_block:
                        value_text = self._get_child_text(value_block, block_map)

            if not key_text and not value_text:
                continue

            page_num = block.get('Page', 1)
            by_page.setdefault(page_num, []).append(f"{key_text}: {value_text}")

        return by_page

    def _extract_tables(self, blocks: List[Dict[str, Any]],
                        block_map: Dict[str, Dict[str, Any]]) -> Dict[int, List[str]]:
        """
        Serialise TABLES blocks into pipe-separated rows, grouped by page

        Args:
            blocks: The Blocks list from a Textract response
            block_map: Mapping of block Id to block

        Returns:
            Mapping of page number to table lines, one line per row
        """
        by_page: Dict[int, List[str]] = {}
        table_index = 0

        for block in blocks:
            if block.get('BlockType') != 'TABLE':
                continue

            table_index += 1
            page_num = block.get('Page', 1)

            # Collect cells into a row -> column -> text grid.
            grid: Dict[int, Dict[int, str]] = {}
            for relationship in block.get('Relationships', []):
                if relationship.get('Type') != 'CHILD':
                    continue
                for cell_id in relationship.get('Ids', []):
                    cell = block_map.get(cell_id)
                    if not cell or cell.get('BlockType') != 'CELL':
                        continue
                    row = cell.get('RowIndex', 1)
                    column = cell.get('ColumnIndex', 1)
                    grid.setdefault(row, {})[column] = self._get_child_text(cell, block_map)

            if not grid:
                continue

            lines = [f"[Table {table_index}]"]
            for row in sorted(grid):
                columns = grid[row]
                lines.append(" | ".join(columns.get(col, "") for col in sorted(columns)))

            by_page.setdefault(page_num, []).extend(lines)

        return by_page

    def _extract_queries(self, blocks: List[Dict[str, Any]],
                         block_map: Dict[str, Dict[str, Any]]) -> Dict[int, List[str]]:
        """
        Serialise QUERY / QUERY_RESULT pairs, grouped by page

        Args:
            blocks: The Blocks list from a Textract response
            block_map: Mapping of block Id to block

        Returns:
            Mapping of page number to "alias: answer" lines
        """
        by_page: Dict[int, List[str]] = {}

        for block in blocks:
            if block.get('BlockType') != 'QUERY':
                continue

            query = block.get('Query', {})
            label = query.get('Alias') or query.get('Text', '')
            page_num = block.get('Page', 1)

            answers: List[str] = []
            for relationship in block.get('Relationships', []):
                if relationship.get('Type') != 'ANSWER':
                    continue
                for answer_id in relationship.get('Ids', []):
                    answer_block = block_map.get(answer_id)
                    if answer_block:
                        answer_text = self._get_child_text(answer_block, block_map)
                        if answer_text:
                            answers.append(answer_text)

            # An unanswered query is a real finding on a partially filled form, so
            # it is recorded rather than dropped.
            by_page.setdefault(page_num, []).append(
                f"{label}: {' | '.join(answers) if answers else '<no answer found>'}"
            )

        return by_page

    def _extract_signatures(self, blocks: List[Dict[str, Any]]) -> Dict[int, List[str]]:
        """
        Serialise SIGNATURE blocks, grouped by page

        Args:
            blocks: The Blocks list from a Textract response

        Returns:
            Mapping of page number to one line per detected signature
        """
        by_page: Dict[int, List[str]] = {}

        for block in blocks:
            if block.get('BlockType') != 'SIGNATURE':
                continue
            page_num = block.get('Page', 1)
            confidence = block.get('Confidence', 0.0)
            by_page.setdefault(page_num, []).append(
                f"Signature detected (confidence {confidence:.1f}%)"
            )

        return by_page

    def _visualise_pdf(
        self, *, pdf_bytes: bytes, blocks: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Render a processed PDF's pages with the text lines Textract detected

        The PDF path used to answer with a 400x600 black rectangle and three
        lines of text, even though every LINE block carries a bounding box and
        the page it sits on - the same geometry the image path already draws.

        Args:
            pdf_bytes: The PDF that was processed
            blocks: The `Blocks` list from the Textract response

        Returns:
            The composed visualisation as an RGB numpy array, which is what the
            Gradio image output takes
        """
        boxes = boxes_from_textract_blocks(blocks=blocks)

        return np.array(
            compose_pdf_visualisation(
                pdf_bytes=pdf_bytes, boxes=boxes, item_noun="text lines"
            )
        )

    def _serialize_feature_blocks(self, blocks: List[Dict[str, Any]],
                                  feature_types: List[str]) -> Dict[int, List[str]]:
        """
        Turn the blocks produced by the requested features into text sections

        The raw LINE text is serialised separately by the caller; this adds the
        structured sections on top of it, so enabling a feature never removes
        information from the text that gets handed to the LLM post-processing step.

        Args:
            blocks: The Blocks list from a Textract response
            feature_types: The feature types that were requested

        Returns:
            Mapping of page number to the section lines for that page. Empty when
            no features were requested.
        """
        if not feature_types or not blocks:
            return {}

        block_map = self._build_block_map(blocks)
        sections: Dict[int, List[str]] = {}

        # LAYOUT changes the reading order of the LINE blocks rather than adding
        # content worth serialising on its own, so it has no section here.
        extractors = [
            ('FORMS', "[Form fields]", lambda: self._extract_form_fields(blocks, block_map)),
            ('TABLES', None, lambda: self._extract_tables(blocks, block_map)),
            ('QUERIES', "[Queries]", lambda: self._extract_queries(blocks, block_map)),
            ('SIGNATURES', "[Signatures]", lambda: self._extract_signatures(blocks)),
        ]

        for feature, heading, extractor in extractors:
            if feature not in feature_types:
                continue
            for page_num, lines in extractor().items():
                if not lines:
                    continue
                page_section = sections.setdefault(page_num, [])
                page_section.append("")
                if heading:
                    page_section.append(heading)
                page_section.extend(lines)

        return sections

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
            # An expired or wrong-account credential looks like a generic upload
            # failure here, so name the profile that actually needs attention.
            message = describe_credential_error(e) or str(e)
            logger.error(f"Failed to upload to S3: {message}")
            raise Exception(f"S3 upload failed: {message}")
    
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
                logger.info("S3 object is ready for Textract processing")
            except Exception as perm_error:
                logger.warning(f"Potential permission issue: {str(perm_error)}")
                
        except Exception as e:
            logger.error(f"S3 object verification failed: {str(e)}")
            raise Exception(f"S3 object not accessible: {str(e)}")
    
    def _structure_pdf_pages(self, pages_content: Dict[int, List[str]],
                             output_schema: Any) -> Tuple[Optional[Dict[str, Any]],
                                                          Optional[Dict[str, int]]]:
        """
        Structure a multi-page PDF's text into one object shaped like the schema

        Pages are batched into as few LLM calls as fit LLM_STRUCTURING_CHAR_BUDGET
        and the results merged. A typical claim form fits in one call, so the model
        sees sections spanning page boundaries whole.

        This replaces structuring each page independently into
        {"pages": {"page_1": ..., ...}}, which produced JSON keyed by page number
        while the schema and ground truth are keyed by field. Every field then read
        as missing and a perfect extraction scored 0%.

        Args:
            pages_content: Mapping of page number to that page's extracted lines
            output_schema: JSON schema the output should conform to

        Returns:
            Tuple of (merged structured object, accumulated token usage). Both are
            None when no page held any text.

        Raises:
            Exception: If a chunk could not be parsed as JSON. Merging the
                       unparsed text into the result would corrupt accuracy
                       numbers instead of reporting a problem.
        """
        # Imported here rather than at module scope, matching how the caller does
        # it: shared.prompt_manager pulls in the Bedrock client stack, which is not
        # needed unless structured output was actually requested.
        from shared.prompt_manager import process_text_with_llm

        page_texts = {
            page_num: "\n".join(lines) for page_num, lines in pages_content.items()
        }
        chunks = group_pages_for_structuring(
            page_texts=page_texts,
            char_budget=LLM_STRUCTURING_CHAR_BUDGET
        )

        if not chunks:
            logger.warning("No page text to structure - Textract returned no LINE blocks")
            return None, None

        logger.info(
            f"Structuring {len(page_texts)} pages of text in {len(chunks)} LLM "
            f"call(s) (budget {LLM_STRUCTURING_CHAR_BUDGET} characters)"
        )

        results: List[Dict[str, Any]] = []
        labels: List[str] = []
        # Keys are camelCase to match what shared.cost_calculator reads. The
        # previous code both built and read snake_case keys against a camelCase
        # source, so multi-page token usage always accumulated zero and the
        # post-processing cost was reported as $0.
        total_token_usage = {'inputTokens': 0, 'outputTokens': 0, 'totalTokens': 0}

        for page_numbers, chunk_text in chunks:
            label = (
                f"page {page_numbers[0]}" if len(page_numbers) == 1
                else f"pages {page_numbers[0]}-{page_numbers[-1]}"
            )
            chunk_result, chunk_tokens = process_text_with_llm(chunk_text, output_schema)

            # process_text_with_llm reports an unparseable model response by
            # returning this shape rather than raising. Identified by both keys
            # together, since a schema could legitimately define one named "error".
            if isinstance(chunk_result, dict) and 'error' in chunk_result and 'raw_text' in chunk_result:
                raise Exception(
                    f"Could not structure {label} as JSON: {chunk_result['error']}"
                )

            results.append(chunk_result)
            labels.append(label)

            if chunk_tokens:
                for token_key in total_token_usage:
                    total_token_usage[token_key] += chunk_tokens.get(token_key, 0)

        merged = merge_structured_results(results=results, labels=labels)
        logger.info(
            f"Structured {len(page_texts)} pages into {len(merged)} top-level fields"
        )

        return merged, total_token_usage

    # Statuses that mean Textract has stopped working on the job. PARTIAL_SUCCESS
    # is a success-with-caveats: some pages were processed and some were not, and
    # the blocks for the ones that worked are real results worth returning.
    TERMINAL_JOB_STATUSES = ('SUCCEEDED', 'PARTIAL_SUCCESS', 'FAILED')

    def _await_async_job(self, get_results, job_id: str, api_name: str) -> str:
        """
        Poll an asynchronous Textract job until it reaches a terminal status

        Polls with exponential backoff on real elapsed wall-clock time. Elapsed
        time is measured with time.monotonic() rather than accumulated from the
        sleep interval, because each poll is a network round trip: summing the
        sleeps alone understates the true wait by the total request time and makes
        the timeout message wrong.

        Args:
            get_results: Bound GetDocumentTextDetection or GetDocumentAnalysis
                         method to call with a JobId keyword
            job_id: Textract job identifier returned by the matching Start call
            api_name: CLI name of the getter, used to build a recovery command in
                      the timeout message

        Returns:
            The terminal job status, one of 'SUCCEEDED' or 'PARTIAL_SUCCESS'

        Raises:
            Exception: If the job reports FAILED, or if no terminal status is
                       reached within TEXTRACT_ASYNC_TIMEOUT_SECONDS
        """
        started_at = time.monotonic()
        wait_interval = TEXTRACT_ASYNC_POLL_INITIAL_SECONDS
        poll_count = 0

        while True:
            elapsed = time.monotonic() - started_at
            if elapsed >= TEXTRACT_ASYNC_TIMEOUT_SECONDS:
                # The job itself is almost certainly still running and will
                # finish: Textract keeps results for 7 days, so name the job id
                # and how to fetch it rather than making the work unrecoverable.
                logger.error(
                    f"Textract job {job_id} did not finish within "
                    f"{TEXTRACT_ASYNC_TIMEOUT_SECONDS}s"
                )
                raise Exception(
                    f"Textract job did not finish within "
                    f"{TEXTRACT_ASYNC_TIMEOUT_SECONDS}s (waited {elapsed:.0f}s over "
                    f"{poll_count} polls). The job is queued or still running, not "
                    f"failed - Textract keeps results for 7 days, so retrieve them "
                    f"with: aws textract {api_name} --job-id {job_id}. "
                    f"Raise OCR_TEXTRACT_ASYNC_TIMEOUT_SECONDS to wait longer."
                )

            # Never sleep past the deadline, so the timeout is honoured to within
            # one request rather than one full backoff interval.
            time.sleep(min(wait_interval, TEXTRACT_ASYNC_TIMEOUT_SECONDS - elapsed))
            poll_count += 1

            # MaxResults=1 keeps each poll small. Without it the poll that finally
            # sees SUCCEEDED downloads a full page of up to 1000 blocks, which is
            # then discarded and re-fetched by the pagination loop.
            get_response = get_results(JobId=job_id, MaxResults=1)
            status = get_response['JobStatus']
            elapsed = time.monotonic() - started_at
            logger.info(f"Job status: {status} (waited {elapsed:.0f}s, poll {poll_count})")

            if status in self.TERMINAL_JOB_STATUSES:
                return self._handle_terminal_status(
                    status=status,
                    get_response=get_response,
                    job_id=job_id,
                    elapsed=elapsed
                )

            if status != 'IN_PROGRESS':
                # An unrecognised status means the API contract changed under us.
                # Guessing at its meaning would risk reporting a failed job as a
                # success, so refuse to interpret it.
                raise Exception(
                    f"Textract job {job_id} returned unrecognised status '{status}'"
                )

            wait_interval = min(
                wait_interval * TEXTRACT_ASYNC_POLL_BACKOFF,
                TEXTRACT_ASYNC_POLL_MAX_SECONDS
            )

    def _handle_terminal_status(self, status: str, get_response: Dict[str, Any],
                                job_id: str, elapsed: float) -> str:
        """
        Interpret a terminal Textract job status

        Args:
            status: Terminal JobStatus value
            get_response: The GetDocument* response that reported the status
            job_id: Textract job identifier, for error messages
            elapsed: Seconds spent waiting, for the completion log line

        Returns:
            The status, when it carries usable results

        Raises:
            Exception: If the status is FAILED, with the service's own explanation
        """
        # Textract explains itself through StatusMessage and Warnings. Raising a
        # bare "job failed" throws away the only description of what went wrong.
        status_message = get_response.get('StatusMessage')
        warnings = get_response.get('Warnings') or []

        if status == 'FAILED':
            detail = status_message or 'no StatusMessage returned by Textract'
            if warnings:
                detail += f"; warnings: {json.dumps(warnings)}"
            logger.error(f"Textract job {job_id} failed: {detail}")
            raise Exception(f"Textract job failed: {detail}")

        if status == 'PARTIAL_SUCCESS':
            # Some pages could not be processed. The rest are still real output,
            # so return them - but say loudly which ones are missing, since
            # otherwise the gap looks like an extraction accuracy problem.
            logger.warning(
                f"Textract job {job_id} partially succeeded after {elapsed:.0f}s - "
                f"some pages could not be processed. "
                f"StatusMessage: {status_message or 'none'}. "
                f"Warnings: {json.dumps(warnings) if warnings else 'none'}"
            )
            return status

        logger.info(f"Textract job completed successfully after {elapsed:.0f}s")
        if warnings:
            logger.warning(f"Textract job {job_id} succeeded with warnings: {json.dumps(warnings)}")
        return status

    def _process_pdf_async(self, textract_client, s3_bucket: str, s3_object_key: str,
                           feature_types: Optional[List[str]] = None,
                           queries_config: Optional[Dict[str, Any]] = None):
        """
        Process PDF using the asynchronous Textract APIs

        With no feature types this uses StartDocumentTextDetection /
        GetDocumentTextDetection. With one or more feature types it uses
        StartDocumentAnalysis / GetDocumentAnalysis, which accept the same
        FeatureTypes and QueriesConfig as the synchronous AnalyzeDocument call.

        Args:
            textract_client: Textract client
            s3_bucket: S3 bucket name
            s3_object_key: S3 object key
            feature_types: Textract feature types to request; empty means text only
            queries_config: QueriesConfig payload, required when QUERIES is requested

        Returns:
            Combined response with all blocks
        """
        feature_types = feature_types or []
        use_analysis = bool(feature_types)

        try:
            document_location = {
                'S3Object': {
                    'Bucket': s3_bucket,
                    'Name': s3_object_key
                }
            }

            # Start the asynchronous job
            if use_analysis:
                logger.info(f"Starting asynchronous Textract analysis job ({feature_types})")
                start_kwargs = {
                    'DocumentLocation': document_location,
                    'FeatureTypes': feature_types
                }
                if queries_config:
                    start_kwargs['QueriesConfig'] = queries_config
                start_response = textract_client.start_document_analysis(**start_kwargs)
            else:
                logger.info("Starting asynchronous Textract text detection job")
                start_response = textract_client.start_document_text_detection(
                    DocumentLocation=document_location
                )

            job_id = start_response['JobId']
            logger.info(f"Textract job started with ID: {job_id}")

            # Both API families expose the same polling and pagination shape, so
            # the rest of this method only needs the right getter.
            get_results = (
                textract_client.get_document_analysis if use_analysis
                else textract_client.get_document_text_detection
            )

            # Block until the job reaches a terminal status, or give up
            self._await_async_job(
                get_results=get_results,
                job_id=job_id,
                api_name=(
                    'get-document-analysis' if use_analysis
                    else 'get-document-text-detection'
                )
            )

            # Collect all result pages
            all_blocks = []
            pages_metadata = {'Pages': 0}
            next_token = None
            
            while True:
                if next_token:
                    get_response = get_results(JobId=job_id, NextToken=next_token)
                else:
                    get_response = get_results(JobId=job_id)

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
