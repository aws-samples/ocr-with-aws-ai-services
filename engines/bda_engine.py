import json
import uuid
import time
import numpy as np
from PIL import ImageDraw
from typing import Dict, Any, Tuple, Optional, List

from engines.base import OCREngine
from shared.aws_client import (
    get_aws_client,
    get_account_id,
    get_current_region,
    describe_credential_error,
)
from shared.image_utils import convert_to_bytes
from shared.config import logger, API_COSTS, POSTPROCESSING_MODEL
from shared.prompt_manager import process_text_with_llm
from shared.blueprint_schema import (
    BlueprintBuild,
    build_blueprint_schema,
    restore_nested_result,
)
from shared.pdf_render import (
    PageBox,
    boxes_from_bda_explainability,
    compose_pdf_visualisation,
)

class BDAEngine(OCREngine):
    """
    Implementation of OCR engine using Amazon Bedrock Document Analysis (BDA)
    """
    
    def __init__(self):
        super().__init__("BDA")
        
    def process_image(self, image, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an image or PDF with Amazon BDA using blueprint or standard processing
        
        Args:
            image: PIL Image, numpy array, path to image, or PDF file
            options: Dictionary of options including:
                - s3_bucket: S3 bucket for BDA processing
                - document_type: Type of document
                - output_schema: JSON schema for structuring the output
                - use_blueprint: Whether to use custom blueprint
                
        Returns:
            Dictionary containing results
        """
        options = options or {}
        s3_bucket = options.get('s3_bucket', '')
        document_type = options.get('document_type', 'generic')
        output_schema = options.get('output_schema')
        use_blueprint = options.get('use_blueprint', False)

        if use_blueprint and not output_schema:
            raise ValueError(
                "BDA custom blueprint processing requires an output schema. "
                "Enable structured output and provide a schema, or disable the "
                "custom blueprint option."
            )
        
        # Check if input is a PDF file
        is_pdf = self._is_pdf_input(image)
        
        if use_blueprint:
            return self._process_with_bda_blueprint(image, s3_bucket, document_type, output_schema, is_pdf)
        else:
            return self._process_with_bda_llm(image, s3_bucket, document_type, output_schema, is_pdf)
    
    def _is_pdf_input(self, image):
        """Check if input is a PDF file"""
        if hasattr(image, 'name') and image.name and image.name.lower().endswith('.pdf'):
            return True
        elif isinstance(image, str) and image.lower().endswith('.pdf'):
            return True
        return False
    
    def _process_with_bda_blueprint(self, image, s3_bucket, document_type, output_schema, is_pdf=False):
        """
        Process with BDA using custom blueprint based on output schema
        """
        return self._process_with_bda(image, s3_bucket, document_type, output_schema, use_blueprint=True, is_pdf=is_pdf)
    
    def _process_with_bda_llm(self, image, s3_bucket, document_type, output_schema, is_pdf=False):
        """
        Process with BDA using default extraction and LLM post-processing
        """
        return self._process_with_bda(image, s3_bucket, document_type, output_schema, use_blueprint=False, is_pdf=is_pdf)
    
    def _convert_schema_to_blueprint_format(
        self, schema_json_str, document_type: str = "generic"
    ) -> Optional[BlueprintBuild]:
        """
        Translate an output schema into a BDA blueprint.

        Args:
            schema_json_str: JSON schema as a string or a dictionary
            document_type: Document type recorded on the blueprint

        Returns:
            A BlueprintBuild carrying the blueprint schema and the field map
            needed to restore the result's nested shape, or None on error.
        """
        try:
            return build_blueprint_schema(
                schema=schema_json_str, document_type=document_type
            )
        except Exception as e:
            logger.error(f"Error converting schema: {str(e)}")
            return None

    def _describe_blueprint(self, *, build: BlueprintBuild) -> str:
        """
        Render a blueprint's fields as readable text for the results panel.

        Groups and tables are shown with their sub-fields indented beneath them,
        and any field that had to be hoisted out of its parent is named, so the
        panel reflects the shape BDA was actually asked for.

        Args:
            build: The translated blueprint.

        Returns:
            A text block listing every extraction field.
        """
        definitions = build.schema.get("definitions", {})

        def sub_fields(*, ref: str) -> Dict[str, Any]:
            """
            Resolve a `$ref` pointer to the properties it names.

            Args:
                ref: A `#/definitions/Name` pointer.

            Returns:
                The referenced properties, empty if the pointer is unknown.
            """
            name = ref.rsplit("/", 1)[-1]
            return definitions.get(name, {}).get("properties", {})

        lines = [f"Extraction Fields ({build.field_count} billable):\n"]

        for prop_name, prop in build.schema["properties"].items():
            original_path = ".".join(build.field_map.get(prop_name, (prop_name,)))
            shown_name = (
                prop_name if original_path == prop_name
                else f"{prop_name}  (extracted for {original_path})"
            )

            if "$ref" in prop:
                lines.append(f"• {shown_name} (group)")
                if prop.get("instruction"):
                    lines.append(f"  - {prop['instruction']}")
                for child_name, child in sub_fields(ref=prop["$ref"]).items():
                    lines.append(f"    · {child_name} ({child.get('type', 'string')})")
                    if child.get("instruction"):
                        lines.append(f"      - {child['instruction']}")

            elif prop.get("type") == "array" and "$ref" in prop.get("items", {}):
                lines.append(f"• {shown_name} (table)")
                if prop.get("instruction"):
                    lines.append(f"  - {prop['instruction']}")
                for column_name, column in sub_fields(ref=prop["items"]["$ref"]).items():
                    lines.append(
                        f"    · {column_name} ({column.get('type', 'string')})"
                    )
                    if column.get("instruction"):
                        lines.append(f"      - {column['instruction']}")

            else:
                lines.append(f"• {shown_name} ({prop.get('type', 'string')})")
                if prop.get("instruction"):
                    lines.append(f"  - {prop['instruction']}")

        if build.hoisted_paths:
            # BDA blueprints allow one level of nesting, so these fields could not
            # stay where the schema puts them. Saying so keeps the difference
            # between the blueprint and the schema visible.
            hoisted = ", ".join(".".join(path) for path in build.hoisted_paths)
            lines.append(
                f"\nExtracted as separate top-level fields because BDA blueprints "
                f"support one level of nesting: {hoisted}.\n"
                f"These are put back in place before the result is compared."
            )

        return "\n".join(lines) + "\n"

    def _restore_blueprint_json(
        self, *, custom_output: Dict[str, Any], build: Optional[BlueprintBuild]
    ) -> Dict[str, Any]:
        """
        Put a blueprint's result back into the shape the output schema declares.

        BDA returns one top-level key per blueprint field. Groups and tables come
        back already shaped, but a field hoisted out of its parent - because BDA
        blueprints support only one level of nesting - comes back at the top
        level. Ground truth is nested, so a result left in the returned shape
        matches nothing and scores 0% however good the extraction was.

        The `inference_result` wrapper is dropped here for the same reason: it is
        one extra level that no truth key can match.

        Args:
            custom_output: The custom output object from BDA.
            build: The translated blueprint, whose field map says where each
                returned key belongs. None when the blueprint was not built by
                this engine, in which case the result is returned as-is.

        Returns:
            The result nested as the output schema declares it.
        """
        inference_result = custom_output['inference_result']

        if build is None:
            logger.warning(
                "No blueprint field map is available, so the BDA result cannot be "
                "restored to the schema's nested shape and will be compared flat")
            return inference_result

        return restore_nested_result(
            inference_result=inference_result, field_map=build.field_map
        )

    def _process_with_bda(self, image, s3_bucket=None, document_type="generic", output_schema=None, use_blueprint=True, is_pdf=False):
        """
        Process image or PDF with Amazon BDA
        
        Args:
            image: Image or PDF to process
            s3_bucket: S3 bucket name for BDA processing
            document_type: Type of document
            output_schema: JSON schema for output structuring
            use_blueprint: Whether to use custom blueprint (True) or post-process with LLM (False)
            is_pdf: Whether the input is a PDF file
            
        Returns:
            Dictionary with processing results
        """
        overall_start_time = time.time()

        # Set up timing context manager
        timing_ctx = self.get_timing_wrapper()
        
        # Handle file processing based on type
        if is_pdf:
            # For PDF files, read the file bytes directly
            if hasattr(image, 'name') and image.name:
                with open(image.name, 'rb') as f:
                    file_bytes = f.read()
            elif isinstance(image, str):
                with open(image, 'rb') as f:
                    file_bytes = f.read()
            else:
                raise ValueError("PDF input must be a file path or file object")
            img_pil = None  # No PIL image for PDF
        else:
            # Convert image to bytes OUTSIDE the timing context
            file_bytes, img_pil = convert_to_bytes(image)
        
        # Start timing for the actual processing
        with timing_ctx:
            
            bda_client = get_aws_client('bedrock-data-automation')
            bda_runtime_client = get_aws_client('bedrock-data-automation-runtime')
            s3_client = get_aws_client('s3')
            
            try:
                account_id = get_account_id()
                current_region = get_current_region()

                if not s3_bucket:
                    raise RuntimeError(
                        "BDA requires an S3 bucket in the same account and region. "
                        "Set OCR_BDA_S3_BUCKET or OCR_S3_BUCKET, or enter one in "
                        "the UI.")
                
                # Validate S3 bucket.
                #
                # Raised rather than returned: the early return here produced a
                # result with neither "operation_type": "error" nor a page count,
                # so the caller read a bucket misconfiguration as a successful run
                # that had somehow processed zero pages. Raising routes it through
                # the handler below, which is the only place that shapes an error.
                try:
                    s3_client.head_bucket(Bucket=s3_bucket)
                except Exception as e:
                    raise RuntimeError(
                        f"S3 bucket '{s3_bucket}' does not exist or is not "
                        f"accessible: {e}") from e
                
                # Upload file to S3
                timestamp = int(time.time())
                random_id = uuid.uuid4().hex[:8]
                input_prefix = "bda-input"
                
                if is_pdf:
                    object_key = f"{input_prefix}/{timestamp}-{random_id}.pdf"
                    content_type = "application/pdf"
                else:
                    object_key = f"{input_prefix}/{timestamp}-{random_id}.jpg"
                    content_type = "image/jpeg"
                
                s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=object_key,
                    Body=file_bytes,
                    ContentType=content_type
                )
                
                input_uri = f"s3://{s3_bucket}/{object_key}"
                output_uri = f"s3://{s3_bucket}/bda-output/{timestamp}-{random_id}/"
                
                logger.info(f"Image uploaded to: {input_uri}")
                logger.info(f"Output will be stored at: {output_uri}")
                
                # Create and manage blueprint if needed
                blueprint_arn = None
                blueprint_build = None
                blueprint_info = "No custom blueprint was used"
                field_count = 0

                if use_blueprint and output_schema:
                    blueprint_build = self._convert_schema_to_blueprint_format(output_schema, document_type)
                    if blueprint_build:
                        blueprint_schema = blueprint_build.schema
                        blueprint_name = f"ocr-{document_type}-{random_id}"
                        logger.info(f"Creating blueprint: {blueprint_name}")
                        
                        blueprint_response = bda_client.create_blueprint(
                            blueprintName=blueprint_name,
                            type='DOCUMENT',
                            blueprintStage='DEVELOPMENT',
                            schema=json.dumps(blueprint_schema)
                        )
                        
                        blueprint_arn = blueprint_response['blueprint']['blueprintArn']
                        logger.info(f"Created blueprint: {blueprint_arn}")
                        
                        # Format blueprint information for display
                        blueprint_info = "Blueprint Configuration:\n\n"
                        blueprint_info += f"Name: {blueprint_name}\n"
                        blueprint_info += "Type: DOCUMENT\n"
                        blueprint_info += "Stage: DEVELOPMENT\n"
                        blueprint_info += f"ARN: {blueprint_arn}\n\n"
                        
                        # Add formatted properties information
                        field_count = blueprint_build.field_count
                        blueprint_info += self._describe_blueprint(build=blueprint_build)
                elif not use_blueprint:
                    blueprint_info = "Using default BDA extraction with LLM post-processing"
                
                # Prepare invocation arguments
                invoke_args = {
                    'inputConfiguration': {
                        's3Uri': input_uri
                    },
                    'outputConfiguration': {
                        's3Uri': output_uri
                    },
                    'dataAutomationProfileArn': f'arn:aws:bedrock:{current_region}:{account_id}:data-automation-profile/us.data-automation-v1'
                }
                
                if blueprint_arn:
                    invoke_args['blueprints'] = [{'blueprintArn': blueprint_arn, 'stage': 'DEVELOPMENT'}]
                else:
                    invoke_args['dataAutomationConfiguration'] = {
                        'dataAutomationProjectArn': f'arn:aws:bedrock:{current_region}:aws:data-automation-project/public-default'
                    }
                
                # Start BDA job
                response = bda_runtime_client.invoke_data_automation_async(**invoke_args)
                invocation_arn = response['invocationArn']
                logger.info(f"BDA job started with invocation ARN: {invocation_arn}")
                
                # Wait for job completion
                max_tries = 30
                delay = 10
                
                for i in range(max_tries):
                    response = bda_runtime_client.get_data_automation_status(invocationArn=invocation_arn)
                    status = response['status']
                    
                    logger.info(f"Job status ({i+1}/{max_tries}): {status}")
                    
                    if status == 'Success':
                        break
                    elif status in ['ClientError', 'ServiceError']:
                        error_message = response.get('error_message', 'Unknown error')
                        raise Exception(f"Job failed with status {status}: {error_message}")
                    
                    if i < max_tries - 1:
                        time.sleep(delay)
                
                if status != 'Success':
                    raise Exception(f"Job timed out after {max_tries * delay} seconds")
                
                # Process results
                job_metadata_s3_location = response['outputConfiguration']['s3Uri']
                bucket, key = self._get_bucket_and_key(job_metadata_s3_location)
                metadata_obj = s3_client.get_object(Bucket=bucket, Key=key)
                metadata_content = metadata_obj['Body'].read().decode('utf-8')
                job_metadata = json.loads(metadata_content)
                
                # Extract text and output from BDA result
                extracted_text, custom_output, standard_output = self._process_bda_results(job_metadata, s3_client)
                # For non-blueprint approach, process with LLM if output schema is provided
                structured_json = None
                token_usage = None
                json_process_time = 0
                
                if not use_blueprint and output_schema and extracted_text:
                    json_start_time = time.time()
                    try:
                        structured_json, token_usage = process_text_with_llm(extracted_text, output_schema)
                        logger.info("Successfully structured BDA output with LLM")
                        json_process_time = time.time() - json_start_time

                        if structured_json:
                            # Use the structured JSON as our display JSON.
                            # Nothing is appended to `blueprint_info` here: it is
                            # only returned when `use_blueprint` is true, so on
                            # this branch it never reaches the panel. The timing
                            # is logged below instead.
                            display_json = structured_json
                            logger.info(
                                f"Post-processed the BDA text with {POSTPROCESSING_MODEL} "
                                f"in {json_process_time:.2f} seconds")
                    except Exception as e:
                        # Structuring was asked for and failed, so there is no
                        # schema-shaped result. Falling through would hand the raw
                        # BDA envelope to the evaluator, which scores it 0% and
                        # reads as poor extraction rather than a failed step.
                        raise RuntimeError(
                            f"Structuring the BDA output against the schema failed, so no "
                            f"schema-shaped JSON was produced: {e}") from e

                    # `process_text_with_llm` signals an unparseable model response by
                    # returning this shape rather than raising. Both keys are checked
                    # together, since a schema could legitimately define one named "error".
                    if structured_json and 'error' in structured_json and 'raw_text' in structured_json:
                        raise RuntimeError(
                            f"The model's response could not be parsed as JSON, so the BDA "
                            f"output was not structured: {structured_json['error']}")
                
                # Update blueprint info with matching result if available
                if use_blueprint and custom_output and 'matched_blueprint' in custom_output:
                    matched_info = custom_output['matched_blueprint']
                    blueprint_info += "\nMatch Results:\n"
                    blueprint_info += f"Matched Blueprint: {matched_info.get('name', 'Unknown')}\n"
                    blueprint_info += f"Confidence: {matched_info.get('confidence', 0) * 100:.1f}%\n"
                    if 'document_class' in custom_output:
                        blueprint_info += f"Document Class: {custom_output['document_class'].get('type', 'Unknown')}\n"
                
                # Determine which JSON to use for display and visualization
                if is_pdf:
                    annotated_image = self._visualise_pdf(
                        pdf_bytes=file_bytes,
                        custom_output=custom_output if use_blueprint else None,
                    )

                    if use_blueprint and custom_output and 'inference_result' in custom_output:
                        display_json = self._restore_blueprint_json(
                            custom_output=custom_output, build=blueprint_build)
                    elif not use_blueprint and structured_json:
                        display_json = structured_json
                    else:
                        display_json = custom_output if custom_output else standard_output
                else:
                    # For images, get dimensions and create annotated images
                    width, height = img_pil.size
                    
                    if use_blueprint and custom_output and 'explainability_info' in custom_output:
                        annotated_image = self._create_annotated_image_with_bda_boxes(
                            img_pil.copy(), width, height, custom_output)
                        
                        if 'inference_result' in custom_output:
                            display_json = self._restore_blueprint_json(
                                custom_output=custom_output, build=blueprint_build)
                        else:
                            display_json = custom_output
                    else:
                        # For non-blueprint approach or if custom output doesn't have explainability info
                        annotated_image = self._create_annotated_image(
                            img_pil.copy(), width, height, standard_output)
                        if not use_blueprint and structured_json:
                            display_json = structured_json
                        else:
                            display_json = standard_output
                
                # Clean up temporary blueprint if it was created
                if blueprint_arn:
                    try:
                        bda_client.delete_blueprint(blueprintArn=blueprint_arn)
                        logger.info(f"Deleted temporary blueprint: {blueprint_arn}")
                    except Exception as e:
                        logger.warning(f"Failed to delete blueprint: {str(e)}")
                
                overall_process_time = time.time() - overall_start_time
                logger.info(f"BDA total processing time: {overall_process_time:.2f}s")

                # Return results - ensure all required fields are included for processor.py
                return {
                    "text": extracted_text if not use_blueprint else blueprint_info,
                    "json": display_json,
                    "raw_json": standard_output,
                    "image": annotated_image,
                    "process_time": overall_process_time,
                    "token_usage": token_usage if not use_blueprint and token_usage else None,
                    "json_process_time": json_process_time,
                    "field_count": field_count,
                    "use_blueprint": use_blueprint,
                    "file_type": "pdf" if is_pdf else "image",
                    "pages": self._page_count(
                        is_pdf=is_pdf,
                        custom_output=custom_output,
                        standard_output=standard_output),
                    "operation_type": "bda"
                }
                
            except Exception as e:
                # Credential failures are reported by botocore without naming the
                # profile at fault, which is the one thing the user needs to know.
                error_message = describe_credential_error(e) or str(e)
                logger.error(f"Error in BDA processing: {error_message}")

                return {
                    "text": f"BDA Error: {error_message}",
                    "json": None,
                    "image": None,
                    "process_time": timing_ctx.process_time,
                    "operation_type": "error",
                    "pages": 0
                }
    
    def _process_bda_results(self, job_metadata, s3_client):
        """Process the results of BDA job from S3"""
        extracted_text = ""
        custom_output = None
        standard_output = None
        
        if 'output_metadata' in job_metadata and len(job_metadata['output_metadata']) > 0:
            asset_metadata = job_metadata['output_metadata'][0]
            
            if 'segment_metadata' in asset_metadata and len(asset_metadata['segment_metadata']) > 0:
                segments_metadata = asset_metadata['segment_metadata']
                segment_metadata = segments_metadata[0]

                if segment_metadata.get('custom_output_status') == 'MATCH' and 'custom_output_path' in segment_metadata:
                    custom_output_path = segment_metadata['custom_output_path']
                    bucket, key = self._get_bucket_and_key(custom_output_path)
                    custom_obj = s3_client.get_object(Bucket=bucket, Key=key)
                    custom_content = custom_obj['Body'].read().decode('utf-8')
                    custom_output = json.loads(custom_content)
                    logger.info("Successfully retrieved custom output")
                    
                    if custom_output and 'inference_result' in custom_output:
                        inference_result = custom_output['inference_result']
                        formatted_text = []
                        
                        for key, value in inference_result.items():
                            if key.startswith("consultants_") and not value:
                                continue
                            formatted_text.append(f"{key}: {value}")
                        
                        extracted_text = "\n".join(formatted_text)
                
                standard_output_path = None
                
                if 'standard_output_path' in segment_metadata:
                    standard_output_path = segment_metadata['standard_output_path']
                elif 'outputs' in segment_metadata and 'standard_output' in segment_metadata['outputs']:
                    standard_output_path = segment_metadata['outputs']['standard_output']
                elif 'outputs' in asset_metadata and 'standard_output' in asset_metadata['outputs']:
                    standard_output_path = asset_metadata['outputs']['standard_output']
                
                if standard_output_path:
                    bucket, key = self._get_bucket_and_key(standard_output_path)
                    output_obj = s3_client.get_object(Bucket=bucket, Key=key)
                    output_content = output_obj['Body'].read().decode('utf-8')
                    standard_output = json.loads(output_content)
                    
                    if not extracted_text:
                        extracted_text = self._extract_text_from_output(standard_output)
        
        return extracted_text, custom_output, standard_output
        
    def _extract_text_from_output(self, standard_output):
        """Extract text content from standard output"""
        extracted_text = ""
        
        if 'document' in standard_output and 'representation' in standard_output['document']:
            rep = standard_output['document']['representation']
            if 'text' in rep:
                return rep['text']
            elif 'markdown' in rep:
                return rep['markdown']
        
        if 'pages' in standard_output:
            for page in standard_output['pages']:
                if 'representation' in page:
                    page_rep = page['representation']
                    if 'text' in page_rep:
                        extracted_text += page_rep['text'] + "\n\n"
                    elif 'markdown' in page_rep:
                        extracted_text += page_rep['markdown'] + "\n\n"
        
        return extracted_text
    
    def _get_bucket_and_key(self, s3_uri):
        """Extract bucket and key from S3 URI"""
        path = s3_uri.replace('s3://', '')
        parts = path.split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        return bucket, key
        
    def _create_annotated_image(self, img_pil, width, height, standard_output):
        """Create basic annotated image with BDA processing indicator"""
        draw = ImageDraw.Draw(img_pil)
        
        border_width = 10
        draw.rectangle(
            [(0, 0), (width, height)],
            outline='#00FF00',
            width=border_width
        )
        
        draw.text(
            (20, 20),
            f"Processed with BDA ({width}x{height})",
            fill='#00FF00'
        )
        
        # Add document structure information
        if standard_output:
            y_pos = 60
            
            # Show document type information
            if 'document' in standard_output and 'statistics' in standard_output['document']:
                stats = standard_output['document']['statistics']
                draw.text(
                    (20, y_pos),
                    f"Document contains {stats.get('element_count', 0)} elements, {stats.get('table_count', 0)} tables",
                    fill='#FFFF00'
                )
                y_pos += 30
            
            # Show count of elements
            if 'elements' in standard_output:
                element_types = {}
                for element in standard_output['elements']:
                    element_type = element.get('type', 'UNKNOWN')
                    element_types[element_type] = element_types.get(element_type, 0) + 1
                
                for element_type, count in element_types.items():
                    draw.text(
                        (20, y_pos),
                        f"- {element_type}: {count}",
                        fill='#FFFF00'
                    )
                    y_pos += 25
        
        return np.array(img_pil)


        
    def _page_count(
        self,
        *,
        is_pdf: bool,
        custom_output: Optional[Dict[str, Any]],
        standard_output: Optional[Dict[str, Any]],
    ) -> int:
        """
        Determine how many pages BDA actually processed.

        This used to be hardcoded to 1, which under-billed every multi-page run:
        `calculate_bda_cost` charges per page, so a seven-page document was
        costed as one. BDA reports the real figure in two different places
        depending on which path ran, so both are read.

        Args:
            is_pdf: Whether the processed input was a PDF rather than an image.
            custom_output: BDA custom output from the blueprint path, or None.
            standard_output: BDA standard output, or None. Its `metadata` carries
                `number_of_pages` for the whole document.

        Returns:
            The number of pages processed. Always 1 for a single image.

        Raises:
            ValueError: If a PDF was processed but neither payload names a page
                count. Returning 1 here is the silent under-billing being fixed,
                so it is an error instead.
        """
        if not is_pdf:
            # One image is one billable unit, and BDA prices images separately.
            return 1

        reported = (standard_output or {}).get("metadata", {}).get("number_of_pages")
        if isinstance(reported, int) and reported > 0:
            return reported

        # The blueprint path can return custom output without standard output.
        # `split_document.page_indices` names every page the blueprint matched,
        # 0-based, so its length is the page count.
        page_indices = (custom_output or {}).get("split_document", {}).get("page_indices")
        if isinstance(page_indices, list) and page_indices:
            return len(set(page_indices))

        raise ValueError(
            "BDA processed a PDF but reported no page count in either "
            "standard_output.metadata.number_of_pages or "
            "custom_output.split_document.page_indices, so the run cannot be costed")

    def _visualise_pdf(
        self, *, pdf_bytes: bytes, custom_output: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Render a processed PDF's pages with the boxes BDA reported.

        The PDF path used to answer with a 400x600 black rectangle, discarding
        the `explainability_info` geometry BDA had already returned. Each page is
        rendered and every extracted value is boxed on the page it was found on,
        coloured per top-level field and labelled with its confidence.

        Args:
            pdf_bytes: The PDF that was processed.
            custom_output: BDA custom output, or None for a run that produced no
                blueprint result and therefore no geometry.

        Returns:
            The composed visualisation as an RGB numpy array, which is what the
            Gradio image output takes.
        """
        boxes: List[PageBox] = []
        if custom_output and custom_output.get("explainability_info"):
            boxes = boxes_from_bda_explainability(
                explainability_info=custom_output["explainability_info"]
            )

        return np.array(
            compose_pdf_visualisation(
                pdf_bytes=pdf_bytes, boxes=boxes, item_noun="values located"
            )
        )

    def _create_annotated_image_with_bda_boxes(self, img_pil, width, height, custom_output):
        """Create annotated image with BDA custom output boxes"""
        draw = ImageDraw.Draw(img_pil)
        
        border_width = 10
        draw.rectangle(
            [(0, 0), (width, height)],
            outline='#00FF00',
            width=border_width
        )
        
        draw.text(
            (20, 20),
            f"Processed with BDA ({width}x{height})",
            fill='#00FF00'
        )
        
        if custom_output and 'explainability_info' in custom_output:
            colors = [
                '#FF0000', '#00FFFF', '#FFFF00', '#FF00FF', 
                '#FFA500', '#0000FF', '#800080', '#008000'
            ]
            
            color_index = 0
            drawn_fields = set()
            
            for field_group in custom_output['explainability_info']:
                for field_name, field_info in field_group.items():
                    if field_name in drawn_fields:
                        continue
                        
                    if 'geometry' in field_info and field_info['geometry']:
                        color = colors[color_index % len(colors)]
                        color_index += 1
                        
                        for geo in field_info['geometry']:
                            box = geo['boundingBox']
                            x1 = int(box['left'] * width)
                            y1 = int(box['top'] * height)
                            x2 = x1 + int(box['width'] * width)
                            y2 = y1 + int(box['height'] * height)
                            
                            draw.rectangle(
                                [(x1, y1), (x2, y2)],
                                outline=color,
                                width=3
                            )
                            
                            field_label = field_name
                            conf = field_info.get('confidence', 0) * 100
                            label_text = f"{field_label} ({conf:.0f}%)"
                            
                            draw.text(
                                (x1, y1 - 15),
                                label_text,
                                fill=color
                            )
                        
                        drawn_fields.add(field_name)
        
        return np.array(img_pil)
    
    def get_cost(self, result: Dict[str, Any]) -> Tuple[str, float]:
        """
        Calculate the cost for BDA processing
        
        Args:
            result: Result dictionary from process_image
            
        Returns:
            Tuple of (HTML representation of cost, actual cost value)
        """
        use_blueprint = result.get('use_blueprint', False)
        document_type = 'document'  # Currently only supporting document type
        page_count = 1  # Currently only supporting single page
        field_count = result.get('field_count', 0)
        
        if use_blueprint:
            # Custom Output with blueprint
            cost_per_unit = API_COSTS['bda']['custom'][document_type]
            
            # Calculate additional cost for extra fields beyond 30
            extra_field_cost = 0
            if field_count > 30:
                extra_field_cost = API_COSTS['bda']['custom']['extra_field'] * (field_count - 30) * page_count
                
            bda_base_cost = (cost_per_unit * page_count) + extra_field_cost
            total_cost = bda_base_cost
            
            # Format HTML output
            html = f'''
            <div class="cost-container">
                <div class="cost-total">${total_cost:.6f} total</div>
                <div class="cost-breakdown">
                    <span>${cost_per_unit:.6f} per {document_type} \u00d7 {page_count} {document_type}s</span>
                    {f"<br><span>${extra_field_cost:.6f} for additional {field_count-30} fields</span>" if field_count > 30 else ""}
                </div>
            </div>
            '''
        else:
            # Standard Output
            cost_per_unit = API_COSTS['bda']['standard'][document_type]
            bda_base_cost = cost_per_unit * page_count
            total_cost = bda_base_cost
            
            # Add LLM cost if applicable
            token_usage = result.get('token_usage')
            if token_usage:
                from shared.cost_calculator import calculate_bedrock_cost
                _, postprocessing_cost = calculate_bedrock_cost(POSTPROCESSING_MODEL, token_usage)
                total_cost += postprocessing_cost
                
                # Format HTML output with LLM cost
                html = f'''
                <div class="cost-container">
                    <div class="cost-total">${total_cost:.6f} total</div>
                    <div class="cost-breakdown">
                        <span>${bda_base_cost:.6f} for BDA standard extraction</span><br>
                        <span>${postprocessing_cost:.6f} for LLM post-processing</span>
                    </div>
                </div>
                '''
            else:
                # Format HTML output without LLM cost
                html = f'''
                <div class="cost-container">
                    <div class="cost-total">${total_cost:.6f} total</div>
                    <div class="cost-breakdown">
                        <span>${cost_per_unit:.6f} per {document_type} \u00d7 {page_count} {document_type}s</span>
                    </div>
                </div>
                '''
        
        return html, total_cost
