# OCR with AWS AI Services

A comprehensive application for comparing OCR (Optical Character Recognition) capabilities across multiple AWS AI services: Amazon Textract, Amazon Bedrock, and Amazon Bedrock Data Automation (BDA).


## Overview

This application provides a unified interface for extracting text and structured data from images using three different AWS AI services:

1. **Amazon Textract**: AWS's dedicated OCR service for extracting text, forms, and tables from documents
   - After calling the Amazon Textract API to extract text, the application uses LLM to structure the extracted data into JSON format according to the provided schema.
2. **Amazon Bedrock**: Using foundation models like Claude for document understanding and extraction
   - Uses foundation models directly for both extraction and JSON structuring in a single step.
3. **Amazon Bedrock Data Automation (BDA)**: AWS's specialized image/document analysis service
   - **Custom Blueprint Method**: Creates a custom document processing blueprint based on the provided JSON schema
   - **LLM Post-processing Method**: Uses standard BDA extraction followed by Bedrock LLM to structure the data (default method)

The application enables side-by-side comparison of these services' accuracy, cost, and processing time across different document types, helping you choose the optimal service for your specific OCR needs.

## Key Features

- **Multi-Engine OCR Processing**: Process the same document with Textract, Bedrock, and BDA simultaneously
- **Interactive UI**: User-friendly interface for testing and comparing OCR engines
- **Performance Comparison**: Side-by-side comparison of extraction quality, processing time, and cost
- **Accuracy Evaluation**: Compare extracted data against ground truth for objective evaluation
- **JSON Schema Support**: Structure extracted data according to custom schemas
- **Cost Calculation**: Real-time cost estimation for each service
- **Batch Processing**: Process multiple sample documents at once
- **Result Visualization**: Visual annotation of detected text elements


## Architecture

The application follows a modular architecture with several key components:

- **Engine Implementations**: Separate modules for each AWS service (Textract, Bedrock, BDA)
- **User Interface**: Gradio-based UI for interactive testing and result visualization
- **Core Processing**: Parallel execution of OCR engines with standardized result handling
- **Sample Management**: Utilities for working with test documents and sample data
- **Evaluation Tools**: Components for accuracy assessment and comparison

## Requirements

- Python 3.10+
- AWS Account with access to:
  - Amazon Textract
  - Amazon Bedrock (with access to supported models)
  - Amazon Bedrock Data Autmation
- AWS credentials configured locally
- S3 bucket for BDA processing

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/aws-samples/ocr-with-aws-ai-services.git
   cd ocr-with-aws-ai-services
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Configure AWS credentials using one of the following methods:
   - AWS CLI: `aws configure`
   - Environment variables
   - Credentials file (~/.aws/credentials)

## Usage

### Starting the Application

Run the application with:

```
python app.py
```

This will start the Gradio web interface, typically accessible at http://localhost:7860 (or the URL displayed in your terminal).

### Using the Interface

1. **Select or Upload an Image**:
   - Choose from sample images in the dropdown, or
   - Upload your own image using the upload control

2. **Select OCR Engines**:
   - Choose one or more OCR engines to use (Textract, Bedrock, BDA)
   - Configure engine-specific options as needed

3. **Set Processing Options**:
   - Document type (generic, form, receipt, table, handwritten)
   - Output JSON schema (for structured data extraction)
   - Model selection for Bedrock
   - S3 bucket and blueprint options for BDA

4. **Process the Image**:
   - Click "Process Sample" to analyze the current image
   - Click "Process All Samples" to batch process all sample images

5. **View Results**:
   - Navigate between tabs to see results from each engine
   - Compare extracted text, structured JSON, and annotated images
   - View performance metrics including processing time, cost, and accuracy
   - Use the "Compare" tab to see detailed comparison with ground truth

Modify these settings as needed for your environment.

## Sample Data

The repository includes sample documents in the `sample/` directory:

- `sample/images/`: Test images for OCR processing
- `sample/schema/`: JSON schemas for structured extraction
- `sample/truth/`: Ground truth data for accuracy evaluation

To add your own samples:

1. Add images to `sample/images/`
2. (Optional) Add corresponding JSON schemas with the same base filename in `sample/schema/`
3. (Optional) Add ground truth data with the same base filename in `sample/truth/`

## Results

Processing results are stored in the `results/` directory, organized by processing run and sample. Each run includes:

- Extracted text
- JSON structured data
- Annotated visualization images
- Processing metadata (time, cost, accuracy)
- Summary reports


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- AWS AI Services teams for providing the underlying OCR capabilities
- Contributors to the Gradio framework for the UI components