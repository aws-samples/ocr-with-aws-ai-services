# Textract Processor Deployment Guide

## CloudFormation Stack Features

This CloudFormation template deploys an OCR document processing solution with the following components:

- **S3 Bucket**: Storage for original images and processing results
- **Lambda Function**: Text extraction using Amazon Textract and JSON structuring with Bedrock AI
- **DynamoDB Table**: Storage for extracted data
- **IAM Role**: Required AWS service permissions

## Post-Deployment Configuration

### 1. S3 Event Notification Setup

After deployment, you must **manually** configure S3 event notifications:

1. Navigate to the created S3 bucket in AWS Console
2. Go to Properties > Event notifications > Create event notification
3. Configure:
   - **Event types**: `All object create events`
   - **Prefix**: `images/`
   - **Destination**: Select Lambda function and choose the created textract-processor function

### 2. S3 Bucket Folder Structure

Prepare the following prefix paths (created automatically on first file upload):

```
bucket-name/
    images/                    # Original image upload location (Event trigger)
    schema/                    # JSON schema files
       schema.json           # Data structure definition
       field_description.json # Field descriptions
    annotated_images/         # Debug images with text regions highlighted
    csv_files/                # Session-based result CSV files
```

### 3. Schema File Upload

Upload the following files to the `schema/` folder:
- `schema.json`: JSON structure definition for data extraction
- `field_description.json`: Description for each field

## Usage

1. Upload image files to the `images/` folder
2. Lambda function automatically processes OCR
3. Check results in DynamoDB

## Result Verification

### DynamoDB Check
1. AWS Console > DynamoDB > Tables
2. Select `Textract-ImageExtractions` table
3. View processed data in Items tab:
   - `sessionId`: Date-based session (YYYY-MM-DD)
   - `imageKey`: Processed image filename
   - `jsonData`: Structured extracted data
   - `confidence`: Text extraction confidence score
   - `processedAt`: Processing timestamp

### S3 Check
- `annotated_images/`: Debug images with text regions highlighted
- `csv_files/`: Consolidated CSV files per session (`YYYY-MM-DD.csv`)