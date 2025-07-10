import os
import io
import base64
from PIL import Image
import gradio as gr
from shared.config import logger

# Try to import PDF processing libraries
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logger.info("PyMuPDF not available - using embedded PDF viewer only")

def handle_file_preview(file):
    """
    Handle preview for uploaded files (images or PDFs)
    
    Args:
        file: Uploaded file object from Gradio
        
    Returns:
        Tuple of (image_preview, pdf_preview, pdf_controls_visible, current_page, total_pages, pdf_path)
    """
    if file is None:
        return (None, "<div style='text-align: center; padding: 50px; color: #666;'>Upload a file to see preview</div>", 
                False, 0, 1, None)
    
    file_path = file.name if hasattr(file, 'name') else str(file)
    file_ext = os.path.splitext(file_path)[1].lower()
    
    logger.info(f"Handling preview for file: {file_path} (extension: {file_ext})")
    
    if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
        # Handle image files
        try:
            image = Image.open(file_path)
            logger.info(f"Loaded image preview: {image.size}")
            return (image, "<div style='text-align: center; padding: 50px; color: #666;'>Image preview shown above</div>", 
                   False, 0, 1, None)
        except Exception as e:
            logger.error(f"Error loading image preview: {str(e)}")
            return (None, f"<div style='text-align: center; padding: 50px; color: #ff6b6b;'>Error loading image: {str(e)}</div>", 
                   False, 0, 1, None)
    
    elif file_ext == '.pdf':
        # Handle PDF files
        try:
            # Get PDF page count
            page_count = get_pdf_page_count(file_path)
            logger.info(f"PDF has {page_count} pages")
            
            # Try to convert PDF to image first (more reliable)
            if HAS_PYMUPDF:
                pdf_image = convert_pdf_to_image(file_path, page_num=0)
                if pdf_image:
                    logger.info(f"Converted PDF to image for preview: {file_path}")
                    return (pdf_image, create_pdf_info_html(file_path, 0, page_count), 
                           page_count > 1, 0, page_count, file_path)
            
            # Fallback to embedded PDF viewer
            pdf_preview_html = create_pdf_preview(file_path)
            return (None, pdf_preview_html, False, 0, page_count, file_path)
        except Exception as e:
            logger.error(f"Error creating PDF preview: {str(e)}")
            return (None, f"<div style='text-align: center; padding: 50px; color: #ff6b6b;'>Error loading PDF: {str(e)}</div>", 
                   False, 0, 1, None)
    
    else:
        return (None, f"<div style='text-align: center; padding: 50px; color: #666;'>File type {file_ext} not supported for preview</div>", 
               False, 0, 1, None)

def create_pdf_preview(pdf_path):
    """
    Create HTML preview for PDF files using multiple fallback methods
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: HTML content for PDF preview
    """
    try:
        # Read PDF file and encode to base64
        with open(pdf_path, 'rb') as pdf_file:
            pdf_data = pdf_file.read()
            pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
            pdf_size = len(pdf_data) / 1024  # Size in KB
        
        # Create HTML with simplified and more compatible PDF viewer
        html_content = f"""
        <div style="width: 100%; height: 500px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; background: white;">
            <div style="background: #f5f5f5; padding: 8px; font-size: 14px; border-bottom: 1px solid #ddd; display: flex; justify-content: space-between; align-items: center;">
                <span>📄 PDF Preview - {os.path.basename(pdf_path)}</span>
                <span style="font-size: 12px; color: #666;">{pdf_size:.1f} KB</span>
            </div>
            
            <div style="height: 460px; width: 100%; position: relative;">
                <iframe 
                    src="data:application/pdf;base64,{pdf_base64}#toolbar=1&navpanes=1&scrollbar=1" 
                    width="100%" 
                    height="100%"
                    style="border: none; display: block;"
                    title="PDF Preview">
                </iframe>
                
                <div id="pdf-fallback" style="display: none; padding: 40px; text-align: center; height: 100%; box-sizing: border-box;">
                    <div style="background: #f8f9fa; padding: 30px; border-radius: 8px; border: 2px dashed #dee2e6;">
                        <h3 style="margin: 0 0 15px 0; color: #495057;">📄 PDF Document Loaded</h3>
                        <p style="margin: 10px 0; color: #6c757d; font-size: 16px;">
                            <strong>{os.path.basename(pdf_path)}</strong>
                        </p>
                        <p style="margin: 10px 0; color: #868e96;">
                            Size: {pdf_size:.1f} KB
                        </p>
                        <p style="margin: 20px 0 0 0; color: #adb5bd; font-size: 14px;">
                            PDF is ready for OCR processing.<br>
                            Preview not available in this browser.
                        </p>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
        // Check if PDF loaded successfully
        setTimeout(function() {{
            var iframe = document.querySelector('iframe[title="PDF Preview"]');
            if (iframe) {{
                iframe.onerror = function() {{
                    document.getElementById('pdf-fallback').style.display = 'block';
                    iframe.style.display = 'none';
                }};
            }}
        }}, 1000);
        </script>
        """
        
        logger.info(f"Created PDF preview for: {pdf_path} ({pdf_size:.1f} KB)")
        return html_content
        
    except Exception as e:
        logger.error(f"Error creating PDF preview: {str(e)}")
        return f"""
        <div style='text-align: center; padding: 50px; color: #ff6b6b; border: 1px solid #ddd; border-radius: 8px;'>
            <p>❌ Error creating PDF preview</p>
            <p style="font-size: 12px; color: #999;">{str(e)}</p>
        </div>
        """

def convert_pdf_to_image(pdf_path, page_num=0, dpi=150):
    """
    Convert PDF page to PIL Image using PyMuPDF
    
    Args:
        pdf_path: Path to the PDF file
        page_num: Page number to convert (0-indexed)
        dpi: Resolution for the conversion
        
    Returns:
        PIL Image object or None if conversion fails
    """
    if not HAS_PYMUPDF:
        return None
        
    try:
        # Open PDF document
        doc = fitz.open(pdf_path)
        
        # Check if page exists
        if page_num >= len(doc):
            page_num = 0
            
        # Get the page
        page = doc[page_num]
        
        # Convert page to image
        zoom = dpi / 72  # PDF default is 72 DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img_data = pix.tobytes("ppm")
        pil_image = Image.open(io.BytesIO(img_data))
        
        doc.close()
        logger.info(f"Converted PDF page {page_num} to image: {pil_image.size}")
        return pil_image
        
    except Exception as e:
        logger.error(f"Error converting PDF to image: {str(e)}")
        return None

def get_pdf_page_count(pdf_path):
    """
    Get the number of pages in a PDF
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        int: Number of pages
    """
    if HAS_PYMUPDF:
        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
            return page_count
        except Exception as e:
            logger.error(f"Error getting PDF page count: {str(e)}")
            return 1
    else:
        return 1

def create_pdf_info_html(pdf_path, current_page=0, total_pages=1):
    """
    Create info HTML for PDF files when displayed as image
    
    Args:
        pdf_path: Path to the PDF file
        current_page: Current page number (0-indexed)
        total_pages: Total number of pages
        
    Returns:
        str: HTML content with PDF info
    """
    try:
        file_size = os.path.getsize(pdf_path) / 1024  # Size in KB
        
        return f"""
        <div style='text-align: center; padding: 20px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; margin-top: 10px;'>
            <h4 style='margin: 0 0 10px 0; color: #495057;'>📄 PDF Document Info</h4>
            <p style='margin: 5px 0; color: #6c757d;'><strong>File:</strong> {os.path.basename(pdf_path)}</p>
            <p style='margin: 5px 0; color: #6c757d;'><strong>Size:</strong> {file_size:.1f} KB</p>
            <p style='margin: 5px 0; color: #6c757d;'><strong>Pages:</strong> {total_pages}</p>
            <p style='margin: 10px 0 0 0; font-size: 14px; color: #868e96;'>Showing page {current_page + 1} of {total_pages}</p>
        </div>
        """
        
    except Exception as e:
        return f"""
        <div style='text-align: center; padding: 20px; color: #6c757d;'>
            📄 PDF loaded successfully
        </div>
        """

def navigate_pdf_page(pdf_path, page_num, total_pages):
    """
    Navigate to a specific page in PDF preview
    
    Args:
        pdf_path: Path to the PDF file
        page_num: Page number to show (0-indexed)
        total_pages: Total number of pages
        
    Returns:
        Tuple of (image, info_html, page_info_html)
    """
    if not pdf_path or not os.path.exists(pdf_path):
        return None, "PDF not found", "<div>Page 1 of 1</div>"
    
    # Ensure page number is within bounds
    page_num = max(0, min(page_num, total_pages - 1))
    
    try:
        if HAS_PYMUPDF:
            pdf_image = convert_pdf_to_image(pdf_path, page_num=page_num)
            if pdf_image:
                info_html = create_pdf_info_html(pdf_path, page_num, total_pages)
                page_info_html = f"<div style='text-align: center; padding: 8px;'>Page {page_num + 1} of {total_pages}</div>"
                return pdf_image, info_html, page_info_html
        
        # Fallback
        page_info_html = f"<div style='text-align: center; padding: 8px;'>Page {page_num + 1} of {total_pages}</div>"
        return None, "Page navigation not available", page_info_html
        
    except Exception as e:
        logger.error(f"Error navigating PDF page: {str(e)}")
        return None, f"Error loading page: {str(e)}", "<div>Error</div>"

def handle_sample_preview(sample_path):
    """
    Handle preview for sample images
    
    Args:
        sample_path: Path to the sample image
        
    Returns:
        Tuple of (image_preview, pdf_preview, image_visible, pdf_visible)
    """
    if not sample_path or not os.path.exists(sample_path):
        return None, "<div style='text-align: center; padding: 50px; color: #666;'>No sample selected</div>", True, False
    
    try:
        image = Image.open(sample_path)
        logger.info(f"Loaded sample image preview: {sample_path}")
        return image, "<div style='text-align: center; padding: 50px; color: #666;'>Sample image preview</div>", True, False
    except Exception as e:
        logger.error(f"Error loading sample preview: {str(e)}")
        return None, f"<div style='text-align: center; padding: 50px; color: #ff6b6b;'>Error loading sample: {str(e)}</div>", True, False