import io
import base64
import html
from pathlib import Path

from gradio.processing_utils import get_upload_folder
from PIL import Image

from shared.config import logger
from shared.sample_paths import SAMPLE_DIR
from shared.ui_theme import banner, note, page_readout

# Try to import PDF processing libraries
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logger.info("PyMuPDF not available - using embedded PDF viewer only")


def _resolve_preview_path(file_path) -> Path:
    """
    Resolve a preview file only from directories the app owns

    Gradio sends uploaded files through its server-side upload folder. Sample
    selections use the repository's sample directory. No other client-provided path
    is a valid preview source.
    """
    candidate = Path(file_path).resolve(strict=True)
    allowed_roots = (
        Path(get_upload_folder()).resolve(),
        (Path.cwd() / SAMPLE_DIR).resolve(),
    )

    if not any(candidate == root or root in candidate.parents for root in allowed_roots):
        raise ValueError("Preview file is outside the upload and sample directories")
    if not candidate.is_file():
        raise ValueError("Preview source is not a file")

    return candidate


def handle_file_preview(file):
    """
    Handle preview for uploaded files (images or PDFs)
    
    Args:
        file: Uploaded file object from Gradio
        
    Returns:
        Tuple of (image_preview, pdf_preview, pdf_controls_visible, current_page, total_pages, pdf_path)
    """
    if file is None:
        return (None, note(text="Upload a file to see a preview", tall=True),
                False, 0, 1, None)


    untrusted_path = file.name if hasattr(file, 'name') else str(file)
    try:
        file_path = _resolve_preview_path(untrusted_path)
    except (OSError, ValueError) as path_error:
        logger.error(f"Rejected preview path: {path_error}")
        return (None, banner(tone="error", text="Could not open that file"),
                False, 0, 1, None)

    file_ext = file_path.suffix.lower()
    
    logger.info(f"Handling preview for file: {file_path} (extension: {file_ext})")
    
    if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
        # Handle image files
        try:
            image = Image.open(file_path)
            logger.info(f"Loaded image preview: {image.size}")
            return (image, note(text="Image preview shown above"),
                   False, 0, 1, None)
        except Exception as e:
            logger.error(f"Error loading image preview: {str(e)}")
            return (None, banner(tone="error", text=f"Could not load image: {e}"),
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
                           page_count > 1, 0, page_count, str(file_path))
            
            # Fallback to embedded PDF viewer
            pdf_preview_html = create_pdf_preview(file_path)
            return (None, pdf_preview_html, False, 0, page_count, str(file_path))
        except Exception as e:
            logger.error(f"Error creating PDF preview: {str(e)}")
            return (None, banner(tone="error", text=f"Could not load PDF: {e}"),
                   False, 0, 1, None)

    else:
        return (None, note(text=f"No preview available for {file_ext} files", tall=True),
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
        
        # This viewer frames a rendered PDF page, which is white, so its chrome stays
        # light in both themes on purpose - a dark bezel around a white page reads as a
        # rendering fault. Unlike the code this replaced it sets background AND text
        # colour together, so it is self-consistent rather than dependent on the theme
        # supplying a light background for it.
        html_content = f"""
        <div style="width: 100%; height: 500px; border: 1px solid #c8ccd4; border-radius: 8px; overflow: hidden; background: #ffffff; color: #1e293b;">
            <div style="background: #eef1f5; color: #1e293b; padding: 8px 10px; font-size: 13px; border-bottom: 1px solid #c8ccd4; display: flex; justify-content: space-between; align-items: center;">
                <span>{html.escape(Path(pdf_path).name)}</span>
                <span style="font-size: 12px; opacity: 0.7;">{pdf_size:.1f} KB</span>
            </div>


            <div style="height: 460px; width: 100%; position: relative;">
                <iframe 
                    src="data:application/pdf;base64,{pdf_base64}#toolbar=1&navpanes=1&scrollbar=1" 
                    width="100%" 
                    height="100%"
                    style="border: none; display: block;"
                    title="PDF Preview">
                </iframe>
                
                <div id="pdf-fallback" style="display: none; padding: 40px; text-align: center; height: 100%; box-sizing: border-box; background: #ffffff; color: #1e293b;">
                    <div style="background: #f4f6f9; color: #1e293b; padding: 30px; border-radius: 8px; border: 1px dashed #a9b1bd;">
                        <p style="margin: 0 0 12px 0; font-size: 15px; font-weight: 600;">
                            {html.escape(Path(pdf_path).name)}
                        </p>
                        <p style="margin: 6px 0; font-size: 13px; opacity: 0.75;">
                            {pdf_size:.1f} KB · ready for OCR processing
                        </p>
                        <p style="margin: 16px 0 0 0; font-size: 13px; opacity: 0.75;">
                            This browser cannot display an embedded PDF.
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
        return banner(tone="error", text=f"Could not create a PDF preview: {e}")

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
        return f"""
        <div class='ocr-card'>
            <h4 class='ocr-card__title'>PDF document</h4>
            <p class='ocr-card__row'><span>File</span><span>{html.escape(Path(pdf_path).name)}</span></p>
            <p class='ocr-card__row'><span>Pages</span><span>{total_pages}</span></p>
            <p class='ocr-card__footer'>Showing page {current_page + 1} of {total_pages}</p>
        </div>
        """

    except Exception as e:
        logger.error(f"Could not read PDF metadata for {pdf_path}: {e}")
        return note(text="PDF loaded; document details unavailable")

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
    if not pdf_path:
        return (None, banner(tone="error", text="PDF not found"),
                page_readout(current_page=0, total_pages=1))

    try:
        resolved_pdf_path = _resolve_preview_path(pdf_path)
    except (OSError, ValueError):
        return (None, banner(tone="error", text="PDF not found"),
                page_readout(current_page=0, total_pages=1))

    # Ensure page number is within bounds
    page_num = max(0, min(page_num, total_pages - 1))
    page_info_html = page_readout(current_page=page_num, total_pages=total_pages)

    try:
        if HAS_PYMUPDF:
            pdf_image = convert_pdf_to_image(resolved_pdf_path, page_num=page_num)
            if pdf_image:
                info_html = create_pdf_info_html(
                    resolved_pdf_path, page_num, total_pages)
                return pdf_image, info_html, page_info_html

        # Fallback
        return None, note(text="Page navigation needs PyMuPDF"), page_info_html

    except Exception as e:
        logger.error(f"Error navigating PDF page: {str(e)}")
        return (None, banner(tone="error", text=f"Could not load page: {e}"),
                page_info_html)

# handle_sample_preview() used to live here. It only called Image.open(), so it could
# not preview a PDF sample, and it duplicated handle_file_preview(): selecting a sample
# assigns the resolved path to the input_image File component, and Gradio's .change()
# fires on programmatic updates too, so handle_file_preview() ran immediately
# afterwards and overwrote its output. Sample selection now leaves previewing entirely
# to the input_image.change handler, which is also the only one that sets up the PDF
# page-navigation state.
