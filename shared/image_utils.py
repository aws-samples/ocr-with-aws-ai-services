import io
import numpy as np
import logging
from PIL import Image
from functools import lru_cache
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

# Global configuration settings
MAX_IMAGE_SIZE = 5 * 1024 * 1024 - 100000  # 5MB minus buffer for Bedrock

def get_image_hash(image):
    """
    Generate a hash for an image to use as cache key
    
    Args:
        image: PIL Image, numpy array, or path to image
        
    Returns:
        String hash of the image content
    """
    if isinstance(image, str):
        with open(image, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    elif isinstance(image, np.ndarray):
        return hashlib.sha256(image.tobytes()).hexdigest()
    elif isinstance(image, Image.Image):
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        return hashlib.sha256(img_bytes.getvalue()).hexdigest()
    else:
        return hashlib.sha256(str(image).encode()).hexdigest()

def get_image_object(image):
    """
    Convert various image formats to PIL Image
    
    Args:
        image: PIL Image, numpy array, or path to image
        
    Returns:
        PIL Image object
    """
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert('RGB')
    elif isinstance(image, str):
        return Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        return image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

def convert_to_bytes(image, max_size=None):
    """
    Convert various image formats to bytes with size limit
    
    Args:
        image: PIL Image, numpy array, or path to image
        max_size: Maximum size in bytes for the output image
        
    Returns:
        Tuple of (image bytes, PIL Image)
    """
    img_pil = get_image_object(image)
    
    # Convert RGBA images to RGB before processing
    if img_pil.mode == 'RGBA':
        # Convert from RGBA mode to RGB (using white background)
        background = Image.new('RGB', img_pil.size, (255, 255, 255))
        background.paste(img_pil, mask=img_pil.split()[3])  # Use alpha channel as mask
        img_pil = background
    elif img_pil.mode != 'RGB':
        # Convert other non-RGB modes to RGB
        img_pil = img_pil.convert('RGB')
    
    # Get original size info for logging
    original_byte_arr = io.BytesIO()
    img_pil.save(original_byte_arr, format='JPEG', quality=85, optimize=True)
    original_size = original_byte_arr.tell()
    logger.info(f"Original image size: {original_size / 1024:.2f}KB ({img_pil.width}x{img_pil.height})")
    
    # If max_size is None, use global default
    if max_size is None:
        max_size = MAX_IMAGE_SIZE
    
    # Resize image if needed
    if max_size and original_size > max_size:
        logger.info(f"Image needs resizing - current size: {original_size / 1024:.2f}KB, max: {max_size / 1024:.2f}KB")
        
        # Progressive approach: reduce quality first, then dimensions if needed
        qualities = [85, 75, 65, 55, 45, 35]
        scale_factors = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        
        # Try reducing quality first
        for quality in qualities:
            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
            current_size = img_byte_arr.tell()
            
            if current_size <= max_size:
                logger.info(f"Reduced quality to {quality}, final size: {current_size / 1024:.2f}KB")
                img_byte_arr.seek(0)
                return img_byte_arr.getvalue(), img_pil
                
        # If quality reduction isn't enough, reduce dimensions
        current_img = img_pil
        for scale in scale_factors:
            new_width = int(img_pil.width * scale)
            new_height = int(img_pil.height * scale)
            resized_img = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Try with lowest quality
            img_byte_arr = io.BytesIO()
            resized_img.save(img_byte_arr, format='JPEG', quality=qualities[-1], optimize=True)
            current_size = img_byte_arr.tell()
            
            if current_size <= max_size:
                logger.info(f"Resized to {new_width}x{new_height}, scale: {scale:.2f}, final size: {current_size / 1024:.2f}KB")
                img_byte_arr.seek(0)
                return img_byte_arr.getvalue(), resized_img
                
            current_img = resized_img
        
        # If we get here, we couldn't reduce enough
        img_byte_arr = io.BytesIO()
        current_img.save(img_byte_arr, format='JPEG', quality=qualities[-1], optimize=True)
        final_size = img_byte_arr.tell()
        logger.warning(f"Failed to reduce image to target size: {final_size / 1024:.2f}KB > {max_size / 1024:.2f}KB")
        img_byte_arr.seek(0)
        return img_byte_arr.getvalue(), current_img
    else:
        # No resizing needed
        img_byte_arr = io.BytesIO()
        img_pil.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
        img_byte_arr.seek(0)
        logger.info(f"No resize needed. Final size: {img_byte_arr.tell() / 1024:.2f}KB")
        return img_byte_arr.getvalue(), img_pil

@lru_cache(maxsize=32)
def get_optimized_image(image_path, max_size=None):
    """
    Cache and optimize images to avoid repeated processing
    
    Args:
        image_path: Path to image file
        max_size: Maximum size in bytes for the output image
        
    Returns:
        Tuple of (image bytes, PIL Image)
    """
    return convert_to_bytes(image_path, max_size)