from abc import ABC, abstractmethod
import time
from typing import Dict, Any, Optional, Tuple

class OCREngine(ABC):
    """
    Base class for all OCR engines
    
    This provides a common interface for different OCR services (Textract, Bedrock, BDA).
    Each implementation should provide the specific logic for that service.
    """
    
    def __init__(self, name: str):
        """
        Initialize the OCR engine
        
        Args:
            name: Name of the engine for display and logging purposes
        """
        self.name = name
        
    @abstractmethod
    def process_image(self, image, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an image and return extracted text and metadata
        
        Args:
            image: PIL Image, numpy array, or path to image file
            options: Dictionary of options specific to the engine
            
        Returns:
            Dictionary containing at minimum:
            - text: Extracted text from the image
            - image: Annotated image showing processing results (numpy array)
            - process_time: Time taken to process the image
        """
        pass
        
    @abstractmethod
    def get_cost(self, result: Dict[str, Any]) -> Tuple[str, float]:
        """
        Calculate the cost for processing
        
        Args:
            result: The result dictionary returned by process_image
            
        Returns:
            Tuple of (HTML representation of cost, actual cost value)
        """
        pass
    
    def get_timing_wrapper(self):
        """
        Create a context manager for timing operations
        
        Returns:
            A context manager that tracks execution time
        """
        class TimingContext:
            def __init__(self_ctx):
                self_ctx.start_time = None
                self_ctx.process_time = 0
                
            def __enter__(self_ctx):
                self_ctx.start_time = time.time()
                return self_ctx
                
            def __exit__(self_ctx, exc_type, exc_val, exc_tb):
                self_ctx.process_time = max(0.001, time.time() - self_ctx.start_time)  # Ensure minimum value of 1ms
                
        return TimingContext()