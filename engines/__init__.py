# Expose OCR engine classes for easy imports
from .base import OCREngine
from .textract_engine import TextractEngine
from .bedrock_engine import BedrockEngine
from .bda_engine import BDAEngine

# These names are re-exported deliberately, so `__all__` declares them as this
# package's public surface. Without it a linter reads every one as dead.
__all__ = [
    "OCREngine",
    "TextractEngine",
    "BedrockEngine",
    "BDAEngine",
]
