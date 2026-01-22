"""OCR module for text extraction from images."""

from ocr.ocr_engine import OCREngine
from ocr.gemini_vision_engine import GeminiVisionEngine
from ocr.hybrid_ocr_engine import HybridOCREngine

# Legacy import for backwards compatibility
try:
    from ocr.llama_vision_engine import LlamaVisionEngine
except ImportError:
    LlamaVisionEngine = None

__all__ = ['OCREngine', 'GeminiVisionEngine', 'HybridOCREngine', 'LlamaVisionEngine']

