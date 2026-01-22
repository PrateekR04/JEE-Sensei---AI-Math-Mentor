"""
Hybrid OCR Engine: Gemini Vision (Primary) + EasyOCR (Fallback)

This engine provides robust math OCR by:
1. First attempting extraction via Gemini Flash (free tier)
2. Falling back to EasyOCR if VLM fails or returns empty results

Benefits:
- VLM provides better context understanding for complex math
- EasyOCR ensures offline fallback capability
- LaTeX extraction available via VLM
"""

import os
import time
from typing import Tuple, Optional
from ocr.ocr_engine import OCREngine


class HybridOCREngine:
    """
    Hybrid OCR engine that uses Gemini Vision as primary
    and EasyOCR as fallback for math text extraction.
    """
    
    def __init__(self, use_vlm: bool = True, use_gpu: bool = False):
        """
        Initialize the hybrid OCR engine.
        
        Args:
            use_vlm: Whether to use VLM (Gemini Vision) as primary. Default True.
            use_gpu: Whether to use GPU for EasyOCR. Default False.
        """
        self.use_vlm = use_vlm
        self.vlm_engine = None
        self.ocr_engine = None
        self.vlm_available = False
        self.vlm_type = None
        
        # Initialize VLM engine if requested
        if use_vlm:
            self._init_vlm_engine()
        
        # Initialize EasyOCR as fallback (always available)
        self._init_ocr_engine(use_gpu)
        
        print(f"✓ Hybrid OCR Engine initialized")
        print(f"  - VLM ({self.vlm_type or 'None'}): {'Enabled' if self.vlm_available else 'Disabled/Unavailable'}")
        print(f"  - EasyOCR Fallback: Enabled")
    
    def _init_vlm_engine(self):
        """Initialize Vision Language Model engine."""
        # Try Gemini first (preferred)
        if not self.vlm_available:
            try:
                from ocr.gemini_vision_engine import GeminiVisionEngine
                self.vlm_engine = GeminiVisionEngine()
                self.vlm_available = True
                self.vlm_type = "Gemini Flash"
            except ImportError as e:
                print(f"⚠️ GeminiVisionEngine import failed: {e}")
            except ValueError as e:
                print(f"⚠️ Gemini init failed (no API key): {e}")
            except Exception as e:
                print(f"⚠️ Gemini init failed: {e}")
        
        # Fallback to Llama Vision if Gemini not available
        if not self.vlm_available:
            try:
                from ocr.llama_vision_engine import LlamaVisionEngine
                self.vlm_engine = LlamaVisionEngine()
                self.vlm_available = True
                self.vlm_type = "Llama Vision"
            except Exception as e:
                print(f"⚠️ Llama Vision also unavailable: {e}")
    
    def _init_ocr_engine(self, use_gpu: bool):
        """Initialize EasyOCR engine."""
        try:
            self.ocr_engine = OCREngine(use_gpu=use_gpu, lang='en')
        except Exception as e:
            print(f"❌ EasyOCR init failed: {e}")
            raise RuntimeError(f"Failed to initialize EasyOCR fallback: {e}")
    
    def extract_text(self, image_path: str, retry_on_rate_limit: bool = True) -> Tuple[str, float]:
        """
        Extract text from image using hybrid approach.
        
        Strategy:
        1. Try Gemini Vision first (better for math context)
        2. Fall back to EasyOCR if VLM fails or returns poor results
        
        Args:
            image_path: Path to the image file
            retry_on_rate_limit: Whether to retry after rate limit errors
            
        Returns:
            Tuple of (extracted_text, confidence)
        """
        vlm_text = None
        vlm_confidence = 0.0
        
        # Try VLM first if available
        if self.use_vlm and self.vlm_available and self.vlm_engine:
            try:
                vlm_text, vlm_confidence = self.vlm_engine.extract_text(image_path)
                
                # Check if VLM result is valid
                if vlm_text and len(vlm_text.strip()) >= 2:
                    print(f"✓ VLM extraction successful: '{vlm_text[:50]}...' (conf: {vlm_confidence:.2f})")
                    return vlm_text, vlm_confidence
                else:
                    print("⚠️ VLM returned empty/short result, falling back to OCR")
                    
            except RuntimeError as e:
                error_msg = str(e)
                if "rate limit" in error_msg.lower() or "429" in error_msg:
                    print(f"⚠️ VLM rate limit hit, falling back to OCR")
                else:
                    print(f"⚠️ VLM extraction failed, falling back to OCR: {e}")
            except Exception as e:
                print(f"⚠️ VLM extraction failed, falling back to OCR: {e}")
        
        # Fallback to EasyOCR
        try:
            ocr_text, ocr_confidence = self.ocr_engine.extract_text(image_path)
            print(f"✓ OCR extraction: '{ocr_text[:50] if ocr_text else ''}...' (conf: {ocr_confidence:.2f})")
            return ocr_text, ocr_confidence
        except Exception as e:
            raise RuntimeError(f"Both VLM and OCR extraction failed: {e}")
    
    def extract_latex(self, image_path: str) -> Tuple[str, float]:
        """
        Extract math content as LaTeX from image.
        
        Note: LaTeX extraction requires VLM. Falls back to plain text
        extraction via OCR if VLM is unavailable.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (latex_string, confidence)
        """
        if self.use_vlm and self.vlm_available and self.vlm_engine:
            try:
                return self.vlm_engine.extract_latex(image_path)
            except Exception as e:
                print(f"⚠️ LaTeX extraction failed: {e}")
                # Fall back to plain text
                text, conf = self.extract_text(image_path)
                return text, conf * 0.7  # Lower confidence for non-LaTeX
        else:
            # VLM not available, return plain text with warning
            print("⚠️ LaTeX extraction requires VLM. Returning plain text.")
            text, conf = self.extract_text(image_path)
            return text, conf * 0.7
    
    def get_status(self) -> dict:
        """
        Get engine status information.
        
        Returns:
            Dictionary with engine status details
        """
        return {
            "vlm_enabled": self.use_vlm,
            "vlm_available": self.vlm_available,
            "vlm_type": self.vlm_type,
            "vlm_model": getattr(self.vlm_engine, 'model_name', None) if self.vlm_engine else None,
            "ocr_engine": "EasyOCR",
            "ocr_available": self.ocr_engine is not None,
            "mode": "hybrid" if self.vlm_available else "ocr_only"
        }
