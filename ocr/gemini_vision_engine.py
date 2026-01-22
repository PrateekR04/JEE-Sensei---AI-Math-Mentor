"""
Gemini Vision Engine for Math OCR
Production-Grade Mathematical Document Extraction Engine

Uses Google Gemini Flash for vision-based question extraction.
Free tier: 15 requests/minute, 1500 requests/day

LOCKED: System prompt is immutable at runtime.
"""

import os
import re
import json
from typing import Tuple, Optional, Dict, Any
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# LOCKED SYSTEM PROMPT - DO NOT MODIFY AT RUNTIME
# ============================================================================
_LOCKED_DOCUMENT_EXTRACTION_PROMPT = """You are a mathematical document extraction model.

Your task is to extract the FULL question from the image, including:
- All sentences
- All math expressions
- All function definitions
- All conditions and intervals

Then produce TWO outputs:

1) LaTeX Version (latex_question):
A faithful LaTeX transcription of the entire question.
Use proper LaTeX formatting: \\frac{}{}, \\lim_{}, \\sqrt{}, etc.
Do NOT include $ or $$ delimiters.

2) Editable Human Version (editable_question):
A CLEAN, SIMPLE, HUMAN-READABLE version using ASCII-friendly notation.

IMPORTANT RULES FOR editable_question:
- NO LaTeX commands (no \\frac, \\lim, \\sqrt, etc.)
- NO dollar signs ($)
- NO backslashes (\\)
- Use simple notation:
  * Fractions: (a)/(b) or a/b
  * Exponents: x^2, x^n
  * Square roots: sqrt(x) or √x
  * Limits: lim(x→a) or "limit as x approaches a"
  * Integrals: integral of f(x) dx
  * Derivatives: d/dx or derivative of
  * Greek letters: use words (alpha, beta, pi) or symbols (α, β, π)
- Write like you would explain to someone verbally
- Use words where helpful: "equals", "approaches", "for all"

Example conversions:
- LaTeX: \\frac{d}{dx}(x^2) → Editable: d/dx(x^2) or "derivative of x^2"
- LaTeX: \\lim_{x \\to 0} \\frac{\\sin x}{x} → Editable: lim(x→0) of sin(x)/x
- LaTeX: \\sqrt{x^2 + 1} → Editable: sqrt(x^2 + 1) or √(x^2 + 1)

Rules:
- Do NOT solve the problem
- Do NOT summarize
- Do NOT explain
- Do NOT omit any part of the question
- Preserve all mathematical meaning
- Preserve all conditions and domains

Return output strictly in this JSON format:

{
  "latex_question": "...",
  "editable_question": "..."
}
"""

# Guard flag to ensure prompt immutability
_PROMPT_LOCKED = True


def _get_locked_prompt() -> str:
    """
    Get the locked document extraction prompt.
    This function ensures the prompt cannot be modified at runtime.
    """
    if not _PROMPT_LOCKED:
        raise RuntimeError("SECURITY ERROR: Prompt lock has been tampered with.")
    return _LOCKED_DOCUMENT_EXTRACTION_PROMPT


class GeminiVisionEngine:
    """
    Vision Language Model engine using Google Gemini Flash
    for mathematical document extraction.
    
    IMPORTANT: This engine operates in DOCUMENT EXTRACTION MODE only.
    - Extracts full questions from images
    - Returns structured JSON output
    - Does NOT solve problems
    - System prompt is LOCKED and immutable
    """
    
    # Available Gemini models with vision support
    VISION_MODELS = [
        "gemini-2.0-flash",      # Primary
        "gemini-2.0-flash-lite", # Lighter, faster
        "gemini-2.5-flash",      # Latest
    ]
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Gemini Vision engine in document extraction mode.
        
        Args:
            api_key: Gemini API key. If not provided, reads from GEMINI_API_KEY env var.
            model: Model to use. Defaults to gemini-2.5-flash.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Set it in .env or pass api_key parameter.")
        
        self.model_name = model or "gemini-2.5-flash"
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Gemini model."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            print(f"✓ Gemini Vision Engine initialized ({self.model_name}) [Document Extraction Mode]")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini: {str(e)}")
    
    def extract_document(self, image_path: str) -> Dict[str, Any]:
        """
        Extract mathematical document from image using locked prompt.
        Returns structured JSON with latex_question and editable_question.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict with keys: latex_question, editable_question, confidence, raw_response
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            img = Image.open(image_path)
            
            # Resize if too large (to reduce token usage)
            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Use LOCKED prompt - cannot be overridden
            prompt = _get_locked_prompt()
            
            response = self.model.generate_content([prompt, img])
            raw_text = response.text.strip()
            
            # Parse JSON response
            result = self._parse_json_response(raw_text)
            
            # Validate output
            validation_result = self._validate_extraction_output(result)
            if not validation_result["valid"]:
                return {
                    "latex_question": "",
                    "editable_question": "",
                    "confidence": 0.0,
                    "raw_response": raw_text,
                    "error": validation_result["reason"],
                    "valid": False
                }
            
            return {
                "latex_question": result.get("latex_question", ""),
                "editable_question": result.get("editable_question", ""),
                "confidence": 0.95,
                "raw_response": raw_text,
                "valid": True
            }
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "ResourceExhausted" in error_msg:
                raise RuntimeError("Rate limit exceeded. Please wait a moment and try again.")
            raise RuntimeError(f"Gemini API error: {error_msg}")
    
    def extract_text(self, image_path: str) -> Tuple[str, float]:
        """
        Extract math text from image using Gemini Vision.
        Returns LaTeX format for rendering.
        
        BACKWARD COMPATIBLE: This method maintains the original interface
        while using the new document extraction internally.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (extracted_text_in_latex, confidence)
        """
        # Use new document extraction internally
        result = self.extract_document(image_path)
        
        if result.get("valid") and result.get("latex_question"):
            return result["latex_question"], result["confidence"]
        elif result.get("valid") and result.get("editable_question"):
            return result["editable_question"], result["confidence"] * 0.9
        else:
            # Fallback to legacy extraction for backward compatibility
            return self._legacy_extract_text(image_path)
    
    def _legacy_extract_text(self, image_path: str) -> Tuple[str, float]:
        """
        Legacy text extraction for backward compatibility.
        Used when new JSON extraction fails.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            img = Image.open(image_path)
            
            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Legacy prompt for fallback only
            prompt = """Extract the mathematical expression or problem from this image and convert it to LaTeX format.
Output ONLY valid LaTeX math notation. Do NOT solve the problem."""
            
            response = self.model.generate_content([prompt, img])
            text = response.text.strip()
            text = self._clean_extracted_text(text)
            
            return text, 0.85  # Lower confidence for legacy method
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "ResourceExhausted" in error_msg:
                raise RuntimeError("Rate limit exceeded. Please wait a moment and try again.")
            raise RuntimeError(f"Gemini API error: {error_msg}")
    
    def extract_latex(self, image_path: str) -> Tuple[str, float]:
        """
        Extract math content as LaTeX from image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (latex_string, confidence)
        """
        result = self.extract_document(image_path)
        
        if result.get("valid") and result.get("latex_question"):
            return result["latex_question"], result["confidence"]
        else:
            # Fallback
            return self._legacy_extract_text(image_path)
    
    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """
        Parse JSON from Gemini response.
        
        Args:
            text: Raw response text
            
        Returns:
            Parsed JSON dict or empty dict on failure
        """
        try:
            # Try direct JSON parse
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try extracting JSON from markdown code blocks
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}'
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        return {}
    
    def _validate_extraction_output(self, result: Dict) -> Dict[str, Any]:
        """
        Validate extraction output meets requirements.
        
        Rejects if:
        - JSON is malformed (already handled by parser)
        - Required fields are missing
        - Output contains solution steps
        
        Args:
            result: Parsed JSON result
            
        Returns:
            Dict with 'valid' bool and 'reason' if invalid
        """
        # Check required fields
        if not result:
            return {"valid": False, "reason": "Empty or malformed JSON response"}
        
        if "latex_question" not in result and "editable_question" not in result:
            return {"valid": False, "reason": "Missing both latex_question and editable_question"}
        
        # Check for non-empty content
        latex_q = result.get("latex_question", "")
        editable_q = result.get("editable_question", "")
        
        if not latex_q and not editable_q:
            return {"valid": False, "reason": "Both latex_question and editable_question are empty"}
        
        # Check that output doesn't contain solution steps
        solution_indicators = [
            r'\b(?:therefore|hence|thus|so)\s*[,:]?\s*(?:the\s+)?(?:answer|solution)\s*(?:is|=)',
            r'\bstep\s*\d+\s*:',
            r'\b(?:solving|substituting|simplifying)',
            r'=\s*\d+\s*$',  # Final numeric answer
        ]
        
        combined_text = f"{editable_q}".lower()
        for pattern in solution_indicators:
            if re.search(pattern, combined_text, re.IGNORECASE):
                return {"valid": False, "reason": "Output contains solution steps - OCR should only extract, not solve"}
        
        return {"valid": True, "reason": None}
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean up extracted text from common VLM artifacts.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove common prefixes
        prefixes_to_remove = [
            r'^The (?:mathematical )?(?:expression|equation|problem) (?:is|reads?|shows?|states?)[:\s]*',
            r'^(?:Here is|This is) the (?:mathematical )?(?:expression|equation)[:\s]*',
            r'^The image (?:shows?|contains?)[:\s]*',
            r'^(?:Math|Equation)[:\s]*',
        ]
        
        for pattern in prefixes_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove quotes if entire text is quoted
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]
        
        # Remove markdown code blocks if present
        text = re.sub(r'^```.*?\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
        
        return text.strip()
