"""
Production-Grade OCR Engine for Handwritten Math
Optimized EasyOCR with handwriting-specific preprocessing
"""

import cv2
import numpy as np
import easyocr
import re
import os
from typing import Tuple, List
from PIL import Image


class OCREngine:
    """
    Optimized OCR engine for handwritten math equations.
    Uses EasyOCR with handwriting-optimized preprocessing.
    """
    
    def __init__(self, use_gpu: bool = False, lang: str = 'en'):
        self.use_gpu = use_gpu
        self.lang = [lang] if isinstance(lang, str) else lang
        self.reader = None
        self._initialize_reader()
    
    def _initialize_reader(self):
        """Initialize EasyOCR with optimized settings."""
        try:
            self.reader = easyocr.Reader(
                self.lang,
                gpu=self.use_gpu,
                verbose=False
            )
            print("✓ EasyOCR initialized for handwritten math")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize EasyOCR: {str(e)}")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocessing optimized for handwritten text on paper.
        
        Strategy:
        - Enhance contrast for faint pencil marks
        - Light denoising to remove paper texture
        - Preserve character shapes (no aggressive thresholding)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast with CLAHE (good for pencil on paper)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Light bilateral filter to reduce paper texture while preserving edges
        denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
        
        # Slight sharpening to make handwriting clearer
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened
    
    def run_ocr(self, processed_img: np.ndarray) -> List[Tuple[str, float, List]]:
        """
        Run EasyOCR with settings optimized for handwritten math.
        """
        results = self.reader.readtext(
            processed_img,
            detail=1,
            paragraph=False,
            # Very relaxed thresholds for handwriting
            min_size=3,
            text_threshold=0.25,  # Lower for faint handwriting
            low_text=0.15,
            link_threshold=0.25,
            # Grouping parameters
            width_ths=0.9,
            height_ths=0.9,
            slope_ths=0.2,
            ycenter_ths=0.5,
            mag_ratio=1.5,
            # Additional handwriting parameters
            decoder='greedy',  # Faster, works well for short text
            beamWidth=5
        )
        
        detections = []
        for item in results:
            if len(item) >= 3:
                bbox, text, conf = item[0], item[1], item[2]
                detections.append((str(text).strip(), float(conf), bbox))
            elif len(item) == 2:
                text, conf = item
                detections.append((str(text).strip(), float(conf), []))
        
        return detections
    
    def reconstruct_equation(self, detections: List[Tuple[str, float, List]]) -> Tuple[str, List[float]]:
        """Reconstruct fragmented equation into single line."""
        if not detections:
            return "", []
        
        # Group by vertical position
        lines = {}
        for text, conf, bbox in detections:
            if not bbox or len(bbox) == 0:
                y_pos = 0
            else:
                y_coords = [p[1] for p in bbox] if isinstance(bbox[0], (list, tuple)) else [bbox[1]]
                y_pos = int(np.mean(y_coords))
            
            line_key = y_pos // 30  # Grouping tolerance
            if line_key not in lines:
                lines[line_key] = []
            lines[line_key].append((text, conf, bbox))
        
        sorted_lines = []
        confidences = []
        
        for line_key in sorted(lines.keys()):
            line_items = lines[line_key]
            # Sort by X position
            sorted_items = sorted(line_items, key=lambda x: self._get_x_pos(x[2]))
            
            line_text = ""
            for i, (text, conf, bbox) in enumerate(sorted_items):
                confidences.append(conf)
                
                # Add spacing logic
                if i > 0 and text:
                    prev_text = sorted_items[i-1][0]
                    # Add space before operators
                    if text[0] in ['+', '-', '=', 'x', '*', '/', '÷', '×', '?']:
                        if not line_text.endswith(' '):
                            line_text += ' '
                    # Add space after operators
                    elif prev_text and prev_text[-1] in ['+', '-', '=', 'x', '*', '/', '÷', '×']:
                        if not line_text.endswith(' '):
                            line_text += ' '
                
                line_text += text
            
            sorted_lines.append(line_text.strip())
        
        final_text = " ".join(sorted_lines)
        return final_text, confidences
    
    def _get_x_pos(self, bbox) -> float:
        """Get X position from bounding box."""
        if not bbox or len(bbox) == 0:
            return 0
        if isinstance(bbox[0], (list, tuple)):
            return np.mean([p[0] for p in bbox])
        return bbox[0]
    
    def clean_math_text(self, text: str) -> str:
        """Clean and normalize math text with handwriting corrections."""
        if not text:
            return text
        
        cleaned = text
        
        # Symbol replacements
        replacements = {
            "×": "x", "÷": "/", "–": "-", "—": "-",
            "＝": "=", "*": "x", "X": "x",
        }
        
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        # Common handwriting OCR errors
        # These are context-aware to avoid breaking words
        # Only apply in math context (near numbers/operators)
        
        # "d" often misread as "2" in handwriting
        cleaned = re.sub(r'\bd\s*x\b', '2x', cleaned, flags=re.IGNORECASE)
        
        # "$" often misread as "5" or "S"
        cleaned = re.sub(r'\$', '5', cleaned)
        
        # ";" or ":" often misread as "5"
        cleaned = re.sub(r'[;:](?=\s*\d|\s*$)', '5', cleaned)
        
        # "%" often appears instead of numbers
        cleaned = re.sub(r'%', '', cleaned)
        
        # Common letter-number confusions in math context
        cleaned = re.sub(r'\bO\b', '0', cleaned)  # O → 0
        cleaned = re.sub(r'\bl\b', '1', cleaned)  # l → 1
        cleaned = re.sub(r'\bI\b', '1', cleaned)  # I → 1
        cleaned = re.sub(r'\bS\b', '5', cleaned)  # S → 5
        
        # Normalize operator spacing
        cleaned = re.sub(r'\s*([+\-x/=])\s*', r' \1 ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove duplicate operators
        cleaned = re.sub(r'([+\-x/=])\s+\1', r'\1', cleaned)
        
        return cleaned.strip()
    
    def compute_confidence(self, confidences: List[float], final_text: str) -> float:
        """Compute realistic confidence score."""
        if not confidences:
            return 0.0
        
        base_conf = sum(confidences) / len(confidences)
        
        # Penalize high fragmentation
        num_fragments = len(confidences)
        text_length = len(final_text.replace(" ", ""))
        
        if text_length > 0:
            avg_chars_per_fragment = text_length / num_fragments
            if avg_chars_per_fragment < 2:
                base_conf *= 0.7
            elif avg_chars_per_fragment < 4:
                base_conf *= 0.85
        
        # Boost for valid math patterns
        math_patterns = [
            r'\d+\s*[a-z]',  # Coefficient like "2x"
            r'[a-z]\s*[+\-x/]',  # Variable with operator
            r'\d+\s*[+\-x/=]',  # Number with operator
            r'[+\-x/]\s*\d+',  # Operator with number
            r'=\s*\d+',  # Equals number
        ]
        
        pattern_matches = sum(1 for p in math_patterns if re.search(p, final_text.lower()))
        
        if pattern_matches >= 3:
            base_conf = min(1.0, base_conf * 1.2)
        elif pattern_matches >= 2:
            base_conf = min(1.0, base_conf * 1.15)
        elif pattern_matches >= 1:
            base_conf = min(1.0, base_conf * 1.1)
        
        return round(min(max(base_conf, 0.0), 1.0), 2)
    
    def extract_text(self, image_path: str) -> Tuple[str, float]:
        """
        Extract handwritten math equation from image.
        
        Pipeline:
        1. Handwriting-optimized preprocessing
        2. Run EasyOCR with relaxed parameters
        3. Reconstruct equation from fragments
        4. Clean and fix common OCR errors
        5. Compute confidence
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (equation_text, confidence)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Stage 1: Handwriting preprocessing
            processed = self.preprocess_image(image_path)
            
            # Stage 2: Run OCR
            detections = self.run_ocr(processed)
            
            if not detections:
                return "", 0.0
            
            # Stage 3: Reconstruct equation
            raw_text, confidences = self.reconstruct_equation(detections)
            
            # Stage 4: Clean and fix OCR errors
            cleaned_text = self.clean_math_text(raw_text)
            
            # Stage 5: Compute confidence
            confidence = self.compute_confidence(confidences, cleaned_text)
            
            return cleaned_text, confidence
            
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")
