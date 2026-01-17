"""
Correction Rules
OCR/ASR correction learning from user edits.
"""

import os
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter


class CorrectionRules:
    """
    Learns correction patterns from user edits to OCR/ASR output.
    
    Stores:
    - Character-level corrections (OCR)
    - Word-level corrections (ASR)
    - Pattern-based rules
    
    Applies learned corrections to new inputs automatically.
    """
    
    def __init__(self, storage_dir: str = None):
        """
        Initialize correction rules.
        
        Args:
            storage_dir: Directory for persistent storage
        """
        if storage_dir is None:
            storage_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "memory_data"
            )
        
        self.storage_dir = storage_dir
        self.rules_file = os.path.join(storage_dir, "correction_rules.json")
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load existing rules
        self.ocr_corrections: Dict[str, Counter] = {}  # wrong -> Counter of corrections
        self.asr_corrections: Dict[str, Counter] = {}  # wrong -> Counter of corrections
        self.pattern_rules: List[Dict] = []  # regex-based rules
        self._load()
    
    def _load(self):
        """Load rules from disk."""
        if os.path.exists(self.rules_file):
            try:
                with open(self.rules_file, 'r') as f:
                    data = json.load(f)
                    
                # Convert Counter format
                self.ocr_corrections = {
                    k: Counter(v) for k, v in data.get("ocr_corrections", {}).items()
                }
                self.asr_corrections = {
                    k: Counter(v) for k, v in data.get("asr_corrections", {}).items()
                }
                self.pattern_rules = data.get("pattern_rules", [])
            except Exception as e:
                print(f"Warning: Could not load correction rules: {e}")
    
    def _save(self):
        """Save rules to disk."""
        try:
            data = {
                "ocr_corrections": {k: dict(v) for k, v in self.ocr_corrections.items()},
                "asr_corrections": {k: dict(v) for k, v in self.asr_corrections.items()},
                "pattern_rules": self.pattern_rules,
                "last_updated": datetime.now().isoformat()
            }
            with open(self.rules_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving correction rules: {e}")
    
    def learn_ocr_correction(self, original: str, corrected: str):
        """
        Learn from an OCR correction.
        
        Args:
            original: Original OCR output (wrong)
            corrected: User's correction (correct)
        """
        # Find differences and learn them
        corrections = self._extract_corrections(original, corrected)
        
        for wrong, correct in corrections:
            if wrong not in self.ocr_corrections:
                self.ocr_corrections[wrong] = Counter()
            self.ocr_corrections[wrong][correct] += 1
        
        self._save()
    
    def learn_asr_correction(self, original: str, corrected: str):
        """
        Learn from an ASR correction.
        
        Args:
            original: Original ASR output (wrong)
            corrected: User's correction (correct)
        """
        # Word-level corrections for ASR
        orig_words = original.lower().split()
        corr_words = corrected.lower().split()
        
        # Simple alignment for word corrections
        corrections = self._align_words(orig_words, corr_words)
        
        for wrong, correct in corrections:
            if wrong not in self.asr_corrections:
                self.asr_corrections[wrong] = Counter()
            self.asr_corrections[wrong][correct] += 1
        
        self._save()
    
    def _extract_corrections(self, original: str, corrected: str) -> List[Tuple[str, str]]:
        """Extract character/token level corrections."""
        corrections = []
        
        # If strings are very different, store whole string correction
        if len(original) != len(corrected) and abs(len(original) - len(corrected)) > len(original) * 0.5:
            corrections.append((original.strip(), corrected.strip()))
            return corrections
        
        # Token-level correction
        orig_tokens = re.findall(r'\S+', original)
        corr_tokens = re.findall(r'\S+', corrected)
        
        if len(orig_tokens) == len(corr_tokens):
            for o, c in zip(orig_tokens, corr_tokens):
                if o != c:
                    corrections.append((o, c))
        else:
            # Store as pattern
            corrections.append((original.strip(), corrected.strip()))
        
        return corrections
    
    def _align_words(self, orig: List[str], corr: List[str]) -> List[Tuple[str, str]]:
        """Simple word alignment for ASR corrections."""
        corrections = []
        
        if len(orig) == len(corr):
            for o, c in zip(orig, corr):
                if o != c:
                    corrections.append((o, c))
        
        return corrections
    
    def add_pattern_rule(self, pattern: str, replacement: str, 
                          rule_type: str = "regex", description: str = ""):
        """
        Add a pattern-based correction rule.
        
        Args:
            pattern: Regex pattern to match
            replacement: Replacement string
            rule_type: Type of rule (regex, exact, prefix, suffix)
            description: Human-readable description
        """
        rule = {
            "pattern": pattern,
            "replacement": replacement,
            "rule_type": rule_type,
            "description": description,
            "created": datetime.now().isoformat(),
            "use_count": 0
        }
        
        self.pattern_rules.append(rule)
        self._save()
    
    def apply_corrections(self, text: str, correction_type: str = "ocr") -> str:
        """
        Apply learned corrections to text.
        
        Args:
            text: Input text
            correction_type: Type of corrections to apply (ocr, asr, all)
            
        Returns:
            Corrected text
        """
        corrected = text
        
        # Apply OCR corrections
        if correction_type in ["ocr", "all"]:
            for wrong, corrections in self.ocr_corrections.items():
                if wrong in corrected:
                    # Use most common correction
                    best_correction = corrections.most_common(1)[0][0]
                    corrected = corrected.replace(wrong, best_correction)
        
        # Apply ASR corrections
        if correction_type in ["asr", "all"]:
            words = corrected.split()
            for i, word in enumerate(words):
                word_lower = word.lower()
                if word_lower in self.asr_corrections:
                    best_correction = self.asr_corrections[word_lower].most_common(1)[0][0]
                    # Preserve original case if possible
                    if word.isupper():
                        best_correction = best_correction.upper()
                    elif word.istitle():
                        best_correction = best_correction.title()
                    words[i] = best_correction
            corrected = " ".join(words)
        
        # Apply pattern rules
        for rule in self.pattern_rules:
            try:
                if rule["rule_type"] == "regex":
                    new_corrected = re.sub(rule["pattern"], rule["replacement"], corrected)
                    if new_corrected != corrected:
                        rule["use_count"] += 1
                        corrected = new_corrected
                elif rule["rule_type"] == "exact":
                    if rule["pattern"] in corrected:
                        corrected = corrected.replace(rule["pattern"], rule["replacement"])
                        rule["use_count"] += 1
            except Exception as e:
                print(f"Error applying rule: {e}")
        
        return corrected
    
    def get_common_corrections(self, correction_type: str = "ocr", 
                                 limit: int = 20) -> List[Dict]:
        """Get most common corrections."""
        corrections_dict = self.ocr_corrections if correction_type == "ocr" else self.asr_corrections
        
        all_corrections = []
        for wrong, corrects in corrections_dict.items():
            for correct, count in corrects.most_common(3):
                all_corrections.append({
                    "wrong": wrong,
                    "correct": correct,
                    "count": count,
                    "type": correction_type
                })
        
        all_corrections.sort(key=lambda x: x["count"], reverse=True)
        return all_corrections[:limit]
    
    def get_stats(self) -> Dict:
        """Get correction rule statistics."""
        return {
            "ocr_rules": len(self.ocr_corrections),
            "asr_rules": len(self.asr_corrections),
            "pattern_rules": len(self.pattern_rules),
            "total_ocr_corrections": sum(
                sum(c.values()) for c in self.ocr_corrections.values()
            ),
            "total_asr_corrections": sum(
                sum(c.values()) for c in self.asr_corrections.values()
            )
        }


def main():
    """Test correction rules."""
    rules = CorrectionRules()
    
    # Learn some OCR corrections
    rules.learn_ocr_correction("2x + 3 = 5", "2x + 3 = 5")  # No change
    rules.learn_ocr_correction("x2 + y = 10", "x² + y = 10")  # Superscript
    rules.learn_ocr_correction("x2 + y = 10", "x² + y = 10")  # Same correction again
    
    # Apply corrections
    test_input = "Solve x2 + 5 = 14"
    corrected = rules.apply_corrections(test_input, "ocr")
    print(f"Original: {test_input}")
    print(f"Corrected: {corrected}")
    
    # Stats
    stats = rules.get_stats()
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
