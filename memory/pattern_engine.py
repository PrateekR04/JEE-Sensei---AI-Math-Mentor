"""
Pattern Engine
Solution pattern extraction and reuse.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import re


class PatternEngine:
    """
    Extracts and reuses solution patterns from solved problems.
    
    Features:
    - Pattern extraction from successful solutions
    - Pattern matching for new problems
    - Strategy recommendation based on similar problems
    - Confidence scoring for pattern matches
    """
    
    def __init__(self, storage_dir: str = None):
        """
        Initialize pattern engine.
        
        Args:
            storage_dir: Directory for persistent storage
        """
        if storage_dir is None:
            storage_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "memory_data"
            )
        
        self.storage_dir = storage_dir
        self.patterns_file = os.path.join(storage_dir, "solution_patterns.json")
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load existing patterns
        self.patterns: List[Dict] = []
        self._load()
        
        # Initialize keyword-based pattern templates
        self._init_templates()
    
    def _load(self):
        """Load patterns from disk."""
        if os.path.exists(self.patterns_file):
            try:
                with open(self.patterns_file, 'r') as f:
                    self.patterns = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load patterns: {e}")
                self.patterns = []
    
    def _save(self):
        """Save patterns to disk."""
        try:
            with open(self.patterns_file, 'w') as f:
                json.dump(self.patterns, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving patterns: {e}")
    
    def _init_templates(self):
        """Initialize built-in pattern templates."""
        self.templates = {
            # Linear equations
            "linear_equation": {
                "keywords": ["solve", "find x", "equation"],
                "pattern": r"(\d*)\s*x\s*([+-])\s*(\d+)\s*=\s*(\d+)",
                "strategy": "isolate_variable",
                "steps": [
                    "Move constant terms to one side",
                    "Divide by coefficient of x",
                    "Verify by substitution"
                ]
            },
            
            # Quadratic equations
            "quadratic_equation": {
                "keywords": ["x²", "x^2", "quadratic"],
                "pattern": r"x\^?2|x²",
                "strategy": "quadratic_formula_or_factoring",
                "steps": [
                    "Identify coefficients a, b, c",
                    "Check discriminant b² - 4ac",
                    "Apply quadratic formula or factor",
                    "Verify roots"
                ]
            },
            
            # System of equations
            "system_equations": {
                "keywords": ["system", "and", "equations", "simultaneously"],
                "pattern": r"(.*=.*)\s+(and|,)\s+(.*=.*)",
                "strategy": "substitution_or_elimination",
                "steps": [
                    "Choose substitution or elimination method",
                    "Solve for one variable",
                    "Substitute to find other variable",
                    "Verify both equations"
                ]
            },
            
            # Derivative
            "derivative": {
                "keywords": ["derivative", "differentiate", "d/dx"],
                "strategy": "apply_differentiation_rules",
                "steps": [
                    "Identify function type",
                    "Apply appropriate rule (power, chain, product, quotient)",
                    "Simplify result"
                ]
            },
            
            # Integration
            "integration": {
                "keywords": ["integrate", "integral", "∫"],
                "strategy": "apply_integration_rules",
                "steps": [
                    "Identify integrand type",
                    "Apply appropriate technique (substitution, parts, etc.)",
                    "Add constant of integration"
                ]
            },
            
            # Probability
            "probability": {
                "keywords": ["probability", "chance", "likely", "dice", "coin"],
                "strategy": "define_sample_space_and_event",
                "steps": [
                    "Define sample space",
                    "Identify favorable outcomes",
                    "Calculate P(E) = favorable/total"
                ]
            },
            
            # Word problem - age
            "age_problem": {
                "keywords": ["years older", "years younger", "age"],
                "strategy": "define_variables_setup_equations",
                "steps": [
                    "Define variable for unknown age",
                    "Express relationships as equations",
                    "Solve system of equations"
                ]
            },
            
            # Word problem - speed/distance
            "motion_problem": {
                "keywords": ["speed", "distance", "time", "km/h", "mph"],
                "strategy": "use_distance_speed_time_formula",
                "steps": [
                    "Use D = S × T",
                    "Set up equations from given conditions",
                    "Solve for unknowns"
                ]
            }
        }
    
    def extract_pattern(self, problem_data: Dict) -> Optional[Dict]:
        """
        Extract a reusable pattern from a solved problem.
        
        Args:
            problem_data: Problem with solution
            
        Returns:
            Extracted pattern or None
        """
        if not problem_data.get("is_correct", True):
            return None  # Only learn from correct solutions
        
        problem_text = problem_data.get("problem_text", "").lower()
        topic = problem_data.get("topic", "")
        intent = problem_data.get("intent", "")
        
        # Match against templates
        matched_template = None
        for name, template in self.templates.items():
            keyword_match = any(kw in problem_text for kw in template["keywords"])
            if keyword_match:
                matched_template = name
                break
        
        # Create pattern
        pattern = {
            "id": f"pat_{len(self.patterns)+1:06d}",
            "template_name": matched_template,
            "topic": topic,
            "intent": intent,
            "keywords": self._extract_keywords(problem_text),
            "problem_structure": self._extract_structure(problem_text),
            "solution_approach": problem_data.get("solution_steps", ""),
            "equations": problem_data.get("modeled_equations", []),
            "success_count": 1,
            "created": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat()
        }
        
        # Check if similar pattern already exists
        existing = self._find_similar_pattern(pattern)
        if existing:
            existing["success_count"] += 1
            existing["last_used"] = datetime.now().isoformat()
            self._save()
            return existing
        
        self.patterns.append(pattern)
        self._save()
        return pattern
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Math-related keywords
        math_keywords = [
            "solve", "find", "calculate", "compute", "evaluate",
            "derivative", "integral", "limit", "probability",
            "equation", "system", "quadratic", "linear",
            "maximize", "minimize", "optimize",
            "sum", "product", "difference", "ratio"
        ]
        
        text_lower = text.lower()
        found = [kw for kw in math_keywords if kw in text_lower]
        return found
    
    def _extract_structure(self, text: str) -> str:
        """Extract structural pattern from problem text."""
        # Replace numbers with placeholders
        structure = re.sub(r'\d+\.?\d*', 'N', text)
        # Replace variables with placeholder
        structure = re.sub(r'\b[a-z]\b', 'V', structure)
        return structure[:100]  # Limit length
    
    def _find_similar_pattern(self, pattern: Dict) -> Optional[Dict]:
        """Find existing similar pattern."""
        for existing in self.patterns:
            if (existing["topic"] == pattern["topic"] and 
                existing["intent"] == pattern["intent"] and
                existing.get("template_name") == pattern.get("template_name")):
                # Check keyword overlap
                common = set(existing["keywords"]) & set(pattern["keywords"])
                if len(common) >= 2:
                    return existing
        return None
    
    def find_matching_pattern(self, problem_text: str, 
                               topic: str = None) -> Tuple[Optional[Dict], float]:
        """
        Find a matching pattern for a new problem.
        
        Args:
            problem_text: New problem text
            topic: Optional topic filter
            
        Returns:
            Tuple of (matched pattern, confidence score)
        """
        problem_lower = problem_text.lower()
        
        best_match = None
        best_score = 0.0
        
        # Check against templates first
        for name, template in self.templates.items():
            keyword_matches = sum(1 for kw in template["keywords"] if kw in problem_lower)
            if keyword_matches > 0:
                score = keyword_matches / len(template["keywords"])
                if score > best_score:
                    best_match = {
                        "type": "template",
                        "template_name": name,
                        "strategy": template["strategy"],
                        "steps": template["steps"]
                    }
                    best_score = score
        
        # Check against learned patterns
        for pattern in self.patterns:
            if topic and pattern.get("topic") != topic:
                continue
            
            keyword_matches = sum(1 for kw in pattern["keywords"] if kw in problem_lower)
            if keyword_matches > 0:
                score = keyword_matches / max(len(pattern["keywords"]), 1)
                # Boost score based on success count
                score *= min(1.0 + pattern["success_count"] * 0.1, 1.5)
                
                if score > best_score:
                    best_match = pattern
                    best_score = score
        
        return best_match, best_score
    
    def get_strategy(self, problem_text: str, topic: str = None) -> Optional[Dict]:
        """
        Get recommended strategy for a problem.
        
        Args:
            problem_text: Problem text
            topic: Optional topic filter
            
        Returns:
            Strategy recommendation with steps
        """
        pattern, score = self.find_matching_pattern(problem_text, topic)
        
        if pattern and score >= 0.5:  # Confidence threshold
            if pattern.get("type") == "template":
                return {
                    "strategy": pattern["strategy"],
                    "steps": pattern["steps"],
                    "confidence": score,
                    "source": "template"
                }
            else:
                return {
                    "strategy": pattern.get("template_name", "learned"),
                    "steps": pattern.get("solution_approach", "").split("\n")[:5],
                    "confidence": score,
                    "source": "learned_pattern"
                }
        
        return None
    
    def get_stats(self) -> Dict:
        """Get pattern statistics."""
        total_patterns = len(self.patterns)
        total_uses = sum(p["success_count"] for p in self.patterns)
        
        # Group by topic
        by_topic = {}
        for p in self.patterns:
            topic = p.get("topic", "unknown")
            if topic not in by_topic:
                by_topic[topic] = 0
            by_topic[topic] += 1
        
        return {
            "total_patterns": total_patterns,
            "total_uses": total_uses,
            "templates": len(self.templates),
            "by_topic": by_topic
        }


def main():
    """Test pattern engine."""
    engine = PatternEngine()
    
    # Test strategy lookup
    problems = [
        "Solve 2x + 5 = 11",
        "Find the derivative of x² + 3x",
        "A father is 25 years older than his son. Find their ages.",
        "What is the probability of getting 2 heads in 3 coin flips?"
    ]
    
    for problem in problems:
        strategy = engine.get_strategy(problem)
        print(f"\nProblem: {problem[:50]}...")
        if strategy:
            print(f"  Strategy: {strategy['strategy']}")
            print(f"  Confidence: {strategy['confidence']:.0%}")
            print(f"  Steps: {strategy['steps'][:2]}...")
        else:
            print("  No matching pattern found")
    
    # Stats
    stats = engine.get_stats()
    print(f"\nStats: {stats}")


if __name__ == "__main__":
    main()
