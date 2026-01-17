"""
Word Problem Validator
Validates word problem models before passing to solver.
"""

import re
from typing import Dict, Tuple, List, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.math_normalizer import MathNormalizer


class WordProblemValidator:
    """
    Validates word problem models for solvability and correctness.
    
    Checks:
    1. Solvability (equations >= unknowns)
    2. Latent variable resolution
    3. Equation syntax (MathNormalizer)
    4. Confidence threshold
    """
    
    CONFIDENCE_THRESHOLD = 0.75
    
    @classmethod
    def validate(cls, model: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate a word problem model.
        
        Args:
            model: Output from WordProblemAgent
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Skip validation for non-word-problems
        if not model.get("is_word_problem"):
            return True, None
        
        # Already flagged as needing clarification
        if model.get("needs_clarification"):
            return False, model.get("reason", "Unknown clarification needed")
        
        # Skip validation for arithmetic problems (no equations)
        if model.get("is_arithmetic"):
            return True, None
        
        # Skip validation for probability (different structure)
        if model.get("modeling_type") == "probability":
            return cls._validate_probability(model)
        
        # Skip validation for optimization (different structure)
        if model.get("modeling_type") == "optimization":
            return cls._validate_optimization(model)
        
        # Standard equation-based validation
        variables = model.get("variables", {})
        equations = model.get("equations", [])
        latent_vars = model.get("latent_variables", [])
        confidence = model.get("modeling_confidence", 0)
        
        # 1. Check confidence threshold
        if confidence < cls.CONFIDENCE_THRESHOLD:
            return False, f"Low modeling confidence ({confidence:.2f} < {cls.CONFIDENCE_THRESHOLD})"
        
        # 2. Check solvability
        valid, err = cls._check_solvability(variables, equations, latent_vars)
        if not valid:
            return False, err
        
        # 3. Check latent variable resolution
        if latent_vars:
            valid, err = cls._check_latent_resolution(latent_vars, equations)
            if not valid:
                return False, err
        
        # 4. Validate equation syntax
        for eq in equations:
            valid, err = cls._validate_equation_syntax(eq)
            if not valid:
                return False, f"Invalid equation '{eq}': {err}"
        
        return True, None
    
    @classmethod
    def _check_solvability(cls, variables: Dict, equations: List[str], 
                          latent_vars: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if system is solvable (not underdetermined)."""
        num_unknowns = len(variables) + len(latent_vars)
        num_equations = len(equations)
        
        if num_unknowns == 0:
            return False, "No unknowns defined"
        
        if num_equations == 0:
            return False, "No equations generated"
        
        if num_unknowns > num_equations:
            return False, f"Underdetermined system: {num_unknowns} unknowns but only {num_equations} equations"
        
        return True, None
    
    @classmethod
    def _check_latent_resolution(cls, latent_vars: List[str], 
                                  equations: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if latent variables can be eliminated."""
        # For each latent variable, check if it appears in a way that allows elimination
        for lv in latent_vars:
            # Count equations containing this latent variable
            count = sum(1 for eq in equations if lv.lower() in eq.lower())
            
            if count < 2:
                # Latent variable must appear in at least 2 equations to be eliminable
                # OR appear in ratio form (D/x - D/y = k)
                has_ratio = any(f"{lv}/" in eq or f"/{lv}" in eq for eq in equations)
                if not has_ratio:
                    return False, f"Latent variable '{lv}' cannot be eliminated (appears in only {count} equation)"
        
        return True, None
    
    @classmethod
    def _validate_equation_syntax(cls, equation: str) -> Tuple[bool, Optional[str]]:
        """Validate equation syntax using MathNormalizer."""
        # Normalize for validation
        eq = equation.strip()
        
        # Must contain =
        if "=" not in eq:
            return False, "Not a valid equation (missing =)"
        
        # Use MathNormalizer for detailed validation
        normalized, error = MathNormalizer.normalize_equation(eq)
        if error:
            return False, error
        
        return True, None
    
    @classmethod
    def _validate_probability(cls, model: Dict) -> Tuple[bool, Optional[str]]:
        """Validate probability model structure."""
        confidence = model.get("modeling_confidence", 0)
        
        if confidence < cls.CONFIDENCE_THRESHOLD:
            return False, f"Low modeling confidence ({confidence:.2f})"
        
        # Check required fields
        if not model.get("sample_space") and not model.get("probability_type"):
            return False, "Missing sample space definition"
        
        if not model.get("event"):
            return False, "Missing event definition"
        
        trials = model.get("trials", 0)
        if trials <= 0 and model.get("probability_type") == "binomial":
            return False, "Missing or invalid trials count"
        
        return True, None
    
    @classmethod
    def _validate_optimization(cls, model: Dict) -> Tuple[bool, Optional[str]]:
        """Validate optimization model structure."""
        confidence = model.get("modeling_confidence", 0)
        
        if confidence < cls.CONFIDENCE_THRESHOLD:
            return False, f"Low modeling confidence ({confidence:.2f})"
        
        if not model.get("objective"):
            return False, "Missing objective function"
        
        if not model.get("constraint"):
            return False, "Missing constraint equation"
        
        # Validate constraint syntax
        constraint = model.get("constraint", "")
        if "=" not in constraint:
            return False, "Constraint must be an equation"
        
        return True, None


def main():
    """Test word problem validator."""
    
    test_cases = [
        # Valid linear system
        {
            "is_word_problem": True,
            "variables": {"x": "first", "y": "second"},
            "equations": ["x + y = 20", "x - y = 4"],
            "modeling_confidence": 0.9
        },
        # Underdetermined
        {
            "is_word_problem": True,
            "variables": {"x": "first", "y": "second"},
            "equations": ["x + y = 20"],
            "modeling_confidence": 0.9
        },
        # Low confidence
        {
            "is_word_problem": True,
            "variables": {"x": "number"},
            "equations": ["x = 5"],
            "modeling_confidence": 0.5
        },
        # Valid with latent variable
        {
            "is_word_problem": True,
            "variables": {"v": "speed"},
            "latent_variables": ["D"],
            "equations": ["D/v - D/(v+10) = 1"],
            "modeling_confidence": 0.85
        },
        # Invalid latent (not eliminable)
        {
            "is_word_problem": True,
            "variables": {"v": "speed"},
            "latent_variables": ["D", "T"],
            "equations": ["D/v = T"],
            "modeling_confidence": 0.85
        },
    ]
    
    print("=" * 60)
    print("WORD PROBLEM VALIDATOR TEST")
    print("=" * 60)
    
    for i, model in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"  Variables: {model.get('variables')}")
        print(f"  Equations: {model.get('equations')}")
        print(f"  Latent: {model.get('latent_variables', [])}")
        
        valid, error = WordProblemValidator.validate(model)
        
        if valid:
            print("  ✓ VALID")
        else:
            print(f"  ✗ INVALID: {error}")


if __name__ == "__main__":
    main()
