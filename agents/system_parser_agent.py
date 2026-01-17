"""
System Parser Agent
Parses system of equations from natural language
"""

import os
import re
import sys
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.math_normalizer import MathNormalizer


class SystemParserAgent:
    """
    Parses system of equations from natural language input.
    Extracts multiple equations and normalizes them for SymPy.
    """
    
    def __init__(self):
        pass
    
    def _extract_equations(self, text: str) -> List[str]:
        """
        Extract multiple equations from text.
        
        Handles formats like:
        - "2x + y = 5 and x - y = 1"
        - "2x + y = 5, x - y = 1"
        - "Solve: 2x+y=5; x-y=1"
        """
        text = text.lower().strip()
        
        # Remove common prefixes
        prefixes = ['solve', 'find', 'determine', 'calculate', 'the system', 
                    'simultaneous equations', 'system of equations']
        for prefix in prefixes:
            text = text.replace(prefix, ' ')
        
        # Split by common delimiters
        # Try "and" first, then comma, then semicolon
        if ' and ' in text:
            parts = text.split(' and ')
        elif ',' in text:
            parts = text.split(',')
        elif ';' in text:
            parts = text.split(';')
        else:
            # Try to find multiple = signs
            parts = [text]
        
        equations = []
        for part in parts:
            part = part.strip()
            
            # Look for equation pattern (contains =)
            if '=' in part:
                # Extract the equation portion
                match = re.search(r'([0-9a-zA-Z+\-*/().^ ]+\s*=\s*[0-9a-zA-Z+\-*/().^ ]+)', part)
                if match:
                    eq = match.group(1).strip()
                    # Clean up
                    eq = ' '.join(eq.split())
                    equations.append(eq)
        
        return equations
    
    def _normalize_equation(self, equation: str) -> str:
        """
        Normalize equation for SymPy.
        """
        # Add implicit multiplication
        equation = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation)
        
        # Remove spaces
        equation = equation.replace(' ', '')
        
        return equation
    
    def _extract_variables(self, equations: List[str]) -> List[str]:
        """
        Extract all variable symbols from equations.
        """
        variables = set()
        for eq in equations:
            # Find single letter variables (excluding e for euler's number)
            for match in re.finditer(r'(?<![a-zA-Z])([a-zA-Z])(?![a-zA-Z])', eq):
                var = match.group(1)
                if var.lower() not in ['e', 'i']:  # Skip constants
                    variables.add(var.lower())
        
        return sorted(list(variables))
    
    def parse(self, problem_text: str) -> Dict[str, Any]:
        """
        Parse system of equations from natural language.
        
        Args:
            problem_text: User's problem text
            
        Returns:
            Dict with type, equations, variables, and any errors
        """
        # Extract equations
        raw_equations = self._extract_equations(problem_text)
        
        if len(raw_equations) < 2:
            return {
                "type": "system",
                "problem_text": problem_text,
                "equations": raw_equations,
                "normalized_equations": [],
                "variables": [],
                "error": "Could not extract multiple equations. Please use format: '2x + y = 5 and x - y = 1'",
                "needs_clarification": True
            }
        
        # Normalize equations
        normalized_equations = []
        normalization_errors = []
        
        for eq in raw_equations:
            normalized = self._normalize_equation(eq)
            # Validate
            norm_result, norm_error = MathNormalizer.normalize_equation(normalized)
            if norm_error:
                normalization_errors.append(f"Error in '{eq}': {norm_error}")
            else:
                normalized_equations.append(norm_result)
        
        if normalization_errors:
            return {
                "type": "system",
                "problem_text": problem_text,
                "equations": raw_equations,
                "normalized_equations": normalized_equations,
                "variables": self._extract_variables(raw_equations),
                "error": "; ".join(normalization_errors),
                "needs_clarification": True
            }
        
        # Extract variables
        variables = self._extract_variables(raw_equations)
        
        return {
            "type": "system",
            "problem_text": problem_text,
            "equations": raw_equations,
            "normalized_equations": normalized_equations,
            "variables": variables,
            "error": None,
            "needs_clarification": False
        }


def main():
    """Test system parser."""
    parser = SystemParserAgent()
    
    test_cases = [
        "Solve 2x + y = 5 and x - y = 1",
        "Find x and y: 3x + 2y = 12, x - y = 1",
        "Solve the system: x + y = 10; 2x - y = 5",
        "x + y + z = 6 and x - y = 2 and y + z = 4"
    ]
    
    print("System Parser Test Results:")
    print("=" * 70)
    
    for case in test_cases:
        print(f"\nInput: {case}")
        result = parser.parse(case)
        print(f"  Equations: {result['equations']}")
        print(f"  Normalized: {result['normalized_equations']}")
        print(f"  Variables: {result['variables']}")
        if result.get('error'):
            print(f"  Error: {result['error']}")


if __name__ == "__main__":
    main()
