"""
Calculus Parser Agent
Parses calculus expressions (derivatives, integrals, limits, optimization)
"""

import os
import re
import sys
from typing import Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CalculusParserAgent:
    """
    Parses calculus problems from natural language.
    Supports: derivatives, integrals, limits, optimization
    """
    
    def __init__(self):
        pass
    
    def _detect_calculus_type(self, text: str) -> str:
        """
        Detect the type of calculus operation.
        """
        text = text.lower()
        
        # Derivative detection
        if any(kw in text for kw in ['derivative', 'differentiate', 'd/dx', 'dy/dx', "f'(x)", "f'"]):
            return "derivative"
        
        # Integral detection
        if any(kw in text for kw in ['integral', 'integrate', 'integration', '∫', 'antiderivative']):
            return "integral"
        
        # Limit detection
        if any(kw in text for kw in ['limit', 'lim', 'approaches', 'tends to', '→', '->']):
            return "limit"
        
        # Optimization detection
        if any(kw in text for kw in ['maximize', 'minimize', 'maximum', 'minimum', 'optimize', 'optimal']):
            return "optimization"
        
        return "unknown"
    
    def _extract_expression(self, text: str, calc_type: str) -> str:
        """
        Extract the mathematical expression from text.
        """
        text_lower = text.lower()
        
        # Clean up the text
        text_clean = text_lower
        
        # Remove common prefixes based on type
        if calc_type == "derivative":
            prefixes = ['find the derivative of', 'derivative of', 'differentiate', 
                       'd/dx', 'dy/dx', "find f'(x) for", "find f' of",
                       "what is the derivative of", "calculate the derivative of",
                       "find the derivative", "how to differentiate", "how do i differentiate", 
                       "how do you differentiate", "how to find derivative of", "how to"]
            # Sort by length descending to match longer prefixes first
            prefixes.sort(key=len, reverse=True)
            for prefix in prefixes:
                text_clean = text_clean.replace(prefix, '')
        
        elif calc_type == "integral":
            prefixes = ['how to integrate', 'how do i integrate', 'how do you integrate',
                       'find the integral of', 'integral of', 'integrate', 
                       'find the antiderivative of', '∫', 'how to']
            prefixes.sort(key=len, reverse=True)
            for prefix in prefixes:
                text_clean = text_clean.replace(prefix, '')
            # Remove dx at the end
            text_clean = re.sub(r'\s*d[a-z]\s*$', '', text_clean)
        
        elif calc_type == "limit":
            # Extract expression before "as" or "when"
            match = re.search(r'(?:limit|lim)\s*(?:of\s+)?(.+?)(?:\s+as\s+|\s+when\s+|\s+->|\s+→)', text_clean)
            if match:
                text_clean = match.group(1)
            else:
                prefixes = ['find the limit of', 'limit of', 'lim', 'evaluate the limit']
                prefixes.sort(key=len, reverse=True)
                for prefix in prefixes:
                    text_clean = text_clean.replace(prefix, '')
        
        elif calc_type == "optimization":
            prefixes = ['maximize', 'minimize', 'find the maximum of', 'find the minimum of',
                       'find the optimal value of', 'optimize']
            prefixes.sort(key=len, reverse=True)
            for prefix in prefixes:
                text_clean = text_clean.replace(prefix, '')
        
        # Clean up
        text_clean = text_clean.strip()
        
        # CRITICAL: Remove equation part (e.g., "= 0" or "= 5")
        # For derivatives, we only need the expression, not the equation
        text_clean = re.sub(r'\s*=\s*[-+]?\d*\.?\d*\s*$', '', text_clean)
        
        # Normalize natural language math expressions
        nl_to_math = [
            (r'\bx\s+square\b', 'x**2'),
            (r'\bx\s+squared\b', 'x**2'),
            (r'\bx\s+cube\b', 'x**3'),
            (r'\bx\s+cubed\b', 'x**3'),
            (r'\by\s+square\b', 'y**2'),
            (r'\by\s+squared\b', 'y**2'),
            (r'\bsquare\s+of\s+x\b', 'x**2'),
            (r'\bcube\s+of\s+x\b', 'x**3'),
        ]
        
        for pattern, replacement in nl_to_math:
            text_clean = re.sub(pattern, replacement, text_clean, flags=re.IGNORECASE)
        
        # Extract mathematical expression pattern
        # First, try to match math functions like sin(x), cos(x), log(x), etc.
        math_funcs = r'(?:sin|cos|tan|cot|sec|csc|log|ln|exp|sqrt|arcsin|arccos|arctan|asin|acos|atan)'
        
        # Pattern 1: Match expressions starting with math functions
        # e.g., sin(x), cos(x) + 1, log(x)/x, etc.
        func_pattern = rf'({math_funcs}\s*\([^)]+\)(?:\s*[+\-*/^]\s*[0-9a-z()*\s^+\-/]+)*)'
        match = re.search(func_pattern, text_clean, re.IGNORECASE)
        if match:
            expr = match.group(1).strip()
            expr = re.sub(r'(\d)([xyt])', r'\1*\2', expr)  # 8x -> 8*x
            expr = expr.replace('^', '**')  # ^ -> **
            expr = expr.replace(' ', '')
            if expr:
                return expr
        
        # Pattern 2: Match polynomial-like expressions
        # Must contain at least one variable (x, y, t) and math symbols
        # Match expressions like: x^3 + 8x + 16, x**2 + 3*x, t^3 + 2t^2
        match = re.search(r'((?:[0-9+\-*/^().\s]*[xyt])+[0-9+\-*/^().\sxyt]*)', text_clean)
        if match:
            expr = match.group(1).strip()
            # Clean up any trailing/leading operators
            expr = re.sub(r'^[+\-*/\s]+', '', expr)
            expr = re.sub(r'[+\-*/\s]+$', '', expr)
            # Normalize
            expr = re.sub(r'(\d)([xyt])', r'\1*\2', expr)  # 8x -> 8*x
            expr = re.sub(r'([xyt])(\d)', r'\1**\2', expr)  # x3 -> x**3 (common typo)
            expr = expr.replace('^', '**')  # ^ -> **
            expr = expr.replace(' ', '')
            if expr:
                return expr
        
        # Fallback: try a more general extraction
        expr = text_clean.strip()
        expr = re.sub(r'(\d)([xyt])', r'\1*\2', expr)
        expr = expr.replace('^', '**')
        expr = expr.replace(' ', '')
        return expr
    
    def _extract_variable(self, text: str, expression: str) -> str:
        """
        Extract the variable of interest.
        """
        text_lower = text.lower()
        
        # Look for explicit variable mention
        match = re.search(r'with respect to\s+([a-z])', text_lower)
        if match:
            return match.group(1)
        
        match = re.search(r'd/d([a-z])', text_lower)
        if match:
            return match.group(1)
        
        # Look for "dx" pattern in integrals
        match = re.search(r'd([a-z])\s*$', text_lower)
        if match:
            return match.group(1)
        
        # Default: find single-letter variables in expression
        vars_found = set(re.findall(r'(?<![a-zA-Z])([a-z])(?![a-zA-Z])', expression.lower()))
        vars_found.discard('e')  # Euler's number
        vars_found.discard('i')  # Imaginary unit
        
        if 'x' in vars_found:
            return 'x'
        elif 't' in vars_found:
            return 't'
        elif vars_found:
            return sorted(vars_found)[0]
        
        return 'x'  # Default
    
    def _extract_limit_point(self, text: str) -> Optional[str]:
        """
        Extract the limit point (e.g., "as x approaches 0").
        """
        text_lower = text.lower()
        
        # Pattern: "as x approaches 0" or "x -> 0" or "x → 0"
        patterns = [
            r'as\s+[a-z]\s+(?:approaches|tends to|goes to)\s+([-+]?\d*\.?\d+|infinity|∞|inf)',
            r'[a-z]\s*(?:->|→)\s*([-+]?\d*\.?\d+|infinity|∞|inf)',
            r'when\s+[a-z]\s*=\s*([-+]?\d*\.?\d+)',
            r'at\s+[a-z]\s*=\s*([-+]?\d*\.?\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                point = match.group(1)
                # Normalize infinity
                if point in ['infinity', '∞', 'inf']:
                    return 'oo'  # SymPy infinity
                return point
        
        return '0'  # Default
    
    def _extract_optimization_type(self, text: str) -> str:
        """
        Determine if maximizing or minimizing.
        """
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ['maximize', 'maximum', 'largest', 'greatest']):
            return "maximize"
        elif any(kw in text_lower for kw in ['minimize', 'minimum', 'smallest', 'least']):
            return "minimize"
        
        return "extremum"  # Find both
    
    def parse(self, problem_text: str) -> Dict[str, Any]:
        """
        Parse calculus problem from natural language.
        
        Args:
            problem_text: User's problem text
            
        Returns:
            Dict with type, expression, variable, and operation-specific fields
        """
        calc_type = self._detect_calculus_type(problem_text)
        
        if calc_type == "unknown":
            return {
                "type": "unknown",
                "problem_text": problem_text,
                "error": "Could not determine calculus operation type",
                "needs_clarification": True
            }
        
        expression = self._extract_expression(problem_text, calc_type)
        variable = self._extract_variable(problem_text, expression)
        
        result = {
            "type": calc_type,
            "problem_text": problem_text,
            "expression": expression,
            "variable": variable,
            "error": None,
            "needs_clarification": False
        }
        
        # Add type-specific fields
        if calc_type == "limit":
            result["limit_point"] = self._extract_limit_point(problem_text)
        
        elif calc_type == "optimization":
            result["optimization_type"] = self._extract_optimization_type(problem_text)
        
        return result


def main():
    """Test calculus parser."""
    parser = CalculusParserAgent()
    
    test_cases = [
        "Find the derivative of x^2 + 3x",
        "Differentiate sin(x) with respect to x",
        "Integrate x^2 dx",
        "Find the integral of 2x + 1",
        "Find the limit of sin(x)/x as x approaches 0",
        "Evaluate lim x->infinity of 1/x",
        "Maximize x(10-x)",
        "Find the minimum of x^2 + 4x + 4"
    ]
    
    print("Calculus Parser Test Results:")
    print("=" * 70)
    
    for case in test_cases:
        print(f"\nInput: {case}")
        result = parser.parse(case)
        print(f"  Type: {result['type']}")
        print(f"  Expression: {result.get('expression', 'N/A')}")
        print(f"  Variable: {result.get('variable', 'N/A')}")
        if result.get('limit_point'):
            print(f"  Limit Point: {result['limit_point']}")
        if result.get('optimization_type'):
            print(f"  Optimization: {result['optimization_type']}")


if __name__ == "__main__":
    main()
