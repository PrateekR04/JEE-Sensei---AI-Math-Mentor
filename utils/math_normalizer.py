"""
Math Expression Normalizer
Production-safe SymPy normalization
"""

import re
from typing import Tuple, Optional


class MathNormalizer:
    """
    Normalizes math expressions for safe symbolic parsing.
    """

    @staticmethod
    def normalize_implicit_multiplication(expr: str) -> str:
        """
        Convert implicit multiplication to explicit.
        """

        expr = expr.replace(" ", "")

        # 2x → 2*x
        expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)

        # 2(x+1) → 2*(x+1)
        expr = re.sub(r'(\d)\(', r'\1*(', expr)

        # (x+1)(x-1) → (x+1)*(x-1)
        expr = re.sub(r'\)\(', r')*(', expr)

        # x(y+1) → x*(y+1)
        expr = re.sub(r'([a-zA-Z])\(', r'\1*(', expr)

        # (x+1)y → (x+1)*y
        expr = re.sub(r'\)([a-zA-Z])', r')*\1', expr)

        return expr


    @staticmethod
    def validate_parentheses(expr: str) -> Tuple[bool, Optional[str]]:
        stack = []
        for i, c in enumerate(expr):
            if c == '(':
                stack.append(i)
            elif c == ')':
                if not stack:
                    return False, f"Unbalanced parenthesis: extra ')' at position {i}"
                stack.pop()

        if stack:
            return False, f"Unbalanced parenthesis: unclosed '(' at position {stack[0]}"

        return True, None


    @staticmethod
    def validate_characters(expr: str) -> Tuple[bool, Optional[str]]:
        """
        Only allow safe math characters
        """
        if not re.fullmatch(r"[0-9a-zA-Z+\-*/=().^]+", expr):
            return False, "Invalid characters in expression"
        return True, None


    @classmethod
    def normalize_equation(cls, equation: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Normalize already-extracted equation.
        """

        equation = equation.strip()

        # Must contain =
        if "=" not in equation:
            return None, "Not a valid equation"

        # Normalize implicit multiplication
        equation = cls.normalize_implicit_multiplication(equation)

        # Validate parentheses
        ok, err = cls.validate_parentheses(equation)
        if not ok:
            return None, err

        # Validate characters
        ok, err = cls.validate_characters(equation)
        if not ok:
            return None, err

        return equation, None
