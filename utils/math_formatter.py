"""
Math Formatting Utilities
Converts SymPy/raw math expressions to human-readable format
with Unicode superscripts and subscripts
"""

import re


# Unicode superscript characters
SUPERSCRIPTS = {
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
    '+': '⁺', '-': '⁻', '(': '⁽', ')': '⁾',
    'n': 'ⁿ', 'i': 'ⁱ'
}

# Unicode subscript characters
SUBSCRIPTS = {
    '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
    '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
    '+': '₊', '-': '₋', '(': '₍', ')': '₎',
    'a': 'ₐ', 'e': 'ₑ', 'i': 'ᵢ', 'n': 'ₙ', 'x': 'ₓ'
}


def to_superscript(text: str) -> str:
    """Convert a string to superscript Unicode characters."""
    result = ""
    for char in str(text):
        result += SUPERSCRIPTS.get(char, char)
    return result


def to_subscript(text: str) -> str:
    """Convert a string to subscript Unicode characters."""
    result = ""
    for char in str(text):
        result += SUBSCRIPTS.get(char, char)
    return result


def format_math_expression(expr: str) -> str:
    """
    Convert a SymPy/raw math expression to human-readable format.
    
    Examples:
        3*x**2 + 8  →  3x² + 8
        x**3 + 2*x  →  x³ + 2x
        sqrt(x)     →  √x
        
    Args:
        expr: Math expression string
        
    Returns:
        Human-readable formatted string
    """
    if not expr:
        return expr
    
    result = str(expr)
    
    # Replace ** with superscripts
    # Match patterns like: x**2, x**3, x**n, (x+1)**2
    def replace_power(match):
        base = match.group(1)
        exponent = match.group(2)
        # Convert exponent to superscript
        sup_exp = to_superscript(exponent)
        return f"{base}{sup_exp}"
    
    # Handle simple powers like x**2, y**3
    result = re.sub(r'([a-zA-Z])\*\*(\d+)', replace_power, result)
    
    # Handle parenthesized powers like (x+1)**2
    result = re.sub(r'\)\*\*(\d+)', lambda m: ')' + to_superscript(m.group(1)), result)
    
    # Remove multiplication signs between numbers and variables
    # 3*x → 3x, 2*y → 2y
    result = re.sub(r'(\d)\*([a-zA-Z])', r'\1\2', result)
    
    # Remove multiplication signs between variable and parenthesis
    # x*(y+1) → x(y+1)
    result = re.sub(r'([a-zA-Z])\*\(', r'\1(', result)
    
    # Replace common math functions with symbols
    result = result.replace('sqrt(', '√(')
    result = result.replace('pi', 'π')
    
    # Clean up extra spaces
    result = re.sub(r'\s+', ' ', result)
    
    return result


def format_answer(answer: str) -> str:
    """
    Format a math answer for display.
    This is a wrapper around format_math_expression with additional cleanup.
    """
    if not answer:
        return answer
    
    formatted = format_math_expression(answer)
    
    # Add spaces around + and - for readability (but not inside superscripts)
    # Only add spaces if not already present
    formatted = re.sub(r'(?<=[^\s⁺⁻₊₋])\+(?=[^\s])', ' + ', formatted)
    formatted = re.sub(r'(?<=[^\s⁺⁻₊₋])-(?=[^\s])', ' - ', formatted)
    
    return formatted


# Test the formatting
if __name__ == "__main__":
    test_expressions = [
        "3*x**2 + 8",
        "x**3 + 2*x**2 + 5*x",
        "2*x + 5",
        "x**2 + y**2",
        "sqrt(x)",
        "3*x**2 + 8*x + 16",
        "sin(x)**2 + cos(x)**2",
        "(x+1)**2",
    ]
    
    print("Math Expression Formatting Test")
    print("=" * 50)
    for expr in test_expressions:
        formatted = format_answer(expr)
        print(f"{expr:30} → {formatted}")
