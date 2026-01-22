"""
LaTeX Preprocessor
Converts LaTeX mathematical notation to plain text for parser processing.
"""

import re
from typing import Tuple


class LaTeXPreprocessor:
    r"""
    Converts LaTeX mathematical expressions to parser-friendly plain text.
    
    Handles:
    - Display and inline math delimiters ($, $$, \[, \], \(, \))
    - Fractions (\frac{a}{b} -> (a)/(b))
    - Greek letters (\alpha, \beta, etc.)
    - Operators (\times, \div, \cdot)
    - Limits, integrals, sums
    - Special symbols
    """
    
    @staticmethod
    def is_latex(text: str) -> bool:
        """Check if text contains LaTeX notation."""
        latex_markers = [
            '\\frac', '\\lim', '\\int', '\\sum', '\\prod',
            '\\left', '\\right', '\\sqrt', '\\alpha', '\\beta',
            '\\theta', '\\infty', '\\to', '\\cdot', '\\times',
            '\\partial', '\\nabla', '\\sin', '\\cos', '\\tan',
            '\\log', '\\ln', '\\exp', '\\leq', '\\geq', '\\neq',
            '\\pm', '\\mp', '\\ldots', '\\cdots', '\\dots',
            '_{', '^{', r'\[', r'\]', r'\(', r'\)',
        ]
        # Also check for $...$ patterns
        if '$' in text and text.count('$') >= 2:
            return True
        return any(marker in text for marker in latex_markers)
    
    @classmethod
    def preprocess(cls, text: str) -> Tuple[str, bool]:
        """
        Convert LaTeX to plain text.
        
        Args:
            text: Input text potentially containing LaTeX
            
        Returns:
            Tuple of (processed_text, was_latex)
        """
        if not cls.is_latex(text):
            return text, False
        
        result = text
        
        # Step 1: Remove math delimiters
        # Remove $$ ... $$ (display math)
        result = re.sub(r'\$\$(.+?)\$\$', r'\1', result, flags=re.DOTALL)
        # Remove $ ... $ (inline math)
        result = re.sub(r'\$(.+?)\$', r'\1', result)
        # Remove \[ ... \] and \( ... \)
        result = result.replace('\\[', '').replace('\\]', '')
        result = result.replace('\\(', '').replace('\\)', '')
        
        # Step 2: Handle \left and \right delimiters
        result = result.replace('\\left(', '(').replace('\\right)', ')')
        result = result.replace('\\left[', '[').replace('\\right]', ']')
        result = result.replace('\\left\\{', '{').replace('\\right\\}', '}')
        result = result.replace('\\left|', '|').replace('\\right|', '|')
        result = result.replace('\\left.', '').replace('\\right.', '')
        
        # Step 3: Handle fractions - \frac{a}{b} -> (a)/(b)
        # Use iterative approach for nested fractions
        max_iterations = 10
        for _ in range(max_iterations):
            # Match \frac{...}{...} where content doesn't contain nested braces at same level
            old_result = result
            result = re.sub(
                r'\\frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
                r'((\1)/(\2))',
                result
            )
            if old_result == result:
                break
        
        # Step 4: Handle limits - \lim_{x \to a} -> lim(x→a)
        result = re.sub(r'\\lim_\{([^{}]+)\\to\s*([^{}]+)\}', r'lim(\1→\2)', result)
        result = re.sub(r'\\lim_\{([^{}]+)\}', r'lim(\1)', result)
        
        # Step 5: Handle sqrt - \sqrt{x} -> sqrt(x), \sqrt[n]{x} -> nthroot(x,n)
        result = re.sub(r'\\sqrt\[([^\]]+)\]\{([^{}]+)\}', r'root(\2,\1)', result)
        result = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', result)
        result = result.replace('\\sqrt', 'sqrt')
        
        # Step 6: Handle subscripts and superscripts
        # x_{n} -> x_n, x^{2} -> x^2
        result = re.sub(r'_\{([^{}]+)\}', r'_\1', result)
        result = re.sub(r'\^\{([^{}]+)\}', r'^\1', result)
        
        # Step 7: Greek letters and symbols
        greek_and_symbols = [
            ('\\alpha', 'α'), ('\\beta', 'β'), ('\\gamma', 'γ'),
            ('\\delta', 'δ'), ('\\epsilon', 'ε'), ('\\theta', 'θ'),
            ('\\lambda', 'λ'), ('\\mu', 'μ'), ('\\pi', 'π'),
            ('\\sigma', 'σ'), ('\\phi', 'φ'), ('\\omega', 'ω'),
            ('\\infty', '∞'), ('\\infinity', '∞'),
            ('\\to', '→'), ('\\rightarrow', '→'), ('\\Rightarrow', '⇒'),
            ('\\leftarrow', '←'), ('\\Leftarrow', '⇐'),
            ('\\leq', '≤'), ('\\le', '≤'), ('\\geq', '≥'), ('\\ge', '≥'),
            ('\\neq', '≠'), ('\\ne', '≠'), ('\\approx', '≈'),
            ('\\pm', '±'), ('\\mp', '∓'),
            ('\\times', '*'), ('\\cdot', '*'), ('\\div', '/'),
            ('\\ldots', '...'), ('\\cdots', '...'), ('\\dots', '...'),
            ('\\forall', '∀'), ('\\exists', '∃'),
            ('\\in', '∈'), ('\\notin', '∉'),
            ('\\subset', '⊂'), ('\\supset', '⊃'),
            ('\\cup', '∪'), ('\\cap', '∩'),
        ]
        for latex, plain in greek_and_symbols:
            result = result.replace(latex, plain)
        
        # Step 8: Trig and log functions - remove backslash
        functions = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc',
                     'sinh', 'cosh', 'tanh', 'arcsin', 'arccos', 'arctan',
                     'log', 'ln', 'exp', 'max', 'min', 'lim', 'sum', 'prod']
        for func in functions:
            result = result.replace(f'\\{func}', func)
        
        # Step 9: Handle integrals and sums
        # \int_{a}^{b} -> integral(a to b)
        result = re.sub(r'\\int_\{([^{}]+)\}\^\{([^{}]+)\}', r'integral(\1 to \2)', result)
        result = result.replace('\\int', 'integral')
        
        # \sum_{i=1}^{n} -> sum(i=1 to n)
        result = re.sub(r'\\sum_\{([^{}]+)\}\^\{([^{}]+)\}', r'sum(\1 to \2)', result)
        result = result.replace('\\sum', 'sum')
        
        # \prod_{i=1}^{n} -> prod(i=1 to n)
        result = re.sub(r'\\prod_\{([^{}]+)\}\^\{([^{}]+)\}', r'prod(\1 to \2)', result)
        result = result.replace('\\prod', 'prod')
        
        # Step 10: Handle partials - \partial -> ∂
        result = result.replace('\\partial', '∂')
        result = result.replace('\\nabla', '∇')
        
        # Step 11: Remove any remaining LaTeX commands (backslash followed by word)
        result = re.sub(r'\\[a-zA-Z]+', '', result)
        
        # Step 12: Clean up braces - remaining {} can be removed
        result = result.replace('{', '').replace('}', '')
        
        # Step 13: Clean up whitespace
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result, True
    
    @classmethod
    def extract_plain_expression(cls, latex_text: str) -> str:
        """
        Extract a plain mathematical expression from LaTeX for solver processing.
        More aggressive cleanup for equation solving.
        """
        result, _ = cls.preprocess(latex_text)
        
        # Additional cleanup for equation solving
        # Remove text like "for all x > 0" at the end
        result = re.sub(r'\s+(for\s+all|where|such\s+that|if|when|given)\s+.*$', '', result, flags=re.IGNORECASE)
        
        # Remove leading "find", "solve", "evaluate" etc.
        result = re.sub(r'^(find|solve|evaluate|compute|calculate|determine)\s+', '', result, flags=re.IGNORECASE)
        
        return result.strip()
