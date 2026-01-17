import os
import re
import sys
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.groq_client import GroqClient
from utils.math_normalizer import MathNormalizer


class ParserAgent:
    """
    Production-grade deterministic math parser using Groq.
    """

    def __init__(self):
        self.llm = GroqClient(model="llama-3.1-8b-instant")

    def extract_equation(self, text: str):
        text = text.lower().strip()

        # Remove trailing "for x", "for y"
        text = re.sub(r"\s+for\s+[a-z]\b", "", text)

        # Look for equation pattern first
        match = re.search(r"([0-9a-zA-Z+\-*/().^ ]+\s*=\s*[0-9a-zA-Z+\-*/().^ ]+)", text)

        if not match:
            return None

        equation = match.group(1).strip()

        # Remove instruction words AFTER extraction
        instruction_words = [
            "solve", "find", "determine", "calculate",
            "evaluate", "compute", "what is", "equation",
            "the", "when", "where", "if", "given", "such that", "that"
        ]

        for word in instruction_words:
            equation = equation.replace(word, " ")

        # Normalize spaces
        equation = " ".join(equation.split()).strip()

        # Normalize implicit multiplication
        equation = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation)

        # Final safety check
        if not re.fullmatch(r"[0-9a-zA-Z+\-*/=().^ ]+", equation):
            return None

        return equation

    def extract_variables(self, equation: str):
        """Extract variable symbols from equation."""
        variables = set()
        for match in re.finditer(r'(?<![a-zA-Z])([a-zA-Z])(?![a-zA-Z])', equation):
            var = match.group(1)
            if var not in ["e", "i"]:
                variables.add(var)
        return sorted(list(variables))

    def classify_topic(self, problem_text: str):
        """Classify problem topic using LLM."""
        prompt = f"""
Classify this math problem into exactly ONE category:

algebra
probability
calculus
linear_algebra

Problem:
{problem_text}

Return only one word.
"""
        try:
            topic = self.llm.generate(prompt, temperature=0.0).lower().strip()
            if topic not in ["algebra", "probability", "calculus", "linear_algebra"]:
                return "algebra"
            return topic
        except:
            return "algebra"

    def parse(self, problem_text: str) -> Dict[str, Any]:
        """Parse math problem into structured format."""
        
        # Get topic first - needed to determine if equation is required
        topic = self.classify_topic(problem_text)
        
        # Override topic for optimization problems (LLM sometimes misclassifies)
        text_lower = problem_text.lower()
        if any(kw in text_lower for kw in ['maximize', 'minimize', 'maximum', 'minimum', 'optimize']):
            topic = "calculus"
        
        equation = self.extract_equation(problem_text)

        if not equation:
            # For calculus and probability, equation is NOT required
            # These problems are handled by domain-specific parsers
            if topic in ["calculus", "probability"]:
                return {
                    "problem_text": problem_text,
                    "topic": topic,
                    "variables": [],
                    "constraints": [],
                    "normalized_equation": None,
                    "normalization_error": None,
                    "needs_clarification": False,  # Allow to proceed
                    "error_type": None
                }
            
            # For algebra/linear_algebra, check if it's a system of equations
            # System problems might not match single equation pattern
            if " and " in problem_text.lower() or problem_text.lower().count("=") >= 2:
                return {
                    "problem_text": problem_text,
                    "topic": "linear_algebra",
                    "variables": [],
                    "constraints": [],
                    "normalized_equation": None,
                    "normalization_error": None,
                    "needs_clarification": False,  # Allow system parser to handle
                    "error_type": None
                }
            
            # Check for direct calculation questions (NEW - square root, factorial, etc.)
            # These don't need equations - they're direct value queries
            direct_calc_keywords = [
                'square root', 'sqrt', 'cube root', 'root of',
                'factorial', 'power of', 'raised to',
                'log of', 'logarithm', 'log10', 'log2', 'ln',
                'sin of', 'cos of', 'tan of', 'sine', 'cosine', 'tangent',
                'absolute value', 'squared', 'cubed'
            ]
            if any(kw in text_lower for kw in direct_calc_keywords):
                return {
                    "problem_text": problem_text,
                    "topic": topic,
                    "variables": [],
                    "constraints": [],
                    "normalized_equation": None,
                    "normalization_error": None,
                    "needs_clarification": False,  # Allow direct calculation handler
                    "error_type": None,
                    "is_direct_calculation": True  # Mark for special handling
                }
            
            # Check for Explanation/Knowledge requests (NEW - "give me a list", "what is", etc.)
            explanation_keywords = [
                'give me', 'list of', 'show me', 'what is', 'explain', 'define',
                'identities', 'formulas', 'theorem', 'tell me about'
            ]
            if any(kw in text_lower for kw in explanation_keywords):
                return {
                    "problem_text": problem_text,
                    "topic": topic,
                    "variables": [],
                    "constraints": [],
                    "normalized_equation": None,
                    "normalization_error": None,
                    "needs_clarification": False,  # Allow explanation handler
                    "error_type": None
                }
            
            # Only block if truly unclear
            return {
                "problem_text": problem_text,
                "topic": topic,
                "variables": [],
                "constraints": [],
                "normalized_equation": None,
                "normalization_error": "No valid equation found",
                "needs_clarification": True,
                "error_type": "no_equation_found"
            }

        normalized_eq, norm_error = MathNormalizer.normalize_equation(equation)

        if norm_error:
            return {
                "problem_text": problem_text,
                "topic": self.classify_topic(problem_text),
                "variables": [],
                "constraints": [equation],
                "normalized_equation": None,
                "normalization_error": norm_error,
                "needs_clarification": True,
                "error_type": "math_parse_error"
            }

        variables = self.extract_variables(equation)
        topic = self.classify_topic(problem_text)

        return {
            "problem_text": problem_text,
            "topic": topic,
            "variables": variables,
            "constraints": [equation],
            "normalized_equation": normalized_eq,
            "normalization_error": None,
            "needs_clarification": False
        }
