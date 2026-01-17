"""
Router Agent
Classifies problem topic and intent, routes to appropriate solver using Groq
UPGRADED: Two-level routing with deterministic intent detection
"""

import json
import os
import re
from typing import Dict
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.groq_client import GroqClient


class RouterAgent:
    """
    Routes problems to appropriate topic-specific solver.
    
    Two-level routing:
    1. Subject: algebra, probability, calculus, linear_algebra
    2. Intent: equation, system_of_equations, derivative, integral, limit, optimization, probability
    """
    
    def __init__(self):
        self.llm = GroqClient(model="llama-3.3-70b-versatile")
    
    def _detect_intent_deterministic(self, problem_text: str) -> str:
        """
        Deterministic intent detection using keyword matching.
        Runs BEFORE LLM for speed and reliability.
        
        Returns:
            Intent string or None if unclear
        """
        text = problem_text.lower()
        
        # System of equations: multiple '=' or contains "and" with equations
        equals_count = text.count('=')
        has_and = ' and ' in text
        has_simultaneous = 'simultaneous' in text or 'system' in text
        
        if equals_count >= 2 or (equals_count >= 1 and has_and) or has_simultaneous:
            return "system_of_equations"
        
        # Derivative detection
        derivative_keywords = ['derivative', 'differentiate', 'd/dx', 'dy/dx', "f'(x)", "f'"]
        if any(kw in text for kw in derivative_keywords):
            return "derivative"
        
        # Integral detection
        integral_keywords = ['integral', 'integrate', 'integration', '∫', 'antiderivative']
        if any(kw in text for kw in integral_keywords):
            return "integral"
        
        # Limit detection
        limit_keywords = ['limit', 'lim', 'approaches', 'tends to', '→', '->']
        if any(kw in text for kw in limit_keywords):
            return "limit"
        
        # Optimization detection
        optimization_keywords = ['maximize', 'minimize', 'maximum', 'minimum', 'optimize', 'optimal', 'greatest', 'least value']
        if any(kw in text for kw in optimization_keywords):
            return "optimization"
        
        # Probability detection
        probability_keywords = ['probability', 'chance', 'likely', 'odds', 'expected', 'random', 'coin', 'dice', 'cards']
        if any(kw in text for kw in probability_keywords):
            return "probability"
        
        # Direct calculation detection (NEW - for questions like "what is the square root of 1")
        direct_calc_keywords = [
            'square root', 'sqrt', 'cube root', 'root of',
            'factorial', 'power of', 'raised to',
            'log of', 'logarithm', 'log10', 'log2', 'ln',
            'sin of', 'cos of', 'tan of', 'sine', 'cosine', 'tangent',
            'absolute value', 'floor', 'ceiling', 'round',
            'what is', 'calculate', 'evaluate', 'compute', 'find the value',
            'how much is', 'what does', 'equal to'
        ]
        question_starters = ['what is', 'calculate', 'find', 'evaluate', 'compute', 'how much']
        math_operations = ['square root', 'sqrt', 'cube root', 'factorial', 'power', 'log', 'sin', 'cos', 'tan', 'root']
        
        # Check if it's a direct calculation question
        has_question_starter = any(text.startswith(qs) or qs in text for qs in question_starters)
        has_math_operation = any(op in text for op in math_operations)
        
        if has_question_starter and has_math_operation and equals_count == 0:
            return "direct_calculation"
        
        # Also catch simple value questions like "5 factorial", "2^10", etc.
        if any(kw in text for kw in ['factorial', 'squared', 'cubed']) and equals_count == 0:
            return "direct_calculation"
        
        # Single equation (fallback if contains '=')
        if equals_count == 1:
            return "equation"
        
        # Cannot determine deterministically
        return None
    
    def _detect_subject_deterministic(self, problem_text: str, intent: str) -> str:
        """
        Deterministic subject detection based on intent and keywords.
        """
        text = problem_text.lower()
        
        # Intent-based subject mapping
        if intent in ["derivative", "integral", "limit", "optimization"]:
            return "calculus"
        
        if intent == "probability":
            return "probability"
        
        if intent == "system_of_equations":
            return "linear_algebra"
        
        if intent == "direct_calculation":
            return "algebra"  # Direct calculations default to algebra
        
        # Keyword-based detection for equations
        if 'matrix' in text or 'matrices' in text or 'determinant' in text:
            return "linear_algebra"
        
        # Default to algebra for single equations
        return "algebra"
    
    def route(self, parsed_problem: Dict) -> Dict[str, any]:
        """
        Classify problem subject and intent, return routing decision.
        
        Args:
            parsed_problem: Parsed problem dict from ParserAgent
            
        Returns:
            Dict with subject, intent, route (for backward compatibility), and confidence
        """
        problem_text = parsed_problem.get("problem_text", "")
        suggested_topic = parsed_problem.get("topic", "")
        
        # Step 1: Deterministic intent detection (FAST)
        intent = self._detect_intent_deterministic(problem_text)
        
        if intent:
            # Deterministic detection succeeded
            subject = self._detect_subject_deterministic(problem_text, intent)
            return {
                "subject": subject,
                "intent": intent,
                "route": subject,  # Backward compatibility
                "confidence": 0.95
            }
        
        # Step 2: LLM fallback for unclear cases
        prompt = f"""You are a math problem router. Classify the subject AND intent.

Problem: {problem_text}
Suggested Topic: {suggested_topic}

Subjects:
- algebra: Linear/quadratic equations, polynomials, inequalities
- probability: Probability, combinations, permutations
- calculus: Limits, derivatives, integrals, optimization
- linear_algebra: Matrices, vectors, systems of equations

Intents:
- equation: Solve a single equation
- system_of_equations: Solve multiple simultaneous equations
- derivative: Find derivative/differentiate
- integral: Find integral/integrate
- limit: Evaluate a limit
- optimization: Find maximum/minimum
- probability: Calculate probability
- direct_calculation: Direct value calculation (square root, factorial, etc.)

Return ONLY valid JSON:
{{
  "subject": "algebra|probability|calculus|linear_algebra",
  "intent": "equation|system_of_equations|derivative|integral|limit|optimization|probability|direct_calculation",
  "confidence": 0.95
}}"""

        try:
            result_text = self.llm.generate(prompt, temperature=0.0)
            
            # Extract JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            
            # Validate subject
            valid_subjects = ["algebra", "probability", "calculus", "linear_algebra"]
            if result.get("subject") not in valid_subjects:
                result["subject"] = suggested_topic or "algebra"
            
            # Validate intent
            valid_intents = ["equation", "system_of_equations", "derivative", "integral", "limit", "optimization", "probability", "direct_calculation"]
            if result.get("intent") not in valid_intents:
                result["intent"] = "equation"  # Default
            
            result.setdefault("confidence", 0.8)
            result["route"] = result["subject"]  # Backward compatibility
            
            return result
            
        except Exception as e:
            # Fallback
            return {
                "subject": suggested_topic or "algebra",
                "intent": "equation",
                "route": suggested_topic or "algebra",  # Backward compatibility
                "confidence": 0.5,
                "error": str(e)
            }


def main():
    """Test router agent with various problem types."""
    router = RouterAgent()
    
    test_cases = [
        {"problem_text": "2x + 3 = 5", "topic": "algebra"},
        {"problem_text": "Solve x² + 5x + 6 = 0", "topic": "algebra"},
        {"problem_text": "Solve 2x + y = 5 and x - y = 1", "topic": "linear_algebra"},
        {"problem_text": "Find derivative of x²", "topic": "calculus"},
        {"problem_text": "Integrate x² dx", "topic": "calculus"},
        {"problem_text": "Find limit of sin(x)/x as x approaches 0", "topic": "calculus"},
        {"problem_text": "Maximize x(10-x)", "topic": "calculus"},
        {"problem_text": "Probability of 2 heads in 3 flips", "topic": "probability"}
    ]
    
    for case in test_cases:
        print(f"\nProblem: {case['problem_text']}")
        print("-" * 60)
        result = router.route(case)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

