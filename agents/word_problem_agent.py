"""
Word Problem Modeling Agent
Converts natural language JEE word problems into symbolic equations.

This agent acts as a TRANSLATOR, not a solver.
It only converts English → Math, then passes to existing solvers.
"""

import os
import re
import sys
import json
from typing import Dict, Any, List, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.groq_client import GroqClient


class WordProblemAgent:
    """
    Models word problems into symbolic equations.
    
    Supports:
    - arithmetic_word_problem: Direct computation, no equations
    - linear_system: System of linear equations
    - nonlinear_system: Product/power relationships
    - number_digits: Digit-based problems
    - speed_distance_time: Motion problems with latent variables
    - work_time: Rate-based work problems
    - optimization: Calculus optimization
    - probability: Probability stories
    """
    
    def __init__(self):
        self.llm = GroqClient(model="llama-3.3-70b-versatile")
        self._init_templates()
    
    def _init_templates(self):
        """Initialize deterministic templates (priority over LLM)."""
        self.templates = {
            "age_problem": {
                "patterns": [r"years?\s+older", r"years?\s+younger", r"years?\s+ago", 
                           r"age\s+of", r"how\s+old", r"present\s+age"],
                "modeling_type": "linear_system"
            },
            "money_problem": {
                "patterns": [r"cost\s+of", r"price", r"₹|rs\.?|rupees?|inr", 
                           r"buys?\s+\d+", r"sells?\s+\d+", r"total\s+cost"],
                "modeling_type": "linear_system"
            },
            "speed_distance": {
                "patterns": [r"km/?h", r"m/?s", r"speed", r"travels?", r"distance",
                           r"takes?\s+\d+\s*h", r"faster", r"slower"],
                "modeling_type": "speed_distance_time"
            },
            "work_time": {
                "patterns": [r"can\s+complete", r"working\s+together", r"days?\s+to\s+finish",
                           r"hours?\s+to\s+complete", r"work\s+done", r"fill\s+the\s+tank"],
                "modeling_type": "work_time"
            },
            "number_digits": {
                "patterns": [r"sum\s+of\s+(the\s+)?digits", r"digits?\s+(are\s+)?reversed",
                           r"two.?digit\s+number", r"three.?digit\s+number"],
                "modeling_type": "number_digits"
            },
            "mixture": {
                "patterns": [r"mixture", r"solution", r"concentration", r"percent\s+acid",
                           r"alloy", r"mixed"],
                "modeling_type": "linear_system"
            },
            "arithmetic": {
                "patterns": [r"profit", r"loss", r"percentage", r"simple\s+interest",
                           r"compound\s+interest", r"discount"],
                "modeling_type": "arithmetic_word_problem"
            },
            "nonlinear": {
                "patterns": [r"product\s+(of|is)", r"square\s+of", r"ratio.*product",
                           r"sum.*product", r"product.*sum"],
                "modeling_type": "nonlinear_system"
            },
            "optimization": {
                "patterns": [r"maximum", r"minimum", r"maximize", r"minimize", 
                           r"largest", r"smallest", r"optimal"],
                "modeling_type": "optimization"
            },
            "probability": {
                "patterns": [r"probability", r"chance", r"likely", r"odds",
                           r"coin", r"dice", r"cards?", r"balls?\s+in\s+a\s+bag"],
                "modeling_type": "probability"
            }
        }
        
        # Currency normalization
        self.currency_patterns = [
            (r"₹\s*", "INR "),
            (r"Rs\.?\s*", "INR "),
            (r"rupees?\s*", "INR "),
            (r"INR\s*", "INR ")
        ]
        
        # Named entity patterns to strip
        self.name_patterns = [
            r"\b(Ram|Shyam|Ramesh|Suresh|Mohan|Sohan|Arun|Varun|A|B|C)\b",
        ]
    
    def _preprocess(self, text: str) -> str:
        """Preprocess text: normalize currency, strip noise."""
        # Normalize currency
        for pattern, replacement in self.currency_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Remove filler words
        fillers = ["suppose", "hypothetically", "assume that", "let's say", 
                   "consider that", "imagine"]
        for filler in fillers:
            text = re.sub(rf"\b{filler}\b", "", text, flags=re.IGNORECASE)
        
        # Normalize spaces
        text = " ".join(text.split())
        return text.strip()
    
    def _detect_template(self, text: str) -> Optional[str]:
        """Detect which template matches (deterministic, fast)."""
        text_lower = text.lower()
        
        for template_name, config in self.templates.items():
            for pattern in config["patterns"]:
                if re.search(pattern, text_lower):
                    return template_name
        
        return None
    
    def _classify_modeling_type(self, text: str, template: Optional[str]) -> str:
        """Classify the modeling type for routing."""
        if template:
            return self.templates[template]["modeling_type"]
        
        # Fallback: check for equation markers
        text_lower = text.lower()
        
        # Check for multiple equations
        if text_lower.count("=") >= 2 or " and " in text_lower:
            return "linear_system"
        
        # Check for single equation
        if "=" in text_lower:
            return "equation"
        
        # Default to linear system for word problems
        return "linear_system"
    
    def _extract_numbers(self, text: str) -> List[Dict]:
        """Extract numbers with context."""
        numbers = []
        # Match numbers with optional units
        pattern = r'(\d+(?:\.\d+)?)\s*(%|km/?h|m/?s|hours?|days?|years?|kg|g|liters?|l)?'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            numbers.append({
                "value": float(match.group(1)),
                "unit": match.group(2) or "",
                "position": match.start()
            })
        return numbers
    
    def _check_solvability(self, variables: Dict, equations: List[str], 
                           latent_vars: List[str]) -> Tuple[bool, str]:
        """Check if the problem is solvable."""
        num_unknowns = len(variables) + len(latent_vars)
        num_equations = len(equations)
        
        if num_unknowns > num_equations:
            return False, f"Underdetermined: {num_unknowns} unknowns but only {num_equations} equations"
        
        return True, ""
    
    def _model_age_problem(self, text: str) -> Dict:
        """Model age-related word problems."""
        prompt = f"""Model this age problem into equations.

Problem: {text}

Rules:
1. Define variables (e.g., a = current age of person A)
2. Write equations from the relationships
3. Use standard form (e.g., a = b + 5, not "a is 5 more than b")

Return ONLY valid JSON:
{{
  "variables": {{"a": "current age of A", "b": "current age of B"}},
  "equations": ["a = b + 5", "a + 3 = 2*(b + 3)"],
  "topic": "linear_algebra"
}}"""

        return self._llm_model(prompt, "linear_system")
    
    def _model_money_problem(self, text: str) -> Dict:
        """Model money/cost word problems."""
        prompt = f"""Model this money problem into equations.

Problem: {text}

Rules:
1. Define variables for unknown costs/prices
2. Write equations from purchase relationships
3. Use multiplication: "5 pens" → "5*p"

Return ONLY valid JSON:
{{
  "variables": {{"p": "cost of one pen", "n": "cost of one notebook"}},
  "equations": ["5*p + 3*n = 120", "3*p + 5*n = 140"],
  "topic": "linear_algebra"
}}"""

        return self._llm_model(prompt, "linear_system")
    
    def _model_speed_distance(self, text: str) -> Dict:
        """Model speed-distance-time problems with latent variables."""
        prompt = f"""Model this motion problem into equations.

Problem: {text}

Rules:
1. Use formula: distance = speed × time
2. If distance is not given, use D as latent variable
3. Express time differences as equations

Return ONLY valid JSON:
{{
  "variables": {{"v": "original speed (km/h)"}},
  "latent_variables": ["D"],
  "equations": ["D/v - D/(v+10) = 1"],
  "topic": "algebra"
}}"""

        return self._llm_model(prompt, "speed_distance_time", has_latent=True)
    
    def _model_work_time(self, text: str) -> Dict:
        """Model work-rate problems."""
        prompt = f"""Model this work problem into equations.

Problem: {text}

Rules:
1. Rate = 1/time (work per day/hour)
2. Combined rate = sum of individual rates
3. Total work = 1 (complete job)

Return ONLY valid JSON:
{{
  "variables": {{"t": "time to complete together (days)"}},
  "equations": ["1/10 + 1/15 = 1/t"],
  "topic": "algebra"
}}"""

        return self._llm_model(prompt, "work_time")
    
    def _model_digit_problem(self, text: str) -> Dict:
        """Model digit-based number problems."""
        prompt = f"""Model this digit problem into equations.

Problem: {text}

Rules:
1. Two-digit number = 10*tens + units
2. Three-digit number = 100*hundreds + 10*tens + units
3. Reversed two-digit = 10*units + tens

Return ONLY valid JSON:
{{
  "variables": {{"x": "tens digit", "y": "units digit"}},
  "equations": ["x + y = 9", "10*y + x = 10*x + y + 27"],
  "topic": "linear_algebra"
}}"""

        return self._llm_model(prompt, "number_digits")
    
    def _model_nonlinear(self, text: str) -> Dict:
        """Model nonlinear (product/power) problems."""
        prompt = f"""Model this problem with product/power relationships.

Problem: {text}

Rules:
1. Use x*y for products
2. Use x**2 for squares
3. Generate polynomial equations

Return ONLY valid JSON:
{{
  "variables": {{"x": "first number", "y": "second number"}},
  "equations": ["x + y = 20", "x*y = 96"],
  "topic": "algebra"
}}"""

        return self._llm_model(prompt, "nonlinear_system")
    
    def _model_optimization(self, text: str) -> Dict:
        """Model optimization problems."""
        # Check for missing constraints
        text_lower = text.lower()
        has_perimeter = re.search(r'perimeter\s*(=|is|of)?\s*\d+', text_lower)
        has_area = re.search(r'area\s*(=|is|of)?\s*\d+', text_lower)
        has_sum = re.search(r'sum\s*(=|is|of)?\s*\d+', text_lower)
        has_constraint = has_perimeter or has_area or has_sum or re.search(r'given\s+that', text_lower)
        
        if not has_constraint and ("perimeter" in text_lower or "given" in text_lower):
            return {
                "is_word_problem": True,
                "needs_clarification": True,
                "reason": "Missing constraint value (e.g., what is the perimeter value?)",
                "modeling_confidence": 0.0
            }
        
        prompt = f"""Model this optimization problem.

Problem: {text}

Rules:
1. Define the objective function to optimize
2. Express constraints
3. Identify what to maximize/minimize

Return ONLY valid JSON:
{{
  "variables": {{"x": "length", "y": "width"}},
  "objective": "x*y",
  "constraint": "2*x + 2*y = 20",
  "optimization_type": "maximize",
  "topic": "calculus"
}}"""

        return self._llm_model(prompt, "optimization")
    
    def _model_probability(self, text: str) -> Dict:
        """Model probability story problems."""
        # Check for required elements
        text_lower = text.lower()
        has_trials = re.search(r'\d+\s*(tosses?|flips?|draws?|picks?|times?)', text_lower)
        has_experiment = re.search(r'(coin|dice|die|cards?|balls?|bag)', text_lower)
        
        if not has_trials and not has_experiment:
            return {
                "is_word_problem": True,
                "needs_clarification": True,
                "reason": "Missing experiment definition (e.g., how many trials?)",
                "modeling_confidence": 0.0
            }
        
        prompt = f"""Model this probability problem.

Problem: {text}

Rules:
1. Define sample space
2. Define the event
3. Specify number of trials

Return ONLY valid JSON:
{{
  "sample_space": "coin flip outcomes",
  "event": "getting exactly 2 heads",
  "trials": 3,
  "probability_type": "binomial",
  "topic": "probability"
}}"""

        return self._llm_model(prompt, "probability")
    
    def _model_arithmetic(self, text: str) -> Dict:
        """Model arithmetic word problems (direct computation)."""
        prompt = f"""Solve this arithmetic problem step by step.

Problem: {text}

Rules:
1. Extract given values
2. Apply formulas (profit = SP - CP, etc.)
3. Compute final answer

Return ONLY valid JSON:
{{
  "is_arithmetic": true,
  "given": {{"cost_price": 40, "selling_price": 50, "quantity": 20}},
  "formula": "profit = (SP - CP) * quantity",
  "answer": 200,
  "working": "Profit per kg = 50 - 40 = 10. Total profit = 10 × 20 = 200"
}}"""

        try:
            result_text = self.llm.generate(prompt, temperature=0.0)
            result = self._parse_json(result_text)
            
            if result and result.get("is_arithmetic"):
                return {
                    "is_word_problem": True,
                    "modeling_type": "arithmetic_word_problem",
                    "is_arithmetic": True,
                    "given": result.get("given", {}),
                    "formula": result.get("formula", ""),
                    "answer": result.get("answer"),
                    "working": result.get("working", ""),
                    "needs_clarification": False,
                    "modeling_confidence": 0.9
                }
        except:
            pass
        
        return {"is_word_problem": True, "needs_clarification": True, 
                "reason": "Could not model arithmetic problem"}
    
    def _llm_model(self, prompt: str, modeling_type: str, has_latent: bool = False) -> Dict:
        """Use LLM to model the problem."""
        try:
            result_text = self.llm.generate(prompt, temperature=0.0)
            result = self._parse_json(result_text)
            
            if not result:
                return {"is_word_problem": True, "needs_clarification": True,
                        "reason": "Could not parse model output"}
            
            # Build output
            output = {
                "is_word_problem": True,
                "modeling_type": modeling_type,
                "variables": result.get("variables", {}),
                "equations": result.get("equations", []),
                "topic": result.get("topic", "algebra"),
                "needs_clarification": False,
                "modeling_confidence": 0.85
            }
            
            # Add latent variables if present
            if has_latent and "latent_variables" in result:
                output["latent_variables"] = result["latent_variables"]
            
            # Add optimization-specific fields
            if modeling_type == "optimization":
                output["objective"] = result.get("objective", "")
                output["constraint"] = result.get("constraint", "")
                output["optimization_type"] = result.get("optimization_type", "maximize")
            
            # Add probability-specific fields
            if modeling_type == "probability":
                output["sample_space"] = result.get("sample_space", "")
                output["event"] = result.get("event", "")
                output["trials"] = result.get("trials", 1)
                output["probability_type"] = result.get("probability_type", "")
            
            return output
            
        except Exception as e:
            return {"is_word_problem": True, "needs_clarification": True,
                    "reason": f"Modeling error: {str(e)}"}
    
    def _parse_json(self, text: str) -> Optional[Dict]:
        """Parse JSON from LLM response."""
        try:
            # Extract JSON block
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            return json.loads(text)
        except:
            # Try direct parse
            try:
                return json.loads(text)
            except:
                return None
    
    def _is_word_problem(self, text: str) -> bool:
        """Determine if input is a word problem vs direct equation."""
        text_lower = text.lower().strip()
        
        # Direct equation patterns (NOT word problems)
        direct_patterns = [
            r"^solve\s+[\d\w+\-*/()=\s^]+$",  # "Solve 2x + 3 = 5"
            r"^find\s+(the\s+)?(derivative|integral|limit)",  # "Find derivative of..."
            r"^\d+\s*[\w+\-*/()=]+\s*=",  # "2x + 3 = 5"
        ]
        
        for pattern in direct_patterns:
            if re.match(pattern, text_lower):
                return False
        
        # Word problem indicators
        word_indicators = [
            "how many", "how much", "find the cost", "find their ages",
            "years older", "years ago", "together", "if the speed",
            "probability of", "buys", "sells",
            "takes", "complete", "mixture", "profit", "loss",
            "sum of two", "product of two", "find the numbers", "two numbers",
            "sum is", "product is", "difference is"
        ]
        
        # Exclude calculus operations - they should go directly to calculus solver
        calculus_exclusions = [
            "derivative", "differentiate", "d/dx",
            "integral", "integrate", 
            "limit", "lim", "as x approaches",
            "maximize", "minimize", "maximum", "minimum"
        ]
        
        for exc in calculus_exclusions:
            if exc in text_lower:
                return False
        
        for indicator in word_indicators:
            if indicator in text_lower:
                return True
        
        # Check sentence structure (word problems have more words)
        words = text.split()
        if len(words) > 15:
            return True
        
        return False
    
    def model(self, text: str) -> Dict[str, Any]:
        """
        Main entry point: Model a word problem into equations.
        
        Args:
            text: User's natural language problem
            
        Returns:
            Dict with modeling results or needs_clarification flag
        """
        # Check if it's actually a word problem
        if not self._is_word_problem(text):
            return {
                "is_word_problem": False,
                "original_text": text
            }
        
        # Preprocess
        processed_text = self._preprocess(text)
        
        # Detect template (deterministic first)
        template = self._detect_template(processed_text)
        modeling_type = self._classify_modeling_type(processed_text, template)
        
        # Build modeling trace
        trace = [f"Detected template: {template or 'none'}",
                 f"Modeling type: {modeling_type}"]
        
        # Route to specific modeler
        if modeling_type == "arithmetic_word_problem":
            result = self._model_arithmetic(processed_text)
        elif template == "age_problem":
            result = self._model_age_problem(processed_text)
        elif template == "money_problem":
            result = self._model_money_problem(processed_text)
        elif template == "speed_distance" or modeling_type == "speed_distance_time":
            result = self._model_speed_distance(processed_text)
        elif template == "work_time" or modeling_type == "work_time":
            result = self._model_work_time(processed_text)
        elif template == "number_digits" or modeling_type == "number_digits":
            result = self._model_digit_problem(processed_text)
        elif template == "nonlinear" or modeling_type == "nonlinear_system":
            result = self._model_nonlinear(processed_text)
        elif modeling_type == "optimization":
            result = self._model_optimization(processed_text)
        elif modeling_type == "probability":
            result = self._model_probability(processed_text)
        else:
            # Generic linear system modeling
            result = self._model_generic(processed_text)
        
        # Add trace
        if "equations" in result:
            trace.append(f"Generated {len(result.get('equations', []))} equations")
        result["modeling_trace"] = trace
        result["original_text"] = text
        
        return result
    
    def _model_generic(self, text: str) -> Dict:
        """Generic word problem modeling for unmatched patterns."""
        prompt = f"""Model this word problem into mathematical equations.

Problem: {text}

Rules:
1. Define clear variables with descriptions
2. Write equations from relationships in the problem
3. Use standard mathematical notation
4. Ensure equations are solvable

Return ONLY valid JSON:
{{
  "variables": {{"x": "description", "y": "description"}},
  "equations": ["equation1", "equation2"],
  "topic": "algebra|linear_algebra|calculus|probability"
}}"""

        return self._llm_model(prompt, "linear_system")


def main():
    """Test word problem agent."""
    agent = WordProblemAgent()
    
    test_cases = [
        # Age problem
        "A father is 30 years older than his son. In 5 years, the father's age will be twice his son's age. Find their current ages.",
        
        # Money problem
        "5 pens and 3 notebooks cost ₹120. 3 pens and 5 notebooks cost ₹140. Find the cost of one pen and one notebook.",
        
        # Speed-distance
        "A train travels 120 km at a certain speed. If the speed was 10 km/h more, it would take 1 hour less. Find the original speed.",
        
        # Digit problem
        "The sum of the digits of a two-digit number is 9. If the digits are reversed, the new number is 27 more than the original. Find the number.",
        
        # Nonlinear
        "The sum of two numbers is 20 and their product is 96. Find the numbers.",
        
        # Arithmetic (direct)
        "A shopkeeper buys apples at ₹40 per kg and sells at ₹50 per kg. If he sells 20 kg, find his profit.",
        
        # Underdetermined (should reject)
        "Find two numbers whose sum is 10.",
        
        # Not a word problem
        "Solve 2x + 3 = 5"
    ]
    
    print("=" * 70)
    print("WORD PROBLEM MODELING AGENT TEST")
    print("=" * 70)
    
    for problem in test_cases:
        print(f"\nProblem: {problem[:60]}...")
        print("-" * 70)
        
        result = agent.model(problem)
        
        if not result.get("is_word_problem"):
            print("  → Not a word problem, pass to direct parser")
        elif result.get("needs_clarification"):
            print(f"  → Needs clarification: {result.get('reason')}")
        elif result.get("is_arithmetic"):
            print(f"  → Arithmetic: {result.get('answer')}")
            print(f"    Working: {result.get('working')}")
        else:
            print(f"  → Type: {result.get('modeling_type')}")
            print(f"  → Variables: {result.get('variables')}")
            print(f"  → Equations: {result.get('equations')}")
            print(f"  → Confidence: {result.get('modeling_confidence')}")


if __name__ == "__main__":
    main()
