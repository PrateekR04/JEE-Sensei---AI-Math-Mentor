"""
Explainer Agent
Generates student-friendly step-by-step explanations using Groq
"""

import json
import os
from typing import Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.groq_client import GroqClient


class ExplainerAgent:
    """
    Generates clear, pedagogical explanations of solutions.
    """
    
    def __init__(self):
        self.llm = GroqClient(model="llama-3.3-70b-versatile")
    
    def explain(self, solution: Dict, verification: Dict, parsed_problem: Dict) -> Dict[str, Any]:
        """
        Generate student-friendly explanation.
        
        Args:
            solution: Solution from SolverAgent
            verification: Verification from VerifierAgent
            parsed_problem: Original parsed problem
            
        Returns:
            Dict with explanation, key_concepts, difficulty
        """
        problem_text = parsed_problem.get("problem_text", "")
        answer = solution.get("answer", "")
        working = solution.get("working", "")
        is_correct = verification.get("is_correct", False)
        
        prompt = f"""You are a friendly math tutor explaining a solution to a student.

Problem: {problem_text}
Answer: {answer}
Working: {working}
Verified: {is_correct}

Create a clear, step-by-step explanation that:
1. Breaks down the solution into simple steps
2. Explains the reasoning behind each step
3. Uses simple language
4. Highlights key concepts

Return ONLY valid JSON:
{{
  "explanation": "Step 1: ...\nStep 2: ...\nStep 3: ...",
  "key_concepts": ["Linear equations", "Inverse operations"],
  "difficulty": "easy|medium|hard"
}}"""

        try:
            result_text = self.llm.generate(prompt, temperature=0.0)
            
            # Extract JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            
            # Ensure required fields
            result.setdefault("explanation", working or "No explanation available")
            result.setdefault("key_concepts", [])
            result.setdefault("difficulty", "medium")
            
            return result
            
        except Exception as e:
            # Fallback explanation
            return {
                "explanation": working or "Solution completed",
                "key_concepts": [],
                "difficulty": "medium",
                "error": str(e)
            }


def main():
    """Test explainer agent."""
    explainer = ExplainerAgent()
    
    solution = {
        "answer": "x = 1",
        "working": "2x + 3 = 5\n2x = 2\nx = 1",
        "sources": ["algebra.txt"]
    }
    
    verification = {
        "is_correct": True,
        "confidence": 0.93
    }
    
    parsed = {
        "problem_text": "2x + 3 = 5. Find x."
    }
    
    print("Generating explanation...")
    print("=" * 60)
    result = explainer.explain(solution, verification, parsed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
