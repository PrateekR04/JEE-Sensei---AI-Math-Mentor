"""
Verifier Agent
Verifies solution correctness using calculator tool
UPGRADED: Supports vector solutions, symbolic expressions, and confidence caps
"""

import json
import os
import re
from typing import Dict, List, Any
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.calculator import Calculator


class VerifierAgent:
    """
    Verifies mathematical solutions for correctness.
    
    UPGRADED: Now supports:
    - Vector solutions (for systems of equations)
    - Symbolic expressions (for calculus)
    - Confidence caps per problem type
    """
    
    # Confidence caps per problem type
    CONFIDENCE_CAPS = {
        "equation": 0.95,
        "system_of_equations": 0.90,
        "derivative": 0.85,
        "integral": 0.85,
        "limit": 0.85,
        "optimization": 0.85,
        "probability": 0.85,
        "default": 0.90
    }
    
    def __init__(self):
        self.calculator = Calculator()
    
    def _get_confidence_cap(self, solution: Dict, parsed_problem: Dict) -> float:
        """
        Get confidence cap based on problem type.
        """
        # Check for calculus types
        calculus_type = solution.get("calculus_type")
        if calculus_type:
            return self.CONFIDENCE_CAPS.get(calculus_type, 0.85)
        
        # Check for system solutions
        if solution.get("system_solutions"):
            return self.CONFIDENCE_CAPS["system_of_equations"]
        
        # Check for probability
        if solution.get("probability_value") is not None:
            return self.CONFIDENCE_CAPS["probability"]
        
        # Default to equation
        return self.CONFIDENCE_CAPS.get("equation", 0.95)
    
    def _verify_system_solution(self, solution: Dict) -> tuple:
        """
        Verify system of equations solution.
        Returns (is_correct, verification_steps)
        """
        steps = []
        is_correct = True
        
        system_solutions = solution.get("system_solutions", [])
        if not system_solutions:
            return False, ["⚠ No system solutions to verify"]
        
        # Check tool calls for verification
        tool_calls = solution.get("tool_calls", [])
        for tc in tool_calls:
            if tc.get("function") == "solve_system" and tc.get("result"):
                steps.append(f"✓ System solver returned: {tc['result']}")
        
        if system_solutions:
            steps.append(f"✓ Found {len(system_solutions)} solution(s)")
            for i, sol in enumerate(system_solutions):
                sol_str = ", ".join([f"{k}={v}" for k, v in sol.items()])
                steps.append(f"  Solution {i+1}: {sol_str}")
        
        return is_correct, steps
    
    def _verify_calculus_solution(self, solution: Dict) -> tuple:
        """
        Verify calculus solution (derivative, integral, limit, optimization).
        Returns (is_correct, verification_steps)
        """
        steps = []
        is_correct = True
        
        calculus_type = solution.get("calculus_type", "unknown")
        answer = solution.get("answer", "")
        
        # Check tool calls
        tool_calls = solution.get("tool_calls", [])
        for tc in tool_calls:
            func = tc.get("function", "")
            if func in ["differentiate", "integrate", "limit", "optimize"]:
                result = tc.get("result")
                if result:
                    steps.append(f"✓ {func.capitalize()} computed: {result}")
                    is_correct = True
                else:
                    steps.append(f"✗ {func.capitalize()} failed")
                    is_correct = False
        
        if not tool_calls:
            # No tool calls, check if answer looks valid
            if answer and answer != "Computation failed":
                steps.append(f"✓ {calculus_type.capitalize()} result: {answer}")
            else:
                steps.append(f"⚠ Could not verify {calculus_type} result")
                is_correct = False
        
        return is_correct, steps
    
    def _verify_probability_solution(self, solution: Dict) -> tuple:
        """
        Verify probability solution.
        Returns (is_correct, verification_steps)
        """
        steps = []
        is_correct = True
        
        prob_value = solution.get("probability_value")
        prob_fraction = solution.get("probability_fraction")
        
        if prob_value is not None:
            # Check probability is in valid range [0, 1]
            if 0 <= prob_value <= 1:
                steps.append(f"✓ Probability {prob_value:.4f} is in valid range [0,1]")
                if prob_fraction:
                    steps.append(f"✓ Expressed as fraction: {prob_fraction}")
            else:
                steps.append(f"✗ Probability {prob_value} is outside valid range [0,1]")
                is_correct = False
        
        # Check tool calls
        tool_calls = solution.get("tool_calls", [])
        for tc in tool_calls:
            if tc.get("function") == "probability":
                result = tc.get("result", {})
                if result.get("success"):
                    steps.append(f"✓ Probability computed using {result.get('method', 'formula')}")
        
        return is_correct, steps
    
    def verify(self, solution: Dict, parsed_problem: Dict) -> Dict[str, Any]:
        """
        Verify solution correctness with STRICT RAG enforcement.
        
        UPGRADED: Now handles:
        - Single equations (existing)
        - System of equations (vector solutions)
        - Calculus (symbolic expressions)
        - Probability
        
        Args:
            solution: Solution from SolverAgent or domain-specific solver
            parsed_problem: Original parsed problem
            
        Returns:
            Dict with is_correct, confidence, issues, verification_steps
        """
        answer = solution.get("answer", "")
        constraints = parsed_problem.get("constraints", [])
        variables = parsed_problem.get("variables", [])
        
        # Check for strict RAG compliance
        has_sufficient_context = solution.get("has_sufficient_context", False)
        citations = solution.get("citations", [])
        sources = solution.get("sources", [])
        
        verification_steps = []
        issues = []
        is_correct = True
        confidence = 0.0
        
        # Get confidence cap for this problem type
        confidence_cap = self._get_confidence_cap(solution, parsed_problem)
        
        try:
            # STRICT RAG ENFORCEMENT CHECKS
            
            # Check 1: Context sufficiency
            if not has_sufficient_context:
                verification_steps.append("✗ INSUFFICIENT CONTEXT: Knowledge base lacks required information")
                issues.append("Insufficient context in knowledge base")
                confidence = 0.0
                is_correct = False
                return {
                    "is_correct": False,
                    "confidence": 0.0,
                    "issues": issues,
                    "verification_steps": verification_steps,
                    "strict_rag_compliant": False
                }
            
            # Check 2: Citation validation
            citation_penalty = 0.0
            if not citations:
                verification_steps.append("✗ MISSING CITATIONS: Solution does not cite source documents")
                issues.append("No citations found - violates strict RAG mode")
                citation_penalty = 0.3
                is_correct = False
            else:
                # Verify citations are valid
                unverified_citations = [c for c in citations if not c.get("verified", True)]
                if unverified_citations:
                    verification_steps.append(
                        f"⚠ UNVERIFIED CITATIONS: {len(unverified_citations)} citations not in retrieved sources"
                    )
                    issues.append("Some citations reference sources not in retrieval results")
                    citation_penalty = 0.15
                else:
                    verification_steps.append(f"✓ All {len(citations)} citations verified against retrieved sources")
                    confidence += 0.2
            
            # Check 3: Verify based on problem type
            
            # System of equations
            if solution.get("system_solutions"):
                sys_correct, sys_steps = self._verify_system_solution(solution)
                verification_steps.extend(sys_steps)
                if sys_correct:
                    confidence += 0.5
                else:
                    is_correct = False
            
            # Calculus problems
            elif solution.get("calculus_type"):
                calc_correct, calc_steps = self._verify_calculus_solution(solution)
                verification_steps.extend(calc_steps)
                if calc_correct:
                    confidence += 0.5
                else:
                    is_correct = False
            
            # Probability problems
            elif solution.get("probability_value") is not None:
                prob_correct, prob_steps = self._verify_probability_solution(solution)
                verification_steps.extend(prob_steps)
                if prob_correct:
                    confidence += 0.5
                else:
                    is_correct = False
            
            # Standard equation (existing logic)
            else:
                # Extract numeric answer if possible
                answer_match = re.search(r'([a-z])\s*=\s*([-+]?\d*\.?\d+)', answer.lower())
                
                if answer_match and constraints:
                    variable = answer_match.group(1)
                    value = float(answer_match.group(2))
                    
                    # Verify against each constraint
                    for constraint in constraints:
                        try:
                            is_valid = self.calculator.verify_solution(
                                constraint, variable, value
                            )
                            
                            if is_valid:
                                verification_steps.append(
                                    f"✓ Verified: {variable}={value} satisfies {constraint}"
                                )
                                confidence += 0.4
                            else:
                                verification_steps.append(
                                    f"✗ Failed: {variable}={value} does not satisfy {constraint}"
                                )
                                issues.append(f"Answer does not satisfy constraint: {constraint}")
                                is_correct = False
                                
                        except Exception as e:
                            verification_steps.append(f"⚠ Could not verify {constraint}: {e}")
                            confidence += 0.1
                else:
                    # No numeric answer to verify
                    verification_steps.append("⚠ No numeric answer found to verify")
                    confidence = 0.5
            
            # Add confidence for sources and tool usage
            if sources:
                confidence += 0.2
                verification_steps.append(f"✓ Solution used {len(sources)} knowledge source(s)")
            
            if solution.get("tool_calls"):
                confidence += 0.1
                verification_steps.append("✓ Solution used calculator/solver tools")
            
            # Apply citation penalty
            confidence = max(0.0, confidence - citation_penalty)
            
            # Apply confidence cap based on problem type
            confidence = min(confidence, confidence_cap)
            confidence = round(confidence, 2)
            
            return {
                "is_correct": is_correct,
                "confidence": confidence,
                "issues": issues,
                "verification_steps": verification_steps,
                "strict_rag_compliant": len(citations) > 0 and has_sufficient_context,
                "confidence_cap_applied": confidence_cap
            }
            
        except Exception as e:
            return {
                "is_correct": False,
                "confidence": 0.3,
                "issues": [f"Verification error: {str(e)}"],
                "verification_steps": ["⚠ Verification failed"]
            }


def main():
    """Test verifier agent with various problem types."""
    verifier = VerifierAgent()
    
    # Test 1: Standard equation
    print("Test 1: Standard Equation")
    solution1 = {
        "answer": "x = 1",
        "working": "2x + 3 = 5\n2x = 2\nx = 1",
        "sources": ["algebra_linear.txt"],
        "citations": [{"source": "algebra_linear.txt", "verified": True}],
        "tool_calls": [{"function": "solve_equation", "result": [1.0]}],
        "has_sufficient_context": True
    }
    parsed1 = {
        "problem_text": "2x + 3 = 5",
        "variables": ["x"],
        "constraints": ["2*x + 3 = 5"]
    }
    result1 = verifier.verify(solution1, parsed1)
    print(json.dumps(result1, indent=2))
    
    # Test 2: System of equations
    print("\nTest 2: System of Equations")
    solution2 = {
        "answer": "x = 2, y = 1",
        "system_solutions": [{"x": 2.0, "y": 1.0}],
        "sources": ["system_of_equations.txt"],
        "citations": [{"source": "system_of_equations.txt", "verified": True}],
        "tool_calls": [{"function": "solve_system", "result": [{"x": 2.0, "y": 1.0}]}],
        "has_sufficient_context": True
    }
    parsed2 = {"problem_text": "2x + y = 5 and x - y = 1"}
    result2 = verifier.verify(solution2, parsed2)
    print(json.dumps(result2, indent=2))
    
    # Test 3: Calculus
    print("\nTest 3: Derivative")
    solution3 = {
        "answer": "2*x + 3",
        "calculus_type": "derivative",
        "sources": ["differentiation_rules.txt"],
        "citations": [{"source": "differentiation_rules.txt", "verified": True}],
        "tool_calls": [{"function": "differentiate", "result": "2*x + 3"}],
        "has_sufficient_context": True
    }
    parsed3 = {"problem_text": "Derivative of x^2 + 3x"}
    result3 = verifier.verify(solution3, parsed3)
    print(json.dumps(result3, indent=2))


if __name__ == "__main__":
    main()

