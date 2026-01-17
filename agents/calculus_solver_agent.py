"""
Calculus Solver Agent
Solves calculus problems (derivatives, integrals, limits, optimization) using SymPy
with Strict RAG enforcement
"""

import json
import os
import re
from typing import Dict, Any
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import (
    symbols, diff, integrate, limit, solve, oo,
    sin, cos, tan, exp, log, sqrt, pi, E,
    simplify, sympify
)
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation
)

from llm.groq_client import GroqClient
from rag.retriever import KnowledgeRetriever
from agents.calculus_parser_agent import CalculusParserAgent
from utils.math_formatter import format_answer, format_math_expression


# Safe parsing transformations
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation
)


class CalculusSolverAgent:
    """
    Solves calculus problems using SymPy.
    Supports: derivatives, integrals, limits, optimization
    Enforces Strict RAG mode.
    """
    
    def __init__(self, similarity_threshold: float = 0.35):
        self.llm = GroqClient(model="llama-3.3-70b-versatile")
        self.retriever = KnowledgeRetriever()
        self.parser = CalculusParserAgent()
        self.similarity_threshold = similarity_threshold
        
        # Local dict for parsing with common math functions
        self.local_dict = {
            'sin': sin, 'cos': cos, 'tan': tan,
            'exp': exp, 'log': log, 'ln': log,
            'sqrt': sqrt, 'pi': pi, 'e': E
        }
    
    def _parse_expression(self, expr_str: str, variable: str):
        """
        Parse expression string to SymPy expression.
        """
        var_sym = symbols(variable)
        self.local_dict[variable] = var_sym
        
        # Clean up expression
        expr_str = expr_str.replace('^', '**')
        
        try:
            return parse_expr(expr_str, transformations=TRANSFORMATIONS, local_dict=self.local_dict), var_sym
        except Exception as e:
            raise ValueError(f"Could not parse expression '{expr_str}': {e}")
    
    def _compute_derivative(self, expression: str, variable: str) -> Dict[str, Any]:
        """
        Compute derivative using SymPy.
        """
        try:
            expr, var = self._parse_expression(expression, variable)
            result = diff(expr, var)
            result_simplified = simplify(result)
            
            return {
                "success": True,
                "result": str(result_simplified),
                "original": str(expr),
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }
    
    def _compute_integral(self, expression: str, variable: str) -> Dict[str, Any]:
        """
        Compute indefinite integral using SymPy.
        """
        try:
            expr, var = self._parse_expression(expression, variable)
            result = integrate(expr, var)
            
            return {
                "success": True,
                "result": f"{str(result)} + C",  # Add constant of integration
                "original": str(expr),
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }
    
    def _compute_limit(self, expression: str, variable: str, point: str) -> Dict[str, Any]:
        """
        Compute limit using SymPy.
        """
        try:
            expr, var = self._parse_expression(expression, variable)
            
            # Parse limit point
            if point == 'oo' or point == 'infinity':
                point_val = oo
            elif point == '-oo' or point == '-infinity':
                point_val = -oo
            else:
                point_val = sympify(point)
            
            result = limit(expr, var, point_val)
            
            return {
                "success": True,
                "result": str(result),
                "original": str(expr),
                "limit_point": str(point_val),
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }
    
    def _compute_optimization(self, expression: str, variable: str, opt_type: str) -> Dict[str, Any]:
        """
        Find maximum/minimum using calculus (set derivative = 0).
        """
        try:
            expr, var = self._parse_expression(expression, variable)
            
            # Find critical points (where derivative = 0)
            derivative = diff(expr, var)
            critical_points = solve(derivative, var)
            
            if not critical_points:
                return {
                    "success": False,
                    "result": None,
                    "error": "No critical points found"
                }
            
            # Evaluate function at critical points
            results = []
            for cp in critical_points:
                try:
                    value = float(expr.subs(var, cp))
                    cp_float = float(cp)
                    results.append({
                        "point": cp_float,
                        "value": value
                    })
                except:
                    results.append({
                        "point": str(cp),
                        "value": str(expr.subs(var, cp))
                    })
            
            # Find max or min
            if opt_type == "maximize":
                best = max(results, key=lambda x: x["value"] if isinstance(x["value"], (int, float)) else 0)
                result_str = f"Maximum value is {best['value']} at {variable} = {best['point']}"
            elif opt_type == "minimize":
                best = min(results, key=lambda x: x["value"] if isinstance(x["value"], (int, float)) else float('inf'))
                result_str = f"Minimum value is {best['value']} at {variable} = {best['point']}"
            else:
                result_str = f"Critical points: {results}"
                best = results[0] if results else {}
            
            return {
                "success": True,
                "result": result_str,
                "critical_points": results,
                "optimal": best,
                "derivative": str(derivative),
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }
    
    def solve(self, parsed_problem: Dict, route: Dict) -> Dict[str, Any]:
        """
        Solve calculus problem with Strict RAG enforcement.
        """
        problem_text = parsed_problem.get("problem_text", "")
        intent = route.get("intent", "derivative")
        
        # Step 1: Parse the calculus problem
        calc_parsed = self.parser.parse(problem_text)
        
        if calc_parsed.get("needs_clarification"):
            return {
                "answer": "Could not parse calculus problem",
                "working": calc_parsed.get("error", "Unknown error"),
                "sources": [],
                "citations": [],
                "tool_calls": [],
                "has_context": False,
                "has_sufficient_context": False
            }
        
        calc_type = calc_parsed.get("type", intent)
        expression = calc_parsed.get("expression", "")
        variable = calc_parsed.get("variable", "x")
        
        # Step 2: Retrieve relevant knowledge
        search_terms = {
            "derivative": "differentiation rules derivative",
            "integral": "integration rules integral",
            "limit": "limit rules evaluation",
            "optimization": "optimization maximum minimum derivative"
        }
        search_query = f"{search_terms.get(calc_type, calc_type)} {problem_text}"
        
        retrieval_result = self.retriever.retrieve_with_threshold(
            search_query,
            k=5,
            threshold=self.similarity_threshold
        )
        
        has_sufficient_context = retrieval_result["has_sufficient_context"]
        retrieved = retrieval_result["results"]
        best_score = retrieval_result["best_score"]
        
        if not has_sufficient_context:
            return {
                "answer": "INSUFFICIENT CONTEXT",
                "working": f"Knowledge base lacks information on {calc_type}. Best similarity: {best_score:.3f}",
                "sources": [],
                "citations": [],
                "tool_calls": [],
                "has_context": False,
                "has_sufficient_context": False,
                "insufficient_context_reason": retrieval_result.get("reason", "No relevant knowledge found")
            }
        
        # Step 3: Compute the result using SymPy
        if calc_type == "derivative":
            compute_result = self._compute_derivative(expression, variable)
            tool_name = "differentiate"
        elif calc_type == "integral":
            compute_result = self._compute_integral(expression, variable)
            tool_name = "integrate"
        elif calc_type == "limit":
            limit_point = calc_parsed.get("limit_point", "0")
            compute_result = self._compute_limit(expression, variable, limit_point)
            tool_name = "limit"
        elif calc_type == "optimization":
            opt_type = calc_parsed.get("optimization_type", "extremum")
            compute_result = self._compute_optimization(expression, variable, opt_type)
            tool_name = "optimize"
        else:
            return {
                "answer": f"Unknown calculus type: {calc_type}",
                "working": "",
                "sources": [],
                "citations": [],
                "tool_calls": [],
                "has_context": True,
                "has_sufficient_context": True
            }
        
        if not compute_result["success"]:
            # SymPy can't compute directly - try using LLM for word problems
            context_parts = [f"[Source: {r['source']}]\n{r['content']}" for r in retrieved]
            context = "\n\n".join(context_parts)
            
            fallback_prompt = f"""Solve this {calc_type} problem step-by-step.
Use ONLY the methods from the retrieved knowledge.

Problem: {problem_text}

Retrieved Knowledge:
{context}

Solve the problem showing your working. At the very end, you MUST write exactly:
"The final answer is: [your answer here]"

For example: "The final answer is: x = 25 meters, y = 25 meters" or "The final answer is: 625 square meters"
"""

            try:
                working = self.llm.generate(fallback_prompt, temperature=0.1)
                
                # Try to extract answer from the LLM response
                # Look for "The final answer is:" pattern first
                final_answer_match = re.search(r'The final answer is[:\s]+([^\n]+)', working, re.IGNORECASE)
                if final_answer_match:
                    extracted_answer = final_answer_match.group(1).strip().rstrip('.')
                else:
                    # Fallback: look for any "answer is" pattern
                    answer_match = re.search(r'(?:answer|result)\s+is[:\s]+([^\n]+)', working, re.IGNORECASE)
                    if answer_match:
                        extracted_answer = answer_match.group(1).strip().rstrip('.')
                    else:
                        extracted_answer = "See explanation"
                
                return {
                    "answer": format_answer(extracted_answer),
                    "working": format_math_expression(working),
                    "sources": [r["source"] for r in retrieved],
                    "citations": [],
                    "tool_calls": [],
                    "has_context": True,
                    "has_sufficient_context": True,
                    "solved_by": "LLM fallback"
                }
            except:
                return {
                    "answer": "Computation failed",
                    "working": f"Error: {compute_result['error']}",
                    "sources": [r["source"] for r in retrieved],
                    "citations": [],
                    "tool_calls": [],
                    "has_context": True,
                    "has_sufficient_context": True,
                    "error": compute_result["error"]
                }
        
        # Step 4: Generate explanation using LLM with RAG
        context_parts = [f"[Source: {r['source']}]\n{r['content']}" for r in retrieved]
        context = "\n\n".join(context_parts)
        
        prompt = f"""Explain this {calc_type} calculation step-by-step.
Use ONLY the rules from the retrieved knowledge.
Cite sources using [Source: filename] format.

Problem: {problem_text}
Expression: {expression}
Variable: {variable}
Computed Result: {compute_result['result']}

Retrieved Knowledge:
{context}

Provide a clear step-by-step explanation with citations."""

        try:
            working = self.llm.generate(prompt, temperature=0.1)
        except:
            working = f"Computed {calc_type} of {expression}.\nResult: {compute_result['result']}"
        
        # Extract citations
        citations = []
        citation_pattern = r'\[Source:\s*([^\]]+)\]'
        matches = re.findall(citation_pattern, working)
        for source in set(matches):
            citations.append({"source": source, "verified": source in [r["source"] for r in retrieved]})
        
        return {
            "answer": format_answer(compute_result["result"]),
            "working": format_math_expression(working),
            "sources": [r["source"] for r in retrieved],
            "citations": citations,
            "tool_calls": [{
                "function": tool_name,
                "args": {"expression": expression, "variable": variable},
                "result": compute_result["result"]
            }],
            "has_context": True,
            "has_sufficient_context": True,
            "similarity_score": best_score,
            "calculus_type": calc_type
        }


def main():
    """Test calculus solver."""
    solver = CalculusSolverAgent()
    
    test_problems = [
        {"problem_text": "Find the derivative of x^2 + 3*x", "topic": "calculus"},
        {"problem_text": "Integrate x^2 dx", "topic": "calculus"},
        {"problem_text": "Find limit of sin(x)/x as x approaches 0", "topic": "calculus"},
        {"problem_text": "Maximize x*(10-x)", "topic": "calculus"},
    ]
    
    for problem in test_problems:
        print(f"\nProblem: {problem['problem_text']}")
        print("=" * 60)
        
        route = {"route": "calculus", "intent": "derivative", "confidence": 0.95}
        result = solver.solve(problem, route)
        
        print(f"Answer: {result['answer']}")
        print(f"Has Context: {result.get('has_sufficient_context')}")


if __name__ == "__main__":
    main()
