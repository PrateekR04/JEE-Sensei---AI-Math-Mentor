"""
System Solver Agent
Solves systems of equations using SymPy with Strict RAG enforcement
"""

import json
import os
import re
from typing import Dict, List, Any
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import symbols, solve, Eq
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)

from llm.groq_client import GroqClient
from rag.retriever import KnowledgeRetriever
from agents.system_parser_agent import SystemParserAgent
from utils.math_formatter import format_answer, format_math_expression


# Safe parsing transformations
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor
)


class SystemSolverAgent:
    """
    Solves systems of equations using SymPy.
    Enforces Strict RAG mode for all solutions.
    """
    
    def __init__(self, similarity_threshold: float = 0.35):
        self.llm = GroqClient(model="llama-3.3-70b-versatile")
        self.retriever = KnowledgeRetriever()
        self.parser = SystemParserAgent()
        self.similarity_threshold = similarity_threshold
    
    def _parse_equation_to_sympy(self, equation: str, local_dict: dict):
        """
        Parse equation string to SymPy Eq object.
        """
        if '=' not in equation:
            raise ValueError(f"Not a valid equation: {equation}")
        
        left, right = equation.split('=')
        left_expr = parse_expr(left.strip(), transformations=TRANSFORMATIONS, local_dict=local_dict)
        right_expr = parse_expr(right.strip(), transformations=TRANSFORMATIONS, local_dict=local_dict)
        
        return Eq(left_expr, right_expr)
    
    def _solve_system(self, equations: List[str], variables: List[str]) -> Dict[str, Any]:
        """
        Solve system of equations using SymPy.
        """
        # Create symbol objects
        sym_dict = {v: symbols(v) for v in variables}
        sym_list = [sym_dict[v] for v in variables]
        
        try:
            # Parse equations
            eq_list = []
            for eq_str in equations:
                eq_obj = self._parse_equation_to_sympy(eq_str, sym_dict)
                eq_list.append(eq_obj)
            
            # Solve
            solutions = solve(eq_list, sym_list, dict=True)
            
            if not solutions:
                return {
                    "success": False,
                    "error": "No solution found",
                    "solutions": []
                }
            
            # Format solutions
            formatted = []
            for sol in solutions:
                sol_dict = {}
                for var_sym, value in sol.items():
                    var_name = str(var_sym)
                    try:
                        sol_dict[var_name] = float(value)
                    except:
                        sol_dict[var_name] = str(value)
                formatted.append(sol_dict)
            
            return {
                "success": True,
                "solutions": formatted,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "solutions": []
            }
    
    def solve(self, parsed_problem: Dict, route: Dict) -> Dict[str, Any]:
        """
        Solve system of equations with Strict RAG enforcement.
        
        Args:
            parsed_problem: Parsed problem from parser
            route: Route from router (contains intent)
            
        Returns:
            Solution dict with answer, working, sources, citations
        """
        problem_text = parsed_problem.get("problem_text", "")
        
        # Check for pre-modeled word problem equations (from WordProblemAgent)
        word_problem_model = parsed_problem.get("word_problem_model")
        
        if word_problem_model and word_problem_model.get("equations"):
            # Use pre-modeled equations from word problem agent
            equations = word_problem_model.get("equations", [])
            variables = list(word_problem_model.get("variables", {}).keys())
            system_parsed = {
                "equations": equations,
                "normalized_equations": equations,
                "variables": variables,
                "needs_clarification": False
            }
        else:
            # Step 1: Parse the system of equations from problem text
            system_parsed = self.parser.parse(problem_text)
            
            if system_parsed.get("needs_clarification"):
                return {
                    "answer": "Could not parse system of equations",
                    "working": system_parsed.get("error", "Unknown parsing error"),
                    "sources": [],
                    "citations": [],
                    "tool_calls": [],
                    "has_context": False,
                    "has_sufficient_context": False
                }
            
            equations = system_parsed.get("normalized_equations", [])
            variables = system_parsed.get("variables", [])
        
        # Step 2: Retrieve knowledge for RAG
        retrieval_result = self.retriever.retrieve_with_threshold(
            f"system of equations solving methods {problem_text}",
            k=5,
            threshold=self.similarity_threshold
        )
        
        has_sufficient_context = retrieval_result["has_sufficient_context"]
        retrieved = retrieval_result["results"]
        best_score = retrieval_result["best_score"]
        
        if not has_sufficient_context:
            return {
                "answer": "INSUFFICIENT CONTEXT",
                "working": f"Knowledge base lacks information on solving systems of equations. Best similarity: {best_score:.3f}",
                "sources": [],
                "citations": [],
                "tool_calls": [],
                "has_context": False,
                "has_sufficient_context": False,
                "insufficient_context_reason": retrieval_result.get("reason", "No relevant knowledge found")
            }
        
        # Step 3: Solve the system using SymPy
        solve_result = self._solve_system(equations, variables)
        
        if not solve_result["success"]:
            return {
                "answer": "Could not solve system",
                "working": f"SymPy error: {solve_result['error']}",
                "sources": [r["source"] for r in retrieved],
                "citations": [],
                "tool_calls": [],
                "has_context": True,
                "has_sufficient_context": True,
                "error": solve_result["error"]
            }
        
        solutions = solve_result["solutions"]
        
        # Step 4: Format the answer
        if len(solutions) == 1:
            sol = solutions[0]
            answer_parts = [f"{var} = {val}" for var, val in sorted(sol.items())]
            answer = ", ".join(answer_parts)
        else:
            answer = f"{len(solutions)} solutions found"
        
        # Step 5: Generate working/explanation using LLM with RAG context
        context_parts = [f"[Source: {r['source']}]\n{r['content']}" for r in retrieved]
        context = "\n\n".join(context_parts)
        
        prompt = f"""Explain how to solve this system of equations step-by-step.
Use ONLY the methods described in the retrieved knowledge.
Cite sources using [Source: filename] format.

System of Equations:
{chr(10).join(system_parsed.get('equations', []))}

Computed Solution (verified by SymPy):
{json.dumps(solutions, indent=2)}

Retrieved Knowledge:
{context}

Provide a clear step-by-step explanation with citations.
At the END, you MUST write on its own line: "**FINAL ANSWER:** x = [value], y = [value]" (or similar for your variables)"""

        try:
            working = self.llm.generate(prompt, temperature=0.1)
        except:
            working = f"Solved using substitution/elimination method.\nSolution: {answer}"
        
        # Extract citations
        citations = []
        citation_pattern = r'\[Source:\s*([^\]]+)\]'
        matches = re.findall(citation_pattern, working)
        for source in set(matches):
            citations.append({"source": source, "verified": source in [r["source"] for r in retrieved]})
        
        return {
            "answer": format_answer(answer),
            "working": format_math_expression(working),
            "sources": [r["source"] for r in retrieved],
            "citations": citations,
            "tool_calls": [{
                "function": "solve_system",
                "args": {"equations": equations, "variables": variables},
                "result": solutions
            }],
            "has_context": True,
            "has_sufficient_context": True,
            "similarity_score": best_score,
            "system_solutions": solutions
        }


def main():
    """Test system solver."""
    solver = SystemSolverAgent()
    
    test_problems = [
        {
            "problem_text": "Solve 2x + y = 5 and x - y = 1",
            "topic": "linear_algebra"
        }
    ]
    
    for problem in test_problems:
        print(f"\nProblem: {problem['problem_text']}")
        print("=" * 60)
        
        route = {"route": "linear_algebra", "intent": "system_of_equations", "confidence": 0.95}
        result = solver.solve(problem, route)
        
        print(f"Answer: {result['answer']}")
        print(f"Has Context: {result.get('has_sufficient_context')}")
        print(f"Sources: {result.get('sources', [])}")


if __name__ == "__main__":
    main()
