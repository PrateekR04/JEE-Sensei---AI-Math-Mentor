"""
Solver Agent
Solves math problems using STRICT RAG mode with Groq
CRITICAL: Only uses retrieved knowledge, never internal model knowledge
"""

import json
import os
import re
from typing import Dict, List, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.groq_client import GroqClient
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retriever import KnowledgeRetriever
from tools.calculator import Calculator
from utils.math_formatter import format_answer, format_math_expression


class SolverAgent:
    """
    Solves math problems using STRICT RAG mode.
    
    STRICT RAG RULES:
    - Only uses retrieved knowledge chunks
    - Never uses model's internal knowledge
    - All formulas must be cited from source documents
    - Refuses to answer if context is insufficient
    """
    
    def __init__(self, similarity_threshold: float = 0.35):
        self.llm = GroqClient(model="llama-3.3-70b-versatile")
        self.retriever = KnowledgeRetriever()
        self.calculator = Calculator()
        self.similarity_threshold = similarity_threshold  # Cosine similarity threshold (0-1)
    
    def _has_sufficient_concepts(self, retrieved_chunks: List[Dict], problem_type: str) -> bool:
        """
        Check if retrieved chunks contain sufficient concepts for the problem type.
        Uses semantic concept detection instead of literal formula matching.
        
        Args:
            retrieved_chunks: List of retrieved document chunks
            problem_type: Type of problem (algebra, calculus, etc.)
            
        Returns:
            True if sufficient concepts found
        """
        # Combine all retrieved content
        combined_text = " ".join([chunk["content"].lower() for chunk in retrieved_chunks])
        
        # Define required concepts by problem type
        concept_sets = {
            "algebra": [
                ["linear", "equation"],
                ["coefficient"],
                ["subtract", "add", "divide", "multiply"],
                ["isolate", "solve"],
                ["variable", "unknown"]
            ],
            "calculus": [
                ["derivative", "differentiate"],
                ["limit"],
                ["integral", "integrate"],
                ["rate", "change"]
            ],
            "probability": [
                ["probability", "chance"],
                ["event", "outcome"],
                ["independent", "dependent"]
            ]
        }
        
        # Get concepts for this problem type
        required_concepts = concept_sets.get(problem_type.lower(), [])
        
        # Count how many concept groups are present
        concepts_found = 0
        for concept_group in required_concepts:
            if any(concept in combined_text for concept in concept_group):
                concepts_found += 1
        
        # Need at least 2 concept groups for sufficient context
        return concepts_found >= 2
    
    def solve(self, parsed_problem: Dict, route: Dict) -> Dict[str, Any]:
        """
        Solve math problem using STRICT RAG mode.
        
        Args:
            parsed_problem: Parsed problem from ParserAgent
            route: Route from RouterAgent
            
        Returns:
            Dict with answer, working, sources, citations, has_sufficient_context
        """
        problem_text = parsed_problem.get("problem_text", "")
        topic = route.get("route", "algebra")
        
        # Step 1: Retrieve relevant knowledge with threshold
        retrieval_result = self.retriever.retrieve_with_threshold(
            problem_text, 
            k=5,  # Get more chunks for better coverage
            threshold=self.similarity_threshold
        )
        
        has_sufficient_context = retrieval_result["has_sufficient_context"]
        retrieved = retrieval_result["results"]
        best_score = retrieval_result["best_score"]
        
        # Step 2: Check if we have sufficient context (STRICT RAG ENFORCEMENT)
        if not has_sufficient_context:
            reason = retrieval_result.get("reason", "No relevant knowledge found")
            return {
                "answer": f"INSUFFICIENT CONTEXT: {reason}",
                "working": f"The knowledge base does not contain sufficient information about this {topic} problem.\\n\\nBest similarity score: {best_score:.3f} (threshold: {self.similarity_threshold})\\n\\nTo solve this problem, please add relevant documents to the knowledge base covering: {topic}",
                "sources": [],
                "citations": [],
                "tool_calls": [],
                "has_context": False,
                "has_sufficient_context": False,
                "insufficient_context_reason": reason,
                "similarity_score": best_score
            }
        
        # Step 3: Build context from retrieved chunks
        context_parts = []
        for i, r in enumerate(retrieved, 1):
            context_parts.append(
                f"[CHUNK {i}] Source: {r['source']} (Score: {r['score']:.3f})\\n{r['content']}"
            )
        context = "\\n\\n".join(context_parts)
        
        # Step 4: Create STRICT RAG prompt
        prompt = f"""STRICT RAG MODE - CRITICAL RULES:

1. You can ONLY use information from the "Retrieved Knowledge" section below
2. You MUST NOT use any formulas or knowledge from your training data
3. Every formula you use MUST be cited with [Source: filename]
4. If the retrieved knowledge doesn't contain the needed formula, respond with:
   "INSUFFICIENT CONTEXT: The knowledge base does not contain information about [specific topic/formula]"
5. Do not make assumptions or use unstated formulas
6. All reasoning steps must reference the source documents

Problem Type: {topic}
Problem: {problem_text}

Retrieved Knowledge (ONLY SOURCE OF TRUTH):
{context}

Instructions:
1. Read the retrieved knowledge carefully
2. Identify which formulas/concepts from the retrieved knowledge apply to this problem
3. For each formula you use, cite it as [Source: filename]
4. For equations, use this format to call calculator: CALC[solve("equation", "variable")]
5. Show your working step-by-step with citations
6. At the END, you MUST write on its own line: "**FINAL ANSWER:** [your answer]"

If any required formula is missing from the retrieved knowledge, stop and respond with "INSUFFICIENT CONTEXT: Missing [formula name]"

Solve the problem now using ONLY the retrieved knowledge above:"""

        try:
            solution_text = self.llm.generate(prompt, temperature=0.1)
            
            
            # Step 5: Check if LLM indicated insufficient context
            if "INSUFFICIENT CONTEXT" in solution_text:
                return {
                    "answer": solution_text.split("\\n")[0],
                    "working": solution_text,
                    "sources": [r["source"] for r in retrieved],
                    "citations": [],
                    "tool_calls": [],
                    "has_context": True,
                    "has_sufficient_context": False,
                    "insufficient_context_reason": "LLM determined context is insufficient",
                    "similarity_score": best_score
                }
            
            # Step 6: Extract citations from solution
            citations = self._extract_citations(solution_text, retrieved)
            
            # Step 7: Execute calculator calls
            tool_calls = []
            if "CALC[" in solution_text:
                solution_text, tool_calls = self._execute_calculator_calls(solution_text)
            
            # Step 8: Extract answer
            answer = self._extract_answer(solution_text)
            
            return {
                "answer": format_answer(answer),
                "working": format_math_expression(solution_text),
                "sources": [r["source"] for r in retrieved],
                "citations": citations,
                "tool_calls": tool_calls,
                "has_context": True,
                "has_sufficient_context": True,
                "similarity_score": best_score
            }
            
        except Exception as e:
            return {
                "answer": "Error solving problem",
                "working": f"Error: {str(e)}",
                "sources": [],
                "citations": [],
                "tool_calls": [],
                "error": str(e),
                "has_context": False,
                "has_sufficient_context": False
            }
    
    def _extract_citations(self, solution_text: str, retrieved: List[Dict]) -> List[Dict[str, str]]:
        """
        Extract citations from solution text.
        
        Returns:
            List of dicts with 'formula' and 'source'
        """
        citations = []
        
        # Pattern: [Source: filename]
        citation_pattern = r'\[Source:\s*([^\]]+)\]'
        matches = re.findall(citation_pattern, solution_text)
        
        # Get unique sources cited
        cited_sources = list(set(matches))
        
        # Verify citations match retrieved sources
        retrieved_sources = [r["source"] for r in retrieved]
        
        for cited_source in cited_sources:
            if cited_source in retrieved_sources:
                citations.append({
                    "source": cited_source,
                    "verified": True
                })
            else:
                citations.append({
                    "source": cited_source,
                    "verified": False,
                    "warning": "Source not in retrieved documents"
                })
        
        return citations
    
    def _execute_calculator_calls(self, solution_text: str) -> tuple:
        """Execute calculator calls and return updated text and tool calls."""
        tool_calls = []
        
        # Extract and execute calculator calls
        calc_pattern = r'CALC\[(.*?)\]'
        matches = re.findall(calc_pattern, solution_text)
        
        for match in matches:
            try:
                # Execute calculator command
                if "solve(" in match:
                    # Extract equation and variable
                    parts = match.split(',')
                    if len(parts) < 2:
                        # Malformed CALC call, skip it
                        tool_calls.append({
                            "function": "solve_equation",
                            "args": match,
                            "error": "Malformed CALC call - expected 'solve(equation, variable)'"
                        })
                        continue
                    
                    equation = parts[0].split('(')[1].strip('\'\"')
                    variable = parts[1].strip().strip('"\' ').rstrip(')').strip('\'\"')
                    
                    # Clean equation string - remove instruction words the LLM might include
                    equation = equation.strip().replace('\\', '').replace('\n', ' ')
                    
                    # Remove common instruction words that LLM might include
                    instruction_words = [
                        'solve', 'find', 'determine', 'calculate', 
                        'evaluate', 'compute', 'the equation', 'equation',
                        'the', 'for', 'when', 'where'
                    ]
                    equation_lower = equation.lower()
                    for word in instruction_words:
                        if equation_lower.startswith(word + ' '):
                            equation = equation[len(word):].strip()
                            equation_lower = equation.lower()
                    
                    # Normalize implicit multiplication (2x → 2*x)
                    equation = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation)
                    
                    result = self.calculator.solve_equation(equation, variable)
                    tool_calls.append({
                        "function": "solve_equation",
                        "args": {"equation": equation, "variable": variable},
                        "result": result
                    })
                    # Replace in solution
                    solution_text = solution_text.replace(
                        f"CALC[{match}]",
                        f"{result}"
                    )
            except Exception as e:
                # Log error but don't crash - continue with other calculator calls
                error_msg = str(e)
                tool_calls.append({
                    "function": "calculator",
                    "args": match,
                    "error": error_msg
                })
                # Replace CALC call with error message in solution
                solution_text = solution_text.replace(
                    f"CALC[{match}]",
                    f"[Calculator Error: {error_msg}]"
                )
        
        return solution_text, tool_calls
    
    def _extract_answer(self, solution_text: str) -> str:
        """Extract final answer from solution text."""
        # Pattern 1: Look for **FINAL ANSWER:** format (preferred)
        final_answer_match = re.search(r'\*\*FINAL ANSWER:\*\*\s*(.+?)(?:\n|$)', solution_text, re.IGNORECASE)
        if final_answer_match:
            return final_answer_match.group(1).strip()
        
        # Pattern 2: Look for "The final answer is:" format
        final_answer_match = re.search(r'[Tt]he final answer is[:\s]+([^\n]+)', solution_text)
        if final_answer_match:
            return final_answer_match.group(1).strip().rstrip('.')
        
        # Pattern 3: Look for "FINAL ANSWER:" without bold
        final_answer_match = re.search(r'FINAL ANSWER[:\s]+([^\n]+)', solution_text, re.IGNORECASE)
        if final_answer_match:
            return final_answer_match.group(1).strip()
        
        lines = solution_text.split('\n')
        
        # Pattern 4: Look for "Answer:" or similar keywords
        for line in reversed(lines):
            if any(keyword in line.lower() for keyword in ['answer:', 'solution:', 'result:']):
                parts = line.split(':', 1)
                if len(parts) > 1:
                    return parts[1].strip()
        
        # Fallback: return last non-empty line
        for line in reversed(lines):
            if line.strip() and not line.strip().startswith('[Source:'):
                return line.strip()
        
        return "See explanation above"


def main():
    """Test solver agent with strict RAG mode."""
    solver = SolverAgent()
    
    # Test 1: Problem that should have context
    parsed = {
        "problem_text": "Solve 2x + 3 = 7 for x",
        "topic": "algebra",
        "variables": ["x"],
        "constraints": ["2*x + 3 = 7"]
    }
    
    route = {
        "route": "algebra",
        "confidence": 0.95
    }
    
    print("=" * 60)
    print("Test 1: Linear equation (should have context)")
    print("=" * 60)
    result = solver.solve(parsed, route)
    print(f"Has sufficient context: {result.get('has_sufficient_context')}")
    print(f"Answer: {result.get('answer')}")
    print(f"Citations: {result.get('citations')}")
    print(f"Sources: {result.get('sources')}")
    
    # Test 2: Problem outside KB
    parsed2 = {
        "problem_text": "Solve the differential equation dy/dx + 2y = e^x",
        "topic": "calculus",
        "variables": ["y"],
        "constraints": []
    }
    
    route2 = {
        "route": "calculus",
        "confidence": 0.95
    }
    
    print("\\n" + "=" * 60)
    print("Test 2: Differential equation (likely outside KB)")
    print("=" * 60)
    result2 = solver.solve(parsed2, route2)
    print(f"Has sufficient context: {result2.get('has_sufficient_context')}")
    print(f"Answer: {result2.get('answer')}")


if __name__ == "__main__":
    main()
