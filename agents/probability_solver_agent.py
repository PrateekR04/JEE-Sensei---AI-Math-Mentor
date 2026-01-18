"""
Probability Solver Agent
Solves probability problems using combinatorics and Strict RAG enforcement
"""

import json
import os
import re
from typing import Dict, Any
import sys
from math import factorial, comb, perm
from fractions import Fraction

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.groq_client import GroqClient
from rag.retriever import KnowledgeRetriever
from agents.probability_parser_agent import ProbabilityParserAgent


class ProbabilitySolverAgent:
    """
    Solves probability problems using combinatorics.
    Enforces Strict RAG mode.
    """
    
    def __init__(self, similarity_threshold: float = 0.35):
        self.llm = GroqClient(model="llama-3.3-70b-versatile")
        self.retriever = KnowledgeRetriever()
        self.parser = ProbabilityParserAgent()
        self.similarity_threshold = similarity_threshold
    
    def _binomial_probability(self, n: int, k: int, p: float = 0.5) -> float:
        """
        Calculate binomial probability: P(X = k) = C(n,k) * p^k * (1-p)^(n-k)
        """
        return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    
    def _binomial_at_least(self, n: int, k: int, p: float = 0.5) -> float:
        """
        Calculate P(X >= k) for binomial distribution.
        """
        total = sum(self._binomial_probability(n, i, p) for i in range(k, n + 1))
        return total
    
    def _solve_coin_flip(self, params: Dict) -> Dict[str, Any]:
        """
        Solve coin flip probability problems.
        """
        if "at_least" in params:
            k = params["at_least"]
            n = params["total_trials"]
            p = params.get("probability_per_trial", 0.5)
            prob = self._binomial_at_least(n, k, p)
            
            return {
                "success": True,
                "probability": prob,
                "fraction": str(Fraction(prob).limit_denominator(1000)),
                "formula": f"P(X >= {k}) = Σ C({n},i) * 0.5^i * 0.5^({n}-i) for i={k} to {n}",
                "method": "binomial_at_least"
            }
        
        elif "favorable" in params and "total_trials" in params:
            k = params["favorable"]
            n = params["total_trials"]
            p = params.get("probability_per_trial", 0.5)
            prob = self._binomial_probability(n, k, p)
            
            # Also express as fraction
            combinations = comb(n, k)
            total_outcomes = 2 ** n
            
            return {
                "success": True,
                "probability": prob,
                "fraction": f"{combinations}/{total_outcomes}",
                "simplified": str(Fraction(combinations, total_outcomes)),
                "formula": f"P(X = {k}) = C({n},{k}) * 0.5^{k} * 0.5^{n-k} = {combinations}/{total_outcomes}",
                "method": "binomial_exact"
            }
        
        return {"success": False, "error": "Could not determine coin flip parameters"}
    
    def _solve_dice(self, params: Dict) -> Dict[str, Any]:
        """
        Solve dice probability problems.
        """
        dice_count = params.get("dice_count", 2)
        target_sum = params.get("target_sum")
        sides = params.get("sides", 6)
        
        if target_sum is None:
            return {"success": False, "error": "No target sum specified"}
        
        if dice_count == 2 and sides == 6:
            # Count favorable outcomes for sum with 2 dice
            favorable = 0
            outcomes = []
            for d1 in range(1, 7):
                for d2 in range(1, 7):
                    if d1 + d2 == target_sum:
                        favorable += 1
                        outcomes.append((d1, d2))
            
            total = 36
            prob = favorable / total
            
            return {
                "success": True,
                "probability": prob,
                "fraction": f"{favorable}/{total}",
                "simplified": str(Fraction(favorable, total)),
                "favorable_outcomes": outcomes,
                "formula": f"P(sum = {target_sum}) = {favorable}/36",
                "method": "enumeration"
            }
        
        return {"success": False, "error": f"Dice problem not supported: {dice_count} dice with {sides} sides"}
    
    def _solve_cards(self, params: Dict) -> Dict[str, Any]:
        """
        Solve card probability problems.
        """
        deck_size = params.get("deck_size", 52)
        favorable = params.get("favorable")
        event = params.get("event", "unknown")
        
        if favorable is None:
            return {"success": False, "error": "Could not determine favorable outcomes"}
        
        prob = favorable / deck_size
        
        return {
            "success": True,
            "probability": prob,
            "fraction": f"{favorable}/{deck_size}",
            "simplified": str(Fraction(favorable, deck_size)),
            "formula": f"P({event}) = {favorable}/{deck_size}",
            "method": "basic_probability"
        }
    
    def _solve_basic(self, params: Dict) -> Dict[str, Any]:
        """
        Solve basic probability: P = favorable / total.
        """
        if "favorable" in params and "total" in params:
            favorable = params["favorable"]
            total = params["total"]
            prob = favorable / total
            
            return {
                "success": True,
                "probability": prob,
                "fraction": f"{favorable}/{total}",
                "simplified": str(Fraction(favorable, total)),
                "formula": f"P = favorable/total = {favorable}/{total}",
                "method": "basic_probability"
            }
        
        return {"success": False, "error": "Could not determine favorable and total outcomes"}
    
    def solve(self, parsed_problem: Dict, route: Dict) -> Dict[str, Any]:
        """
        Solve probability problem with Strict RAG enforcement.
        """
        problem_text = parsed_problem.get("problem_text", "")
        
        # Step 1: Parse the probability problem
        prob_parsed = self.parser.parse(problem_text)
        
        if prob_parsed.get("needs_clarification"):
            return {
                "answer": "Could not parse probability problem",
                "working": prob_parsed.get("error", "Unknown error"),
                "sources": [],
                "citations": [],
                "tool_calls": [],
                "has_context": False,
                "has_sufficient_context": False
            }
        
        scenario = prob_parsed.get("scenario", "general")
        params = prob_parsed.get("params", {})
        
        # Step 2: Retrieve relevant knowledge
        retrieval_result = self.retriever.retrieve_with_threshold(
            f"probability {scenario} calculation {problem_text}",
            k=5,
            threshold=self.similarity_threshold
        )
        
        has_sufficient_context = retrieval_result["has_sufficient_context"]
        retrieved = retrieval_result["results"]
        best_score = retrieval_result["best_score"]
        
        if not has_sufficient_context:
            return {
                "answer": "INSUFFICIENT CONTEXT",
                "working": f"Knowledge base lacks probability information. Best similarity: {best_score:.3f}",
                "sources": [],
                "citations": [],
                "tool_calls": [],
                "has_context": False,
                "has_sufficient_context": False,
                "insufficient_context_reason": retrieval_result.get("reason", "No relevant knowledge found")
            }
        
        # Step 3: Compute the probability
        if scenario == "coin":
            compute_result = self._solve_coin_flip(params)
        elif scenario == "dice":
            compute_result = self._solve_dice(params)
        elif scenario == "cards":
            compute_result = self._solve_cards(params)
        else:
            compute_result = self._solve_basic(params)
        
        if not compute_result.get("success"):
            return {
                "answer": "Could not compute probability",
                "working": compute_result.get("error", "Unknown computation error"),
                "sources": [r["source"] for r in retrieved],
                "citations": [],
                "tool_calls": [],
                "has_context": True,
                "has_sufficient_context": True,
                "error": compute_result.get("error")
            }
        
        # Format answer
        prob = compute_result["probability"]
        fraction = compute_result.get("simplified", compute_result.get("fraction", str(prob)))
        answer = f"{fraction} (≈ {prob:.4f})"
        
        # Step 4: Generate explanation using LLM with RAG
        context_parts = [f"[Source: {r['source']}]\n{r['content']}" for r in retrieved]
        context = "\n\n".join(context_parts)
        
        prompt = f"""Explain this probability calculation step-by-step.
Use ONLY the formulas from the retrieved knowledge.
Cite sources using [Source: filename] format.

Problem: {problem_text}
Scenario: {scenario}
Computed Result: {answer}
Formula Used: {compute_result.get('formula', 'N/A')}

Retrieved Knowledge:
{context}

Provide a clear step-by-step explanation with citations.
At the END, you MUST write on its own line: "**FINAL ANSWER:** [probability as fraction and decimal]"
For example: "**FINAL ANSWER:** 1/4 (0.25 or 25%)""""

        try:
            working = self.llm.generate(prompt, temperature=0.1)
        except:
            working = f"Using {compute_result.get('method', 'probability formula')}:\n{compute_result.get('formula', '')}\nResult: {answer}"
        
        # Extract citations
        citations = []
        citation_pattern = r'\[Source:\s*([^\]]+)\]'
        matches = re.findall(citation_pattern, working)
        for source in set(matches):
            citations.append({"source": source, "verified": source in [r["source"] for r in retrieved]})
        
        return {
            "answer": answer,
            "working": working,
            "sources": [r["source"] for r in retrieved],
            "citations": citations,
            "tool_calls": [{
                "function": "probability",
                "args": {"scenario": scenario, "params": params},
                "result": compute_result
            }],
            "has_context": True,
            "has_sufficient_context": True,
            "similarity_score": best_score,
            "probability_value": prob,
            "probability_fraction": fraction
        }


def main():
    """Test probability solver."""
    solver = ProbabilitySolverAgent()
    
    test_problems = [
        {"problem_text": "What is the probability of getting 2 heads in 3 coin flips?", "topic": "probability"},
        {"problem_text": "Probability of rolling a sum of 7 with two dice", "topic": "probability"},
        {"problem_text": "What is the probability of drawing an ace from a standard deck?", "topic": "probability"},
    ]
    
    for problem in test_problems:
        print(f"\nProblem: {problem['problem_text']}")
        print("=" * 60)
        
        route = {"route": "probability", "intent": "probability", "confidence": 0.95}
        result = solver.solve(problem, route)
        
        print(f"Answer: {result['answer']}")
        print(f"Has Context: {result.get('has_sufficient_context')}")


if __name__ == "__main__":
    main()
