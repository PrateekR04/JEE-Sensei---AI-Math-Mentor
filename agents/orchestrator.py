"""
Orchestrator
Coordinates all agents to solve math problems
UPGRADED: Intent-based routing to domain-specific solvers
"""

import json
import os
from typing import Dict, Any
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.parser_agent import ParserAgent
from agents.router_agent import RouterAgent
from agents.solver_agent import SolverAgent
from agents.verifier_agent import VerifierAgent
from agents.explainer_agent import ExplainerAgent

# Import new domain-specific solvers
from agents.system_solver_agent import SystemSolverAgent
from agents.calculus_solver_agent import CalculusSolverAgent
from agents.probability_solver_agent import ProbabilitySolverAgent

# Import Word Problem Modeling Agent (NEW - Day 4)
from agents.word_problem_agent import WordProblemAgent
from utils.word_problem_validator import WordProblemValidator

# Import Memory & Self-Learning Layer (NEW - Day 4)
from memory.memory_store import MemoryStore
from memory.problem_memory import ProblemMemory
from memory.feedback_handler import FeedbackHandler
from memory.user_profile import UserProfile
from memory.pattern_engine import PatternEngine
from memory.correction_rules import CorrectionRules


class MathMentorOrchestrator:
    """
    Orchestrates multi-agent pipeline for solving math problems.
    
    UPGRADED: Now supports intent-based routing to specialized solvers:
    - equation: Uses existing SolverAgent (DO NOT TOUCH)
    - system_of_equations: Uses SystemSolverAgent
    - derivative/integral/limit/optimization: Uses CalculusSolverAgent
    - probability: Uses ProbabilitySolverAgent
    """
    
    def __init__(self):
        # Initialize core agents (existing)
        self.parser = ParserAgent()
        self.router = RouterAgent()
        self.solver = SolverAgent()  # Existing equation solver - DO NOT MODIFY
        self.verifier = VerifierAgent()
        self.explainer = ExplainerAgent()
        
        # Initialize new domain-specific solvers
        self.system_solver = SystemSolverAgent()
        self.calculus_solver = CalculusSolverAgent()
        self.probability_solver = ProbabilitySolverAgent()
        
        # Initialize Word Problem Agent (NEW - Day 4)
        self.word_problem_agent = WordProblemAgent()
        
        # Initialize Memory & Self-Learning Layer (NEW - Day 4)
        self.memory_store = MemoryStore()
        self.problem_memory = ProblemMemory()
        self.feedback_handler = FeedbackHandler()
        self.user_profile = UserProfile()
        self.pattern_engine = PatternEngine()
        self.correction_rules = CorrectionRules()
        
        print("✓ Math Mentor Orchestrator initialized with memory layer")
    
    def _select_solver(self, route: Dict):
        """
        Select appropriate solver based on intent.
        
        CRITICAL: Preserves existing equation path.
        """
        intent = route.get("intent", "equation")
        
        # EXISTING PATH - DO NOT TOUCH
        if intent == "equation":
            return self.solver, "SolverAgent"
        
        # NEW DOMAIN-SPECIFIC PATHS
        elif intent == "system_of_equations":
            return self.system_solver, "SystemSolverAgent"
        
        elif intent in ["derivative", "integral", "limit", "optimization"]:
            return self.calculus_solver, "CalculusSolverAgent"
        
        elif intent == "probability":
            return self.probability_solver, "ProbabilitySolverAgent"
        
        # Fallback to existing solver
        else:
            return self.solver, "SolverAgent"
    
    def solve_problem(self, user_input: str) -> Dict[str, Any]:
        """
        Complete pipeline to solve a math problem.
        
        Args:
            user_input: User's math problem text
            
        Returns:
            Dict with answer, explanation, confidence, sources, trace
        """
        trace = []
        
        try:
            # Pre-check intent to avoid incorrect memory lookup for explanations
            # (Explanation queries should strictly match knowledge base, not previous numerical problems)
            pre_route = self.router.route({"problem_text": user_input})
            is_explanation_intent = pre_route.get("intent") == "explanation"
            
            # Step -1: Memory Lookup - Check for similar solved problems (NEW - Day 4)
            # Skip memory for explanations to avoid "Explain chain rule" matching "Explain integration of sin(x^2)"
            similar = {"found": False}
            if not is_explanation_intent:
                print("🧠 Checking memory for similar problems...")
                similar = self.check_similar_problems(user_input, threshold=0.85)
            else:
                print("🧠 Skipping memory for explanation query")
            
            if similar.get("found") and similar["best_match"]["similarity"] > 0.90:
                # High-confidence match - reuse the solution
                match = similar["best_match"]
                print(f"✨ Found highly similar problem (similarity: {match['similarity']:.0%})")
                
                metadata = match.get("metadata", {})
                
                # Extract answer - try multiple fields
                stored_answer = metadata.get("answer", "")
                solution_steps = metadata.get("solution_steps", "")
                
                # If answer is empty or indicates failure, try to extract from solution
                if not stored_answer or stored_answer in ["", "See similar problem", "Computation failed", "No answer found"]:
                    # Try to extract answer from solution_steps
                    if solution_steps:
                        # Look for common answer patterns in the solution
                        import re
                        
                        # Prioritize patterns that indicate final answer (in order of preference)
                        patterns = [
                            # "Therefore, x = 3" or "Therefore, the answer is 3"
                            r'[Tt]herefore[,:]?\s*(?:the\s+)?(?:side|answer|result|value|x|y)?\s*(?:is|=|should be)?\s*(\d+(?:\.\d+)?(?:\s*(?:cm|m|units)?)?)',
                            # "the side should be 3 cm"
                            r'(?:side|answer|result|value)\s+(?:should be|is|=)\s*(\d+(?:\.\d+)?(?:\s*(?:cm|m|units)?)?)',
                            # "x = 18/6 = 3" - look for the LAST equals sign value
                            r'x\s*=\s*[\d/\s]+\s*=\s*(\d+(?:\.\d+)?)',
                            # "we find that x = 3"
                            r'(?:we\s+)?(?:find|get|have)\s+(?:that\s+)?x\s*=\s*(\d+(?:\.\d+)?)',
                            # "is 3 cm" at the end
                            r'(?:is|be)\s+(\d+(?:\.\d+)?)\s*(?:cm|m|units)?\s*(?:for|$)',
                        ]
                        
                        for pattern in patterns:
                            match_result = re.search(pattern, solution_steps, re.IGNORECASE)
                            if match_result:
                                stored_answer = match_result.group(1).strip()
                                # Add units if not present and it's a measurement problem
                                if stored_answer.isdigit() and 'cm' in solution_steps.lower():
                                    stored_answer += " cm"
                                break
                        
                        if not stored_answer or stored_answer in ["", "Computation failed", "0"]:
                            stored_answer = "See explanation below"
                
                return {
                    "answer": stored_answer,
                    "explanation": f"**From Memory:** This problem is very similar to one I've solved before.\n\n{solution_steps}",
                    "confidence": match["similarity"],
                    "sources": ["Memory: Previously solved problem"],
                    "trace": [{"agent": "MemoryLookup", "output": {
                        "action": "reused_solution",
                        "similarity": match["similarity"],
                        "original_problem": metadata.get("problem_text", "")
                    }}],
                    "status": "success",
                    "from_memory": True,
                    "similar_problem": metadata.get("problem_text", "")
                }
            elif similar.get("found"):
                # Found similar but not confident enough - just log for strategy hints
                print(f"📚 Found related problems (best: {similar['best_match']['similarity']:.0%})")
                trace.append({
                    "agent": "MemoryLookup", 
                    "output": {"found_similar": True, "similarity": similar["best_match"]["similarity"]}
                })
            
            # Step 0: Word Problem Modeling (NEW - Day 4)
            print("📖 Checking for word problem...")
            wp_result = self.word_problem_agent.model(user_input)
            
            if wp_result.get("is_word_problem"):
                trace.append({"agent": "WordProblemAgent", "output": wp_result})
                
                # Validate the word problem model
                valid, err = WordProblemValidator.validate(wp_result)
                if not valid:
                    return {
                        "answer": "Clarification needed",
                        "explanation": err,
                        "confidence": 0.0,
                        "sources": [],
                        "trace": trace,
                        "status": "needs_clarification",
                        "modeling_trace": wp_result.get("modeling_trace", [])
                    }
                
                # Handle arithmetic word problems directly
                if wp_result.get("is_arithmetic"):
                    print("🔢 Arithmetic word problem - computing directly...")
                    return {
                        "answer": str(wp_result.get("answer", "")),
                        "explanation": wp_result.get("working", ""),
                        "confidence": wp_result.get("modeling_confidence", 0.9),
                        "sources": [],
                        "trace": trace,
                        "status": "success",
                        "intent": "arithmetic",
                        "subject": "algebra",
                        "is_word_problem": True
                    }
                
                # For equation-based word problems, create synthetic parsed input
                print(f"📐 Modeled as {wp_result.get('modeling_type')}...")
                
                # Route based on modeling type
                modeling_type = wp_result.get("modeling_type", "linear_system")
                
                if modeling_type in ["linear_system", "nonlinear_system", "number_digits"]:
                    # System of equations path
                    equations = wp_result.get("equations", [])
                    variables = list(wp_result.get("variables", {}).keys())
                    
                    # Create parsed input for system solver
                    parsed = {
                        "problem_text": user_input,
                        "topic": "linear_algebra",
                        "variables": variables,
                        "constraints": equations,
                        "normalized_equation": None,
                        "needs_clarification": False,
                        "word_problem_model": wp_result
                    }
                    route = {
                        "subject": "linear_algebra",
                        "intent": "system_of_equations",
                        "route": "linear_algebra",
                        "confidence": wp_result.get("modeling_confidence", 0.85)
                    }
                    
                    trace.append({"agent": "Parser", "output": parsed})
                    trace.append({"agent": "Router", "output": route})
                    
                    # Skip to solver selection
                    solver, solver_name = self._select_solver(route)
                    print(f"🔧 Solving with {solver_name}...")
                    
                    solution = solver.solve(parsed, route)
                    trace.append({"agent": solver_name, "output": solution})
                    
                    # Continue with verification below
                    intent = route.get("intent")
                    subject = route.get("subject")
                    
                elif modeling_type == "optimization":
                    # Optimization path - use calculus solver
                    parsed = {
                        "problem_text": user_input,
                        "topic": "calculus",
                        "optimization": True,
                        "objective": wp_result.get("objective", ""),
                        "constraint": wp_result.get("constraint", ""),
                        "optimization_type": wp_result.get("optimization_type", "maximize"),
                        "needs_clarification": False,
                        "word_problem_model": wp_result
                    }
                    route = {
                        "subject": "calculus",
                        "intent": "optimization",
                        "route": "calculus",
                        "confidence": wp_result.get("modeling_confidence", 0.85)
                    }
                    
                    trace.append({"agent": "Parser", "output": parsed})
                    trace.append({"agent": "Router", "output": route})
                    
                    solver, solver_name = self._select_solver(route)
                    print(f"🔧 Solving with {solver_name}...")
                    
                    solution = solver.solve(parsed, route)
                    trace.append({"agent": solver_name, "output": solution})
                    
                    intent = "optimization"
                    subject = "calculus"
                    
                elif modeling_type == "probability":
                    # Probability path
                    parsed = {
                        "problem_text": user_input,
                        "topic": "probability",
                        "sample_space": wp_result.get("sample_space", ""),
                        "event": wp_result.get("event", ""),
                        "trials": wp_result.get("trials", 1),
                        "needs_clarification": False,
                        "word_problem_model": wp_result
                    }
                    route = {
                        "subject": "probability",
                        "intent": "probability",
                        "route": "probability",
                        "confidence": wp_result.get("modeling_confidence", 0.85)
                    }
                    
                    trace.append({"agent": "Parser", "output": parsed})
                    trace.append({"agent": "Router", "output": route})
                    
                    solver, solver_name = self._select_solver(route)
                    print(f"🔧 Solving with {solver_name}...")
                    
                    solution = solver.solve(parsed, route)
                    trace.append({"agent": solver_name, "output": solution})
                    
                    intent = "probability"
                    subject = "probability"
                    
                else:
                    # Default: try with existing parser
                    # Fall through to standard parsing
                    parsed = None
                
                # If we handled it above, continue with verification
                if parsed is not None:
                    # Check if solver found insufficient context
                    if not solution.get("has_sufficient_context", True):
                        return {
                            "answer": solution.get("answer", "Insufficient context"),
                            "explanation": solution.get("working", "The knowledge base does not contain sufficient information."),
                            "confidence": 0.0,
                            "sources": solution.get("sources", []),
                            "trace": trace,
                            "status": "insufficient_knowledge",
                            "is_word_problem": True,
                            "word_problem_model": wp_result
                        }
                    
                    # Verify and explain
                    print("✅ Verifying solution...")
                    verification = self.verifier.verify(solution, parsed)
                    trace.append({"agent": "Verifier", "output": verification})
                    
                    print("📝 Generating explanation...")
                    explanation = self.explainer.explain(solution, verification, parsed)
                    trace.append({"agent": "Explainer", "output": explanation})
                    
                    return {
                        "answer": solution.get("answer", "No answer found"),
                        "explanation": explanation.get("explanation", ""),
                        "confidence": verification.get("confidence", 0.0),
                        "sources": solution.get("sources", []),
                        "citations": solution.get("citations", []),
                        "trace": trace,
                        "status": "success",
                        "key_concepts": explanation.get("key_concepts", []),
                        "difficulty": explanation.get("difficulty", "medium"),
                        "verification_steps": verification.get("verification_steps", []),
                        "is_correct": verification.get("is_correct", False),
                        "strict_rag_compliant": verification.get("strict_rag_compliant", False),
                        "intent": intent,
                        "subject": subject,
                        "is_word_problem": True,
                        "word_problem_model": wp_result
                    }
            
            # Standard path for non-word-problems (EXISTING - DO NOT MODIFY)
            # Step 1: Parse
            print("🔍 Parsing problem...")
            parsed = self.parser.parse(user_input)
            trace.append({"agent": "Parser", "output": parsed})
            
            if parsed.get("needs_clarification"):
                return {
                    "answer": "Clarification needed",
                    "explanation": parsed.get("normalization_error", "The problem is ambiguous. Please provide more details."),
                    "confidence": 0.0,
                    "sources": [],
                    "trace": trace,
                    "status": "needs_clarification"
                }
            
            # Step 2: Route (now with intent)
            print("🧭 Routing to topic solver...")
            route = self.router.route(parsed)
            trace.append({"agent": "Router", "output": route})
            
            intent = route.get("intent", "equation")
            subject = route.get("subject", route.get("route", "algebra"))
            
            # Handle direct calculations (NEW - for square root, factorial, etc. questions)
            if intent == "direct_calculation":
                print("🔢 Handling direct calculation...")
                
                # Use retriever to get relevant knowledge
                from rag.retriever import KnowledgeRetriever
                retriever = KnowledgeRetriever()
                relevant_docs = retriever.retrieve(user_input, k=3)
                
                # Build context from RAG (results are dicts with 'content' and 'source')
                context = "\n".join([doc.get("content", "") for doc in relevant_docs]) if relevant_docs else ""
                sources = [doc.get("source", "knowledge_base") for doc in relevant_docs] if relevant_docs else []
                
                # Use LLM to compute and explain the answer
                from llm.groq_client import GroqClient
                llm = GroqClient(model="llama-3.3-70b-versatile")
                
                prompt = f"""You are a helpful math tutor. Answer this math question directly and show your work.

Question: {user_input}

Relevant Knowledge:
{context}

Instructions:
1. Compute the exact answer
2. Show the calculation steps
3. Explain the concept briefly

Format your response as:
ANSWER: [the numerical or exact answer]
STEPS:
[step-by-step calculation]
EXPLANATION:
[brief concept explanation]"""

                try:
                    response = llm.generate(prompt, temperature=0.0)
                    
                    # Parse the response
                    answer = ""
                    steps = ""
                    explanation_text = ""
                    
                    if "ANSWER:" in response:
                        parts = response.split("ANSWER:")
                        if len(parts) > 1:
                            rest = parts[1]
                            if "STEPS:" in rest:
                                answer = rest.split("STEPS:")[0].strip()
                                rest2 = rest.split("STEPS:")[1]
                                if "EXPLANATION:" in rest2:
                                    steps = rest2.split("EXPLANATION:")[0].strip()
                                    explanation_text = rest2.split("EXPLANATION:")[1].strip()
                                else:
                                    steps = rest2.strip()
                            else:
                                answer = rest.strip()
                    else:
                        # Fallback: use entire response
                        answer = response.strip()
                    
                    full_explanation = f"**Answer:** {answer}\n\n**Calculation:**\n{steps}\n\n**Explanation:**\n{explanation_text}" if steps else response
                    
                    trace.append({"agent": "DirectCalculation", "output": {
                        "answer": answer,
                        "steps": steps,
                        "sources": sources
                    }})
                    
                    return {
                        "answer": answer,
                        "explanation": full_explanation,
                        "confidence": 0.95 if relevant_docs else 0.85,
                        "sources": sources,
                        "citations": [],
                        "trace": trace,
                        "status": "success",
                        "key_concepts": [],
                        "difficulty": "easy",
                        "verification_steps": [],
                        "is_correct": True,
                        "strict_rag_compliant": bool(relevant_docs),
                        "intent": intent,
                        "subject": subject
                    }
                    
                except Exception as e:
                    return {
                        "answer": "Error computing result",
                        "explanation": f"An error occurred: {str(e)}",
                        "confidence": 0.0,
                        "sources": [],
                        "trace": trace,
                        "status": "error",
                        "error": str(e)
                    }
            
            # Handle EXPLANATION queries (concept questions like "What is chain rule?")
            if intent == "explanation":
                print("📚 Handling concept explanation question...")
                
                # Use retriever to get relevant knowledge
                from rag.retriever import KnowledgeRetriever
                retriever = KnowledgeRetriever()
                relevant_docs = retriever.retrieve(user_input, k=5)
                
                # Build context from RAG
                context = "\n".join([doc.get("content", "") for doc in relevant_docs]) if relevant_docs else ""
                sources = [doc.get("source", "knowledge_base") for doc in relevant_docs] if relevant_docs else []
                
                # Use LLM to explain the concept
                from llm.groq_client import GroqClient
                llm = GroqClient(model="llama-3.3-70b-versatile")
                
                prompt = f"""You are a helpful JEE math tutor. Answer this concept/explanation question.

Question: {user_input}

Relevant Knowledge from textbooks:
{context}

Instructions:
1. Explain the concept clearly and thoroughly
2. Cite your sources using [Source: filename] format where applicable
3. If the question asks for an example, provide a clear worked example
4. Use proper mathematical notation:
   - Use '^' for exponents (e.g., x^2 not x2)
   - Use standard algebra notation (e.g., 2ab instead of 2*a*b, 5x instead of 5*x)
   - Use '/' for division
5. Make it easy to understand for a student

Provide your explanation:"""

                try:
                    response = llm.generate(prompt, temperature=0.3)
                    
                    trace.append({"agent": "ConceptExplainer", "output": {"response": response[:200] + "..."}})
                    
                    return {
                        "answer": "See explanation below",
                        "explanation": response,
                        "confidence": 0.9 if relevant_docs else 0.75,
                        "sources": sources,
                        "citations": [],
                        "trace": trace,
                        "status": "success",
                        "key_concepts": [],
                        "difficulty": "medium",
                        "verification_steps": [],
                        "is_correct": True,
                        "strict_rag_compliant": bool(relevant_docs),
                        "intent": intent,
                        "subject": subject
                    }
                    
                except Exception as e:
                    return {
                        "answer": "Error generating explanation",
                        "explanation": f"An error occurred: {str(e)}",
                        "confidence": 0.0,
                        "sources": [],
                        "trace": trace,
                        "status": "error",
                        "error": str(e)
                    }
            
            # Step 3: Select solver based on intent
            solver, solver_name = self._select_solver(route)
            print(f"🔧 Solving with {solver_name} (intent: {intent})...")
            
            # Step 4: Solve
            solution = solver.solve(parsed, route)
            trace.append({"agent": solver_name, "output": solution})
            
            # Check if solver found insufficient context (STRICT RAG)
            if not solution.get("has_sufficient_context", True):
                return {
                    "answer": solution.get("answer", "Insufficient context"),
                    "explanation": solution.get("working", "The knowledge base does not contain sufficient information to solve this problem."),
                    "confidence": 0.0,
                    "sources": solution.get("sources", []),
                    "trace": trace,
                    "status": "insufficient_knowledge",
                    "insufficient_context_reason": solution.get("insufficient_context_reason", "Unknown"),
                    "similarity_score": solution.get("similarity_score", 0.0),
                    "key_concepts": [],
                    "difficulty": "unknown",
                    "verification_steps": [],
                    "is_correct": False,
                    "intent": intent,
                    "subject": subject
                }
            
            # Step 5: Verify
            print("✅ Verifying solution...")
            verification = self.verifier.verify(solution, parsed)
            trace.append({"agent": "Verifier", "output": verification})
            
            # Step 6: Explain
            print("📝 Generating explanation...")
            explanation = self.explainer.explain(solution, verification, parsed)
            trace.append({"agent": "Explainer", "output": explanation})
            
            # Compile final result
            return {
                "answer": solution.get("answer", "No answer found"),
                "explanation": explanation.get("explanation", ""),
                "confidence": verification.get("confidence", 0.0),
                "sources": solution.get("sources", []),
                "citations": solution.get("citations", []),
                "trace": trace,
                "status": "success",
                "key_concepts": explanation.get("key_concepts", []),
                "difficulty": explanation.get("difficulty", "medium"),
                "verification_steps": verification.get("verification_steps", []),
                "is_correct": verification.get("is_correct", False),
                "strict_rag_compliant": verification.get("strict_rag_compliant", False),
                "intent": intent,
                "subject": subject
            }
            
        except Exception as e:
            return {
                "answer": "Error processing problem",
                "explanation": f"An error occurred: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "trace": trace,
                "status": "error",
                "error": str(e)
            }
    
    # ==================== MEMORY METHODS (NEW - Day 4) ====================
    
    def check_similar_problems(self, problem_text: str, threshold: float = 0.85) -> Dict:
        """
        Check memory for similar solved problems.
        
        Args:
            problem_text: New problem text
            threshold: Minimum similarity threshold
            
        Returns:
            Dict with similar problems or empty if none found
        """
        similar = self.memory_store.find_similar(
            problem_text, 
            n_results=3, 
            threshold=threshold
        )
        
        if similar:
            return {
                "found": True,
                "results": similar,
                "best_match": similar[0]
            }
        
        return {"found": False, "results": []}
    
    def store_solved_problem(self, problem_text: str, result: Dict, 
                              user_id: str = "default") -> str:
        """
        Store a solved problem to memory.
        
        Args:
            problem_text: Original problem text
            result: Solution result dict
            user_id: User identifier
            
        Returns:
            Problem ID
        """
        # Store to vector memory
        self.memory_store.store_problem({
            "problem_text": problem_text,
            "answer": result.get("answer", ""),
            "topic": result.get("subject", ""),
            "intent": result.get("intent", ""),
            "equations": result.get("word_problem_model", {}).get("equations", []),
            "solution_steps": result.get("explanation", ""),
            "confidence": result.get("confidence", 0.0),
            "user_id": user_id
        })
        
        # Store to problem history
        problem_id = self.problem_memory.store({
            "original_input": problem_text,
            "input_type": "text",
            "parsed_question": problem_text,
            "is_word_problem": result.get("is_word_problem", False),
            "modeled_equations": result.get("word_problem_model", {}).get("equations", []),
            "variables": result.get("word_problem_model", {}).get("variables", {}),
            "topic": result.get("subject", ""),
            "intent": result.get("intent", ""),
            "rag_sources": result.get("sources", []),
            "final_answer": result.get("answer", ""),
            "solution_steps": result.get("explanation", ""),
            "verifier_result": result.get("is_correct"),
            "confidence": result.get("confidence", 0.0),
            "execution_trace": result.get("trace", []),
            "user_id": user_id
        })
        
        # Update user profile
        topic = result.get("subject", "algebra")
        intent = result.get("intent", "equation")
        confidence = result.get("confidence", 0.0)
        
        # Record attempt (assume correct until feedback)
        self.user_profile.record_attempt(
            user_id=user_id,
            topic=topic,
            intent=intent,
            is_correct=True,  # Default, updated by feedback
            confidence=confidence,
            problem_id=problem_id
        )
        
        # Extract pattern from successful solution
        if result.get("status") == "success":
            self.pattern_engine.extract_pattern({
                "problem_text": problem_text,
                "topic": topic,
                "intent": intent,
                "modeled_equations": result.get("word_problem_model", {}).get("equations", []),
                "solution_steps": result.get("explanation", ""),
                "is_correct": True
            })
        
        return problem_id
    
    def submit_feedback(self, problem_id: str, is_correct: bool,
                        correction: str = None, error_category: str = None,
                        user_id: str = "default") -> bool:
        """
        Submit feedback for a solved problem.
        
        Args:
            problem_id: Problem ID to submit feedback for
            is_correct: Whether the solution was correct
            correction: User's correction if incorrect
            error_category: Type of error (ocr, parsing, modeling, reasoning)
            user_id: User identifier
            
        Returns:
            True if feedback was recorded
        """
        # Get problem data
        problem = self.problem_memory.get(problem_id)
        
        # Store feedback
        self.feedback_handler.store_feedback(
            problem_id=problem_id,
            is_correct=is_correct,
            correction=correction,
            error_category=error_category,
            problem_data=problem,
            user_id=user_id
        )
        
        # Update problem memory
        self.problem_memory.update_feedback(problem_id, is_correct, correction)
        
        # Update vector store
        self.memory_store.update_problem_feedback(problem_id, is_correct, correction)
        
        # Update user profile
        if problem:
            topic = problem.get("topic", "algebra")
            intent = problem.get("intent", "equation")
            confidence = problem.get("confidence", 0.0)
            
            # Re-record with correct feedback
            self.user_profile.record_attempt(
                user_id=user_id,
                topic=topic,
                intent=intent,
                is_correct=is_correct,
                confidence=confidence,
                problem_id=problem_id
            )
        
        # Learn from correction if provided
        if not is_correct and correction:
            if error_category == "ocr":
                original = problem.get("original_input", "") if problem else ""
                self.correction_rules.learn_ocr_correction(original, correction)
            elif error_category == "asr":
                original = problem.get("original_input", "") if problem else ""
                self.correction_rules.learn_asr_correction(original, correction)
        
        return True
    
    def get_user_summary(self, user_id: str = "default") -> Dict:
        """Get user profile summary."""
        return self.user_profile.get_summary(user_id)
    
    def get_strategy_hint(self, problem_text: str) -> Dict:
        """Get strategy hint for a problem."""
        return self.pattern_engine.get_strategy(problem_text)
    
    def apply_corrections(self, text: str, correction_type: str = "all") -> str:
        """Apply learned corrections to input text."""
        return self.correction_rules.apply_corrections(text, correction_type)
    
    def get_memory_stats(self) -> Dict:
        """Get memory layer statistics."""
        return {
            "problems_stored": self.problem_memory.count(),
            "vector_problems": self.memory_store.get_problem_count(),
            "patterns": self.pattern_engine.get_stats(),
            "feedback": self.feedback_handler.get_stats(),
            "correction_rules": self.correction_rules.get_stats()
        }



def main():
    """Test orchestrator with various problem types."""
    orchestrator = MathMentorOrchestrator()
    
    test_problems = [
        "Solve 2x + 3 = 5",
        "Solve 2x + y = 5 and x - y = 1",
        "Find the derivative of x^2 + 3x",
        "Probability of getting 2 heads in 3 coin flips"
    ]
    
    for problem in test_problems:
        print("\n" + "=" * 60)
        print(f"Problem: {problem}")
        print("=" * 60)
        
        result = orchestrator.solve_problem(problem)
        
        print(f"\nStatus: {result['status']}")
        print(f"Intent: {result.get('intent', 'N/A')}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result.get('confidence', 0):.0%}")


if __name__ == "__main__":
    main()

