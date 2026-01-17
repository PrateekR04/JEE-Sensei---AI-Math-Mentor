"""
Problem Memory
Database of solved problems with full execution traces.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid


class ProblemMemory:
    """
    Stores solved problems with complete execution history.
    
    Each problem record contains:
    - Original input (text/image/audio path)
    - Parsed question
    - Modeled equations (if word problem)
    - Retrieved RAG context
    - Final answer
    - Step-by-step solution
    - Verifier result
    - Confidence score
    - User feedback
    """
    
    def __init__(self, storage_dir: str = None):
        """
        Initialize problem memory.
        
        Args:
            storage_dir: Directory for persistent storage
        """
        if storage_dir is None:
            storage_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "memory_data"
            )
        
        self.storage_dir = storage_dir
        self.problems_file = os.path.join(storage_dir, "problem_history.json")
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load existing problems
        self.problems: Dict[str, Dict] = {}
        self._load()
    
    def _load(self):
        """Load problems from disk."""
        if os.path.exists(self.problems_file):
            try:
                with open(self.problems_file, 'r') as f:
                    self.problems = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load problem history: {e}")
                self.problems = {}
    
    def _save(self):
        """Save problems to disk."""
        try:
            with open(self.problems_file, 'w') as f:
                json.dump(self.problems, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving problem history: {e}")
    
    def store(self, problem_data: Dict[str, Any]) -> str:
        """
        Store a solved problem.
        
        Args:
            problem_data: Complete problem execution data
            
        Returns:
            Problem ID
        """
        problem_id = str(uuid.uuid4())[:8]
        
        record = {
            "id": problem_id,
            "timestamp": datetime.now().isoformat(),
            
            # Input
            "original_input": problem_data.get("original_input", ""),
            "input_type": problem_data.get("input_type", "text"),  # text/image/audio
            "parsed_question": problem_data.get("parsed_question", ""),
            
            # Word problem modeling
            "is_word_problem": problem_data.get("is_word_problem", False),
            "modeled_equations": problem_data.get("modeled_equations", []),
            "variables": problem_data.get("variables", {}),
            
            # Routing
            "topic": problem_data.get("topic", ""),
            "intent": problem_data.get("intent", ""),
            
            # RAG context
            "rag_context": problem_data.get("rag_context", []),
            "rag_sources": problem_data.get("rag_sources", []),
            
            # Solution
            "final_answer": problem_data.get("final_answer", ""),
            "solution_steps": problem_data.get("solution_steps", ""),
            "explanation": problem_data.get("explanation", ""),
            
            # Verification
            "verifier_result": problem_data.get("verifier_result", None),
            "is_correct": problem_data.get("is_correct", None),
            "confidence": problem_data.get("confidence", 0.0),
            
            # Feedback (to be filled later)
            "user_feedback": None,
            "user_correction": None,
            "feedback_timestamp": None,
            
            # User info
            "user_id": problem_data.get("user_id", "default"),
            
            # Trace
            "execution_trace": problem_data.get("execution_trace", [])
        }
        
        self.problems[problem_id] = record
        self._save()
        
        return problem_id
    
    def get(self, problem_id: str) -> Optional[Dict]:
        """Get a problem by ID."""
        return self.problems.get(problem_id)
    
    def update_feedback(self, problem_id: str, is_correct: bool, 
                        correction: str = None) -> bool:
        """
        Update feedback for a problem.
        
        Args:
            problem_id: ID of the problem
            is_correct: Whether the solution was correct
            correction: User's correction if incorrect
            
        Returns:
            True if updated successfully
        """
        if problem_id not in self.problems:
            return False
        
        self.problems[problem_id]["user_feedback"] = "correct" if is_correct else "incorrect"
        self.problems[problem_id]["feedback_timestamp"] = datetime.now().isoformat()
        
        if correction:
            self.problems[problem_id]["user_correction"] = correction
        
        self._save()
        return True
    
    def get_by_user(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get problems for a specific user."""
        user_problems = [
            p for p in self.problems.values() 
            if p.get("user_id") == user_id
        ]
        
        # Sort by timestamp (newest first)
        user_problems.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return user_problems[:limit]
    
    def get_by_topic(self, topic: str, limit: int = 50) -> List[Dict]:
        """Get problems for a specific topic."""
        topic_problems = [
            p for p in self.problems.values()
            if p.get("topic", "").lower() == topic.lower()
        ]
        
        topic_problems.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return topic_problems[:limit]
    
    def get_incorrect(self, user_id: str = None) -> List[Dict]:
        """Get problems that were marked incorrect."""
        incorrect = []
        for p in self.problems.values():
            if p.get("user_feedback") == "incorrect":
                if user_id is None or p.get("user_id") == user_id:
                    incorrect.append(p)
        return incorrect
    
    def get_stats(self, user_id: str = None) -> Dict:
        """Get statistics about stored problems."""
        if user_id:
            problems = [p for p in self.problems.values() if p.get("user_id") == user_id]
        else:
            problems = list(self.problems.values())
        
        total = len(problems)
        if total == 0:
            return {"total": 0, "correct": 0, "incorrect": 0, "pending": 0, "accuracy": 0.0}
        
        correct = sum(1 for p in problems if p.get("user_feedback") == "correct")
        incorrect = sum(1 for p in problems if p.get("user_feedback") == "incorrect")
        pending = total - correct - incorrect
        
        reviewed = correct + incorrect
        accuracy = correct / reviewed if reviewed > 0 else 0.0
        
        # Topic breakdown
        topic_stats = {}
        for p in problems:
            topic = p.get("topic", "unknown")
            if topic not in topic_stats:
                topic_stats[topic] = {"total": 0, "correct": 0}
            topic_stats[topic]["total"] += 1
            if p.get("user_feedback") == "correct":
                topic_stats[topic]["correct"] += 1
        
        return {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "pending": pending,
            "accuracy": accuracy,
            "topic_breakdown": topic_stats
        }
    
    def get_recent(self, limit: int = 10) -> List[Dict]:
        """Get most recent problems."""
        all_problems = list(self.problems.values())
        all_problems.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return all_problems[:limit]
    
    def count(self) -> int:
        """Get total number of stored problems."""
        return len(self.problems)


def main():
    """Test problem memory."""
    memory = ProblemMemory()
    
    # Store a test problem
    problem_id = memory.store({
        "original_input": "Solve 2x + 3 = 7",
        "input_type": "text",
        "parsed_question": "Solve the equation 2x + 3 = 7 for x",
        "topic": "algebra",
        "intent": "equation",
        "final_answer": "x = 2",
        "confidence": 0.95,
        "user_id": "test_user"
    })
    
    print(f"Stored problem: {problem_id}")
    
    # Retrieve
    problem = memory.get(problem_id)
    print(f"Retrieved: {problem['original_input']} → {problem['final_answer']}")
    
    # Update feedback
    memory.update_feedback(problem_id, is_correct=True)
    
    # Stats
    stats = memory.get_stats()
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
